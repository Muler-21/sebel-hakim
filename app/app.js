/* Sebel Hakim (ሰብል ሐኪም) — offline crop disease diagnosis in the browser. Created by Mulusew Kassa Bitew (Muler). */
"use strict";

const ORT_CDN = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/";
const NOT_SURE = 0.5;          // calibrated confidence below this → "not sure"
const $ = (s, r = document) => r.querySelector(s);
const el = (tag, attrs = {}, ...kids) => {
  const e = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === "class") e.className = v; else if (k === "text") e.textContent = v;
    else if (k.startsWith("on")) e.addEventListener(k.slice(2), v); else e.setAttribute(k, v);
  }
  for (const k of kids.flat()) if (k != null) e.append(k);
  return e;
};

const S = {
  lang: "en", kb: null, dis: {}, crops: {}, meta: null, session: null, loading: null,
  modelClasses: [], last: null, offline: false,
};

// ------------------------------------------------------------- language
function t(key) {
  const L = I18N[S.lang] || I18N.en;
  return key.split(".").reduce((o, k) => (o || {})[k], L) ?? key.split(".").reduce((o, k) => (o || {})[k], I18N.en) ?? key;
}
const nm = (obj) => (obj && (obj[S.lang] || obj.en)) || "";
function applyLang() {
  document.documentElement.lang = S.lang;
  document.querySelectorAll("[data-i]").forEach((n) => { n.textContent = t(n.dataset.i); });
  $("#q").placeholder = t("search");
  fillCropSelects();
  if (S.last) renderResult(S.last);
  renderLibrary();
  renderAbout();
  setModelState();
}

// ------------------------------------------------------------- data
async function loadKB() {
  S.kb = await (await fetch("kb.json")).json();
  for (const c of S.kb.crops) S.crops[c.id] = c;
  for (const d of S.kb.diseases) S.dis[d.id] = d;
}

async function loadMeta() {
  try {
    const r = await fetch("model/labels.json", { cache: "no-cache" });
    if (!r.ok) throw 0;
    S.meta = await r.json();
    S.modelClasses = S.meta.classes.map((c) => c.id);
  } catch { S.meta = null; }
}

function loadSession() {
  if (S.session) return Promise.resolve(S.session);
  if (S.loading) return S.loading;
  S.loading = (async () => {
    setModelState("loading");
    while (!window.ort) await new Promise((r) => setTimeout(r, 50));
    ort.env.wasm.wasmPaths = ORT_CDN;
    ort.env.wasm.numThreads = self.crossOriginIsolated ? Math.min(4, navigator.hardwareConcurrency || 1) : 1;
    const file = (S.meta.files && (S.meta.files.int8 || S.meta.files.fp32)) || "model.onnx";
    S.session = await ort.InferenceSession.create("model/" + file,
      { executionProviders: ["wasm"], graphOptimizationLevel: "all" });
    // warm-up run so the first real diagnosis is already fast
    const sz = S.meta.img;
    await S.session.run({ [S.session.inputNames[0]]: new ort.Tensor("float32", new Float32Array(VIEWS * 3 * sz * sz), [VIEWS, 3, sz, sz]) });
    setModelState("ready");
    return S.session;
  })().catch((e) => { S.loading = null; setModelState("error"); throw e; });
  return S.loading;
}

function setModelState(s) {
  const n = $("#model-state");
  const shoot = $(".shoot");
  n.classList.remove("bad");
  const off = $("#a-offline");
  if (off) off.textContent = S.offline ? "✓ " + t("offline_ready") : t("offline_pending");
  if (!S.meta) { n.textContent = t("no_model"); shoot.classList.add("disabled"); return; }
  shoot.classList.remove("disabled");
  if (s === "loading") { n.textContent = t("loading_model"); return; }
  if (s === "error") { n.textContent = t("model_error"); n.classList.add("bad"); return; }
  const parts = [];
  if (S.meta.placeholder) { parts.push(t("placeholder")); n.classList.add("bad"); }
  if (S.offline) parts.push("✓ " + t("offline_ready"));
  n.textContent = parts.join(" ");
}

// ask the service worker whether the app, AI engine and model are all stored for offline use
function checkOffline() {
  if (!navigator.serviceWorker || !navigator.serviceWorker.controller) return;
  navigator.serviceWorker.controller.postMessage("offline-status");
}

// ------------------------------------------------------------- image → tensor
// Turn the photo into 4 views the model looks at together (test-time augmentation):
// the standard centre crop (as in training), its mirror image, a closer look at the centre,
// and the whole frame. Averaging them is more accurate than one look, at almost no extra time.
const VIEWS = 4;
async function toTensor(file) {
  const size = S.meta.img;
  let bmp = await createImageBitmap(file, { imageOrientation: "from-image" });
  if (Math.min(bmp.width, bmp.height) > size * 4) {        // shrink big camera photos once (faster)
    const f = (size * 4) / Math.min(bmp.width, bmp.height);
    const small = await createImageBitmap(bmp, { resizeWidth: Math.round(bmp.width * f),
      resizeHeight: Math.round(bmp.height * f), resizeQuality: "high" });
    bmp.close && bmp.close(); bmp = small;
  }
  const c = document.createElement("canvas");
  c.width = size; c.height = size;
  const g = c.getContext("2d", { willReadFrequently: true });
  g.imageSmoothingQuality = "high";
  const n = size * size, out = new Float32Array(VIEWS * 3 * n);
  const [m0, m1, m2] = S.meta.mean, [s0, s1, s2] = S.meta.std;
  const draw = (k, shortSide, mirror, squash) => {
    g.setTransform(1, 0, 0, 1, 0, 0);
    g.clearRect(0, 0, size, size);
    if (mirror) g.setTransform(-1, 0, 0, 1, size, 0);
    if (squash) g.drawImage(bmp, 0, 0, size, size);
    else {
      const sc = shortSide / Math.min(bmp.width, bmp.height);
      const w = bmp.width * sc, h = bmp.height * sc;
      g.drawImage(bmp, (size - w) / 2, (size - h) / 2, w, h);
    }
    const px = g.getImageData(0, 0, size, size).data, o = k * 3 * n;
    for (let i = 0; i < n; i++) {
      out[o + i] = (px[i * 4] / 255 - m0) / s0;
      out[o + n + i] = (px[i * 4 + 1] / 255 - m1) / s1;
      out[o + 2 * n + i] = (px[i * 4 + 2] / 255 - m2) / s2;
    }
  };
  draw(0, size * 1.14, false, false);   // standard view (matches training)
  draw(1, size * 1.14, true, false);    // mirrored
  draw(2, size * 1.45, false, false);   // closer look at the centre
  draw(3, 0, false, true);              // the whole photo
  bmp.close && bmp.close();
  return new ort.Tensor("float32", out, [VIEWS, 3, size, size]);
}

// average the views' logits into one answer
function mergeViews(data, views) {
  const C = data.length / views, m = new Float32Array(C);
  for (let v = 0; v < views; v++) for (let c = 0; c < C; c++) m[c] += data[v * C + c] / views;
  return m;
}

function softmax(logits, temp, allowed) {
  let max = -Infinity;
  for (let i = 0; i < logits.length; i++) if (allowed[i] && logits[i] > max) max = logits[i];
  const p = new Float32Array(logits.length);
  let sum = 0;
  for (let i = 0; i < logits.length; i++) if (allowed[i]) { p[i] = Math.exp((logits[i] - max) / temp); sum += p[i]; }
  for (let i = 0; i < p.length; i++) p[i] /= sum;
  return p;
}

// Rank diseases and plants from the raw model output. `crop` restricts the diagnosis to one plant.
// which plant each of the 4 views points to; if they disagree, the photo is probably something the AI was
// not trained on (a fruit, a whole tree, a new crop), so the app must not sound sure
function viewCrops(raw) {
  if (!raw) return [];
  const C = S.modelClasses.length, res = [];
  for (let v = 0; v < raw.length / C; v++) {
    const sl = raw.subarray(v * C, (v + 1) * C);
    const pr = softmax(sl, S.meta.temperature || 1, S.modelClasses.map(() => true)), by = {};
    S.modelClasses.forEach((id, i) => { const c = id.split(".")[0]; by[c] = (by[c] || 0) + pr[i]; });
    res.push(Object.entries(by).sort((a, b) => b[1] - a[1])[0][0]);
  }
  return res;
}

function rank(logits, crop, raw) {
  const T = S.meta.temperature || 1;
  const all = softmax(logits, T, S.modelClasses.map(() => true));
  const plantP = {};
  S.modelClasses.forEach((id, i) => { const c = id.split(".")[0]; plantP[c] = (plantP[c] || 0) + all[i]; });
  const plants = Object.entries(plantP).sort((a, b) => b[1] - a[1]).map(([id, p]) => ({ id, p }));
  // step 1: the plant (chosen by the user, or the most likely one); step 2: the disease within that plant
  const use = crop || plants[0].id;
  const p = softmax(logits, T, S.modelClasses.map((id) => id.startsWith(use + ".")));
  const top = [...p.keys()].sort((a, b) => p[b] - p[a]).slice(0, 3).map((i) => ({ id: S.modelClasses[i], p: p[i] }));
  const plant = crop ? { id: crop, p: plantP[crop] || 0, chosen: true } : plants[0];
  const views = viewCrops(raw);
  const viewsDisagree = views.some((c) => c !== plants[0].id);
  // weak overall response = the photo looks unlike anything in training (tuned on real field photos: on
  // PlantDoc leaves this flags ~16% as "not sure"; a mango fruit, tree photos and screenshots are all flagged)
  let maxL = -Infinity; for (const v of logits) if (v > maxL) maxL = v;
  const weak = maxL < (S.meta.ood_logit || 6.3);
  const plantUnsure = !crop && (plants[0].p < 0.7 || viewsDisagree || weak);
  return { top, plants, plant, plantUnsure, unsure: top[0].p < NOT_SURE || plantUnsure };
}

async function diagnose(file) {
  if (!S.meta) return;
  $("#capture").hidden = true; $("#result").hidden = true; $("#busy").hidden = false;
  try {
    await loadSession();
    const t0 = performance.now();
    const x = await toTensor(file);
    const out = await S.session.run({ [S.session.inputNames[0]]: x });
    const raw = out[S.session.outputNames[0]].data;
    const logits = mergeViews(raw, VIEWS);
    const ms = performance.now() - t0;
    S.last = { logits, raw, ms, url: URL.createObjectURL(file), crop: $("#crop").value };
    renderResult(S.last);
  } catch (e) {
    console.error(e);
    $("#capture").hidden = false;
    setModelState("error");
  } finally {
    $("#busy").hidden = true;
  }
}

// ------------------------------------------------------------- rendering
function list(items) { return el("ul", {}, (items || []).map((x) => el("li", { text: x }))); }

function advice(d) {
  const box = el("div");
  if (d.type === "healthy") {
    box.append(el("div", { class: "healthy-box" }, el("h3", { text: t("healthy_title") }), el("p", { text: t("healthy_text") })));
    return box;
  }
  if (d.severity === "critical" || /report/i.test(d.notes || "")) box.append(el("p", { class: "report", text: t("report") }));
  box.append(el("h3", { text: t("what_to_do"), style: "margin-top:22px" }));
  const ladder = el("ol", { class: "ladder" });
  if (d.cultural) ladder.append(el("li", {}, el("h3", { text: t("step_cultural") }), list(d.cultural)));
  if (d.biological) ladder.append(el("li", {}, el("h3", { text: t("step_biological") }), list(d.biological)));
  if (d.chemical) ladder.append(el("li", { class: "chem" }, el("h3", { text: t("step_chemical") }), list(d.chemical),
    el("p", { class: "safety", text: t("chemical_safety") })));
  box.append(ladder);
  const more = (key, body) => body && el("details", { class: "more" }, el("summary", { text: t(key) }), body);
  box.append(
    more("symptoms", d.symptoms && list(d.symptoms)),
    more("spread", d.spread && el("p", { text: d.spread })),
    more("prevention", (d.prevention || d.notes) && el("div", {}, d.prevention && el("p", { text: d.prevention }), d.notes && el("p", { text: d.notes }))),
  );
  if (S.lang !== "en") box.append(el("p", { class: "lang-note", text: t("advice_lang") }));
  return box;
}

// a list of every plant the AI knows, for when the right one is not among the top 3
function allPlants(r, R) {
  const sel = el("select", { class: "all-plants", "aria-label": t("change_plant") });
  sel.append(el("option", { value: "", text: "＋ " + t("other_plant") }));
  const known = new Set(S.modelClasses.map((id) => id.split(".")[0]));
  S.kb.crops.filter((c) => known.has(c.id) && c.id !== "other")
    .sort((a, b) => nm(a.name).localeCompare(nm(b.name)))
    .forEach((c) => sel.append(el("option", { value: c.id, text: nm(c.name) })));
  sel.onchange = () => { if (sel.value) { r.crop = sel.value; renderResult(r); window.scrollTo({ top: 0, behavior: "smooth" }); } };
  return sel;
}

function plantBlock(r, R) {
  const box = el("div", { class: "plant" });
  const known = R.plants.filter((x) => x.id !== "other");
  const unknown = R.plant.id === "other";
  const chips = (list) => el("div", { class: "chips" }, list.map((x) => el("button", {
    type: "button", class: "chip" + (x.id === R.plant.id ? " on" : ""),
    onclick: () => { r.crop = x.id; renderResult(r); window.scrollTo({ top: 0, behavior: "smooth" }); },
  }, el("span", { text: nm(S.crops[x.id]?.name) }), el("small", { text: Math.round(x.p * 100) + "%" }))));
  box.append(el("p", { class: "kicker", text: t("plant_label") }));
  if (unknown) {
    box.append(el("h2", { text: t("unknown_plant_title") }),
      el("p", { class: "muted", text: t("unknown_plant_text").replace("{n}", known.length) }));
    box.append(el("p", { class: "ask", text: t("is_it") }), chips(known.slice(0, 3)));
    return box;
  }
  const how = R.plant.chosen ? t("you_chose") : R.plantUnsure ? t("plant_guess") : `${Math.round(R.plant.p * 100)}% ${t("plant_sure")}`;
  box.append(el("h2", { class: "plant-name" }, el("span", { text: nm(S.crops[R.plant.id]?.name) + (R.plantUnsure ? "?" : "") }),
    el("small", { text: how, class: R.plantUnsure ? "guess" : "" })));
  box.append(el("p", { class: "ask", text: R.plantUnsure ? t("is_it") : t("change_plant") }),
    chips(known.slice(0, 3).concat(R.plant.chosen && !known.slice(0, 3).some((x) => x.id === R.plant.id) ? [R.plant] : [])),
    allPlants(r, R));
  return box;
}

function renderResult(r) {
  const R = rank(r.logits, r.crop, r.raw);
  $("#photo").src = r.url;
  $("#r-time").textContent = "";
  $("#r-plant").replaceChildren(plantBlock(r, R));
  const unknown = R.plant.id === "other";
  $("#r-diag").hidden = unknown;
  if (unknown) { $("#capture").hidden = true; $("#result").hidden = false; return; }
  const top = R.top[0], d = S.dis[top.id];
  $("#r-name").textContent = nm(d.name);
  $("#r-crop").textContent = d.type !== "healthy" ? t("type." + d.type) : "";
  const pct = Math.round(top.p * 100);
  // when even the plant is a guess, a disease percentage would look falsely precise: hide it
  $("#r-bar").style.width = (R.plantUnsure ? Math.min(pct, 35) : pct) + "%";
  $("#r-bar").style.background = R.unsure ? "var(--gold)" : "var(--leaf)";
  $("#r-conf").textContent = (R.plantUnsure ? "" : `${pct}% ${t("confidence")}`) + (S.meta?.placeholder ? " — " + t("placeholder") : "");
  $("#r-time").textContent = r.ms ? t("done_in").replace("{s}", (r.ms / 1000).toLocaleString(S.lang, { maximumFractionDigits: 1 })) : "";
  $("#r-unsure").hidden = !R.unsure;
  $("#r-unsure").textContent = t("not_sure");
  const sev = $("#r-sev");
  sev.className = "sev " + (d.severity || "");
  sev.textContent = d.type === "healthy" ? "" : t("severity." + (d.severity || "medium"));
  $("#r-advice").replaceChildren(advice(d));
  const others = R.top.slice(1).filter((o) => o.p >= 0.03 && o.id.split(".")[0] !== "other");
  $("#r-others").replaceChildren(...(others.length ? [el("p", { class: "alt-title", text: t("others") }),
    ...others.map((o) => el("button", { class: "alt", type: "button", onclick: () => {
      $("#r-advice").replaceChildren(advice(S.dis[o.id]));
      $("#r-name").textContent = nm(S.dis[o.id].name);
      $("#r-advice").scrollIntoView({ behavior: "smooth" });
    } }, el("b", { text: nm(S.dis[o.id].name) }), el("span", { text: Math.round(o.p * 100) + "%" })))] : []));
  $("#capture").hidden = true; $("#result").hidden = false;
}

function fillCropSelects() {
  const known = new Set((S.meta ? S.modelClasses : S.kb.diseases.map((d) => d.id)).map((id) => id.split(".")[0]));
  const groups = {};
  for (const c of S.kb.crops) if (known.has(c.id) && c.id !== "other") (groups[c.group] ||= []).push(c);
  const build = (sel, firstLabel, allCrops) => {
    const keep = sel.value;
    sel.replaceChildren(el("option", { value: "", text: firstLabel }));
    for (const [g, cs] of Object.entries(groups)) {
      const og = el("optgroup", { label: nm(S.kb.groups[g]) });
      cs.sort((a, b) => nm(a.name).localeCompare(nm(b.name))).forEach((c) => og.append(el("option", { value: c.id, text: nm(c.name) })));
      sel.append(og);
    }
    if (allCrops) for (const c of S.kb.crops) if (!known.has(c.id)) sel.append(el("option", { value: c.id, text: nm(c.name) }));
    sel.value = keep;
  };
  build($("#crop"), t("any_crop"), false);
  // library shows every crop in the knowledge base
  const lib = $("#lib-crop"), keep = lib.value;
  lib.replaceChildren(el("option", { value: "", text: t("all_crops") }));
  const all = {};
  for (const c of S.kb.crops) if (c.id !== "other") (all[c.group] ||= []).push(c);
  for (const [g, cs] of Object.entries(all)) {
    const og = el("optgroup", { label: nm(S.kb.groups[g]) });
    cs.forEach((c) => og.append(el("option", { value: c.id, text: nm(c.name) })));
    lib.append(og);
  }
  lib.value = keep;
}

function renderLibrary() {
  const q = $("#q").value.trim().toLowerCase(), crop = $("#lib-crop").value;
  const ul = $("#lib-list");
  ul.replaceChildren();
  let lastCrop = null;
  const items = S.kb.diseases.filter((d) => d.type !== "healthy" && d.crop !== "other" && (!crop || d.crop === crop) && (!q ||
    [nm(d.name), d.name.en, d.scientific, nm(S.crops[d.crop].name), S.crops[d.crop].name.en].join(" ").toLowerCase().includes(q)));
  for (const d of items) {
    if (d.crop !== lastCrop) { ul.append(el("li", { class: "crop-h", text: nm(S.crops[d.crop].name) })); lastCrop = d.crop; }
    ul.append(el("li", {}, el("button", { type: "button", onclick: () => openDisease(d.id) },
      el("span", { text: nm(d.name) }), el("i", { class: "dot " + (d.severity || ""), title: t("severity." + d.severity) }))));
  }
}

function openDisease(id) {
  const d = S.dis[id], box = $("#lib-detail");
  box.replaceChildren(
    el("button", { class: "back", type: "button", text: "← " + t("tab_library"), onclick: () => { box.hidden = true; $("#lib-list").hidden = false; $(".filters").hidden = false; } }),
    el("h2", { text: nm(d.name) }),
    d.scientific ? el("p", { class: "sci", text: d.scientific }) : null,
    el("p", { class: "muted", text: `${nm(S.crops[d.crop].name)} — ${t("type." + d.type)}` }),
    el("p", { class: "sev " + d.severity, text: t("severity." + d.severity) }),
    advice(d));
  // in the library, show the symptoms open by default
  box.querySelector("details")?.setAttribute("open", "");
  $("#lib-list").hidden = true; $(".filters").hidden = true; box.hidden = false;
  window.scrollTo({ top: 0 });
}

function renderAbout() {
  $("#a-model").textContent = S.meta ? S.meta.model : t("model_none");
  $("#a-classes").textContent = S.meta ? S.modelClasses.filter((c) => !c.endsWith(".healthy") && !c.startsWith("other.")).length : "0";
  $("#a-plants").textContent = S.meta ? new Set(S.modelClasses.map((c) => c.split(".")[0]).filter((c) => c !== "other")).size : "0";
  $("#a-kb").textContent = S.kb.diseases.filter((d) => d.type !== "healthy" && d.crop !== "other").length;
}

function show(view) {
  for (const v of ["diagnose", "library", "about"]) $("#v-" + v).hidden = v !== view;
  document.querySelectorAll(".tabs button").forEach((b) => b.toggleAttribute("aria-current", b.dataset.view === view));
  if (view === "diagnose") window.scrollTo({ top: 0 });
}

// ------------------------------------------------------------- start
async function main() {
  const saved = localStorage.getItem("lang");
  const nav = (navigator.language || "en").slice(0, 2);
  S.lang = saved || (I18N[nav] ? nav : "en");
  const sel = $("#lang");
  for (const [k, v] of Object.entries(I18N)) sel.append(el("option", { value: k, text: v.lang }));
  sel.value = S.lang;
  sel.addEventListener("change", () => { S.lang = sel.value; try { localStorage.setItem("lang", S.lang); } catch {} applyLang(); });

  await Promise.all([loadKB(), loadMeta()]);
  applyLang();

  for (const id of ["cam", "gal"]) $("#" + id).addEventListener("change", (e) => {
    const f = e.target.files && e.target.files[0];
    e.target.value = "";
    if (f) diagnose(f);
  });
  $(".shoot").addEventListener("click", (e) => { if (!S.meta) e.preventDefault(); });
  $("#again").addEventListener("click", () => { $("#result").hidden = true; $("#capture").hidden = false; S.last = null; window.scrollTo({ top: 0 }); });
  $("#q").addEventListener("input", renderLibrary);
  $("#lib-crop").addEventListener("change", renderLibrary);
  document.querySelectorAll(".tabs button").forEach((b) => b.addEventListener("click", () => show(b.dataset.view)));

  // warm up the model in the background so the first diagnosis is fast
  if (S.meta) (window.requestIdleCallback || setTimeout)(() => loadSession().catch(() => {}));
  if ("serviceWorker" in navigator) {
    navigator.serviceWorker.addEventListener("message", (e) => {
      if (e.data && e.data.type === "offline-status") { S.offline = e.data.ready; setModelState(S.session ? "ready" : undefined); }
    });
    navigator.serviceWorker.register("sw.js").then(() => navigator.serviceWorker.ready).then(() => {
      checkOffline();
      navigator.serviceWorker.addEventListener("controllerchange", checkOffline);
      setTimeout(checkOffline, 4000);
    }).catch(() => {});
  }
}
main();
