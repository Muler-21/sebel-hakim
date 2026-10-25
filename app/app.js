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
    S.session = await ort.InferenceSession.create("model/" + file, { executionProviders: ["wasm"] });
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
async function toTensor(file) {
  const size = S.meta.img;
  const bmp = await createImageBitmap(file, { imageOrientation: "from-image" });
  const short = Math.round(size * 1.14);                 // same as training: resize then centre-crop
  const scale = short / Math.min(bmp.width, bmp.height);
  const w = Math.round(bmp.width * scale), h = Math.round(bmp.height * scale);
  const c = document.createElement("canvas");
  c.width = size; c.height = size;
  const g = c.getContext("2d", { willReadFrequently: true });
  g.imageSmoothingQuality = "high";
  g.drawImage(bmp, (size - w) / 2, (size - h) / 2, w, h);
  bmp.close && bmp.close();
  const px = g.getImageData(0, 0, size, size).data;
  const n = size * size, out = new Float32Array(3 * n);
  const [m0, m1, m2] = S.meta.mean, [s0, s1, s2] = S.meta.std;
  for (let i = 0; i < n; i++) {
    out[i] = (px[i * 4] / 255 - m0) / s0;
    out[n + i] = (px[i * 4 + 1] / 255 - m1) / s1;
    out[2 * n + i] = (px[i * 4 + 2] / 255 - m2) / s2;
  }
  return new ort.Tensor("float32", out, [1, 3, size, size]);
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

async function diagnose(file) {
  if (!S.meta) return;
  $("#capture").hidden = true; $("#result").hidden = true; $("#busy").hidden = false;
  try {
    await loadSession();
    const x = await toTensor(file);
    const out = await S.session.run({ [S.session.inputNames[0]]: x });
    const logits = out[S.session.outputNames[0]].data;
    const crop = $("#crop").value;
    const allowed = S.modelClasses.map((id) => !crop || id.startsWith(crop + "."));
    const p = softmax(logits, S.meta.temperature || 1, allowed);
    const order = [...p.keys()].sort((a, b) => p[b] - p[a]).slice(0, 3);
    const top = order.map((i) => ({ id: S.modelClasses[i], p: p[i] }));
    // how sure are we about the crop itself?
    const cropP = {};
    S.modelClasses.forEach((id, i) => { const c = id.split(".")[0]; cropP[c] = (cropP[c] || 0) + p[i]; });
    const cropSure = Math.max(...Object.values(cropP));
    S.last = { top, url: URL.createObjectURL(file), unsure: top[0].p < NOT_SURE || (!crop && cropSure < 0.6) };
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

function renderResult(r) {
  const top = r.top[0], d = S.dis[top.id];
  $("#photo").src = r.url;
  $("#r-name").textContent = nm(d.name);
  $("#r-crop").textContent = [nm(S.crops[d.crop]?.name), d.type !== "healthy" ? t("type." + d.type) : ""].filter(Boolean).join(" — ");
  const pct = Math.round(top.p * 100);
  $("#r-bar").style.width = pct + "%";
  $("#r-bar").style.background = r.unsure ? "var(--gold)" : "var(--leaf)";
  $("#r-conf").textContent = `${pct}% ${t("confidence")}` + (S.meta?.placeholder ? " — " + t("placeholder") : "");
  $("#r-unsure").hidden = !r.unsure;
  $("#r-unsure").textContent = t("not_sure");
  const sev = $("#r-sev");
  sev.className = "sev " + (d.severity || "");
  sev.textContent = d.type === "healthy" ? "" : t("severity." + (d.severity || "medium"));
  $("#r-advice").replaceChildren(advice(d));
  const others = r.top.slice(1).filter((o) => o.p >= 0.03);
  $("#r-others").replaceChildren(...(others.length ? [el("p", { class: "alt-title", text: t("others") }),
    ...others.map((o) => el("button", { class: "alt", type: "button", onclick: () => {
      const swapped = { ...r, top: [o, ...r.top.filter((x) => x !== o)] };
      renderResult(swapped); window.scrollTo({ top: 0, behavior: "smooth" });
    } }, el("b", { text: `${nm(S.dis[o.id].name)} (${nm(S.crops[S.dis[o.id].crop]?.name)})` }), el("span", { text: Math.round(o.p * 100) + "%" })))] : []));
  $("#capture").hidden = true; $("#result").hidden = false;
}

function fillCropSelects() {
  const known = new Set((S.meta ? S.modelClasses : S.kb.diseases.map((d) => d.id)).map((id) => id.split(".")[0]));
  const groups = {};
  for (const c of S.kb.crops) if (known.has(c.id)) (groups[c.group] ||= []).push(c);
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
  for (const c of S.kb.crops) (all[c.group] ||= []).push(c);
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
  const items = S.kb.diseases.filter((d) => d.type !== "healthy" && (!crop || d.crop === crop) && (!q ||
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
  $("#a-classes").textContent = S.meta ? S.modelClasses.filter((c) => !c.endsWith(".healthy")).length : "0";
  $("#a-kb").textContent = S.kb.diseases.filter((d) => d.type !== "healthy").length;
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
