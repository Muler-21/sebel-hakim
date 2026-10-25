// Sebel Hakim offline support.
// On install it stores everything the app needs: the app itself, the disease library,
// the AI engine (onnxruntime-web) and the AI model. After that it works with no internet.
const CACHE = "sebel-hakim-v3";
const ORT = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/";
const ORT_FILES = ["ort.min.js", "ort-wasm-simd-threaded.wasm", "ort-wasm-simd-threaded.mjs"].map((f) => ORT + f);
const SHELL = ["./", "index.html", "styles.css", "app.js", "i18n.js", "kb.json", "icon.svg",
  "manifest.webmanifest", "model/labels.json"];

async function modelFiles() {
  try {
    const meta = await (await fetch("model/labels.json", { cache: "no-cache" })).json();
    const f = meta.files || {};
    return ["model/" + (f.int8 || f.fp32 || "model.onnx")];
  } catch { return []; }
}

self.addEventListener("install", (e) => {
  e.waitUntil((async () => {
    const c = await caches.open(CACHE);
    await c.addAll(SHELL.map((u) => new Request(u, { cache: "reload" })));
    // big files: try each one; a failure here is retried on the next visit
    for (const u of [...ORT_FILES, ...(await modelFiles())]) {
      try { await c.add(new Request(u, { mode: u.startsWith("http") ? "cors" : "same-origin", cache: "reload" })); }
      catch (err) { console.warn("offline cache: could not store", u, err); }
    }
    await self.skipWaiting();
  })());
});

self.addEventListener("activate", (e) => {
  e.waitUntil((async () => {
    for (const k of await caches.keys()) if (k !== CACHE) await caches.delete(k);
    await self.clients.claim();
  })());
});

self.addEventListener("fetch", (e) => {
  const req = e.request;
  if (req.method !== "GET") return;
  const url = new URL(req.url);
  const isBig = url.href.startsWith(ORT) || url.pathname.endsWith(".onnx");
  if (isBig) {
    // AI engine and model: cache first (large, versioned)
    e.respondWith((async () => {
      const hit = await caches.match(req.url);
      if (hit) return hit;
      const res = await fetch(req);
      if (res.ok) { const copy = res.clone(); caches.open(CACHE).then((c) => c.put(req.url, copy)); }
      return res;
    })());
    return;
  }
  if (url.origin !== location.origin) return;
  // the app itself: network first so updates arrive, cache when offline
  e.respondWith((async () => {
    try {
      const res = await fetch(req);
      if (res.ok) { const copy = res.clone(); caches.open(CACHE).then((c) => c.put(req, copy)); }
      return res;
    } catch {
      return (await caches.match(req, { ignoreSearch: true })) ||
        (req.mode === "navigate" ? await caches.match("index.html") : Response.error());
    }
  })());
});

// the page asks "is everything stored for offline use?"
self.addEventListener("message", async (e) => {
  if (e.data !== "offline-status") return;
  const c = await caches.open(CACHE);
  const need = [...SHELL, ...ORT_FILES, ...(await modelFiles())];
  const missing = [];
  for (const u of need) if (!(await c.match(u))) missing.push(u);
  e.source.postMessage({ type: "offline-status", ready: missing.length === 0, missing });
});
