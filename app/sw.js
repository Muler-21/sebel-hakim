// Offline support: app shell and model cached on first visit; network updates when online.
const CACHE = "agrodoc-v1";
const SHELL = ["./", "index.html", "styles.css", "app.js", "i18n.js", "kb.json", "icon.svg", "manifest.webmanifest"];

self.addEventListener("install", (e) => {
  e.waitUntil(caches.open(CACHE).then((c) => c.addAll(SHELL)).then(() => self.skipWaiting()));
});
self.addEventListener("activate", (e) => {
  e.waitUntil(caches.keys().then((ks) => Promise.all(ks.filter((k) => k !== CACHE).map((k) => caches.delete(k))))
    .then(() => self.clients.claim()));
});
self.addEventListener("fetch", (e) => {
  const req = e.request;
  if (req.method !== "GET") return;
  const url = new URL(req.url);
  const big = url.pathname.endsWith(".onnx") || url.pathname.endsWith(".wasm") || url.hostname === "cdn.jsdelivr.net";
  if (big) {
    // model + runtime: cache first (large, rarely change)
    e.respondWith(caches.match(req).then((hit) => hit || fetch(req).then((res) => {
      if (res.ok || res.type === "opaque") { const copy = res.clone(); caches.open(CACHE).then((c) => c.put(req, copy)); }
      return res;
    })));
    return;
  }
  // everything else: network first, cache as fallback (so updates arrive when online)
  e.respondWith(fetch(req).then((res) => {
    if (res.ok && url.origin === location.origin) { const copy = res.clone(); caches.open(CACHE).then((c) => c.put(req, copy)); }
    return res;
  }).catch(() => caches.match(req).then((hit) => hit || caches.match("index.html"))));
});
