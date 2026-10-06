// Shared Leaflet helpers.
window.LandMap = (function () {
  const GRADE_COLORS = { A: "#1d4e89", B: "#2a9d8f", C: "#c08a1e" };
  const CLUSTER_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2"];
  const HALF_CELL_DEG = 0.009; // ~1 km

  function create(elementId, options) {
    const map = L.map(elementId, { preferCanvas: true, ...options });
    L.tileLayer("https://tile.openstreetmap.org/{z}/{x}/{y}.png", {
      maxZoom: 19,
      attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors',
    }).addTo(map);
    return map;
  }

  function cellBounds(lat, lon) {
    const dLon = HALF_CELL_DEG / Math.cos((lat * Math.PI) / 180);
    return [[lat - HALF_CELL_DEG, lon - dLon], [lat + HALF_CELL_DEG, lon + dLon]];
  }

  function fmt(n) {
    return Number(n).toLocaleString("th-TH", { maximumFractionDigits: 0 });
  }

  async function fetchCells(map) {
    const b = map.getBounds();
    const bbox = [b.getWest(), b.getSouth(), b.getEast(), b.getNorth()].map((v) => v.toFixed(4)).join(",");
    const res = await fetch(`/api/cells?bbox=${bbox}`);
    if (!res.ok) return { cells: [], truncated: false };
    return res.json();
  }

  function readJson(id) {
    const el = document.getElementById(id);
    return el ? JSON.parse(el.textContent) : null;
  }

  return { create, cellBounds, fmt, fetchCells, readJson, GRADE_COLORS, CLUSTER_COLORS };
})();
