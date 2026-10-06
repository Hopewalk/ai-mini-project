(function () {
  const latInput = document.getElementById("lat");
  const lonInput = document.getElementById("lon");
  const picked = document.getElementById("picked");
  const coverage = LandMap.readJson("coverage-data") || [];

  const map = LandMap.create("picker-map").setView([13.0, 100.5], 6);
  const cellsLayer = L.layerGroup().addTo(map);
  const provinceLayer = L.layerGroup().addTo(map);
  let marker = null;

  coverage.forEach((c) => {
    L.circleMarker([c.lat, c.lon], { radius: 9, color: "#154d40", weight: 2, fillColor: "#9fd8c6", fillOpacity: 0.8 })
      .bindTooltip(`${c.name} — คลิกเพื่อซูม`)
      .on("click", () => map.setView([c.lat, c.lon], 10))
      .addTo(provinceLayer);
  });

  function setPoint(lat, lon, pan) {
    latInput.value = lat.toFixed(5);
    lonInput.value = lon.toFixed(5);
    picked.classList.add("is-set");
    picked.classList.remove("is-invalid");
    picked.textContent = `📍 เลือกแล้ว (${lat.toFixed(4)}, ${lon.toFixed(4)}) — คลิกที่อื่นเพื่อเปลี่ยน`;
    if (marker) marker.setLatLng([lat, lon]);
    else marker = L.marker([lat, lon]).addTo(map);
    if (pan) map.setView([lat, lon], Math.max(map.getZoom(), 12));
  }

  async function refreshCells() {
    cellsLayer.clearLayers();
    const showCells = map.getZoom() >= 9;
    if (showCells) map.removeLayer(provinceLayer);
    else provinceLayer.addTo(map);
    if (!showCells) return;
    const data = await LandMap.fetchCells(map);
    data.cells.forEach((c) => {
      L.rectangle(LandMap.cellBounds(c.lat, c.lon), {
        color: LandMap.GRADE_COLORS[c.grade] || "#888", weight: 0.5, fillOpacity: 0.25, interactive: false,
      }).addTo(cellsLayer);
    });
  }

  map.on("click", (e) => setPoint(e.latlng.lat, e.latlng.lng, false));
  map.on("moveend", refreshCells);

  document.querySelectorAll(".sample-btn").forEach((btn) =>
    btn.addEventListener("click", () => setPoint(parseFloat(btn.dataset.lat), parseFloat(btn.dataset.lon), true))
  );

  if (latInput.value && lonInput.value) setPoint(parseFloat(latInput.value), parseFloat(lonInput.value), true);
})();
