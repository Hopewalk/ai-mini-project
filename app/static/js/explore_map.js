(function () {
  const coverage = LandMap.readJson("coverage-data") || [];
  const clusterNames = LandMap.readJson("cluster-names") || {};
  const status = document.getElementById("map-status");
  const map = LandMap.create("explore-map").setView([13.0, 100.5], 6);
  const cellsLayer = L.layerGroup().addTo(map);
  const provinceLayer = L.layerGroup().addTo(map);
  const MIN_ZOOM = 8;
  // log-scale price ramp (THB / sq.wah)
  const PRICE_STOPS = [[100, "#f7fcb9"], [500, "#addd8e"], [2000, "#41ab5d"], [10000, "#238443"],
                       [50000, "#005a32"], [Infinity, "#002d19"]];
  let colorBy = "grade";
  let cells = [];
  let legend = null;

  coverage.forEach((c) => {
    L.circleMarker([c.lat, c.lon], { radius: 10, color: "#154d40", weight: 2, fillColor: "#9fd8c6", fillOpacity: 0.85 })
      .bindTooltip(`${c.name} — คลิกเพื่อซูม`)
      .on("click", () => map.setView([c.lat, c.lon], 10))
      .addTo(provinceLayer);
  });

  function colorOf(c) {
    if (colorBy === "grade") return LandMap.GRADE_COLORS[c.grade] || "#888";
    if (colorBy === "cluster") return LandMap.CLUSTER_COLORS[c.cluster % LandMap.CLUSTER_COLORS.length];
    return PRICE_STOPS.find(([limit]) => c.price < limit)[1];
  }

  function renderLegend() {
    if (legend) legend.remove();
    legend = L.control({ position: "bottomright" });
    legend.onAdd = () => {
      const div = L.DomUtil.create("div", "map-legend");
      let rows = [];
      if (colorBy === "grade") {
        rows = ["A", "B", "C"].map((g) => `<i style="background:${LandMap.GRADE_COLORS[g]}"></i>${LandMap.GRADE_NAMES[g]}`);
        rows.unshift("<strong>ระดับราคา</strong>");
      } else if (colorBy === "cluster") {
        rows = Object.entries(clusterNames).map(([id, name]) =>
          `<i style="background:${LandMap.CLUSTER_COLORS[id % LandMap.CLUSTER_COLORS.length]}"></i>${name}`);
        rows.unshift("<strong>ประเภททำเล</strong>");
      } else {
        let lower = 0;
        rows = PRICE_STOPS.map(([limit, color]) => {
          const label = limit === Infinity ? `≥ ${LandMap.fmt(lower)}` : `${LandMap.fmt(lower)} – ${LandMap.fmt(limit)}`;
          lower = limit;
          return `<i style="background:${color}"></i>${label}`;
        });
        rows.unshift("<strong>ราคาประเมิน (บาท/ตร.ว.)</strong>");
      }
      div.innerHTML = rows.join("<br>");
      return div;
    };
    legend.addTo(map);
  }

  function popupHtml(c) {
    return `<strong>${c.province}${c.amphoe ? " › " + c.amphoe : ""}</strong><br>
      ราคาประเมิน (ค่ากลาง): <strong>${LandMap.fmt(c.price)}</strong> บาท/ตร.ว.<br>
      ระดับ: ${LandMap.GRADE_NAMES[c.grade] || "–"}<br>
      ทำเล: ${c.cluster_name || "–"}<br>
      <a href="/?lat=${c.lat.toFixed(5)}&lon=${c.lon.toFixed(5)}">ดูวิเคราะห์ละเอียด →</a>`;
  }

  function draw() {
    cellsLayer.clearLayers();
    cells.forEach((c) => {
      L.rectangle(LandMap.cellBounds(c.lat, c.lon), { color: colorOf(c), weight: 0.5, fillOpacity: 0.55 })
        .bindPopup(popupHtml(c)).addTo(cellsLayer);
    });
  }

  async function refresh() {
    if (map.getZoom() < MIN_ZOOM) {
      cells = [];
      draw();
      provinceLayer.addTo(map);
      status.textContent = "คลิกจุดสีเขียว (จังหวัดที่มีข้อมูล) หรือซูมเข้า เพื่อดูราคาแต่ละพื้นที่";
      return;
    }
    map.removeLayer(provinceLayer);
    const data = await LandMap.fetchCells(map);
    cells = data.cells;
    draw();
    status.textContent = `แสดง ${LandMap.fmt(cells.length)} พื้นที่ — คลิกสี่เหลี่ยมเพื่อดูราคา`
      + (data.truncated ? " (แสดงไม่ครบ — ซูมเข้าเพื่อดูทั้งหมด)" : "");
  }

  document.querySelectorAll("input[name=colorBy]").forEach((el) =>
    el.addEventListener("change", () => { colorBy = el.value; draw(); renderLegend(); })
  );
  map.on("moveend", refresh);
  renderLegend();
  refresh();
})();
