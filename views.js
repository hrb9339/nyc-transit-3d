// Map views: the default 3D city, a top-down night-lights view (VIIRS), the hexagonal
// analysis grid, distance to the nearest subway station (straight line and walking),
// and population density.
// Data comes from scripts/build_hexgrid.py, scripts/gee_ntl.py and data/features/*.csv.

const Views = (() => {
  const NYC_BOUNDS = [[-74.26, 40.49], [-73.70, 40.92]];
  const NTL_RAMP = ["#000000", "#1a0b3d", "#5c1a6b", "#b8326b", "#f0703a", "#fbc54a", "#fff7d6"];
  // Straight-line distance from cell center to nearest station, in meters.
  const DIST_BANDS = [
    { max: 400, color: "#1a9850", label: "0–400 m" },
    { max: 800, color: "#a6d96a", label: "400–800 m" },
    { max: 1600, color: "#fee08b", label: "800–1,600 m" },
    { max: 3200, color: "#f46d43", label: "1,600–3,200 m" },
    { max: Infinity, color: "#a50026", label: "3,200 m+" },
  ];
  // Residents per km² of land (2020 Census blocks, data/features/demographics.csv).
  const POP_BANDS = [
    { max: 1000, color: "#2b1145", label: "under 1,000" },
    { max: 5000, color: "#5c1a6b", label: "1,000–5,000" },
    { max: 10000, color: "#a3306e", label: "5,000–10,000" },
    { max: 20000, color: "#e0583f", label: "10,000–20,000" },
    { max: 40000, color: "#f9a23a", label: "20,000–40,000" },
    { max: Infinity, color: "#fde68a", label: "40,000+" },
  ];

  let map;
  let current = "city";
  let ntlMeta = null;   // null until loaded; false if not generated yet
  let hasHexNtl = false;
  const stats = {};     // per view: { median, shares: [fraction of cells per band] }

  const CITY_LAYERS = ["buildings-3d"];
  const NIGHT_LAYERS = ["ntl-raster"];
  const HEX_LAYERS = ["hex-fill", "hex-line"];
  const DIST_LAYERS = ["dist-fill", "dist-line"];
  const WALK_LAYERS = ["walk-fill", "walk-line"];
  const POP_LAYERS = ["pop-fill", "pop-line"];

  // ---------- Data ----------

  async function loadNtl() {
    try {
      const r = await fetch("data/ntl/ntl_web.json", { cache: "no-store" });
      if (!r.ok) throw new Error(r.status);
      ntlMeta = await r.json();
    } catch {
      ntlMeta = false;
      return;
    }
    const [w, s, e, n] = ntlMeta.bounds;
    map.addSource("ntl", {
      type: "image",
      url: "data/ntl/ntl_web.png",
      coordinates: [[w, n], [e, n], [e, s], [w, s]],
    });
    map.addLayer({
      id: "ntl-raster", type: "raster", source: "ntl",
      layout: { visibility: "none" },
      paint: { "raster-opacity": 0.95, "raster-resampling": "linear", "raster-fade-duration": 0 },
    }, "subway-glow");
  }

  // Feature tables start with '#' comment lines (sources, column meanings), then plain CSV.
  async function loadFeatureCsv(name) {
    try {
      const r = await fetch(`data/features/${name}.csv`, { cache: "no-store" });
      if (!r.ok) throw new Error(r.status);
      const lines = (await r.text()).split(/\r?\n/).filter((l) => l && !l.startsWith("#"));
      const cols = lines[0].split(",");
      const rows = new Map();
      for (const line of lines.slice(1)) {
        const vals = line.split(",");
        rows.set(vals[0], Object.fromEntries(cols.map((c, i) => [c, vals[i] === "" ? null : +vals[i]])));
      }
      return rows;
    } catch {
      return new Map();
    }
  }

  function bandStats(values, bands) {
    const v = values.filter((x) => x != null && !Number.isNaN(x)).sort((a, b) => a - b);
    let lower = -Infinity;
    return {
      median: v[Math.floor(v.length / 2)],
      shares: bands.map(({ max }) => {
        const share = v.filter((d) => d > lower && d <= max).length / v.length;
        lower = max;
        return share;
      }),
    };
  }

  function bandFill(prop, bands) {
    const expr = ["step", ["coalesce", ["get", prop], -1], "#1b2230", 0, bands[0].color];
    bands.slice(0, -1).forEach((band, i) => expr.push(band.max + 0.001, bands[i + 1].color));
    return expr;
  }

  async function loadHexes() {
    const [grid, demo, walk, stud] = await Promise.all([
      fetch("data/hexgrid.geojson", { cache: "no-store" }).then((r) => r.json()),
      loadFeatureCsv("demographics"),
      loadFeatureCsv("walk"),
      loadFeatureCsv("students"),
    ]);
    hasHexNtl = grid.features.some((f) => f.properties.ntl_mean != null);
    for (const f of grid.features) {
      const p = f.properties;
      const d = demo.get(p.hex_id) || {};
      const w = walk.get(p.hex_id) || {};
      const s = stud.get(p.hex_id) || {};
      p.population = d.population ?? null;
      p.jobs = d.jobs ?? null;
      p.pop_density = d.pop_density ?? null;
      p.students = s.students ?? null;
      p.students_college = s.students_college ?? null;
      p.walk_m = w.walk_to_entrance_m ?? null;
      p.walk_capped = w.walk_capped ?? 0;
    }
    map.addSource("hexgrid", { type: "geojson", data: grid });
    // Heatmap views use the hexagon centers too; the hexgrid source copies the data, so this is safe.
    await Heatmaps.init(map, grid, loadFeatureCsv);

    const props = grid.features.map((f) => f.properties);
    stats.distance = bandStats(props.map((p) => p.nearest_station_m), DIST_BANDS);
    stats.walk = bandStats(props.map((p) => p.walk_m), DIST_BANDS);
    stats.population = bandStats(props.map((p) => p.pop_density), POP_BANDS);
    stats.population.total = props.reduce((s, p) => s + (p.population || 0), 0);

    // Night lights on a log scale (radiance spans ~1 to 300+), else station count.
    const fill = hasHexNtl
      ? ["interpolate", ["linear"], ["ln", ["+", ["coalesce", ["get", "ntl_mean"], 0], 1]],
          0, NTL_RAMP[0], 1.5, NTL_RAMP[1], 2.5, NTL_RAMP[2], 3.3, NTL_RAMP[3], 4.1, NTL_RAMP[4], 4.9, NTL_RAMP[5], 5.7, NTL_RAMP[6]]
      : ["step", ["get", "stations"], "#1b2230", 1, "#2f5f8f", 2, "#4f8fd0", 4, "#9fd0ff"];

    map.addLayer({
      id: "hex-fill", type: "fill", source: "hexgrid",
      layout: { visibility: "none" },
      paint: { "fill-color": fill, "fill-opacity": 0.82 },
    }, "subway-glow");
    map.addLayer({
      id: "hex-line", type: "line", source: "hexgrid",
      layout: { visibility: "none" },
      paint: {
        "line-color": ["case", [">", ["get", "stations"], 0], "rgba(255,255,255,0.75)", "rgba(255,255,255,0.22)"],
        "line-width": ["case", [">", ["get", "stations"], 0], 1.4, 0.7],
      },
    });

    // Banded choropleths: straight-line distance, walking distance, population density.
    for (const [id, prop, bands] of [
      ["dist", "nearest_station_m", DIST_BANDS],
      ["walk", "walk_m", DIST_BANDS],
      ["pop", "pop_density", POP_BANDS],
    ]) {
      map.addLayer({
        id: `${id}-fill`, type: "fill", source: "hexgrid",
        layout: { visibility: "none" },
        paint: { "fill-color": bandFill(prop, bands), "fill-opacity": 0.88 },
      }, "subway-glow");
      map.addLayer({
        id: `${id}-line`, type: "line", source: "hexgrid",
        layout: { visibility: "none" },
        paint: { "line-color": "rgba(0,0,0,0.3)", "line-width": 0.5 },
      }, "subway-glow");
    }

    const popup = new maplibregl.Popup({ closeButton: false, closeOnClick: false, offset: 8 });
    const showPopup = (e) => {
      const p = e.features[0].properties;
      const names = JSON.parse(p.station_names || "[]");
      const rows = [
        ["Borough", p.borough],
        ["Land area", `${p.land_km2} km²`],
        ["Subway stations", names.length ? `${p.stations} (${names.slice(0, 3).join(", ")}${names.length > 3 ? "…" : ""})` : "none"],
        ["Nearest station", `${(p.nearest_station_m / 1000).toFixed(2)} km from center`],
      ];
      if (p.walk_m != null) {
        rows.push(["Walk to entrance", p.walk_capped ? "over 5 km" : `${(p.walk_m / 1000).toFixed(2)} km along streets`]);
      }
      if (p.population != null) {
        rows.push(["Population (2020)", `${Math.round(p.population).toLocaleString()} (${Math.round(p.pop_density).toLocaleString()} per km²)`]);
        rows.push(["Jobs (2021)", Math.round(p.jobs).toLocaleString()]);
      }
      if (p.students != null) {
        rows.push(["Students", `${Math.round(p.students).toLocaleString()}${p.students_college ? ` (${Math.round(p.students_college).toLocaleString()} college)` : ""}`]);
      }
      if (p.ntl_mean != null) {
        rows.push(["Mean radiance", `${(+p.ntl_mean).toFixed(1)} nW/cm²/sr`]);
        rows.push(["Sum of lights", `${(+p.ntl_sum).toFixed(0)}`]);
        rows.push(["Lit / bright pixels", `${(+p.lit_pixels).toFixed(1)} / ${(+p.bright_pixels).toFixed(1)} of ${(+p.pixels).toFixed(1)}`]);
      }
      const wrap = document.createElement("div");
      wrap.innerHTML = `<div class="popup-title"></div><table class="popup-table"></table>`;
      wrap.querySelector(".popup-title").textContent = `Hexagon ${p.hex_id}`;
      const table = wrap.querySelector("table");
      for (const [k, v] of rows) {
        const tr = table.insertRow();
        tr.insertCell().textContent = k;
        tr.insertCell().textContent = v;
      }
      popup.setLngLat(e.lngLat).setDOMContent(wrap).addTo(map);
    };
    for (const layer of ["hex-fill", "dist-fill", "walk-fill", "pop-fill"]) {
      map.on("mousemove", layer, showPopup);
      map.on("mouseleave", layer, () => popup.remove());
    }
  }

  // ---------- Switching ----------

  function show(ids, on) {
    ids.forEach((id) => map.getLayer(id) && map.setLayoutProperty(id, "visibility", on ? "visible" : "none"));
  }

  function setView(name) {
    if (name === "night" && ntlMeta === false) {
      legend(`<b>Night-light data not generated yet.</b><br>Run <code>python scripts/gee_ntl.py --project YOUR-ID</code> after signing in to Earth Engine.`);
      return;
    }
    current = name;
    document.querySelectorAll("#views-mode button").forEach((b) => b.classList.toggle("active", b.dataset.view === name));

    show(CITY_LAYERS, name === "city" && document.getElementById("tg-buildings").checked);
    show(NIGHT_LAYERS, name === "night");
    show(HEX_LAYERS, name === "hex");
    show(DIST_LAYERS, name === "distance");
    show(WALK_LAYERS, name === "walk");
    show(POP_LAYERS, name === "population");
    Heatmaps.show(name);

    // Let the lights, the banded colors or the heatmaps read as the main picture.
    const muted = ["night", "distance", "walk", "population", ...Heatmaps.views].includes(name);
    map.setPaintProperty("subway-lines", "line-opacity", muted ? 0.35 : 1);
    map.setPaintProperty("subway-glow", "line-opacity", muted ? 0 : 0.35);

    if (name === "city") {
      legend(null);
      map.easeTo({ center: [-73.94, 40.72], zoom: 11, pitch: 50, bearing: -20, duration: 1500 });
    } else {
      map.fitBounds(NYC_BOUNDS, { pitch: 0, bearing: 0, padding: 30, duration: 1500 });
      if (Heatmaps.views.includes(name)) {
        legend("&nbsp;");
        Heatmaps.renderLegend(name, document.getElementById("legend"));
        return;
      }
      legend({ night: nightLegend, hex: hexLegend, distance: distanceLegend, walk: walkLegend, population: populationLegend }[name]());
    }
  }

  // ---------- Legend ----------

  function legend(html) {
    const el = document.getElementById("legend");
    el.hidden = !html;
    if (html) el.innerHTML = html;
  }

  const gradient = `linear-gradient(to right, ${NTL_RAMP.join(", ")})`;

  function nightLegend() {
    return `<div class="legend-title">Night-time light</div>
      <div class="legend-bar" style="background:${gradient}"></div>
      <div class="legend-scale"><span>0</span><span>6</span><span>45</span><span>300+</span></div>
      <div class="legend-note">Radiance, nW/cm²/sr (log scale) · ${ntlMeta.source}</div>`;
  }

  function bandRows(bands, s) {
    return bands.map((band, i) => `<div class="legend-band">
        <i style="background:${band.color}"></i><span>${band.label}</span>
        <b>${Math.round(s.shares[i] * 100)}%</b>
      </div>`).join("");
  }

  function distanceLegend() {
    const s = stats.distance;
    return `<div class="legend-title">Distance to nearest subway station</div>
      ${bandRows(DIST_BANDS, s)}
      <div class="legend-note">Share of hexagons, straight-line from cell center · median ${s.median.toLocaleString()} m</div>`;
  }

  function walkLegend() {
    const s = stats.walk;
    if (s.median == null) return `<b>Walking data not found.</b><br>Run <code>python scripts/features_walk.py</code>.`;
    return `<div class="legend-title">Walking distance to a subway entrance</div>
      ${bandRows(DIST_BANDS, s)}
      <div class="legend-note">Share of hexagons, along streets and paths from cell center to the nearest entrance ·
        median ${Math.round(s.median).toLocaleString()} m (straight line: ${stats.distance.median.toLocaleString()} m) · capped at 5 km</div>`;
  }

  function populationLegend() {
    const s = stats.population;
    if (s.median == null) return `<b>Population data not found.</b><br>Run <code>python scripts/features_demographics.py</code>.`;
    return `<div class="legend-title">Population density</div>
      ${bandRows(POP_BANDS, s)}
      <div class="legend-note">Residents per km² of land, share of hexagons · 2020 Census blocks ·
        ${(s.total / 1e6).toFixed(2)} million people in the grid</div>`;
  }

  function hexLegend() {
    if (hasHexNtl) {
      return `<div class="legend-title">Hex grid · mean night light</div>
        <div class="legend-bar" style="background:${gradient}"></div>
        <div class="legend-scale"><span>0</span><span>6</span><span>45</span><span>300+</span></div>
        <div class="legend-note">500 m-radius hexagons, cut to the shoreline · bright outline = has a subway station</div>`;
    }
    return `<div class="legend-title">Hex grid · subway stations</div>
      <div class="legend-swatches">
        <span><i style="background:#1b2230"></i>0</span><span><i style="background:#2f5f8f"></i>1</span>
        <span><i style="background:#4f8fd0"></i>2–3</span><span><i style="background:#9fd0ff"></i>4+</span>
      </div>
      <div class="legend-note">500 m-radius hexagons, cut to the shoreline · run scripts/gee_ntl.py to color by night light</div>`;
  }

  return {
    async init(mapInstance) {
      map = mapInstance;
      await Promise.all([loadNtl(), loadHexes()]);
      document.querySelectorAll("#views-mode button").forEach((b) => (b.onclick = () => setView(b.dataset.view)));
    },
    get current() { return current; },
  };
})();
