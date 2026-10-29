// Map views: the default 3D city, a top-down night-lights view (VIIRS), and the
// hexagonal analysis grid. Data comes from scripts/build_hexgrid.py and scripts/gee_ntl.py.

const Views = (() => {
  const NYC_BOUNDS = [[-74.26, 40.49], [-73.70, 40.92]];
  const NTL_RAMP = ["#000000", "#1a0b3d", "#5c1a6b", "#b8326b", "#f0703a", "#fbc54a", "#fff7d6"];

  let map;
  let current = "city";
  let ntlMeta = null;   // null until loaded; false if not generated yet
  let hasHexNtl = false;

  const CITY_LAYERS = ["buildings-3d"];
  const NIGHT_LAYERS = ["ntl-raster"];
  const HEX_LAYERS = ["hex-fill", "hex-line"];

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

  async function loadHexes() {
    const grid = await fetch("data/hexgrid.geojson", { cache: "no-store" }).then((r) => r.json());
    hasHexNtl = grid.features.some((f) => f.properties.ntl_mean != null);
    map.addSource("hexgrid", { type: "geojson", data: grid });

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

    const popup = new maplibregl.Popup({ closeButton: false, closeOnClick: false, offset: 8 });
    map.on("mousemove", "hex-fill", (e) => {
      if (current !== "hex") return;
      const p = e.features[0].properties;
      const names = JSON.parse(p.station_names || "[]");
      const rows = [
        ["Borough", p.borough],
        ["Land area", `${p.land_km2} km²`],
        ["Subway stations", names.length ? `${p.stations} (${names.slice(0, 3).join(", ")}${names.length > 3 ? "…" : ""})` : "none"],
        ["Nearest station", `${(p.nearest_station_m / 1000).toFixed(2)} km from center`],
      ];
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
    });
    map.on("mouseleave", "hex-fill", () => popup.remove());
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

    // Let the lights read as the main picture in the night view.
    map.setPaintProperty("subway-lines", "line-opacity", name === "night" ? 0.35 : 1);
    map.setPaintProperty("subway-glow", "line-opacity", name === "night" ? 0 : 0.35);

    if (name === "city") {
      legend(null);
      map.easeTo({ center: [-73.94, 40.72], zoom: 11, pitch: 50, bearing: -20, duration: 1500 });
    } else {
      map.fitBounds(NYC_BOUNDS, { pitch: 0, bearing: 0, padding: 30, duration: 1500 });
      legend(name === "night" ? nightLegend() : hexLegend());
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
