// NYC Transit 3D — MapLibre GL + OpenFreeMap (no API key required).

const STYLE_URL = "https://tiles.openfreemap.org/styles/dark";

const VIEWS = {
  "Overview":        { center: [-73.94, 40.72], zoom: 11, pitch: 50, bearing: -20 },
  "Midtown":         { center: [-73.985, 40.755], zoom: 15, pitch: 65, bearing: 29 },
  "Lower Manhattan": { center: [-74.010, 40.708], zoom: 15.3, pitch: 65, bearing: -30 },
  "Brooklyn":        { center: [-73.985, 40.692], zoom: 14.5, pitch: 60, bearing: 10 },
  "Queens / LIC":    { center: [-73.945, 40.747], zoom: 14.5, pitch: 60, bearing: -40 },
  "Bronx":           { center: [-73.900, 40.835], zoom: 13.5, pitch: 55, bearing: 0 },
};

const map = new maplibregl.Map({
  container: "map",
  style: STYLE_URL,
  ...VIEWS["Overview"],
  maxPitch: 80,
  antialias: true,
});

map.addControl(new maplibregl.NavigationControl({ visualizePitch: true }), "top-right");
map.addControl(new maplibregl.ScaleControl({ unit: "metric" }), "bottom-left");

let routes = [];                 // line features' properties, in MTA sort order
const hiddenRoutes = new Set();  // route_ids currently toggled off

map.on("load", async () => {
  const [lines, stations] = await Promise.all([
    fetch("data/subway-lines.geojson").then((r) => r.json()),
    fetch("data/subway-stations.geojson").then((r) => r.json()),
  ]);
  routes = lines.features.map((f) => f.properties);

  addBuildings();
  addSubway(lines, stations);
  buildPanel();
  setupInteractions();
  Realtime.init(map, routes)
    .then(applyRouteFilter)
    .catch((err) => console.error("Live trains unavailable", err));
  Views.init(map);

  map.setSky({
    "sky-color": "#0b1a2e",
    "horizon-color": "#1d2b44",
    "fog-color": "#0e1116",
    "sky-horizon-blend": 0.6,
    "horizon-fog-blend": 0.7,
    "fog-ground-blend": 0.4,
  });
});

function addBuildings() {
  // Replace the flat building footprints with extruded ones.
  if (map.getLayer("building")) map.removeLayer("building");

  map.addLayer({
    id: "buildings-3d",
    type: "fill-extrusion",
    source: "openmaptiles",
    "source-layer": "building",
    minzoom: 12,
    paint: {
      "fill-extrusion-color": [
        "interpolate", ["linear"], ["coalesce", ["get", "render_height"], 0],
        0, "#1f2633",
        50, "#2b3546",
        150, "#3a4860",
        300, "#4f6283",
      ],
      "fill-extrusion-height": ["coalesce", ["get", "render_height"], 0],
      "fill-extrusion-base": ["coalesce", ["get", "render_min_height"], 0],
      "fill-extrusion-opacity": 0.85,
    },
  });
}

function addSubway(lines, stations) {
  map.addSource("subway-lines", { type: "geojson", data: lines });
  map.addSource("subway-stations", { type: "geojson", data: stations });

  // Soft glow underneath each line.
  map.addLayer({
    id: "subway-glow",
    type: "line",
    source: "subway-lines",
    layout: { "line-join": "round", "line-cap": "round" },
    paint: {
      "line-color": ["get", "color"],
      "line-width": ["interpolate", ["linear"], ["zoom"], 10, 6, 15, 18],
      "line-blur": ["interpolate", ["linear"], ["zoom"], 10, 4, 15, 12],
      "line-opacity": 0.35,
    },
  });

  map.addLayer({
    id: "subway-lines",
    type: "line",
    source: "subway-lines",
    layout: { "line-join": "round", "line-cap": "round" },
    paint: {
      "line-color": ["get", "color"],
      "line-width": ["interpolate", ["linear"], ["zoom"], 10, 1.5, 13, 3, 16, 6],
    },
  });

  map.addLayer({
    id: "subway-stations",
    type: "circle",
    source: "subway-stations",
    minzoom: 11,
    paint: {
      "circle-radius": ["interpolate", ["linear"], ["zoom"], 11, 2, 15, 6],
      "circle-color": "#ffffff",
      "circle-stroke-color": "#0e1116",
      "circle-stroke-width": 1.5,
    },
  });

  map.addLayer({
    id: "subway-station-labels",
    type: "symbol",
    source: "subway-stations",
    minzoom: 14,
    layout: {
      "text-field": ["get", "name"],
      "text-font": ["Noto Sans Regular"],
      "text-size": 11,
      "text-offset": [0, 1.2],
      "text-anchor": "top",
      "text-optional": true,
    },
    paint: {
      "text-color": "#e6e9ef",
      "text-halo-color": "#0e1116",
      "text-halo-width": 1.5,
    },
  });
}

// ---------- Panel ----------

function bulletEl(props, tag = "button") {
  const el = document.createElement(tag);
  el.className = "bullet" + (props.name.length > 1 ? " wide" : "");
  el.textContent = props.name;
  el.style.background = props.color;
  el.style.color = props.text_color;
  el.title = props.long_name;
  return el;
}

function buildPanel() {
  const routesEl = document.getElementById("routes");
  for (const props of routes) {
    const b = bulletEl(props);
    b.type = "button";
    b.dataset.route = props.route_id;
    b.addEventListener("click", () => {
      hiddenRoutes.has(props.route_id) ? hiddenRoutes.delete(props.route_id) : hiddenRoutes.add(props.route_id);
      applyRouteFilter();
    });
    routesEl.appendChild(b);
  }

  document.getElementById("all-on").onclick = () => { hiddenRoutes.clear(); applyRouteFilter(); };
  document.getElementById("all-off").onclick = () => {
    routes.forEach((r) => hiddenRoutes.add(r.route_id));
    applyRouteFilter();
  };

  const viewsEl = document.getElementById("views");
  for (const [name, view] of Object.entries(VIEWS)) {
    const b = document.createElement("button");
    b.type = "button";
    b.textContent = name;
    b.onclick = () => map.flyTo({ ...view, duration: 3000, essential: true });
    viewsEl.appendChild(b);
  }

  // Buildings only belong to the 3D city view.
  document.getElementById("tg-buildings").onchange = (e) =>
    map.setLayoutProperty("buildings-3d", "visibility", e.target.checked && Views.current === "city" ? "visible" : "none");
  bindVisibility("tg-stations", ["subway-stations", "subway-station-labels"]);
  document.getElementById("tg-trains").onchange = (e) => Realtime.setEnabled(e.target.checked);
  document.getElementById("tg-rotate").onchange = (e) => (e.target.checked ? startRotate() : stopRotate());

  const panel = document.getElementById("panel");
  document.getElementById("panel-toggle").onclick = () => panel.classList.remove("hidden");
  document.getElementById("panel-close").onclick = () => panel.classList.add("hidden");
  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape") panel.classList.add("hidden");
  });
}

function applyRouteFilter() {
  const filter = hiddenRoutes.size
    ? ["!", ["in", ["get", "route_id"], ["literal", [...hiddenRoutes]]]]
    : null;
  ["subway-lines", "subway-glow", ...Realtime.LAYERS].forEach((id) => {
    if (map.getLayer(id)) map.setFilter(id, filter);
  });
  document.querySelectorAll("#routes .bullet").forEach((b) => {
    b.classList.toggle("off", hiddenRoutes.has(b.dataset.route));
  });
}

function bindVisibility(checkboxId, layerIds) {
  document.getElementById(checkboxId).onchange = (e) => {
    const v = e.target.checked ? "visible" : "none";
    layerIds.forEach((id) => map.setLayoutProperty(id, "visibility", v));
  };
}

// ---------- Interactions ----------

function setupInteractions() {
  const popup = new maplibregl.Popup({ closeButton: false, closeOnClick: false, offset: 10 });

  map.on("mousemove", "subway-lines", (e) => {
    // Trains sit on top of the lines and have their own popup.
    if (map.getLayer("trains") && map.queryRenderedFeatures(e.point, { layers: ["trains"] }).length) {
      popup.remove();
      return;
    }
    map.getCanvas().style.cursor = "pointer";
    const p = e.features[0].properties;
    popup.setLngLat(e.lngLat).setHTML(
      `<div class="popup-title">${p.name} train</div><div class="popup-meta">${p.long_name}</div>`
    ).addTo(map);
  });
  map.on("mouseleave", "subway-lines", () => {
    map.getCanvas().style.cursor = "";
    popup.remove();
  });

  map.on("click", "subway-stations", (e) => {
    const p = e.features[0].properties;
    const wrap = document.createElement("div");
    wrap.innerHTML = `<div class="popup-title"></div><div class="popup-routes"></div><div class="popup-meta"></div>`;
    wrap.querySelector(".popup-title").textContent = p.name;
    const routesEl = wrap.querySelector(".popup-routes");
    for (const name of String(p.routes).split(" ").filter(Boolean)) {
      const r = routes.find((x) => x.name === name);
      if (r) routesEl.appendChild(bulletEl(r, "span"));
    }
    wrap.querySelector(".popup-meta").textContent =
      `${p.line} · ${p.structure}${p.ada ? " · ♿ accessible" : ""}`;
    new maplibregl.Popup({ offset: 10 }).setLngLat(e.features[0].geometry.coordinates).setDOMContent(wrap).addTo(map);
  });
  map.on("mouseenter", "subway-stations", () => (map.getCanvas().style.cursor = "pointer"));
  map.on("mouseleave", "subway-stations", () => (map.getCanvas().style.cursor = ""));

  // Stop auto-rotate as soon as the user grabs the map.
  map.on("mousedown", stopRotate);
  map.on("touchstart", stopRotate);
}

// ---------- Auto-rotate ----------

let rotating = false;

function startRotate() {
  rotating = true;
  const step = () => {
    if (!rotating) return;
    map.rotateTo(map.getBearing() + 0.08, { duration: 0 });
    requestAnimationFrame(step);
  };
  step();
}

function stopRotate() {
  rotating = false;
  document.getElementById("tg-rotate").checked = false;
}
