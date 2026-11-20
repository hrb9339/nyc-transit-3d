// "Ridership by hour" view: subway entries at each station complex in each clock hour,
// averaged over the last 6 months. Each station is a circle whose area is proportional to
// its entries in the selected hour; a day-type switch, an hour slider, a clickable citywide
// chart and a play button step through the day.
// Ridership is counted at stations, so it is drawn at stations, not spread over hexagons
// or smoothed into a heatmap (which would suggest coverage where there is no station).
// Data: data/ridership/hourly_web.json (scripts/features_ridership_hourly.py).

const Ridership = (() => {
  const RAMP = ["#5c1a6b", "#a3306e", "#e0583f", "#f9a23a", "#fde68a"];
  const GRADIENT = `linear-gradient(to right, ${RAMP.join(", ")})`;
  const DAY_TYPES = { weekday: "Weekday", saturday: "Saturday", sunday: "Sunday" };
  const LAYERS = ["ridership-circles"];
  // Circle radius = sqrt(entries per hour) × this factor, which grows with zoom.
  const SCALE_STOPS = [[10, 0.2], [12, 0.33], [14, 0.6], [16, 1.2]];

  let map;
  let popup;
  let rides = null;   // hourly_web.json, plus byId
  let norm = 1;       // entries per hour at the top of the color scale
  const state = { day: "weekday", hour: 8, playing: null };

  const fmtHour = (h) => `${h % 12 === 0 ? 12 : h % 12}${h < 12 || h === 24 ? " am" : " pm"}`;
  const hourLabel = (h) => `${fmtHour(h)}–${fmtHour(h + 1)}`;
  const n0 = (x) => Math.round(x).toLocaleString();

  function scaleAt(zoom) {
    if (zoom <= SCALE_STOPS[0][0]) return SCALE_STOPS[0][1];
    for (let i = 1; i < SCALE_STOPS.length; i++) {
      const [z1, s1] = SCALE_STOPS[i];
      const [z0, s0] = SCALE_STOPS[i - 1];
      if (zoom <= z1) return s0 + ((zoom - z0) / (z1 - z0)) * (s1 - s0);
    }
    return SCALE_STOPS[SCALE_STOPS.length - 1][1];
  }

  // 24 bars; the selected hour highlighted. Returns an SVG string.
  function bars(values, hour, { w = 252, h = 54, clickable = false } = {}) {
    const max = Math.max(...values, 1);
    const bw = w / 24;
    return `<svg class="hour-bars" width="${w}" height="${h}" viewBox="0 0 ${w} ${h}">${values.map((v, i) => {
      const bh = Math.max(1, (v / max) * (h - 2));
      return `<rect ${clickable ? `data-hour="${i}"` : ""} x="${i * bw + 1}" y="${h - bh}" width="${bw - 2}" height="${bh}"
        rx="1.5" fill="${i === hour ? "#fbc54a" : "#4f5d78"}"><title>${hourLabel(i)}: ${n0(v)}</title></rect>`;
    }).join("")}</svg>`;
  }

  function features() {
    return {
      type: "FeatureCollection",
      features: rides.complexes.map((c) => ({
        type: "Feature",
        geometry: { type: "Point", coordinates: [c.lon, c.lat] },
        properties: { id: c.id, v: c[state.day][state.hour] },
      })),
    };
  }

  async function load() {
    try {
      const r = await fetch("data/ridership/hourly_web.json", { cache: "no-store" });
      if (!r.ok) throw new Error(r.status);
      rides = await r.json();
    } catch {
      return;
    }
    rides.byId = new Map(rides.complexes.map((c) => [c.id, c]));
    // Color saturates at the 99th percentile of all weekday station-hours (rush hour at big hubs).
    const all = rides.complexes.flatMap((c) => c.weekday).filter((x) => x > 0).sort((a, b) => a - b);
    norm = all[Math.floor(all.length * 0.99)] || 1;

    map.addSource("ridership-hourly", { type: "geojson", data: features() });
    map.addLayer({
      id: "ridership-circles", type: "circle", source: "ridership-hourly",
      layout: { visibility: "none", "circle-sort-key": ["-", 0, ["get", "v"]] },   // small circles on top
      paint: {
        // ["zoom"] may only appear at the top level, so interpolate between per-zoom size formulas.
        "circle-radius": ["interpolate", ["linear"], ["zoom"],
          ...SCALE_STOPS.flatMap(([z, k]) => [z, ["max", 1.5, ["*", ["sqrt", ["get", "v"]], k]]])],
        "circle-color": ["interpolate", ["linear"], ["sqrt", ["/", ["get", "v"], norm]],
          0, RAMP[0], 0.25, RAMP[1], 0.5, RAMP[2], 0.75, RAMP[3], 1, RAMP[4]],
        "circle-opacity": 0.8,
        "circle-stroke-color": "rgba(14,17,22,0.8)",
        "circle-stroke-width": 0.6,
      },
    });

    map.on("mousemove", "ridership-circles", (e) => {
      const c = rides.byId.get(e.features[0].properties.id);
      const day = c[state.day];
      const wrap = document.createElement("div");
      wrap.innerHTML = `<div class="popup-title"></div><table class="popup-table"></table>${bars(day, state.hour, { w: 220, h: 40 })}`;
      wrap.querySelector(".popup-title").textContent = c.name;
      const table = wrap.querySelector("table");
      for (const [k, v] of [
        [`${DAY_TYPES[state.day]} ${hourLabel(state.hour)}`, `${n0(day[state.hour])} entries`],
        [`${DAY_TYPES[state.day]} total`, `${n0(day.reduce((s, x) => s + x, 0))} entries`],
      ]) {
        const tr = table.insertRow();
        tr.insertCell().textContent = k;
        tr.insertCell().textContent = v;
      }
      popup.setLngLat(e.lngLat).setDOMContent(wrap).addTo(map);
    });
    map.on("mouseleave", "ridership-circles", () => popup.remove());
    // Keep the size key in the legend true to the circles on screen.
    map.on("zoomend", () => {
      const el = document.getElementById("legend");
      if (map.getLayer("ridership-circles") && map.getLayoutProperty("ridership-circles", "visibility") === "visible"
          && el.querySelector("[data-sizes]")) {
        el.querySelector("[data-sizes]").innerHTML = sizeKey();
      }
    });
  }

  // ---------- Legend + controls ----------

  function sizeKey() {
    const k = scaleAt(map.getZoom());
    return [100, 1000, 5000, 10000].map((v) => {
      const r = Math.max(1.5, Math.sqrt(v) * k);
      return `<span class="size-key"><svg width="${2 * r + 2}" height="${2 * r + 2}">
        <circle cx="${r + 1}" cy="${r + 1}" r="${r}" fill="none" stroke="#c9cfdb" stroke-width="1"/></svg>
        ${v.toLocaleString()}</span>`;
    }).join("");
  }

  function setHour(h, el) {
    state.hour = ((h % 24) + 24) % 24;
    map.getSource("ridership-hourly").setData(features());
    if (el) update(el);
  }

  function stopPlay(el) {
    clearInterval(state.playing);
    state.playing = null;
    const b = el && el.querySelector("[data-play]");
    if (b) b.textContent = "▶ Play";
  }

  function renderLegend(el) {
    if (!rides) {
      el.innerHTML = `<b>Hourly ridership not found.</b><br>Run <code>python scripts/features_ridership_hourly.py</code>.`;
      return;
    }
    el.innerHTML = `<div class="legend-title">Subway entries by hour</div>
      <div class="seg">${Object.entries(DAY_TYPES).map(([k, v]) =>
        `<button type="button" data-day="${k}">${v}</button>`).join("")}</div>
      <div class="hour-head"><b data-hour-label></b><span data-hour-total></span></div>
      <div data-bars></div>
      <div class="hour-row">
        <button type="button" data-play>▶ Play</button>
        <input type="range" min="0" max="23" step="1" data-slider aria-label="Hour of day" />
      </div>
      <div class="size-keys" data-sizes>${sizeKey()}</div>
      <div class="legend-bar" style="background:${GRADIENT}"></div>
      <div class="legend-scale"><span>few</span><span>${n0(norm)}+ entries/hour</span></div>
      <div class="legend-note">Circle area = mean entries at the station complex in that clock hour, ${rides.period}
        (${rides.days.weekday} weekdays, ${rides.days.saturday} Saturdays, ${rides.days.sunday} Sundays) ·
        ${rides.source}</div>`;
    el.querySelectorAll("[data-day]").forEach((b) => (b.onclick = () => {
      state.day = b.dataset.day;
      setHour(state.hour, el);
    }));
    el.querySelector("[data-slider]").oninput = (e) => {
      stopPlay(el);
      setHour(+e.target.value, el);
    };
    el.querySelector("[data-play]").onclick = (e) => {
      if (state.playing) return stopPlay(el);
      e.target.textContent = "❚❚ Pause";
      state.playing = setInterval(() => setHour(state.hour + 1, el), 800);
    };
    update(el);
  }

  function update(el) {
    const city = rides.citywide[state.day];
    el.querySelectorAll("[data-day]").forEach((b) => b.classList.toggle("active", b.dataset.day === state.day));
    el.querySelector("[data-hour-label]").textContent = `${DAY_TYPES[state.day]}, ${hourLabel(state.hour)}`;
    el.querySelector("[data-hour-total]").textContent = `${n0(city[state.hour])} entries citywide`;
    el.querySelector("[data-slider]").value = state.hour;
    const box = el.querySelector("[data-bars]");
    box.innerHTML = bars(city, state.hour, { clickable: true });
    box.querySelectorAll("rect").forEach((r) => (r.onclick = () => {
      stopPlay(el);
      setHour(+r.dataset.hour, el);
    }));
  }

  // ---------- Station and train dots ----------

  // Fade the regular station and live-train dots so the ridership circles read clearly.
  const DIM = [
    ["subway-stations", ["circle-opacity", "circle-stroke-opacity"], 0.15],
    ["trains-halo", ["circle-opacity"], 0],
    ["trains", ["circle-opacity", "circle-stroke-opacity"], 0.25],
    ["subway-station-labels", ["text-opacity"], 0.5],
    ["trains-label", ["text-opacity"], 0],
  ];
  const saved = new Map();
  function dimNetwork(on) {
    for (const [id, props, value] of DIM) {
      if (!map.getLayer(id)) continue;
      for (const prop of props) {
        const key = `${id}|${prop}`;
        if (!saved.has(key)) saved.set(key, map.getPaintProperty(id, prop) ?? 1);
        map.setPaintProperty(id, prop, on ? value : saved.get(key));
      }
    }
  }

  return {
    async init(mapInstance) {
      map = mapInstance;
      popup = new maplibregl.Popup({ closeButton: false, closeOnClick: false, offset: 8, maxWidth: "260px" });
      await load();
    },
    show(name) {
      const on = name === "ridership";
      LAYERS.forEach((id) => map.getLayer(id) && map.setLayoutProperty(id, "visibility", on ? "visible" : "none"));
      dimNetwork(on);
      if (!on) stopPlay(document.getElementById("legend"));
    },
    renderLegend,
  };
})();
