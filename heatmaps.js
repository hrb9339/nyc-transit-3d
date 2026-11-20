// Heatmap views:
//  - ridership: subway entries per station complex by hour of day (last 6 months), with a
//    day-type switch, an hour slider and a play button that steps through the day
//  - students:  K-12 and college enrollment at each school / campus
//  - people:    residents, workers, students, or the daytime population W + S + a·R per km²
// Data: data/ridership/hourly_web.json (scripts/features_ridership_hourly.py),
//       data/students/students_web.json (scripts/features_students.py),
//       data/features/demographics.csv + students.csv on the hexagon centers.

const Heatmaps = (() => {
  const RAMP = ["interpolate", ["linear"], ["heatmap-density"],
    0, "rgba(0,0,0,0)", 0.08, "rgba(60,20,110,0.55)", 0.25, "#7b2a8c", 0.45, "#c43c6a",
    0.65, "#f0703a", 0.85, "#fbc54a", 1, "#fff7d6"];
  const GRADIENT = "linear-gradient(to right, rgba(60,20,110,0.55), #7b2a8c, #c43c6a, #f0703a, #fbc54a, #fff7d6)";
  // Radius that grows with the map, so a blob covers the same ground at every zoom.
  const groundRadius = (pxAtZoom10) => ["interpolate", ["exponential", 2], ["zoom"], 9, pxAtZoom10 / 2, 16, pxAtZoom10 * 64];

  const DAY_TYPES = { weekday: "Weekday", saturday: "Saturday", sunday: "Sunday" };
  const KINDS = {
    public: { label: "Public", color: "#4da3ff" },
    charter: { label: "Charter", color: "#a78bfa" },
    private: { label: "Private", color: "#34d399" },
    college: { label: "College", color: "#fbbf24" },
  };
  const PEOPLE = {
    residents: { label: "Residents", note: "where people live (2020 Census blocks)" },
    workers: { label: "Workers", note: "where jobs are (LODES 2021)" },
    students: { label: "Students", note: "where schools and colleges are" },
    daytime: { label: "Daytime", note: "workers + students + a × residents" },
  };

  const LAYERS = {
    ridership: ["ridership-heat", "ridership-dots"],
    students: ["students-heat", "students-dots"],
    people: ["people-heat"],
  };

  let map;
  let popup;
  const state = { day: "weekday", hour: 8, playing: null, kinds: new Set(Object.keys(KINDS)), people: "daytime", a: 0.5 };
  let rides = null;      // hourly_web.json
  let rideNorm = 1;      // hourly entries that saturate the heatmap
  let students = null;   // students_web.json
  let people = null;     // { features, totals, norms }

  // ---------- Formatting ----------

  const fmtHour = (h) => `${h % 12 === 0 ? 12 : h % 12}${h < 12 || h === 24 ? " am" : " pm"}`;
  const hourLabel = (h) => `${fmtHour(h)}–${fmtHour(h + 1)}`;
  const n0 = (x) => Math.round(x).toLocaleString();
  const short = (x) => (x >= 1e6 ? `${(x / 1e6).toFixed(2)}M` : x >= 1e3 ? `${Math.round(x / 1e3)}k` : `${Math.round(x)}`);

  function quantile(values, q) {
    const v = values.filter((x) => x > 0).sort((a, b) => a - b);
    return v.length ? v[Math.min(v.length - 1, Math.floor(q * v.length))] : 1;
  }

  // 24 bars; the selected hour highlighted. Returns an SVG string.
  function bars(values, hour, { w = 252, h = 54, color = "#4f5d78", hi = "#fbc54a", clickable = false } = {}) {
    const max = Math.max(...values, 1);
    const bw = w / 24;
    return `<svg class="hour-bars" width="${w}" height="${h}" viewBox="0 0 ${w} ${h}">${values.map((v, i) => {
      const bh = Math.max(1, (v / max) * (h - 2));
      return `<rect ${clickable ? `data-hour="${i}"` : ""} x="${i * bw + 1}" y="${h - bh}" width="${bw - 2}" height="${bh}"
        rx="1.5" fill="${i === hour ? hi : color}"><title>${hourLabel(i)}: ${n0(v)}</title></rect>`;
    }).join("")}</svg>`;
  }

  // ---------- Data ----------

  async function getJson(url) {
    try {
      const r = await fetch(url, { cache: "no-store" });
      if (!r.ok) throw new Error(r.status);
      return await r.json();
    } catch {
      return null;
    }
  }

  function rideFeatures() {
    return {
      type: "FeatureCollection",
      features: rides.complexes.map((c) => ({
        type: "Feature",
        geometry: { type: "Point", coordinates: [c.lon, c.lat] },
        properties: { id: c.id, v: c[state.day][state.hour] },
      })),
    };
  }

  async function loadRidership() {
    rides = await getJson("data/ridership/hourly_web.json");
    if (!rides) return;
    rides.byId = new Map(rides.complexes.map((c) => [c.id, c]));
    // The 99th percentile of every weekday station-hour: rush hour at the big hubs saturates.
    rideNorm = quantile(rides.complexes.flatMap((c) => c.weekday), 0.99);
    map.addSource("ridership-hourly", { type: "geojson", data: rideFeatures() });
    map.addLayer({
      id: "ridership-heat", type: "heatmap", source: "ridership-hourly",
      layout: { visibility: "none" },
      paint: {
        "heatmap-weight": ["min", 1.5, ["/", ["get", "v"], rideNorm]],
        "heatmap-radius": ["interpolate", ["exponential", 2], ["zoom"], 9, 16, 11, 34, 13, 90, 16, 400],
        "heatmap-intensity": ["interpolate", ["linear"], ["zoom"], 9, 1.6, 13, 1],
        "heatmap-color": RAMP,
        "heatmap-opacity": 0.9,
      },
    }, "subway-lines");
    map.addLayer({
      id: "ridership-dots", type: "circle", source: "ridership-hourly",
      minzoom: 12,
      layout: { visibility: "none" },
      paint: {
        "circle-radius": ["interpolate", ["linear"], ["sqrt", ["get", "v"]], 0, 2, 30, 8, 100, 22],
        "circle-color": "rgba(255,255,255,0.12)",
        "circle-stroke-color": "rgba(255,255,255,0.7)",
        "circle-stroke-width": 1,
      },
    });
    hover("ridership-dots", (p) => {
      const c = rides.byId.get(p.id);
      const day = c[state.day];
      return {
        title: c.name,
        rows: [
          [`${DAY_TYPES[state.day]} ${hourLabel(state.hour)}`, `${n0(day[state.hour])} entries`],
          [`${DAY_TYPES[state.day]} total`, `${n0(day.reduce((s, x) => s + x, 0))} entries`],
        ],
        extra: bars(day, state.hour, { w: 220, h: 40 }),
      };
    });
  }

  async function loadStudents() {
    students = await getJson("data/students/students_web.json");
    if (!students) return;
    const fc = {
      type: "FeatureCollection",
      features: students.points.map(([lon, lat, n, kind, name]) => ({
        type: "Feature", geometry: { type: "Point", coordinates: [lon, lat] }, properties: { n, kind, name },
      })),
    };
    map.addSource("students", { type: "geojson", data: fc });
    const norm = quantile(students.points.map((p) => p[2]), 0.97);
    map.addLayer({
      id: "students-heat", type: "heatmap", source: "students",
      layout: { visibility: "none" },
      paint: {
        "heatmap-weight": ["min", 2, ["/", ["get", "n"], norm]],
        "heatmap-radius": ["interpolate", ["exponential", 2], ["zoom"], 9, 8, 12, 34, 16, 260],
        "heatmap-intensity": 0.6,
        "heatmap-color": RAMP,
        "heatmap-opacity": 0.9,
      },
    }, "subway-lines");
    map.addLayer({
      id: "students-dots", type: "circle", source: "students",
      minzoom: 12.5,
      layout: { visibility: "none" },
      paint: {
        "circle-radius": ["interpolate", ["linear"], ["sqrt", ["get", "n"]], 0, 2, 30, 6, 230, 24],
        "circle-color": ["match", ["get", "kind"], ...Object.entries(KINDS).flatMap(([k, v]) => [k, v.color]), "#999"],
        "circle-opacity": 0.55,
        "circle-stroke-color": "rgba(0,0,0,0.5)",
        "circle-stroke-width": 0.5,
      },
    });
    hover("students-dots", (p) => ({
      title: p.name,
      rows: [["Type", KINDS[p.kind].label], ["Students", n0(p.n)]],
    }));
  }

  async function loadPeople(grid, loadFeatureCsv) {
    const [demo, stud] = await Promise.all([loadFeatureCsv("demographics"), loadFeatureCsv("students")]);
    if (!demo.size) return;
    const feats = grid.features.map((f) => {
      const p = f.properties;
      const d = demo.get(p.hex_id) || {};
      const s = stud.get(p.hex_id) || {};
      const land = Math.max(p.land_km2 || 0, 0.05);   // tiny shoreline slivers would blow up densities
      return {
        type: "Feature",
        geometry: { type: "Point", coordinates: p.center },
        properties: { land, residents: d.population || 0, workers: d.jobs || 0, students: s.students || 0 },
      };
    });
    people = { features: feats, hasStudents: stud.size > 0 };
    map.addSource("people", { type: "geojson", data: peopleData() });
    map.addLayer({
      id: "people-heat", type: "heatmap", source: "people",
      layout: { visibility: "none" },
      paint: {
        "heatmap-weight": ["min", 1.5, ["/", ["get", "d"], ["get", "norm"]]],
        // Hex centers are ~870 m apart; a radius of ~3 spacings blends them into a smooth
        // surface (smaller shows the honeycomb) at every zoom.
        "heatmap-radius": groundRadius(26),
        "heatmap-intensity": 0.35,
        "heatmap-color": RAMP,
        "heatmap-opacity": 0.92,
      },
    }, "subway-lines");
  }

  // Density per km² for the current mode, with the 97th percentile as the saturation point.
  function peopleData() {
    const val = (p) => (state.people === "daytime" ? p.workers + p.students + state.a * p.residents : p[state.people]);
    const dens = people.features.map((f) => val(f.properties) / f.properties.land);
    const norm = quantile(dens, 0.97);
    people.norm = norm;
    people.total = people.features.reduce((s, f) => s + val(f.properties), 0);
    return {
      type: "FeatureCollection",
      features: people.features.map((f, i) => ({ ...f, properties: { d: dens[i], norm } })),
    };
  }

  // ---------- Popups ----------

  function hover(layer, describe) {
    map.on("mousemove", layer, (e) => {
      const { title, rows, extra } = describe(e.features[0].properties);
      const wrap = document.createElement("div");
      wrap.innerHTML = `<div class="popup-title"></div><table class="popup-table"></table>${extra || ""}`;
      wrap.querySelector(".popup-title").textContent = title;
      const table = wrap.querySelector("table");
      for (const [k, v] of rows) {
        const tr = table.insertRow();
        tr.insertCell().textContent = k;
        tr.insertCell().textContent = v;
      }
      popup.setLngLat(e.lngLat).setDOMContent(wrap).addTo(map);
    });
    map.on("mouseleave", layer, () => popup.remove());
  }

  // ---------- Ridership controls ----------

  function setHour(h, el) {
    state.hour = ((h % 24) + 24) % 24;
    map.getSource("ridership-hourly").setData(rideFeatures());
    if (el) updateRidershipLegend(el);
  }

  function stopPlay(el) {
    clearInterval(state.playing);
    state.playing = null;
    const b = el && el.querySelector("[data-play]");
    if (b) b.textContent = "▶ Play";
  }

  function ridershipLegend(el) {
    if (!rides) {
      el.innerHTML = `<b>Hourly ridership not found.</b><br>Run <code>python scripts/features_ridership_hourly.py</code>.`;
      return;
    }
    el.innerHTML = `<div class="legend-title">Subway entries by hour</div>
      <div class="seg" data-seg="day">${Object.entries(DAY_TYPES).map(([k, v]) =>
        `<button type="button" data-day="${k}">${v}</button>`).join("")}</div>
      <div class="hour-head"><b data-hour-label></b><span data-hour-total></span></div>
      <div data-bars></div>
      <div class="hour-row">
        <button type="button" data-play>▶ Play</button>
        <input type="range" min="0" max="23" step="1" data-slider aria-label="Hour of day" />
      </div>
      <div class="legend-bar" style="background:${GRADIENT}"></div>
      <div class="legend-scale"><span>few</span><span>${n0(rideNorm)}+ entries/hour at a station</span></div>
      <div class="legend-note">Mean entries per station complex in each clock hour, ${rides.period}
        (${rides.days.weekday} weekdays, ${rides.days.saturday} Saturdays, ${rides.days.sunday} Sundays) ·
        ${rides.source} · zoom in for stations</div>`;
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
    updateRidershipLegend(el);
  }

  function updateRidershipLegend(el) {
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

  // ---------- Students controls ----------

  function studentsLegend(el) {
    if (!students) {
      el.innerHTML = `<b>Student data not found.</b><br>Run <code>python scripts/features_students.py</code>.`;
      return;
    }
    const t = students.totals;
    const total = Object.values(t).reduce((s, x) => s + x, 0);
    el.innerHTML = `<div class="legend-title">Students · where schools and colleges are</div>
      ${Object.entries(KINDS).map(([k, v]) => `<label class="legend-band">
        <input type="checkbox" data-kind="${k}" ${state.kinds.has(k) ? "checked" : ""} />
        <i style="background:${v.color}"></i><span>${v.label}</span><b>${short(t[k] || 0)}</b></label>`).join("")}
      <div class="legend-bar" style="background:${GRADIENT}; margin-top:8px"></div>
      <div class="legend-scale"><span>fewer</span><span>more students</span></div>
      <div class="legend-note">${n0(total)} students at ${n0(students.points.length)} schools and colleges ·
        enrollment counted at the school, not at home · ${students.sources} · zoom in for schools</div>`;
    el.querySelectorAll("[data-kind]").forEach((cb) => (cb.onchange = () => {
      cb.checked ? state.kinds.add(cb.dataset.kind) : state.kinds.delete(cb.dataset.kind);
      const filter = ["in", ["get", "kind"], ["literal", [...state.kinds]]];
      LAYERS.students.forEach((id) => map.setFilter(id, filter));
    }));
  }

  // ---------- People controls ----------

  function peopleLegend(el) {
    if (!people) {
      el.innerHTML = `<b>Population data not found.</b><br>Run <code>python scripts/features_demographics.py</code>.`;
      return;
    }
    el.innerHTML = `<div class="legend-title">Where people are · heatmap</div>
      <div class="seg" data-seg="people">${Object.entries(PEOPLE).map(([k, v]) =>
        `<button type="button" data-people="${k}">${v.label}</button>`).join("")}</div>
      <div class="a-row" ${state.people === "daytime" ? "" : "hidden"}>
        <span>a = <b data-a-label></b></span>
        <input type="range" min="0" max="1" step="0.05" value="${state.a}" data-a aria-label="Share of residents home during the day" />
      </div>
      <div class="legend-bar" style="background:${GRADIENT}"></div>
      <div class="legend-scale"><span>0</span><span data-norm></span></div>
      <div class="legend-note" data-people-note></div>`;
    el.querySelectorAll("[data-people]").forEach((b) => (b.onclick = () => {
      state.people = b.dataset.people;
      refreshPeople(el);
    }));
    el.querySelector("[data-a]").oninput = (e) => {
      state.a = +e.target.value;
      refreshPeople(el);
    };
    refreshPeople(el);
  }

  function refreshPeople(el) {
    map.getSource("people").setData(peopleData());
    el.querySelectorAll("[data-people]").forEach((b) => b.classList.toggle("active", b.dataset.people === state.people));
    el.querySelector(".a-row").hidden = state.people !== "daytime";
    el.querySelector("[data-a-label]").textContent = state.a.toFixed(2);
    el.querySelector("[data-norm]").textContent = `${n0(people.norm)}+ per km²`;
    const missing = state.people !== "residents" && state.people !== "workers" && !people.hasStudents
      ? " · student data missing: run scripts/features_students.py" : "";
    el.querySelector("[data-people-note]").innerHTML = state.people === "daytime"
      ? `Weekday daytime population Pop_day = W + S + a·R per km² of land: workers + students + the share a of
         residents assumed to stay near home (an assumption, not learned) · ${short(people.total)} people${missing}`
      : `${PEOPLE[state.people].label} per km² of land, ${PEOPLE[state.people].note} · ${short(people.total)} in the grid${missing}`;
  }

  // Station and live-train dots would hide the heatmap, so fade them while a heatmap is shown.
  const DIM = [
    ["subway-stations", ["circle-opacity", "circle-stroke-opacity"], 0.2],
    ["trains-halo", ["circle-opacity"], 0],
    ["trains", ["circle-opacity", "circle-stroke-opacity"], 0.25],
    ["subway-station-labels", ["text-opacity"], 0.4],
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

  // ---------- Public ----------

  return {
    views: Object.keys(LAYERS),
    async init(mapInstance, grid, loadFeatureCsv) {
      map = mapInstance;
      popup = new maplibregl.Popup({ closeButton: false, closeOnClick: false, offset: 8, maxWidth: "260px" });
      await Promise.all([loadRidership(), loadStudents(), loadPeople(grid, loadFeatureCsv)]);
    },
    // Show this module's layers for `name` (or hide them all for any other view).
    show(name) {
      for (const [view, ids] of Object.entries(LAYERS)) {
        ids.forEach((id) => map.getLayer(id) && map.setLayoutProperty(id, "visibility", view === name ? "visible" : "none"));
      }
      dimNetwork(name in LAYERS);
      if (name !== "ridership") stopPlay(document.getElementById("legend"));
    },
    renderLegend(name, el) {
      ({ ridership: ridershipLegend, students: studentsLegend, people: peopleLegend })[name](el);
    },
  };
})();
