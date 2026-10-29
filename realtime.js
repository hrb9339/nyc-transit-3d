// Live subway trains from the MTA GTFS-realtime feeds (no API key, CORS-enabled).
//
// The feeds don't carry GPS positions, only each train's upcoming stops and
// predicted arrival times. We place every train between its previous and next
// station, interpolated by time, along the real track geometry from network.json.

const FEED_BASE = "https://api-endpoint.mta.info/Dataservice/mtagtfsfeeds/nyct%2F";
const FEEDS = ["gtfs", "gtfs-ace", "gtfs-bdfm", "gtfs-g", "gtfs-jz", "gtfs-nqrw", "gtfs-l", "gtfs-si"];
const REFRESH_MS = 30000;
const FRAME_MS = 60;
const STOPPED_AT = 1;

const Realtime = (() => {
  let map, network, routeColors;
  let trains = [];       // [{id, route, shape, stops: [{stop, arr, dep}], vpStop, status}]  let clockOffset = 0;   // seconds to add to local clock if it disagrees badly with the MTA's
  let enabled = true;
  let lastFrame = 0;
  const segLengths = new Map();

  // ---------- Minimal protobuf reader (only what GTFS-realtime needs) ----------

  function varint(buf, pos) {
    let result = 0, mult = 1, b;
    do {
      b = buf[pos.i++];
      result += (b & 0x7f) * mult;
      mult *= 128;
    } while (b & 0x80);
    return result;
  }

  // Returns {fieldNumber: [values]}; length-delimited values are Uint8Array views.
  function decode(buf) {
    const out = {};
    const pos = { i: 0 };
    while (pos.i < buf.length) {
      const key = varint(buf, pos);
      const field = Math.floor(key / 8), wire = key & 7;
      let value;
      if (wire === 0) value = varint(buf, pos);
      else if (wire === 2) {
        const len = varint(buf, pos);
        value = buf.subarray(pos.i, pos.i + len);
        pos.i += len;
      } else if (wire === 1) { pos.i += 8; continue; }
      else if (wire === 5) { pos.i += 4; continue; }
      else break;
      (out[field] ||= []).push(value);
    }
    return out;
  }

  const text = (bytes) => (bytes ? new TextDecoder().decode(bytes) : "");

  // ---------- Feed parsing ----------

  function parseFeed(buf) {
    const feed = decode(buf);
    const header = feed[1] ? decode(feed[1][0]) : {};
    const updates = new Map(), vehicles = new Map();

    for (const raw of feed[2] || []) {
      const entity = decode(raw);
      if (entity[3]) {
        const tu = decode(entity[3][0]);
        const trip = decode(tu[1][0]);
        const id = text(trip[1]?.[0]);
        const stops = (tu[2] || []).map((s) => {
          const u = decode(s);
          const arr = u[2] ? decode(u[2][0])[2]?.[0] : undefined;
          const dep = u[3] ? decode(u[3][0])[2]?.[0] : undefined;
          return { stop: text(u[4]?.[0]), arr: arr ?? dep, dep: dep ?? arr };
        }).filter((s) => s.stop && s.arr);
        updates.set(id, { id, route: text(trip[5]?.[0]), stops });
      }
      if (entity[4]) {
        const vp = decode(entity[4][0]);
        const id = text(decode(vp[1][0])[1]?.[0]);
        vehicles.set(id, { vpStop: text(vp[7]?.[0]), status: vp[4]?.[0] ?? 2 });
      }
    }

    // Only trips with a vehicle entry are actually running; the rest are scheduled-but-not-departed.
    const out = [];
    for (const [id, v] of vehicles) {
      const u = updates.get(id);
      if (!u) continue;
      out.push({ ...u, ...v, shape: id.split("_").slice(1).join("_") });
    }
    return { trains: out, timestamp: header[3]?.[0] };
  }

  async function refresh() {
    const results = await Promise.allSettled(
      FEEDS.map((f) => fetch(FEED_BASE + f, { cache: "no-store" }).then((r) => {
        if (!r.ok) throw new Error(`${f}: HTTP ${r.status}`);
        return r.arrayBuffer();
      }))
    );

    const next = [];
    let failed = 0, newest = 0;
    for (const r of results) {
      if (r.status !== "fulfilled") { failed++; continue; }
      const parsed = parseFeed(new Uint8Array(r.value));
      next.push(...parsed.trains);
      newest = Math.max(newest, parsed.timestamp || 0);
    }
    if (next.length || failed === FEEDS.length) trains = next;

    // Feed timestamps lag real time by up to ~30s, so only correct gross clock errors.
    const skew = newest - Date.now() / 1000;
    clockOffset = Math.abs(skew) > 120 ? skew : 0;

    updateStatus(failed);
  }

  // ---------- Positioning ----------

  function prevStop(train, stopId) {
    const seq = network.shapes[train.shape];
    const i = seq ? seq.indexOf(stopId) : -1;
    if (i > 0) return seq[i - 1];
    return network.pred[`${train.route}|${stopId}`];
  }

  function stopPoint(stopId) {
    const s = network.stops[stopId];
    return s ? [s[0], s[1]] : null;
  }

  function along(key, coords, f) {
    let lengths = segLengths.get(key);
    if (!lengths) {
      lengths = [0];
      for (let i = 1; i < coords.length; i++) {
        const dx = (coords[i][0] - coords[i - 1][0]) * 0.758;
        const dy = coords[i][1] - coords[i - 1][1];
        lengths.push(lengths[i - 1] + Math.hypot(dx, dy));
      }
      segLengths.set(key, lengths);
    }
    const target = f * lengths[lengths.length - 1];
    let i = 1;
    while (i < lengths.length - 1 && lengths[i] < target) i++;
    const span = lengths[i] - lengths[i - 1] || 1;
    const t = (target - lengths[i - 1]) / span;
    const a = coords[i - 1], b = coords[i];
    return [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t];
  }

  // Returns {coord, next, eta} or null if the train can't be placed.
  function locate(train, now) {
    const stops = train.stops;
    if (!stops.length) {
      const p = stopPoint(train.vpStop);
      return p && { coord: p, next: train.vpStop, eta: 0 };
    }

    const i = stops.findIndex((s) => s.arr > now);
    if (i === -1) {
      const last = stops[stops.length - 1];
      const p = stopPoint(last.stop);
      return p && { coord: p, next: last.stop, eta: 0 };
    }

    const to = stops[i];
    const toPoint = stopPoint(to.stop);
    if (!toPoint) return null;
    if (train.status === STOPPED_AT && train.vpStop === to.stop) {
      return { coord: toPoint, next: to.stop, eta: 0 };
    }

    const from = i > 0 ? stops[i - 1].stop : prevStop(train, to.stop);
    const fromPoint = from && stopPoint(from);
    if (!fromPoint) return { coord: toPoint, next: to.stop, eta: to.arr - now };
    const key = `${from}|${to.stop}`;
    const seg = network.segments[key];

    const t0 = i > 0 ? stops[i - 1].dep : to.arr - (seg ? seg[0] : 120);
    const f = Math.min(1, Math.max(0, (now - t0) / Math.max(1, to.arr - t0)));
    const coords = seg ? seg[1] : [fromPoint, toPoint];
    return { coord: along(key, coords, f), next: to.stop, eta: to.arr - now };
  }

  // ---------- Rendering ----------

  function frame(ts) {
    requestAnimationFrame(frame);
    if (!enabled || ts - lastFrame < FRAME_MS) return;
    lastFrame = ts;

    const now = Date.now() / 1000 + clockOffset;
    const features = [];
    for (const t of trains) {
      const loc = locate(t, now);
      if (!loc || !loc.coord) continue;
      const route = routeColors[t.route] || { color: "#808183", text_color: "#FFFFFF", name: t.route };
      const dest = t.stops.length ? t.stops[t.stops.length - 1].stop : t.vpStop;
      features.push({
        type: "Feature",
        geometry: { type: "Point", coordinates: loc.coord },
        properties: {
          route_id: t.route,
          name: route.name,
          color: route.color,
          text_color: route.text_color,
          next: network.stops[loc.next]?.[2] || "",
          dest: network.stops[dest]?.[2] || "",
          eta: Math.round(loc.eta),
        },
      });
    }
    map.getSource("trains").setData({ type: "FeatureCollection", features });
  }

  function updateStatus(failed) {
    const tag = document.getElementById("tag-count");
    if (tag) tag.textContent = `${trains.length} trains`;
    const el = document.getElementById("live-status");
    if (!el) return;
    const time = new Date().toLocaleTimeString();
    el.textContent = `${trains.length} trains · updated ${time}` + (failed ? ` · ${failed} feed(s) failed` : "");
    el.classList.toggle("warn", failed > 0);
  }

  function addLayers() {
    map.addSource("trains", { type: "geojson", data: { type: "FeatureCollection", features: [] } });

    map.addLayer({
      id: "trains-halo",
      type: "circle",
      source: "trains",
      paint: {
        "circle-radius": ["interpolate", ["linear"], ["zoom"], 10, 6, 15, 16],
        "circle-color": ["get", "color"],
        "circle-blur": 1,
        "circle-opacity": 0.6,
      },
    });

    map.addLayer({
      id: "trains",
      type: "circle",
      source: "trains",
      paint: {
        "circle-radius": ["interpolate", ["linear"], ["zoom"], 10, 3, 13, 5, 16, 10],
        "circle-color": ["get", "color"],
        "circle-stroke-color": "#ffffff",
        "circle-stroke-width": ["interpolate", ["linear"], ["zoom"], 10, 1, 16, 2],
      },
    });

    map.addLayer({
      id: "trains-label",
      type: "symbol",
      source: "trains",
      minzoom: 14,
      layout: {
        "text-field": ["get", "name"],
        "text-font": ["Noto Sans Bold"],
        "text-size": 11,
        "text-allow-overlap": true,
      },
      paint: { "text-color": ["get", "text_color"] },
    });

    const popup = new maplibregl.Popup({ closeButton: false, closeOnClick: false, offset: 12 });
    map.on("mousemove", "trains", (e) => {
      map.getCanvas().style.cursor = "pointer";
      const p = e.features[0].properties;
      const eta = p.eta > 30 ? `in ${Math.round(p.eta / 60)} min` : "now";
      const wrap = document.createElement("div");
      wrap.innerHTML = `<div class="popup-title"></div><div class="popup-meta"></div>`;
      wrap.querySelector(".popup-title").textContent = `${p.name} train to ${p.dest}`;
      wrap.querySelector(".popup-meta").textContent = `Next: ${p.next} (${eta})`;
      popup.setLngLat(e.features[0].geometry.coordinates).setDOMContent(wrap).addTo(map);
    });
    map.on("mouseleave", "trains", () => {
      map.getCanvas().style.cursor = "";
      popup.remove();
    });
  }

  return {
    LAYERS: ["trains-halo", "trains", "trains-label"],

    async init(mapInstance, routes) {
      map = mapInstance;
      routeColors = Object.fromEntries(routes.map((r) => [r.route_id, r]));
      network = await fetch("data/network.json").then((r) => r.json());
      addLayers();
      await refresh();
      setInterval(refresh, REFRESH_MS);
      requestAnimationFrame(frame);
    },

    setEnabled(on) {
      enabled = on;
      this.LAYERS.forEach((id) => map.setLayoutProperty(id, "visibility", on ? "visible" : "none"));
    },
  };
})();
