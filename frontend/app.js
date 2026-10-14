const REFRESH_MS = 5000;
const $ = (id) => document.getElementById(id);
const fmt = (v, d = 2) => (v == null ? "–" : Number(v).toFixed(d));

const KPIS = [
  ["Dominante frequentie", (o) => fmt(o.dominant_frequency_hz, 3), "Hz"],
  ["Trilling systeem uit (RMS)", (o) => fmt(o.rms_system_off, 3), "m/s²"],
  ["Trilling systeem aan (RMS)", (o) => fmt(o.rms_system_on, 3), "m/s²"],
  ["Demping", (o) => fmt(o.reduction_percent, 0), "%"],
  ["Systeem aan", (o) => fmt(o.active_fraction == null ? null : o.active_fraction * 100, 0), "% van periode"],
  ["Kabelspanning gem.", (o) => fmt(o.tension_mean_kn, 1), "kN"],
  ["Kabelspanning piek", (o) => fmt(o.tension_peak_kn, 1), "kN"],
  ["Spanningsamplitude", (o) => fmt(o.tension_amplitude_kn, 1), "kN"],
];

function renderKpis(o) {
  $("kpis").innerHTML = KPIS.map(
    ([label, f, unit]) =>
      `<div class="card"><div class="label">${label}</div><div class="value">${f(o)} <span class="unit">${unit}</span></div></div>`
  ).join("");
}

const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
const charts = {};

// Gearceerde banden waar het systeem aan staat
function activeBands(u, ts, active) {
  const ctx = u.ctx;
  ctx.save();
  ctx.fillStyle = css("--accent") + "22";
  let start = null;
  for (let i = 0; i <= ts.length; i++) {
    const on = i < ts.length && active[i] === 1;
    if (on && start === null) start = ts[i];
    if (!on && start !== null) {
      const x0 = u.valToPos(start, "x", true);
      const x1 = u.valToPos(ts[Math.min(i, ts.length - 1)], "x", true);
      ctx.fillRect(x0, u.bbox.top, x1 - x0, u.bbox.height);
      start = null;
    }
  }
  ctx.restore();
}

function draw(key, el, ts, ys, label, active) {
  const width = el.clientWidth || 800;
  charts[key]?.destroy();
  el.innerHTML = "";
  charts[key] = new uPlot(
    {
      width,
      height: 220,
      scales: { x: { time: true } },
      series: [{}, { label, stroke: css("--accent"), width: 1.5 }],
      axes: [{ stroke: css("--muted") }, { stroke: css("--muted") }],
      hooks: active ? { drawClear: [(u) => activeBands(u, ts, active)] } : {},
    },
    [ts, ys],
    el
  );
}

async function refresh() {
  const minutes = $("minutes").value;
  try {
    const [o, s] = await Promise.all([
      fetch(`/api/overview?minutes=${minutes}`).then((r) => r.ok ? r.json() : Promise.reject(r.status)),
      fetch(`/api/timeseries?minutes=${minutes}`).then((r) => r.ok ? r.json() : Promise.reject(r.status)),
    ]);
    renderKpis(o);
    draw("accel", $("chart-accel"), s.t, s.accel_mag, "versnelling", s.system_active);
    draw("tension", $("chart-tension"), s.t, s.tension_kn, "kabelspanning", s.system_active);
    $("status").className = "";
    $("status").textContent = "bijgewerkt " + new Date().toLocaleTimeString("nl-NL");
  } catch (e) {
    $("status").className = "error";
    $("status").textContent = "geen data (" + e + ")";
  }
}

$("minutes").addEventListener("change", refresh);
refresh();
setInterval(refresh, REFRESH_MS);
