const CHART_REFRESH_MS = 5000; // grafieken: de hele gekozen periode
const LIVE_REFRESH_MS = 1000; // statusbalk en joysticks: alleen de laatste seconden
const HEALTH_REFRESH_MS = 30000; // bron en bewaartijd: verandert hooguit na een herstart van de server
const STICK_RADIUS = 90; // straal van de buitenste ring in SVG-eenheden (= volle schaal)

const $ = (id) => document.getElementById(id);
const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
const minutes = () => $("minutes").value;

// Sessie verlopen (401): de pagina opnieuw laden stuurt de gebruiker via /auth/login terug naar Authentik.
let reloading = false;
async function getJson(url) {
  const r = await fetch(url, { headers: { Accept: "application/json" } });
  if (r.status === 401) {
    if (!reloading) {
      reloading = true;
      setFooter("session expired, logging in again…", true);
      location.reload();
    }
    throw new Error("401");
  }
  if (!r.ok) throw new Error(r.status);
  return r.json();
}

// Herhaal fn pas ms na afloop van de vorige aanroep, zodat trage antwoorden niet opstapelen.
function every(ms, fn) {
  const tick = async () => {
    if (!reloading) await fn();
    setTimeout(tick, ms);
  };
  tick();
}

function setFooter(text, isError = false) {
  $("footer").textContent = text;
  $("footer").className = "footer" + (isError ? " error" : "");
}

// ---- Statusbalk: de systeemmodus -------------------------------------------------------------

// Groen als het systeem kracht op de toren zet (compensatie aan: CT low, damping), rood in de andere modi.
const STATES = {
  on: "status--on",
  off: "status--off",
  nodata: "status--nodata",
};
let currentStatus = null;
function setStatus(state, text) {
  const label = text.toUpperCase();
  if (currentStatus === state + label) return; // alleen bij een wijziging, anders leest een schermlezer elke seconde voor
  currentStatus = state + label;
  $("status").className = "status " + STATES[state];
  $("status-label").textContent = label;
}

// ---- Grafieken ------------------------------------------------------------------------------

const clock = new Intl.DateTimeFormat("en-GB", { hour: "2-digit", minute: "2-digit", second: "2-digit" });
const clockShort = new Intl.DateTimeFormat("en-GB", { hour: "2-digit", minute: "2-digit" });

// Seconden tonen zolang de ticks dichter dan een minuut op elkaar staan, anders herhaalt elk label dezelfde minuut.
// Dat hangt af van de data die er is, niet van de gekozen periode: een UDP-cache die net vult heeft maar seconden.
function timeLabels(u, splits, axisIdx, space, incr) {
  return splits.map((v) => (incr >= 60 ? clockShort : clock).format(v * 1000));
}

// [kolom in /api/timeseries, naam, kleurtoken, opties]. X en Y houden in alle grafieken dezelfde kleur; skid 1
// trekt in X en skid 2 in Y, dus ze delen de kleur van die richting. Opties: scale "y2" zet de lijn op de rechteras
// (minSpan2 is de minimale hoogte daarvan), dash maakt de lijn gestippeld.
const DASHED = { scale: "y2", dash: [6, 4] };
const CHARTS = {
  overlay: {
    el: "chart-overlay",
    minSpan: 5,
    minSpan2: 4,
    series: [
      ["tension_skid1_kn", "Skid 1 tension (kN)", "--series-1"],
      ["tension_skid2_kn", "Skid 2 tension (kN)", "--series-2"],
      ["tower_x", "Tower X (mm)", "--series-1", DASHED],
      ["tower_y", "Tower Y (mm)", "--series-2", DASHED],
    ],
  },
  tension: {
    el: "chart-tension",
    minSpan: 5,
    series: [["tension_skid1_kn", "Skid 1 (kN)", "--series-1"], ["tension_skid2_kn", "Skid 2 (kN)", "--series-2"]],
  },
  tower: { el: "chart-tower", minSpan: 4, series: [["tower_x", "X (mm)", "--series-1"], ["tower_y", "Y (mm)", "--series-2"]] },
  vessel: { el: "chart-vessel", minSpan: 40, series: [["vessel_x", "X (mm)", "--series-1"], ["vessel_y", "Y (mm)", "--series-2"]] },
};
const charts = {}; // key -> { u: uPlot, active: [0/1, ...] }

// Zonder minimale hoogte zoomt de as bij een vlak signaal in op de meetruis.
function paddedRange(minSpan) {
  return (u, min, max) => {
    if (min == null || max == null) return [0, 1];
    const span = Math.max(max - min, minSpan);
    const mid = (max + min) / 2;
    const pad = span * 0.08;
    return [mid - span / 2 - pad, mid + span / 2 + pad];
  };
}

// Groene achtergrond waar de compensatie aan stond: zo zie je direct het effect van het systeem.
function drawBands(u, active, color) {
  const ts = u.data[0];
  if (!ts || !ts.length || !active) return;
  const ctx = u.ctx;
  ctx.save();
  ctx.fillStyle = color;
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

const chartHeight = () => (window.innerWidth < 700 ? 200 : 230);

function buildChart(def, data) {
  const el = $(def.el);
  el.innerHTML = "";
  const chart = { active: data.compensation_active };
  const shade = css("--shade");
  const axis = { stroke: css("--muted"), grid: { stroke: css("--grid"), width: 1 }, ticks: { show: false }, border: { show: false } };
  chart.u = new uPlot(
    {
      width: el.clientWidth || 600,
      height: chartHeight(),
      padding: [8, 26, 0, 0], // rechts de helft van een tijdlabel, anders wordt het laatste label afgekapt
      scales: {
        x: { time: true },
        y: { range: paddedRange(def.minSpan) },
        ...(def.minSpan2 ? { y2: { range: paddedRange(def.minSpan2) } } : {}),
      },
      series: [
        { label: "Time", value: (u, v) => (v == null ? "–" : clock.format(v * 1000)) },
        ...def.series.map(([, label, color, options]) => ({
          label,
          stroke: css(color),
          width: 2,
          points: { show: false },
          ...options,
        })),
      ],
      axes: [
        { ...axis, size: 32, space: 70, values: timeLabels },
        { ...axis, size: 52 },
        ...(def.minSpan2 ? [{ ...axis, scale: "y2", side: 1, size: 52, grid: { show: false } }] : []),
      ],
      cursor: { drag: { x: false, y: false } },
      hooks: { drawClear: [(u) => drawBands(u, chart.active, shade)] },
    },
    [data.t, ...def.series.map(([column]) => data[column])],
    el
  );
  new ResizeObserver(() => {
    const width = el.clientWidth;
    if (width && width !== chart.u.width) chart.u.setSize({ width, height: chartHeight() });
  }).observe(el);
  return chart;
}

function renderChart(key, data) {
  const def = CHARTS[key];
  if (!data.t.length) {
    charts[key]?.u.destroy();
    delete charts[key];
    $(def.el).innerHTML = '<p class="nodata">No data in this period</p>';
    return;
  }
  if (charts[key]) {
    charts[key].active = data.compensation_active;
    charts[key].u.setData([data.t, ...def.series.map(([column]) => data[column])]);
  } else {
    charts[key] = buildChart(def, data);
  }
}

async function refreshCharts() {
  const m = minutes(); // de keuze kan veranderen terwijl de aanvraag loopt
  try {
    const data = await getJson(`/api/timeseries?minutes=${m}`);
    if (m !== minutes()) return; // verouderd antwoord: de nieuwe keuze heeft een eigen aanvraag
    for (const key of Object.keys(CHARTS)) renderChart(key, data);
    setFooter("updated " + clock.format(Date.now()));
  } catch (e) {
    if (!reloading && m === minutes()) setFooter("no data (" + e.message + ")", true);
  }
}

// Een ander kleurenschema (donker/licht) heeft andere kleuren nodig: grafieken opnieuw opbouwen.
matchMedia("(prefers-color-scheme: dark)").addEventListener("change", () => {
  for (const key of Object.keys(charts)) {
    charts[key].u.destroy();
    delete charts[key];
  }
  refreshCharts();
});

// ---- Joystick-cirkels -----------------------------------------------------------------------

const STICK_LABELS = { tower: "Tower", vessel: "Vessel" };

function renderStick(key, m) {
  const root = $("stick-" + key);
  const label = STICK_LABELS[key];
  const svg = root.querySelector("svg");
  if (!m) {
    root.classList.add("stick--stale");
    root.classList.remove("stick--over");
    root.querySelector(".value").textContent = "–";
    svg.setAttribute("aria-label", `${label}: no current data`);
    return;
  }
  root.classList.remove("stick--stale");

  // Meetwaarde (mm) naar SVG-eenheden; buiten de ring wordt het stipje op de rand vastgezet.
  const scale = STICK_RADIUS / m.fullscale;
  const toSvg = ([x, y]) => {
    const px = x * scale;
    const py = -y * scale; // SVG-y wijst omlaag, onze Y omhoog
    const r = Math.hypot(px, py);
    return r > STICK_RADIUS ? [(px * STICK_RADIUS) / r, (py * STICK_RADIUS) / r] : [px, py];
  };
  const [dx, dy] = toSvg([m.x, m.y]);
  root.querySelector(".dot").style.transform = `translate(${dx}px, ${dy}px)`;

  // Staart van de afgelopen seconden in drie stukken, die naar nu toe donkerder worden.
  const points = m.trail.map(toSvg);
  const third = Math.ceil(points.length / 3);
  [0, 1, 2].forEach((i) => {
    const part = points.slice(Math.max(i * third - 1, 0), (i + 1) * third);
    root.querySelector(`.trail--${i + 1}`).setAttribute("points", part.map((p) => p.join(",")).join(" "));
  });

  const magnitude = Math.hypot(m.x, m.y);
  root.classList.toggle("stick--over", magnitude > m.fullscale);
  const text = `${magnitude >= 100 ? magnitude.toFixed(0) : magnitude.toFixed(1)} mm`;
  root.querySelector(".value").textContent = text;
  root.querySelector(".scale").textContent = `scale ± ${m.fullscale} mm`;
  svg.setAttribute("aria-label", `${label}: ${text} deviation from the mean`);
}

async function refreshLive() {
  const m = minutes();
  try {
    const live = await getJson(`/api/live?minutes=${m}`);
    if (m !== minutes()) return; // zie refreshCharts
    if (!live.ready || live.stale) throw new Error("no current data");
    setStatus(live.compensation_active ? "on" : "off", live.mode.name);
    renderStick("tower", live.tower);
    renderStick("vessel", live.vessel);
  } catch (e) {
    if (m !== minutes()) return;
    setStatus("nodata", "No data");
    renderStick("tower", null);
    renderStick("vessel", null);
  }
}

// ---- Periode: alleen aanbieden wat de bron kan leveren --------------------------------------

const PERIOD_STEPS = [0.5, 1, 2, 5, 10, 30, 60]; // minuten; een torenoscillatie duurt ~6 s, dus 1 minuut is de standaard
const DEFAULT_PERIOD = 1;
const periodLabel = (m) => (m < 1 ? `last ${Math.round(m * 60)} s` : `last ${m} min`);
const MAX_PERIOD = 60; // langer dan dit weigert de API (422)

// De stappen tot en met retention, plus retention zelf als die er niet in zit (12 -> 1, 2, 5, 10, 12).
function periodChoices(retention) {
  const choices = PERIOD_STEPS.filter((m) => m <= retention);
  if (!choices.includes(retention)) choices.push(retention);
  return choices;
}

// 1 minuut als dat kan, anders de langste periode die er is.
function defaultPeriod(choices) {
  return choices.includes(DEFAULT_PERIOD) ? DEFAULT_PERIOD : choices[choices.length - 1];
}

function refreshAll() {
  refreshCharts();
  refreshLive();
}

// De HTML heeft vaste opties (10/30/60) als terugval. Zodra de bron meldt hoe ver terug ze data heeft komen
// die in de plaats; zonder bruikbaar getal blijven ze staan.
function applyRetention(value) {
  if (!Number.isInteger(value) || value < 1) return;
  const choices = periodChoices(Math.min(value, MAX_PERIOD));
  const select = $("minutes");
  // Ongewijzigd laten staan: bij vervangen klapt een geopende keuzelijst dicht, en dit wordt steeds herhaald.
  if (choices.join() === Array.from(select.options, (o) => o.value).join()) return;
  const before = Number(select.value);
  const selected = choices.includes(before) ? before : defaultPeriod(choices);
  // selected-attribuut mee: zo is de keuze ook in een DOM-dump te zien
  select.replaceChildren(...choices.map((m) => new Option(periodLabel(m), m, m === selected, m === selected)));
  if (selected !== before) refreshAll(); // de grafieken tonen nog de oude periode: niet wachten op de volgende ronde
}

// ---- Gebruiker, bron en bewaartijd: fouten hier mogen het dashboard niet blokkeren -------------

async function loadUser() {
  try {
    const me = await getJson("/api/me");
    if (me.auth) {
      $("user-name").textContent = me.name || me.username || ""; // textContent: nooit als HTML
      $("user").hidden = false;
    }
  } catch (e) {
    /* getJson regelt 401; andere fouten laten we stil */
  }
}

// Los van loadUser: een fout bij /api/me mag de demo-melding en de periodes niet tegenhouden. Een mislukte
// aanvraag blijft stil; de vaste opties werken, en bij de volgende ronde proberen we het opnieuw.
async function refreshHealth() {
  try {
    const health = await getJson("/api/health");
    $("demo-banner").hidden = health.source !== "mock";
    applyRetention(health.retention_minutes);
  } catch (e) {
    /* getJson regelt 401 */
  }
}

$("minutes").addEventListener("change", refreshAll);
loadUser();
every(HEALTH_REFRESH_MS, refreshHealth);
every(CHART_REFRESH_MS, refreshCharts);
every(LIVE_REFRESH_MS, refreshLive);
