const RISK_COLORS = { high: '#E4572E', medium: '#F2B705', low: '#3FA34D' };

let map, markers = {}, localities = [], currentFilter = 'all', historyChart = null;

async function init() {
  localities = await (await fetch('/api/localities')).json();

  safe(renderStats);
  safe(renderList);
  safe(() => document.getElementById('filter-group').addEventListener('click', onFilterClick));
  safe(() => document.getElementById('predict-form').addEventListener('submit', onPredict));

  // Map (Leaflet) and metrics load independently — if the CDN or an API call
  // fails, the rest of the dashboard still works.
  try { renderMap(); } catch (err) {
    console.error('Map failed to load', err);
    document.getElementById('map').innerHTML =
      '<div class="empty-state">Map tiles unavailable (no internet access to the tile CDN). Everything else still works.</div>';
  }
  renderMetrics().catch(err => console.error('Metrics failed to load', err));
}

function safe(fn) {
  try { fn(); } catch (err) { console.error(err); }
}

function renderStats() {
  const counts = { high: 0, medium: 0, low: 0 };
  localities.forEach(l => counts[l.predicted_risk]++);
  document.getElementById('stat-total').textContent = localities.length;
  document.getElementById('stat-high').textContent = counts.high;
  document.getElementById('stat-medium').textContent = counts.medium;
  document.getElementById('stat-low').textContent = counts.low;
}

function renderMap() {
  map = L.map('map', { scrollWheelZoom: false }).setView([11.0, 78.4], 6.6);
  L.tileLayer('https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png', {
    attribution: '&copy; OpenStreetMap &copy; CARTO',
    maxZoom: 12,
  }).addTo(map);

  localities.forEach(loc => {
    const color = RISK_COLORS[loc.predicted_risk];
    const marker = L.circleMarker([loc.lat, loc.lon], {
      radius: 8, fillColor: color, color: color, weight: 1, fillOpacity: 0.75,
    }).addTo(map);
    marker.bindPopup(
      `<div class="popup-title">${prettyName(loc.locality)}</div>
       <div class="popup-risk" style="color:${color}">${loc.predicted_risk.toUpperCase()} RISK</div>
       <div class="popup-risk">Depth: ${loc.groundwater_depth_m} m · ${loc.land_use}</div>`
    );
    marker.on('click', () => selectLocality(loc.locality));
    markers[loc.locality] = marker;
  });
}

function prettyName(name) {
  return name.replace(/_/g, ' ');
}

function renderList() {
  const container = document.getElementById('locality-list');
  container.innerHTML = '';
  const filtered = currentFilter === 'all' ? localities : localities.filter(l => l.predicted_risk === currentFilter);
  const sorted = [...filtered].sort((a, b) => b.groundwater_depth_m - a.groundwater_depth_m);

  sorted.forEach(loc => {
    const card = document.createElement('div');
    card.className = `locality-card risk-${loc.predicted_risk}`;
    const depthPct = Math.min((loc.groundwater_depth_m / 30) * 100, 100);
    card.innerHTML = `
      <div class="strata-mini">
        <div class="water-fill" style="height:${depthPct < 90 ? 100 - depthPct : 10}%"></div>
        <div class="water-line" style="top:${depthPct}%"></div>
      </div>
      <div class="card-main">
        <div class="card-name">${prettyName(loc.locality)}</div>
        <div class="card-meta">${loc.land_use} · depth ${loc.groundwater_depth_m}m · ${loc.rainfall_mm}mm rain</div>
      </div>
      <div class="card-risk-badge risk-${loc.predicted_risk}">${loc.predicted_risk}</div>
    `;
    card.addEventListener('click', () => selectLocality(loc.locality));
    container.appendChild(card);
  });
}

function onFilterClick(e) {
  const btn = e.target.closest('.filter-btn');
  if (!btn) return;
  document.querySelectorAll('.filter-btn').forEach(b => b.classList.remove('active'));
  btn.classList.add('active');
  currentFilter = btn.dataset.filter;
  renderList();
}

async function selectLocality(name) {
  const loc = localities.find(l => l.locality === name);
  const history = await (await fetch(`/api/history/${encodeURIComponent(name)}`)).json();

  document.getElementById('detail-title').textContent = prettyName(name);
  document.getElementById('detail-sub').innerHTML =
    `Predicted risk (next 1–2 yrs): <b style="color:${RISK_COLORS[loc.predicted_risk]}">${loc.predicted_risk.toUpperCase()}</b>
     &nbsp;·&nbsp; confidence ${(Math.max(loc.prob_low, loc.prob_medium, loc.prob_high) * 100).toFixed(0)}%`;

  const body = document.getElementById('detail-body');
  body.innerHTML = `
    <div class="detail-grid">
      <div class="detail-metric"><div class="m-label">Groundwater depth</div><div class="m-value">${loc.groundwater_depth_m} m</div></div>
      <div class="detail-metric"><div class="m-label">Rainfall (latest yr)</div><div class="m-value">${loc.rainfall_mm} mm</div></div>
      <div class="detail-metric"><div class="m-label">Borewells</div><div class="m-value">${loc.num_borewells}</div></div>
      <div class="detail-metric"><div class="m-label">Land use</div><div class="m-value" style="text-transform:capitalize">${loc.land_use}</div></div>
    </div>
    <div class="chart-wrap"><canvas id="history-chart"></canvas></div>
  `;

  if (markers[name]) { map.panTo(markers[name].getLatLng()); markers[name].openPopup(); }

  const ctx = document.getElementById('history-chart').getContext('2d');
  if (historyChart) historyChart.destroy();
  historyChart = new Chart(ctx, {
    data: {
      labels: history.map(h => h.year),
      datasets: [
        {
          type: 'line', label: 'Groundwater depth (m, right axis)', yAxisID: 'y1',
          data: history.map(h => h.groundwater_depth_m),
          borderColor: '#5FB4D6', backgroundColor: '#5FB4D6', tension: 0.3, pointRadius: 2,
        },
        {
          type: 'bar', label: 'Rainfall (mm, left axis)', yAxisID: 'y',
          data: history.map(h => h.rainfall_mm),
          backgroundColor: 'rgba(138,90,52,0.55)',
        },
      ],
    },
    options: {
      responsive: true, maintainAspectRatio: false,
      interaction: { mode: 'index', intersect: false },
      plugins: { legend: { labels: { color: '#8FA3A0', font: { family: 'IBM Plex Mono', size: 10 } } } },
      scales: {
        x: { ticks: { color: '#8FA3A0' }, grid: { color: '#2A3538' } },
        y: { position: 'left', ticks: { color: '#8FA3A0' }, grid: { color: '#2A3538' }, title: { display: true, text: 'Rainfall (mm)', color: '#8FA3A0' } },
        y1: { position: 'right', reverse: true, ticks: { color: '#8FA3A0' }, grid: { display: false }, title: { display: true, text: 'Depth to water (m, inverted)', color: '#8FA3A0' } },
      },
    },
  });
}

async function onPredict(e) {
  e.preventDefault();
  const form = e.target;
  const payload = {
    rainfall_mm: +form.rainfall_mm.value,
    rainfall_mm_last_year: +form.rainfall_mm_last_year.value,
    num_borewells: +form.num_borewells.value,
    num_borewells_last_year: +form.num_borewells_last_year.value,
    avg_pump_hours_per_day: +form.avg_pump_hours_per_day.value,
    impervious_pct: +form.impervious_pct.value,
    groundwater_depth_m: +form.groundwater_depth_m.value,
    groundwater_depth_m_last_year: +form.groundwater_depth_m_last_year.value,
    land_use: form.land_use.value,
  };
  const res = await fetch('/api/predict', {
    method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload),
  });
  const result = await res.json();
  if (result.error) {
    document.getElementById('predict-result').innerHTML = `<div class="empty-state">${result.error}</div>`;
    return;
  }
  const color = RISK_COLORS[result.predicted_risk];
  document.getElementById('predict-result').innerHTML = `
    <div class="result-card">
      <div class="result-badge" style="background:${color}22;color:${color}">${result.predicted_risk.toUpperCase()}</div>
      <div class="result-probs">
        <div>Low<b style="color:${RISK_COLORS.low}">${(result.probabilities.low*100).toFixed(0)}%</b></div>
        <div>Medium<b style="color:${RISK_COLORS.medium}">${(result.probabilities.medium*100).toFixed(0)}%</b></div>
        <div>High<b style="color:${RISK_COLORS.high}">${(result.probabilities.high*100).toFixed(0)}%</b></div>
      </div>
    </div>`;
}

async function renderMetrics() {
  const m = await (await fetch('/api/metrics')).json();
  const container = document.getElementById('model-metrics');

  const rows = Object.entries(m.classification_report)
    .filter(([k]) => ['low', 'medium', 'high'].includes(k))
    .map(([cls, v]) => `<tr><td style="color:${RISK_COLORS[cls]}">${cls}</td><td>${v.precision.toFixed(2)}</td><td>${v.recall.toFixed(2)}</td><td>${v['f1-score'].toFixed(2)}</td><td>${v.support}</td></tr>`)
    .join('');

  const tableHtml = `
    <div>
      <p style="font-family:var(--font-mono); font-size:0.8rem; color:var(--text-dim); margin-bottom:10px;">
        Overall test accuracy: <b style="color:var(--text)">${(m.test_accuracy*100).toFixed(1)}%</b>
        &nbsp;(${m.n_train} train / ${m.n_test} test rows, held-out by year)
      </p>
      <table class="metrics-table">
        <thead><tr><th>Class</th><th>Precision</th><th>Recall</th><th>F1</th><th>Support</th></tr></thead>
        <tbody>${rows}</tbody>
      </table>
    </div>`;

  const maxImp = Math.max(...Object.values(m.feature_importance));
  const impHtml = `
    <div>
      <p style="font-family:var(--font-mono); font-size:0.8rem; color:var(--text-dim); margin-bottom:10px;">Feature importance</p>
      ${Object.entries(m.feature_importance).sort((a,b)=>b[1]-a[1]).map(([f, v]) => `
        <div class="importance-bar-row">
          <div class="importance-label">${f}</div>
          <div class="importance-track"><div class="importance-fill" style="width:${(v/maxImp*100).toFixed(0)}%"></div></div>
        </div>`).join('')}
    </div>`;

  container.innerHTML = tableHtml + impHtml;
}

init();
