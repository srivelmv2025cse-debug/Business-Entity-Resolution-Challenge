/* ============================================================
   Business Entity Resolution – Frontend Application
   Author: Member 3
   ============================================================ */

const API = 'http://localhost:8000';

// ── State ──────────────────────────────────────────────────
const state = {
  currentPage: 'dashboard',
  stats: null,
  thresholdCurve: null,
  charts: {},
  pollTimers: {},
};

// ── Router ─────────────────────────────────────────────────
function navigate(page) {
  document.querySelectorAll('.page').forEach(p => p.classList.remove('active'));
  document.querySelectorAll('.nav-item').forEach(n => n.classList.remove('active'));

  const pageEl = document.getElementById(`page-${page}`);
  if (pageEl) pageEl.classList.add('active');
  const navEl = document.querySelector(`[data-page="${page}"]`);
  if (navEl) navEl.classList.add('active');

  document.getElementById('topbar-title').textContent = PAGE_TITLES[page] || page;
  state.currentPage = page;
  window.location.hash = page;

  const loaders = { dashboard: loadDashboard, candidates: loadCandidates, output: loadOutput, evaluation: loadEvaluation };
  if (loaders[page]) loaders[page]();
}

const PAGE_TITLES = {
  dashboard:    '📊 Dashboard',
  upload:       '📁 Data Upload',
  preprocessing:'🔧 Preprocessing',
  candidates:   '🔍 Candidate Generation',
  model:        '🤖 ML Model',
  matching:     '🔗 Entity Matching',
  evaluation:   '📈 Evaluation',
  output:       '📤 Output',
  validation:   '✅ Validation',
  methodology:  '📖 Methodology',
  architecture: '🏗 System Architecture',
};

// ── API Helper ─────────────────────────────────────────────
async function apiFetch(endpoint, opts = {}) {
  try {
    const r = await fetch(`${API}${endpoint}`, opts);
    if (!r.ok) {
      const err = await r.json().catch(() => ({ detail: r.statusText }));
      throw new Error(err.detail || r.statusText);
    }
    return await r.json();
  } catch (e) {
    console.warn('API Error:', endpoint, e.message);
    throw e;
  }
}

function fmt(v, decimals = 2) {
  if (v == null) return '—';
  if (typeof v === 'number') return v.toFixed(decimals);
  return v;
}

function fmtPct(v) { return v == null ? '—' : (v * 100).toFixed(1) + '%'; }
function fmtBytes(b) {
  if (!b) return '0 B';
  const u = ['B','KB','MB','GB'];
  let i = 0;
  while (b >= 1024 && i < u.length - 1) { b /= 1024; i++; }
  return b.toFixed(1) + ' ' + u[i];
}

// ── DASHBOARD ──────────────────────────────────────────────
async function loadDashboard() {
  try {
    const [stats, modelStatus] = await Promise.all([
      apiFetch('/api/data/stats'),
      apiFetch('/api/model/status').catch(() => null),
    ]);
    state.stats = stats;

    setCard('card-s1',  stats.train.source1.rows || stats.test.source1.rows || 0);
    setCard('card-s2',  stats.train.source2.rows || stats.test.source2.rows || 0);
    setCard('card-s3',  stats.train.source3.rows || stats.test.source3.rows || 0);
    setCard('card-cands', stats.output.candidate_pairs.rows || 0);

    // matched / singleton
    if (stats.output.matching_results.exists) {
      const preview = await apiFetch('/api/output/preview?file=matching_results');
      const rows = preview.preview || [];
      let matches = 0, singletons = 0;
      rows.forEach(r => {
        const ids = (r.matched_entity_ids || '').trim();
        if (ids) matches += ids.split(',').filter(Boolean).length;
        else singletons++;
      });
      setCard('card-matches', matches);
      setCard('card-singletons', singletons);
    } else {
      setCard('card-matches', '—');
      setCard('card-singletons', '—');
    }

    // Model status
    if (modelStatus) {
      setCard('card-model', modelStatus.model_exists ? 'Ready' : 'Not trained');
      const f05 = modelStatus.val_f05;
      setCard('card-f05', f05 != null ? (f05 * 100).toFixed(1) + '%' : '—');
      updateF05Hero(f05, modelStatus);
    }
  } catch (e) {
    showAlert('dash-alert', 'Could not load dashboard: ' + e.message, 'warn');
  }
}

function setCard(id, value) {
  const el = document.getElementById(id);
  if (el) el.textContent = value;
}

function updateF05Hero(f05, status) {
  const el = document.getElementById('f05-score');
  if (!el) return;
  el.textContent = f05 != null ? (f05 * 100).toFixed(1) + '%' : '—';
  const sub = document.getElementById('f05-threshold');
  if (sub && status) sub.textContent = `Threshold: ${fmt(status.threshold, 4) ?? '—'} · Status: ${status.training_status}`;
}

// ── UPLOAD ─────────────────────────────────────────────────
function initUpload() {
  const FILES = [
    { id: 'upload-train-s1', target: 'train_source1',      label: 'train_source1.tsv' },
    { id: 'upload-train-s2', target: 'train_source2',      label: 'train_source2.tsv' },
    { id: 'upload-train-s3', target: 'train_source3',      label: 'train_source3.tsv' },
    { id: 'upload-train-gt', target: 'train_ground_truth', label: 'train_ground_truth.tsv' },
    { id: 'upload-test-s1',  target: 'test_source1',       label: 'test_source1.tsv' },
    { id: 'upload-test-s2',  target: 'test_source2',       label: 'test_source2.tsv' },
    { id: 'upload-test-s3',  target: 'test_source3',       label: 'test_source3.tsv' },
  ];

  FILES.forEach(f => {
    const zone = document.getElementById(f.id);
    if (!zone) return;
    zone.addEventListener('click', () => {
      const inp = document.createElement('input');
      inp.type = 'file'; inp.accept = '.tsv,.txt,text/tab-separated-values';
      inp.onchange = e => uploadFile(e.target.files[0], f.target, zone);
      inp.click();
    });
    zone.addEventListener('dragover', e => { e.preventDefault(); zone.classList.add('dragover'); });
    zone.addEventListener('dragleave', () => zone.classList.remove('dragover'));
    zone.addEventListener('drop', e => {
      e.preventDefault(); zone.classList.remove('dragover');
      uploadFile(e.dataTransfer.files[0], f.target, zone);
    });
  });
}

async function uploadFile(file, target, zone) {
  if (!file) return;
  const status = zone.querySelector('.upload-status');
  if (status) { status.textContent = '⏳ Uploading…'; status.className = 'upload-status text-sm text-accent mt-8'; }

  const fd = new FormData();
  fd.append('file', file);
  try {
    const r = await fetch(`${API}/api/data/upload?target=${target}`, { method: 'POST', body: fd });
    const data = await r.json();
    if (!r.ok) throw new Error(data.detail || 'Upload failed');
    if (status) {
      status.textContent = `✅ ${data.rows} rows, ${data.cols} cols – ${fmtBytes(data.size_bytes)}`;
      status.className = 'upload-status text-sm text-success mt-8';
    }
  } catch (e) {
    if (status) { status.textContent = '❌ ' + e.message; status.className = 'upload-status text-sm text-danger mt-8'; }
  }
}

// ── PREPROCESSING ──────────────────────────────────────────
async function runPreprocessing() {
  const btn  = document.getElementById('btn-preprocess');
  const stat = document.getElementById('preprocess-status');
  btn.disabled = true;
  stat.textContent = '⏳ Starting preprocessing + blocking…';
  stat.className = 'alert alert-info';

  try {
    await apiFetch('/api/pipeline/preprocess', { method: 'POST' });
    pollPipelineStatus(stat, btn);
  } catch (e) {
    stat.textContent = '❌ ' + e.message;
    stat.className = 'alert alert-danger';
    btn.disabled = false;
  }
}

function pollPipelineStatus(stat, btn) {
  clearInterval(state.pollTimers.pipeline);
  state.pollTimers.pipeline = setInterval(async () => {
    try {
      const s = await apiFetch('/api/pipeline/status');
      if (s.preprocess === 'done' && s.blocking === 'done') {
        stat.textContent = `✅ Pipeline complete at ${s.last_run} — candidate_pairs.tsv generated`;
        stat.className = 'alert alert-success';
        btn.disabled = false;
        clearInterval(state.pollTimers.pipeline);
        loadDashboard();
        loadCandidates();
      } else if (s.preprocess === 'error' || s.blocking === 'error') {
        stat.textContent = '❌ Pipeline error: ' + (s.error || 'unknown');
        stat.className = 'alert alert-danger';
        btn.disabled = false;
        clearInterval(state.pollTimers.pipeline);
      } else {
        stat.textContent = `⏳ Preprocessing: ${s.preprocess}   Blocking: ${s.blocking}`;
      }
    } catch { /* retry */ }
  }, 1500);
}

// ── CANDIDATES ─────────────────────────────────────────────
async function loadCandidates() {
  try {
    const stats = await apiFetch('/api/candidates/stats');
    setEl('cand-total-s1',   stats.total_s1);
    setEl('cand-total',      stats.total_candidates);
    setEl('cand-avg',        stats.avg_per_s1);
    setEl('cand-reduction',  stats.reduction_ratio != null ? (stats.reduction_ratio * 100).toFixed(1) + '%' : '—');

    const stratList = document.getElementById('blocking-strategies');
    if (stratList && stats.strategies) {
      stratList.innerHTML = stats.strategies.map(s =>
        `<div class="pipeline-step step-done">
          <span class="step-icon">✓</span>
          <span class="step-name">${s.name}</span>
          <span class="badge badge-active">${s.status}</span>
        </div>`
      ).join('');
    }
  } catch (e) {
    console.warn('Candidates stats error:', e.message);
  }
}

async function searchCandidate() {
  const id  = document.getElementById('cand-search-input').value.trim();
  const out = document.getElementById('cand-search-result');
  if (!id) return;
  out.innerHTML = '<div class="spinner"></div>';
  try {
    const data = await apiFetch(`/api/candidates/search?s1_id=${encodeURIComponent(id)}`);
    out.innerHTML = renderCandidateSearch(data);
  } catch (e) {
    out.innerHTML = `<div class="alert alert-warn">⚠️ ${e.message}</div>`;
  }
}

function renderCandidateSearch(data) {
  if (!data.s1_entity) return '<div class="alert alert-warn">Entity not found in source data.</div>';
  const s1 = data.s1_entity;
  let html = `
    <div class="entity-card mb-12">
      <div class="entity-header">
        <div>
          <div class="entity-id">SOURCE 1</div>
          <div class="entity-name">${esc(s1.business_name)}</div>
          <div class="entity-meta">${esc(s1.business_address)} · ${esc(s1.country)}</div>
        </div>
        <span class="badge badge-active">${esc(s1.entity_id)}</span>
      </div>
    </div>
    <div class="panel-title">🔍 ${data.candidates.length} Candidate(s)</div>`;

  if (!data.candidates.length) {
    return html + '<div class="alert alert-info">No candidates found for this entity.</div>';
  }
  data.candidates.forEach(c => {
    html += `
    <div class="entity-card">
      <div class="entity-header">
        <div>
          <div class="entity-id">${esc(c.source?.toUpperCase() || '')} · ${esc(c.entity_id)}</div>
          <div class="entity-name">${esc(c.business_name)}</div>
          <div class="entity-meta">${esc(c.business_address)} · ${esc(c.country)}</div>
        </div>
      </div>
    </div>`;
  });
  return html;
}

// ── MODEL ──────────────────────────────────────────────────
async function trainModel() {
  const btn  = document.getElementById('btn-train');
  const stat = document.getElementById('model-status');
  btn.disabled = true;
  stat.textContent = '⏳ Training started…';
  stat.className = 'alert alert-info';

  try {
    await apiFetch('/api/model/train', { method: 'POST' });
    pollModelStatus(stat, btn);
  } catch (e) {
    stat.textContent = '❌ ' + e.message;
    stat.className = 'alert alert-danger';
    btn.disabled = false;
  }
}

function pollModelStatus(stat, btn) {
  clearInterval(state.pollTimers.model);
  state.pollTimers.model = setInterval(async () => {
    try {
      const s = await apiFetch('/api/model/status');
      if (s.training_status === 'done') {
        stat.textContent = `✅ Model trained. Threshold: ${fmt(s.threshold, 4)} · F0.5: ${fmtPct(s.val_f05)}`;
        stat.className = 'alert alert-success';
        btn.disabled = false;
        clearInterval(state.pollTimers.model);
        loadModelMetrics();
        loadDashboard();
      } else if (s.training_status === 'error') {
        stat.textContent = '❌ Training error: ' + (s.training_error || 'unknown');
        stat.className = 'alert alert-danger';
        btn.disabled = false;
        clearInterval(state.pollTimers.model);
      } else if (s.training_status === 'running') {
        stat.textContent = '⏳ Training in progress…';
      }
    } catch { /* retry */ }
  }, 2000);
}

async function loadModelMetrics() {
  try {
    const m = await apiFetch('/api/model/metrics');
    setEl('metric-f05',       fmtPct(m.f05));
    setEl('metric-precision', fmtPct(m.precision));
    setEl('metric-recall',    fmtPct(m.recall));
    setEl('metric-threshold', fmt(m.threshold, 4));
    setEl('metric-matches',   m.match_count ?? '—');
    setEl('metric-singletons',m.singleton_count ?? '—');
    setEl('metric-false-merges', m.false_merges ?? '—');
    setEl('metric-train-pairs', m.n_train_pairs ?? '—');
  } catch { /* metrics not yet available */ }
}

async function runPredict() {
  const btn  = document.getElementById('btn-predict');
  const stat = document.getElementById('predict-status');
  btn.disabled = true;
  stat.textContent = '⏳ Running predictions…';
  stat.className = 'alert alert-info';
  try {
    const r = await apiFetch('/api/model/predict', { method: 'POST' });
    stat.textContent = `✅ Done. Matches: ${r.match_count} · Singletons: ${r.singleton_count} · Entities: ${r.n_entities}`;
    stat.className = 'alert alert-success';
    loadDashboard();
  } catch (e) {
    stat.textContent = '❌ ' + e.message;
    stat.className = 'alert alert-danger';
  }
  btn.disabled = false;
}

// ── ENTITY MATCHING ────────────────────────────────────────
async function lookupEntity() {
  const id  = document.getElementById('match-search-input').value.trim();
  const out = document.getElementById('match-result');
  if (!id) return;
  out.innerHTML = '<div class="flex items-center gap-8"><span class="spinner"></span> Looking up…</div>';
  try {
    const data = await apiFetch(`/api/entity/lookup?s1_id=${encodeURIComponent(id)}`);
    out.innerHTML = renderMatchResult(data);
  } catch (e) {
    out.innerHTML = `<div class="alert alert-warn">⚠️ ${e.message}</div>`;
  }
}

function renderMatchResult(data) {
  const s1 = data.s1_entity;
  if (!s1) return '<div class="alert alert-warn">Entity not found.</div>';

  let html = `
    <div class="panel">
      <div class="panel-title">SOURCE 1</div>
      <div class="entity-name">${esc(s1.business_name)}</div>
      <div class="entity-meta mt-8">${esc(s1.business_address)}</div>
      <div class="entity-meta">${esc(s1.country)}</div>
    </div>
    <div class="panel-title">MATCHING CANDIDATES <span class="badge badge-idle">${data.candidates.length}</span></div>`;

  if (!data.candidates.length) {
    return html + '<div class="alert alert-info">🔍 No candidates — SINGLETON (no match predicted)</div>';
  }

  data.candidates.forEach(c => {
    const prob    = c.probability;
    const pclass  = prob >= 0.7 ? 'prob-high' : prob >= 0.4 ? 'prob-med' : 'prob-low';
    const badge   = c.decision === 'MATCH' ? 'badge-match' : 'badge-no-match';
    html += `
      <div class="entity-card">
        <div class="entity-header">
          <div>
            <div class="entity-id">${esc(c.entity_id)}</div>
            <div class="entity-name">${esc(c.business_name)}</div>
            <div class="entity-meta">${esc(c.business_address)} · ${esc(c.country)}</div>
          </div>
          <span class="badge ${badge}">${c.decision}</span>
        </div>
        <div class="prob-bar-wrap"><div class="prob-bar ${pclass}" style="width:${(prob*100).toFixed(1)}%"></div></div>
        <div class="flex gap-12 text-xs text-muted mt-8">
          <span>Probability: <strong>${(prob*100).toFixed(1)}%</strong></span>
          <span>Name sim: ${(c.name_similarity*100).toFixed(0)}%</span>
          <span>Addr sim: ${(c.address_similarity*100).toFixed(0)}%</span>
          <span>Country match: ${c.country_match ? '✅' : '❌'}</span>
        </div>
      </div>`;
  });

  const hasMatch = data.has_match;
  html += `<div class="alert ${hasMatch ? 'alert-success' : 'alert-warn'} mt-12">
    ${hasMatch ? '✅ MATCH PREDICTED' : '⚪ NO MATCH — SINGLETON'}
    · Threshold: ${fmt(data.threshold, 4)}
  </div>`;

  return html;
}

// ── EVALUATION ─────────────────────────────────────────────
async function loadEvaluation() {
  try {
    const m = await apiFetch('/api/model/metrics');
    setEl('eval-f05',       fmtPct(m.f05));
    setEl('eval-precision', fmtPct(m.precision));
    setEl('eval-recall',    fmtPct(m.recall));
    setEl('eval-threshold', fmt(m.threshold, 4));
    setEl('eval-matches',   m.match_count ?? '—');
    setEl('eval-singletons',m.singleton_count ?? '—');
    setEl('eval-false-merges', m.false_merges ?? '—');
  } catch { /* not trained yet */ }

  // Load threshold curve
  try {
    const curve = await apiFetch('/api/evaluation/threshold-curve');
    if (curve.available) {
      state.thresholdCurve = curve;
      renderThresholdCharts(curve);
    } else {
      document.getElementById('eval-curve-status').textContent = curve.reason || 'Curve not available';
    }
  } catch (e) {
    console.warn('Curve error:', e.message);
  }
}

function renderThresholdCharts(curve) {
  const labels = curve.curve.map(p => p.threshold);
  const f05s   = curve.curve.map(p => p.f05);
  const precs  = curve.curve.map(p => p.precision);
  const recs   = curve.curve.map(p => p.recall);
  const best   = curve.best_threshold;

  const common = {
    type: 'line',
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: { legend: { labels: { color: '#8b949e' } } },
      scales: {
        x: { ticks: { color: '#8b949e' }, grid: { color: '#21262d' } },
        y: { ticks: { color: '#8b949e' }, grid: { color: '#21262d' }, min: 0, max: 1 },
      },
    },
  };

  const mkDataset = (label, data, color) => ({
    label, data,
    borderColor: color, backgroundColor: color + '22',
    tension: 0.3, pointRadius: 4,
  });

  const annotate = (chart) => {
    const idx = labels.indexOf(best);
    if (idx >= 0) {
      chart.data.datasets.push({
        label: `Best (${best})`,
        data: labels.map((_, i) => i === idx ? chart.data.datasets[0].data[idx] : null),
        pointBackgroundColor: '#f0883e',
        pointRadius: 8,
        showLine: false,
      });
    }
    chart.update();
  };

  // F0.5 chart
  if (state.charts.f05) state.charts.f05.destroy();
  const ctx1 = document.getElementById('chart-f05')?.getContext('2d');
  if (ctx1) {
    state.charts.f05 = new Chart(ctx1, {
      ...common,
      data: { labels, datasets: [mkDataset('F0.5', f05s, '#58a6ff')] },
    });
    annotate(state.charts.f05);
  }

  // Precision chart
  if (state.charts.precision) state.charts.precision.destroy();
  const ctx2 = document.getElementById('chart-precision')?.getContext('2d');
  if (ctx2) {
    state.charts.precision = new Chart(ctx2, {
      ...common,
      data: { labels, datasets: [mkDataset('Precision', precs, '#3fb950')] },
    });
    annotate(state.charts.precision);
  }

  // Recall chart
  if (state.charts.recall) state.charts.recall.destroy();
  const ctx3 = document.getElementById('chart-recall')?.getContext('2d');
  if (ctx3) {
    state.charts.recall = new Chart(ctx3, {
      ...common,
      data: { labels, datasets: [mkDataset('Recall', recs, '#bc8cff')] },
    });
    annotate(state.charts.recall);
  }
}

// ── OUTPUT ─────────────────────────────────────────────────
async function loadOutput() {
  await Promise.all([
    loadFilePreview('candidate_pairs', 'preview-cp'),
    loadFilePreview('matching_results', 'preview-mr'),
  ]);
}

async function loadFilePreview(file, containerId) {
  const el = document.getElementById(containerId);
  if (!el) return;
  el.innerHTML = '<div class="flex items-center gap-8"><span class="spinner"></span> Loading…</div>';
  try {
    const d = await apiFetch(`/api/output/preview?file=${file}`);
    if (!d.preview.length) { el.innerHTML = '<div class="alert alert-warn">File is empty</div>'; return; }
    const cols = d.columns;
    let html = `<div class="text-xs text-muted mb-8">${d.total_rows} rows</div>
      <div class="table-wrap">
      <table><thead><tr>${cols.map(c => `<th>${esc(c)}</th>`).join('')}</tr></thead><tbody>`;
    d.preview.forEach(row => {
      html += '<tr>' + cols.map(c => `<td class="mono">${esc(row[c] ?? '')}</td>`).join('') + '</tr>';
    });
    html += '</tbody></table></div>';
    if (d.total_rows > 100) html += `<div class="text-xs text-muted mt-8">Showing first 100 of ${d.total_rows} rows</div>`;
    el.innerHTML = html;
  } catch (e) {
    el.innerHTML = `<div class="alert alert-warn">⚠️ ${e.message}</div>`;
  }
}

function downloadFile(file) {
  window.open(`${API}/api/output/download?file=${file}`, '_blank');
}

// ── VALIDATION ─────────────────────────────────────────────
async function runValidation() {
  const btn = document.getElementById('btn-validate');
  const out = document.getElementById('validation-output');
  btn.disabled = true;
  out.innerHTML = '<div class="flex items-center gap-8"><span class="spinner"></span> Running validation checks…</div>';
  try {
    const r = await apiFetch('/api/validate', { method: 'POST' });
    out.innerHTML = renderValidationResults(r);
  } catch (e) {
    out.innerHTML = `<div class="alert alert-danger">❌ ${e.message}</div>`;
  }
  btn.disabled = false;
}

function renderValidationResults(data) {
  const allPassed = data.all_passed;
  let html = `<div class="alert ${allPassed ? 'alert-success' : 'alert-danger'} mb-16">
    ${allPassed ? '✅ ALL CHECKS PASSED' : '❌ VALIDATION FAILED — See details below'}
  </div>`;
  data.checks.forEach(c => {
    html += `<div class="check-item ${c.passed ? 'check-pass' : 'check-fail'}">
      <span class="check-icon">${c.passed ? '✅' : '❌'}</span>
      <div class="flex-col">
        <span class="check-name">${esc(c.name)}</span>
        ${c.detail ? `<span class="check-detail">${esc(c.detail)}</span>` : ''}
      </div>
      <span class="badge ${c.passed ? 'badge-pass' : 'badge-fail'}">${c.passed ? 'PASS' : 'FAIL'}</span>
    </div>`;
  });
  return html;
}

// ── SUBMISSION PACKAGE ─────────────────────────────────────
async function createPackage() {
  const btn  = document.getElementById('btn-package');
  const stat = document.getElementById('package-status');
  btn.disabled = true;
  stat.textContent = '⏳ Creating submission zip…';
  stat.className = 'alert alert-info';
  try {
    const r = await apiFetch('/api/submission/package', { method: 'POST' });
    stat.textContent = `✅ Package created: ${fmtBytes(r.size_bytes)}`;
    stat.className = 'alert alert-success';
    document.getElementById('btn-dl-package').classList.remove('hidden');
  } catch (e) {
    stat.textContent = '❌ ' + e.message;
    stat.className = 'alert alert-danger';
  }
  btn.disabled = false;
}

// ── UTILS ──────────────────────────────────────────────────
function setEl(id, value) {
  const el = document.getElementById(id);
  if (el) el.textContent = value ?? '—';
}

function showAlert(id, msg, type = 'info') {
  const el = document.getElementById(id);
  if (el) { el.textContent = msg; el.className = `alert alert-${type}`; }
}

function esc(str) {
  if (str == null) return '';
  return String(str).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');
}

// ── INIT ──────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
  // Wire nav items
  document.querySelectorAll('.nav-item[data-page]').forEach(el => {
    el.addEventListener('click', () => navigate(el.dataset.page));
  });

  // Init upload zones
  initUpload();

  // Load model metrics on model page init
  document.querySelector('[data-page="model"]')?.addEventListener('click', loadModelMetrics);

  // Hash-based routing
  const hash = window.location.hash.replace('#', '') || 'dashboard';
  navigate(hash);

  // Auto-refresh dashboard every 30s
  setInterval(() => { if (state.currentPage === 'dashboard') loadDashboard(); }, 30000);
});
