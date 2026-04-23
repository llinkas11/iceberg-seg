/**
 * admin.js — Admin dashboard logic for iceberg-labeler.
 */

'use strict';

let adminToken = null;
let labelers   = [];
let regions    = [];
let szaBins    = [];

// ── Init ──────────────────────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
  const stored = localStorage.getItem('adminToken');
  if (stored) {
    adminToken = stored;
    bootAdmin();
  } else {
    document.getElementById('admin-login-form').style.display = 'block';
  }
});

async function adminLogin() {
  const token = document.getElementById('admin-token-input').value.trim();
  if (!token) return;

  try {
    await adminFetch('/api/admin/stats');
    adminToken = token;
    localStorage.setItem('adminToken', token);
    document.getElementById('admin-login-form').style.display = 'none';
    bootAdmin();
  } catch {
    showAdminAlert('Invalid admin token', 'danger');
  }
}

async function bootAdmin() {
  document.getElementById('admin-content').style.display = 'block';
  await Promise.all([
    loadStats(),
    loadLabelers(),
    loadRegions(),
  ]);
  loadChipBrowser();
}

// ── Stats ─────────────────────────────────────────────────────────────────────
async function loadStats() {
  try {
    const s = await adminFetch('/api/admin/stats');
    document.getElementById('stat-chips').textContent    = s.total_chips;
    document.getElementById('stat-assigned').textContent = s.total_assigned;
    document.getElementById('stat-complete').textContent = s.total_complete;
    document.getElementById('stat-pending').textContent  = s.total_pending;
    document.getElementById('stat-labelers').textContent = s.labeler_count;
    const pct = s.total_assigned > 0
      ? Math.round(s.total_complete / s.total_assigned * 100) : 0;
    document.getElementById('stat-pct').textContent = pct + '%';
  } catch (err) {
    showAdminAlert('Failed to load stats: ' + err.message, 'danger');
  }
}

// ── Labelers ──────────────────────────────────────────────────────────────────
async function loadLabelers() {
  try {
    labelers = await adminFetch('/api/admin/labelers');
    const tbody = document.getElementById('labelers-tbody');
    tbody.innerHTML = '';

    // Populate assign dropdown
    const sel = document.getElementById('assign-labeler-select');
    sel.innerHTML = '<option value="">-- Select labeler --</option>';

    for (const l of labelers) {
      const pct   = l.total_assigned > 0
        ? Math.round(l.complete / l.total_assigned * 100) : 0;
      const tr    = document.createElement('tr');
      tr.innerHTML = `
        <td>${l.name}</td>
        <td>${l.group_name || '—'}</td>
        <td>${l.total_assigned}</td>
        <td><span class="badge badge-complete">${l.complete}</span></td>
        <td><span class="badge badge-pending">${l.pending}</span></td>
        <td><span class="badge badge-skipped">${l.skipped}</span></td>
        <td>
          <div class="mini-progress">
            <div class="mini-bar-bg">
              <div class="mini-bar-fill" style="width:${pct}%"></div>
            </div>
            <span>${pct}%</span>
          </div>
        </td>
        <td>${new Date(l.created_at).toLocaleDateString()}</td>
      `;
      tbody.appendChild(tr);

      const opt = document.createElement('option');
      opt.value       = l.id;
      opt.textContent = l.name;
      sel.appendChild(opt);
    }
  } catch (err) {
    showAdminAlert('Failed to load labelers: ' + err.message, 'danger');
  }
}

// ── Regions / SZA bins ────────────────────────────────────────────────────────
async function loadRegions() {
  try {
    const data = await adminFetch('/api/admin/regions');
    regions  = data.regions;
    szaBins  = data.sza_bins;

    const rSel = document.getElementById('assign-region-select');
    const sSel = document.getElementById('assign-sza-select');
    const brSel = document.getElementById('browse-region-select');
    const bsSel = document.getElementById('browse-sza-select');

    [rSel, brSel].forEach(sel => {
      sel.innerHTML = '<option value="">All regions</option>';
      regions.forEach(r => { sel.innerHTML += `<option value="${r}">${r}</option>`; });
    });

    [sSel, bsSel].forEach(sel => {
      sel.innerHTML = '<option value="">All SZA bins</option>';
      szaBins.forEach(s => { sel.innerHTML += `<option value="${s}">${s}</option>`; });
    });
  } catch (err) {
    // non-fatal — may have no chips yet
  }
}

// ── Chip browser ──────────────────────────────────────────────────────────────
async function loadChipBrowser() {
  const region  = document.getElementById('browse-region-select')?.value || '';
  const szaBin  = document.getElementById('browse-sza-select')?.value   || '';
  const limit   = 100;

  let url = `/api/admin/chips?limit=${limit}`;
  if (region)  url += `&region=${encodeURIComponent(region)}`;
  if (szaBin)  url += `&sza_bin=${encodeURIComponent(szaBin)}`;

  try {
    const data  = await adminFetch(url);
    const tbody = document.getElementById('chips-tbody');
    tbody.innerHTML = '';

    document.getElementById('chip-count').textContent = `${data.chips.length} of ${data.total}`;

    for (const chip of data.chips) {
      const asgns = chip.assignments;
      const nDone = asgns.filter(a => a.status === 'complete').length;
      const statusHtml = asgns.length === 0
        ? '<span class="badge badge-pending">unassigned</span>'
        : asgns.map(a =>
            `<span class="badge badge-${a.status === 'complete' ? 'complete' :
              a.status === 'in_progress' ? 'inprog' :
              a.status === 'skipped'     ? 'skipped' : 'pending'}"
            >${a.status}</span>`
          ).join(' ');

      const tr = document.createElement('tr');
      tr.innerHTML = `
        <td>${chip.id}</td>
        <td style="font-size:11px;font-family:monospace;max-width:260px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap"
            title="${chip.filename}">${chip.filename}</td>
        <td>${chip.region || '—'}</td>
        <td>${chip.sza_bin || '—'}</td>
        <td>${chip.prediction_count}</td>
        <td>${statusHtml}</td>
      `;
      tbody.appendChild(tr);
    }
  } catch (err) {
    showAdminAlert('Failed to load chips: ' + err.message, 'danger');
  }
}

// ── Assign chips ──────────────────────────────────────────────────────────────
async function assignChips() {
  const labeler_id = parseInt(document.getElementById('assign-labeler-select').value);
  const num_chips  = parseInt(document.getElementById('assign-n-input').value);
  const region     = document.getElementById('assign-region-select').value || null;
  const sza_bin    = document.getElementById('assign-sza-select').value   || null;

  if (!labeler_id || !num_chips) {
    showAdminAlert('Select a labeler and number of chips', 'danger'); return;
  }

  try {
    const resp = await adminFetch('/api/admin/assign', {
      method: 'POST',
      body: JSON.stringify({ labeler_id, num_chips, region, sza_bin }),
    });
    showAdminAlert(`Assigned ${resp.assigned} chips to ${resp.labeler_name}`, 'success');
    await loadStats();
    await loadLabelers();
    loadChipBrowser();
  } catch (err) {
    showAdminAlert('Assignment failed: ' + err.message, 'danger');
  }
}

// ── Export ────────────────────────────────────────────────────────────────────
function exportCOCO() {
  const url = `/api/export/coco?` +
    new URLSearchParams({ t: Date.now() }).toString();
  triggerDownload(url);
}

function exportCSV() {
  const url = `/api/export/csv?` +
    new URLSearchParams({ t: Date.now() }).toString();
  triggerDownload(url);
}

function triggerDownload(url) {
  const a = document.createElement('a');
  a.href  = url;
  // Include auth header by fetching with adminFetch → create Blob
  adminFetch(url.replace(/\?.*/, ''), {}, true).then(resp => {
    resp.blob().then(blob => {
      const blobUrl = URL.createObjectURL(blob);
      a.href = blobUrl;
      const cd = resp.headers.get('Content-Disposition') || '';
      const m  = cd.match(/filename="([^"]+)"/);
      a.download = m ? m[1] : 'export';
      a.click();
      URL.revokeObjectURL(blobUrl);
    });
  }).catch(err => showAdminAlert('Export failed: ' + err.message, 'danger'));
}

// ── Admin logout ──────────────────────────────────────────────────────────────
function adminLogout() {
  localStorage.removeItem('adminToken');
  location.reload();
}

// ── API helper ────────────────────────────────────────────────────────────────
async function adminFetch(url, opts = {}, rawResponse = false) {
  const headers = {
    'Content-Type':  'application/json',
    'Authorization': `Bearer ${adminToken || document.getElementById('admin-token-input')?.value?.trim()}`,
    ...opts.headers,
  };
  const resp = await fetch(url, { ...opts, headers });
  if (!resp.ok) {
    const err = await resp.json().catch(() => ({ detail: resp.statusText }));
    throw new Error(err.detail || `HTTP ${resp.status}`);
  }
  if (rawResponse) return resp;
  return resp.json();
}

// ── Alert helper ──────────────────────────────────────────────────────────────
function showAdminAlert(msg, type = 'info') {
  const el = document.getElementById('admin-alert');
  if (!el) return;
  el.className     = `alert alert-${type}`;
  el.textContent   = msg;
  el.style.display = 'block';
  if (type !== 'danger') setTimeout(() => { el.style.display = 'none'; }, 4000);
}
