/**
 * annotator.js — Core annotation logic for iceberg-labeler.
 *
 * Uses Leaflet.js with CRS.Simple for pixel-coordinate image display,
 * and Leaflet.draw for interactive polygon editing.
 *
 * Coordinate conventions:
 *   Storage (pixel):  [col, row]  where col=x, row=y, (0,0)=top-left
 *   Leaflet (latLng): [H-row, col]  (y-axis flipped so image shows correctly)
 */

'use strict';

// ── State ─────────────────────────────────────────────────────────────────────
const state = {
  labeler:        null,   // { labeler_id, name, token, is_admin }
  chip:           null,   // current ChipWithPredictions
  progress:       { complete: 0, total: 0 },

  overlayVisible: true,
  overlayOpacity: 0.55,

  // polygon layers indexed by prediction_id (or "new_N" for drawn polygons)
  polyLayers:     {},     // id → { layer, prediction_id, class_name, action, coords }
  selectedPoly:   null,   // id of currently selected polygon
  drawClass:      'iceberg',   // class to apply when drawing a new polygon
  newPolyCounter: 0,
};

let map           = null;
let imageOverlay  = null;
let drawnItems    = null;
let drawControl   = null;
let isDrawing     = false;

const CHIP_W = 256;
const CHIP_H = 256;

// ── Coordinate helpers ────────────────────────────────────────────────────────
function pixelToLatLng(col, row) {
  return L.latLng(CHIP_H - row, col);
}

function latLngToPixel(latLng) {
  return [latLng.lng, CHIP_H - latLng.lat];   // [col, row]
}

function pixelCoordsToLatLngs(coords) {
  return coords.map(([col, row]) => pixelToLatLng(col, row));
}

function latLngsToPixelCoords(latLngs) {
  return latLngs.map(ll => latLngToPixel(ll));
}

function polygonArea(coords) {
  let area = 0;
  const n = coords.length;
  for (let i = 0; i < n; i++) {
    const [x1, y1] = coords[i];
    const [x2, y2] = coords[(i + 1) % n];
    area += x1 * y2 - x2 * y1;
  }
  return Math.abs(area) / 2;
}

// ── Map initialisation ────────────────────────────────────────────────────────
function initMap() {
  if (map) { map.remove(); map = null; }

  map = L.map('leaflet-map', {
    crs:          L.CRS.Simple,
    minZoom:      -3,
    maxZoom:       4,
    zoomSnap:      0.25,
    zoomControl:  true,
    attributionControl: false,
  });

  drawnItems = new L.FeatureGroup().addTo(map);

  // Draw control (polygon tool only)
  drawControl = new L.Control.Draw({
    draw: {
      polygon:   { shapeOptions: { color: '#2dc653', weight: 2 } },
      polyline:  false,
      rectangle: false,
      circle:    false,
      circlemarker: false,
      marker:    false,
    },
    edit: { featureGroup: drawnItems, edit: false, remove: false },
  });

  map.on(L.Draw.Event.CREATED, onPolygonDrawn);

  return map;
}

// ── Load chip onto map ────────────────────────────────────────────────────────
function loadChip(chip) {
  state.chip = chip;
  state.polyLayers  = {};
  state.selectedPoly = null;
  state.newPolyCounter = 0;

  // Remove old layers
  if (imageOverlay) { imageOverlay.remove(); }
  drawnItems.clearLayers();

  const bounds = [[0, 0], [CHIP_H, CHIP_W]];
  const imgUrl = `/api/chips/image/${chip.id}?t=${Date.now()}`;

  imageOverlay = L.imageOverlay(imgUrl, bounds, { opacity: 1 }).addTo(map);
  map.fitBounds(bounds, { padding: [20, 20] });

  // Render predictions
  for (const pred of chip.predictions) {
    addPredictionLayer(pred);
  }

  updateOverlayVisibility();
  renderPolygonList();
  updateTopBar();
}

// ── Add one prediction polygon ────────────────────────────────────────────────
function addPredictionLayer(pred) {
  const latLngs = pixelCoordsToLatLngs(pred.pixel_coords);
  const color   = pred.class_name === 'iceberg' ? '#3a86ff' : '#fb8500';

  const layer = L.polygon(latLngs, {
    color:       color,
    weight:      1.5,
    opacity:     state.overlayOpacity,
    fillOpacity: state.overlayOpacity * 0.4,
    fillColor:   color,
  });

  layer.addTo(drawnItems);
  layer.on('click', () => selectPolygon(String(pred.id)));

  const id = String(pred.id);
  state.polyLayers[id] = {
    layer,
    prediction_id: pred.id,
    class_name:    pred.class_name,
    action:        'pending',   // pending | accepted | rejected | modified
    coords:        pred.pixel_coords,
    isNew:         false,
  };
}

// ── Polygon drawn by labeler ──────────────────────────────────────────────────
function onPolygonDrawn(e) {
  stopDrawMode();
  const latLngs  = e.layer.getLatLngs()[0];
  const coords   = latLngsToPixelCoords(latLngs);
  const color    = state.drawClass === 'iceberg' ? '#2dc653' : '#fb5c00';
  const id       = `new_${++state.newPolyCounter}`;

  const layer = L.polygon(latLngs, {
    color:       color,
    weight:      2,
    opacity:     1,
    fillOpacity: 0.35,
    fillColor:   color,
  });

  layer.addTo(drawnItems);
  layer.on('click', () => selectPolygon(id));

  state.polyLayers[id] = {
    layer,
    prediction_id: null,
    class_name:    state.drawClass,
    action:        'added',
    coords,
    isNew:         true,
  };

  renderPolygonList();
  selectPolygon(id);
}

// ── Draw mode ─────────────────────────────────────────────────────────────────
function startDrawMode(cls) {
  state.drawClass = cls;
  isDrawing = true;
  if (!map.hasControl(drawControl)) {
    map.addControl(drawControl);
  }
  new L.Draw.Polygon(map, drawControl.options.draw.polygon).enable();
  document.getElementById('draw-hint').textContent =
    'Click to place vertices. Double-click to finish. Drawing: iceberg';
  document.getElementById('draw-hint').style.display = 'block';
}

function stopDrawMode() {
  isDrawing = false;
  document.getElementById('draw-hint').style.display = 'none';
}

// ── Select polygon ────────────────────────────────────────────────────────────
function selectPolygon(id) {
  // Deselect previous
  if (state.selectedPoly && state.polyLayers[state.selectedPoly]) {
    refreshLayerStyle(state.selectedPoly);
  }
  state.selectedPoly = id;

  const entry = state.polyLayers[id];
  if (entry) {
    entry.layer.setStyle({ weight: 3, opacity: 1, fillOpacity: 0.5 });
    // Scroll list item into view
    const el = document.getElementById(`poly-item-${id}`);
    if (el) el.scrollIntoView({ block: 'nearest' });
  }
  renderPolygonList();
}

function refreshLayerStyle(id) {
  const entry = state.polyLayers[id];
  if (!entry) return;
  const { action, class_name, isNew } = entry;

  let color, opacity, fillOpacity;
  if (action === 'accepted') {
    color = '#2dc653'; opacity = 0.9; fillOpacity = 0.25;
  } else if (action === 'rejected') {
    color = '#e63946'; opacity = 0.5; fillOpacity = 0.1;
  } else if (isNew || action === 'added') {
    color = '#2dc653'; opacity = state.overlayOpacity; fillOpacity = 0.3;
  } else {
    color = class_name === 'iceberg' ? '#3a86ff' : '#fb8500';
    opacity = state.overlayOpacity; fillOpacity = state.overlayOpacity * 0.4;
  }

  entry.layer.setStyle({ color, weight: 1.5, opacity, fillOpacity, fillColor: color });
}

function setPolygonAction(id, action) {
  if (!state.polyLayers[id]) return;
  state.polyLayers[id].action = action;
  refreshLayerStyle(id);

  // Auto-advance selection to next pending polygon
  const ids = Object.keys(state.polyLayers);
  const nextPending = ids.find(pid => pid !== id &&
    state.polyLayers[pid].action === 'pending');
  if (nextPending) {
    selectPolygon(nextPending);
  } else {
    selectPolygon(null);
  }
  renderPolygonList();
}

// ── Overlay visibility & opacity ──────────────────────────────────────────────
function updateOverlayVisibility() {
  const vis = state.overlayVisible;
  for (const id in state.polyLayers) {
    const entry = state.polyLayers[id];
    if (vis) { entry.layer.addTo(drawnItems); }
    else      { drawnItems.removeLayer(entry.layer); }
  }
}

function updateOverlayOpacity() {
  state.overlayOpacity = parseFloat(
    document.getElementById('opacity-slider').value
  );
  for (const id in state.polyLayers) {
    if (state.polyLayers[id].action === 'pending') {
      refreshLayerStyle(id);
    }
  }
}

// ── Render polygon sidebar list ────────────────────────────────────────────────
function renderPolygonList() {
  const container = document.getElementById('polygon-list');
  if (!container) return;
  container.innerHTML = '';

  const ids = Object.keys(state.polyLayers);
  if (ids.length === 0) {
    container.innerHTML = '<p style="color:var(--c-subtext);padding:12px;font-size:12px;">No predictions for this chip</p>';
    return;
  }

  for (const id of ids) {
    const entry  = state.polyLayers[id];
    const area   = polygonArea(entry.coords);
    const isSel  = id === state.selectedPoly;
    const dotCls = entry.isNew ? 'new' : entry.class_name;

    const item = document.createElement('div');
    item.id        = `poly-item-${id}`;
    item.className = `polygon-item ${isSel ? 'selected' : ''} ${entry.action !== 'pending' ? entry.action : ''}`;
    item.onclick   = () => selectPolygon(id);

    item.innerHTML = `
      <div class="poly-class-dot ${dotCls}"></div>
      <div class="poly-info">
        <div class="poly-label">${entry.isNew ? '+ ' : ''}${entry.class_name}</div>
        <div class="poly-area">${area.toFixed(0)} px² ${entry.action !== 'pending' ? `· ${entry.action}` : ''}</div>
      </div>
      <div class="poly-actions">
        <button class="poly-btn poly-btn-accept" title="Accept (A)" onclick="event.stopPropagation();setPolygonAction('${id}','accepted')">✓</button>
        <button class="poly-btn poly-btn-reject" title="Reject (R)" onclick="event.stopPropagation();setPolygonAction('${id}','rejected')">✗</button>
      </div>
    `;
    container.appendChild(item);
  }
}

// ── Top bar updates ───────────────────────────────────────────────────────────
function updateTopBar() {
  const chip = state.chip;
  if (!chip) return;
  document.getElementById('chip-info').textContent =
    `${chip.filename}  ·  ${chip.region || ''}  ·  ${chip.sza_bin || ''}  ·  ${chip.prediction_count} predictions`;

  const { complete, total } = state.progress;
  const pct = total > 0 ? (complete / total * 100) : 0;
  document.getElementById('progress-fill').style.width = pct + '%';
  document.getElementById('progress-text').textContent = `${complete}/${total}`;
}

// ── Submit actions ────────────────────────────────────────────────────────────
async function submitVerdict(verdict) {
  if (!state.chip) return;

  const btn = document.getElementById(`btn-${verdict}`);
  if (btn) { btn.disabled = true; btn.textContent = 'Saving…'; }

  let polygon_decisions = [];

  if (verdict === 'accepted' || verdict === 'rejected') {
    polygon_decisions = state.chip.predictions.map(p => ({
      prediction_id: p.id,
      action:        verdict === 'accepted' ? 'accepted' : 'rejected',
      class_name:    p.class_name,
      pixel_coords:  p.pixel_coords,
    }));
  } else if (verdict === 'edited') {
    // Validate: every prediction must have a decision
    for (const id in state.polyLayers) {
      const entry = state.polyLayers[id];
      if (entry.action === 'pending') {
        showAlert('Please accept or reject every polygon before submitting.', 'danger');
        if (btn) { btn.disabled = false; btn.textContent = 'Submit Edit'; }
        return;
      }
      polygon_decisions.push({
        prediction_id: entry.prediction_id,
        action:        entry.action,
        class_name:    entry.class_name,
        pixel_coords:  entry.coords,
      });
    }
  } else if (verdict === 'skipped') {
    polygon_decisions = [];
  }

  const notes = document.getElementById('notes-field')?.value || '';
  const tags  = typeof getSelectedTags === 'function' ? getSelectedTags() : [];

  try {
    const resp = await apiFetch('/api/annotations', {
      method: 'POST',
      body: JSON.stringify({
        chip_id:           state.chip.id,
        assignment_id:     state.chip.assignment_id,
        verdict,
        polygon_decisions,
        notes,
        tags,
      }),
    });

    state.progress.complete = Math.min(state.progress.complete + 1, state.progress.total);

    if (resp.next_chip_available) {
      await loadNextChip();
    } else {
      showDoneState();
    }
  } catch (err) {
    showAlert(err.message || 'Submission failed', 'danger');
    if (btn) { btn.disabled = false; btn.textContent = btn._origText || verdict; }
  }
}

async function loadNextChip() {
  try {
    const chip = await apiFetch('/api/chips/next');
    if (!chip) { showDoneState(); return; }

    document.getElementById('notes-field').value = '';
    hideAlert();
    loadChip(chip);
  } catch (err) {
    showAlert('Could not load next chip: ' + err.message, 'danger');
  }
}

function showDoneState() {
  document.getElementById('leaflet-map').innerHTML = `
    <div class="empty-state">
      <div class="icon">🎉</div>
      <h2>All done!</h2>
      <p>You've completed all your assigned chips.<br>
         Thank you for your contributions.</p>
    </div>
  `;
  document.getElementById('chip-info').textContent = 'Queue complete';
}

// ── Accept / Reject all helpers ───────────────────────────────────────────────
function acceptAll() {
  for (const id in state.polyLayers) {
    state.polyLayers[id].action = 'accepted';
    refreshLayerStyle(id);
  }
  renderPolygonList();
  submitVerdict('accepted');
}

function rejectAll() {
  for (const id in state.polyLayers) {
    state.polyLayers[id].action = 'rejected';
    refreshLayerStyle(id);
  }
  renderPolygonList();
  submitVerdict('rejected');
}

// ── Keyboard shortcuts ────────────────────────────────────────────────────────
document.addEventListener('keydown', e => {
  if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return;
  if (!state.chip) return;

  switch (e.key.toLowerCase()) {
    case 'a': acceptAll(); break;
    case 'x': rejectAll(); break;
    case 's': submitVerdict('skipped'); break;
    case 'i': startDrawMode('iceberg'); break;
    case 'escape': stopDrawMode(); break;

    // Accept/reject selected polygon
    case 'arrowup':
    case 'arrowdown': {
      e.preventDefault();
      const ids   = Object.keys(state.polyLayers);
      const idx   = ids.indexOf(state.selectedPoly);
      const next  = e.key === 'ArrowDown' ? idx + 1 : idx - 1;
      if (next >= 0 && next < ids.length) selectPolygon(ids[next]);
      break;
    }
    case 'enter': {
      if (state.selectedPoly) setPolygonAction(state.selectedPoly, 'accepted');
      break;
    }
    case 'delete':
    case 'backspace': {
      if (state.selectedPoly) setPolygonAction(state.selectedPoly, 'rejected');
      break;
    }
  }
});

// ── API helpers ───────────────────────────────────────────────────────────────
async function apiFetch(url, opts = {}) {
  const headers = { 'Content-Type': 'application/json' };
  if (state.labeler?.token) {
    headers['Authorization'] = `Bearer ${state.labeler.token}`;
  }
  const resp = await fetch(url, { ...opts, headers: { ...headers, ...opts.headers } });
  if (!resp.ok) {
    const err = await resp.json().catch(() => ({ detail: resp.statusText }));
    throw new Error(err.detail || `HTTP ${resp.status}`);
  }
  if (resp.status === 204) return null;
  return resp.json();
}

// ── UI helpers ────────────────────────────────────────────────────────────────
function showAlert(msg, type = 'danger') {
  const el = document.getElementById('app-alert');
  if (!el) return;
  el.className = `alert alert-${type}`;
  el.textContent = msg;
  el.style.display = 'block';
}

function hideAlert() {
  const el = document.getElementById('app-alert');
  if (el) el.style.display = 'none';
}

// ── Export for HTML ───────────────────────────────────────────────────────────
window.annotator = {
  state, apiFetch, initMap, loadChip,
  acceptAll, rejectAll, submitVerdict, loadNextChip,
  selectPolygon, setPolygonAction,
  startDrawMode, stopDrawMode,
  updateOverlayVisibility, updateOverlayOpacity,
  renderPolygonList,
};
