// ═══════════════════════════════════════════════════════════
//  services/api.js  –  All HTTP calls to the Flask backend
//
//  Base URL is read from .env:
//    REACT_APP_API_URL=http://localhost:5000
//  Falls back to http://localhost:5000 if not set.
// ═══════════════════════════════════════════════════════════

// Vite exposes env vars via import.meta.env (not process.env)
const BASE_URL =
  import.meta.env.VITE_API_URL || 'http://localhost:5000';

// ── Generic fetch wrapper ────────────────────────────────────
async function apiFetch(endpoint, options = {}) {
  const url = `${BASE_URL}${endpoint}`;
  try {
    const res = await fetch(url, options);
    const data = await res.json();
    if (!res.ok) {
      throw new Error(data.error || `HTTP ${res.status}`);
    }
    return data;
  } catch (err) {
    if (err.name === 'TypeError') {
      // Network error — server is down
      throw new Error('Cannot reach the Flask server. Is it running?');
    }
    throw err;
  }
}

// ── GET /health ──────────────────────────────────────────────
export async function checkHealth() {
  return apiFetch('/health');
}

// ── GET /model-info ──────────────────────────────────────────
export async function getModelInfo() {
  return apiFetch('/model-info');
}

// ── POST /predict ────────────────────────────────────────────
/**
 * Send a single MRI image file to /predict.
 *
 * @param {File} imageFile  - File object from <input type="file">
 * @param {Function} onProgress - optional progress callback (not used but kept for future)
 * @returns {Promise<{
 *   pred_mask_b64: string,
 *   overlay_b64: string,
 *   original_b64: string,
 *   prob_map_b64: string,
 *   tumor_coverage: number,
 *   inference_ms: number
 * }>}
 */
export async function predictImage(imageFile) {
  const form = new FormData();
  form.append('file', imageFile);

  return apiFetch('/predict', {
    method: 'POST',
    body: form,
  });
}

// ── POST /predict-with-mask ──────────────────────────────────
/**
 * Send an MRI image + ground-truth mask to get metrics too.
 *
 * @param {File} imageFile  - MRI image
 * @param {File} maskFile   - Ground-truth binary mask
 * @returns {Promise<{
 *   ...same as predictImage...,
 *   gt_mask_b64: string,
 *   metrics: { dice: number, iou: number, accuracy: number }
 * }>}
 */
export async function predictWithMask(imageFile, maskFile) {
  const form = new FormData();
  form.append('image', imageFile);
  form.append('mask',  maskFile);

  return apiFetch('/predict-with-mask', {
    method: 'POST',
    body: form,
  });
}

// ── Helpers ──────────────────────────────────────────────────

/**
 * Convert a base64 PNG string to a data URI usable in <img src>.
 * React usage:  <img src={b64ToDataUri(result.overlay_b64)} />
 */
export function b64ToDataUri(b64) {
  return `data:image/png;base64,${b64}`;
}

/**
 * Trigger a browser download of a base64 image.
 * @param {string} b64      - base64 PNG string
 * @param {string} filename - e.g. 'predicted_mask.png'
 */
export function downloadB64Image(b64, filename = 'result.png') {
  const link = document.createElement('a');
  link.href     = b64ToDataUri(b64);
  link.download = filename;
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
}