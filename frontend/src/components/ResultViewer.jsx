// ═══════════════════════════════════════════════════════════
//  ResultViewer.jsx  –  Side-by-side prediction result display
//
//  Panels (tabbed):
//    Original  |  Predicted Mask  |  Overlay  |  Heatmap  |  Ground Truth*
//  * Ground Truth tab only shown when gt_mask_b64 is provided
// ═══════════════════════════════════════════════════════════

import { useState } from 'react';
import { b64ToDataUri, downloadB64Image } from '../services/api.js';

const PANELS = [
  { key: 'original',  label: 'Original',  color: '#38bdf8' },
  { key: 'overlay',   label: 'Overlay',   color: '#2dd4bf' },
  { key: 'mask',      label: 'Pred. Mask',color: '#818cf8' },
  { key: 'heatmap',   label: 'Heatmap',   color: '#f59e0b' },
  { key: 'gt',        label: 'Ground Truth', color: '#4ade80' },
];

export default function ResultViewer({ result }) {
  /*
   * result shape (from api.js):
   *   original_b64   : string
   *   overlay_b64    : string
   *   pred_mask_b64  : string
   *   prob_map_b64   : string
   *   gt_mask_b64?   : string   (only when mask was uploaded)
   *   tumor_coverage : number
   *   inference_ms   : number
   *   metrics?       : { dice, iou, accuracy }
   */

  const [activePanel, setActivePanel] = useState('overlay');
  const [compareMode, setCompareMode] = useState(false);

  if (!result) return null;

  const hasGT = !!result.gt_mask_b64;

  // Map panel key → base64 string
  const imageMap = {
    original : result.original_b64,
    overlay  : result.overlay_b64,
    mask     : result.pred_mask_b64,
    heatmap  : result.prob_map_b64,
    gt       : result.gt_mask_b64,
  };

  const visiblePanels = PANELS.filter(p => p.key !== 'gt' || hasGT);
  const active = visiblePanels.find(p => p.key === activePanel)
              || visiblePanels[0];

  const handleDownload = () => {
    const b64 = imageMap[active.key];
    if (b64) downloadB64Image(b64, `${active.key}_result.png`);
  };

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 14 }}>

      {/* ── Tab bar ─────────────────────────────────────────── */}
      <div style={{
        display: 'flex', alignItems: 'center',
        gap: 6, flexWrap: 'wrap',
      }}>
        {visiblePanels.map(panel => (
          <button
            key={panel.key}
            onClick={() => { setActivePanel(panel.key); setCompareMode(false); }}
            style={{
              padding: '6px 14px',
              borderRadius: 8,
              border: `1px solid ${activePanel === panel.key
                ? panel.color + '50'
                : 'var(--border)'}`,
              background: activePanel === panel.key
                ? panel.color + '15'
                : 'transparent',
              color: activePanel === panel.key
                ? panel.color
                : 'var(--text-muted)',
              fontFamily: 'var(--font-mono)',
              fontSize: 11.5,
              cursor: 'pointer',
              transition: 'all 0.18s ease',
              fontWeight: activePanel === panel.key ? 600 : 400,
            }}
          >
            {panel.label}
          </button>
        ))}

        {/* Compare toggle — only show when GT is available */}
        {hasGT && (
          <button
            onClick={() => setCompareMode(c => !c)}
            style={{
              marginLeft: 'auto',
              padding: '6px 14px',
              borderRadius: 8,
              border: `1px solid ${compareMode ? 'var(--teal)' : 'var(--border)'}`,
              background: compareMode ? 'rgba(45,212,191,0.12)' : 'transparent',
              color: compareMode ? 'var(--teal)' : 'var(--text-muted)',
              fontFamily: 'var(--font-mono)', fontSize: 11.5,
              cursor: 'pointer', transition: 'all 0.18s',
              display: 'flex', alignItems: 'center', gap: 6,
            }}
          >
            <CompareIcon /> Compare
          </button>
        )}
      </div>

      {/* ── Image display ────────────────────────────────────── */}
      {compareMode && hasGT ? (
        /* Side-by-side compare: Prediction vs Ground Truth */
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 10 }}>
          {[
            { b64: result.overlay_b64,  label: 'Prediction Overlay', color: '#2dd4bf' },
            { b64: result.gt_mask_b64,  label: 'Ground Truth Mask',   color: '#4ade80' },
          ].map(({ b64, label, color }) => (
            <ImagePanel key={label} b64={b64} label={label} color={color}
                        tumorCoverage={null} />
          ))}
        </div>
      ) : (
        /* Single panel view */
        <ImagePanel
          b64={imageMap[active.key]}
          label={active.label}
          color={active.color}
          tumorCoverage={active.key === 'overlay' ? result.tumor_coverage : null}
          panelKey={active.key}
        />
      )}

      {/* ── Action buttons ──────────────────────────────────── */}
      <div style={{ display: 'flex', gap: 8 }}>
        <button
          onClick={handleDownload}
          style={{
            padding: '9px 18px',
            borderRadius: 'var(--radius-md)',
            border: '1px solid var(--border)',
            background: 'var(--bg-elevated)',
            color: 'var(--text-secondary)',
            fontFamily: 'var(--font-body)', fontSize: 13,
            cursor: 'pointer', transition: 'all 0.2s',
            display: 'flex', alignItems: 'center', gap: 7,
          }}
          onMouseEnter={e => {
            e.currentTarget.style.borderColor = 'var(--border-hover)';
            e.currentTarget.style.color = 'var(--text-primary)';
          }}
          onMouseLeave={e => {
            e.currentTarget.style.borderColor = 'var(--border)';
            e.currentTarget.style.color = 'var(--text-secondary)';
          }}
        >
          <DownloadIcon /> Download {active.label}
        </button>

        {/* Download all */}
        <button
          onClick={() => {
            Object.entries(imageMap).forEach(([key, b64]) => {
              if (b64) setTimeout(() => downloadB64Image(b64, `${key}.png`), 0);
            });
          }}
          style={{
            padding: '9px 18px',
            borderRadius: 'var(--radius-md)',
            border: '1px solid var(--border)',
            background: 'transparent',
            color: 'var(--text-muted)',
            fontFamily: 'var(--font-body)', fontSize: 13,
            cursor: 'pointer', transition: 'all 0.2s',
          }}
        >
          Download All
        </button>
      </div>
    </div>
  );
}

// ── Single image panel ───────────────────────────────────────
function ImagePanel({ b64, label, color, tumorCoverage, panelKey }) {
  const [zoom, setZoom] = useState(false);

  if (!b64) {
    return (
      <div style={{
        aspectRatio: '1 / 1',
        borderRadius: 'var(--radius-lg)',
        background: 'var(--bg-card)',
        border: '1px solid var(--border)',
        display: 'flex', alignItems: 'center', justifyContent: 'center',
      }}>
        <span style={{
          fontFamily: 'var(--font-mono)', fontSize: 12,
          color: 'var(--text-muted)',
        }}>No image</span>
      </div>
    );
  }

  return (
    <>
      <div style={{
        position: 'relative',
        borderRadius: 'var(--radius-lg)',
        overflow: 'hidden',
        border: `1px solid ${color}25`,
        background: '#000',
        cursor: 'zoom-in',
        boxShadow: `0 0 30px ${color}10`,
      }}
        onClick={() => setZoom(true)}
      >
        <img
          src={b64ToDataUri(b64)}
          alt={label}
          style={{
            width: '100%',
            display: 'block',
            objectFit: 'contain',
          }}
        />

        {/* Label chip */}
        <div style={{
          position: 'absolute', top: 12, left: 12,
          padding: '4px 10px', borderRadius: 100,
          background: 'rgba(4,8,15,0.75)',
          backdropFilter: 'blur(8px)',
          border: `1px solid ${color}40`,
          fontFamily: 'var(--font-mono)', fontSize: 10.5,
          color: color, fontWeight: 500, letterSpacing: '0.04em',
        }}>
          {label}
        </div>

        {/* Coverage badge on overlay panel */}
        {tumorCoverage !== null && tumorCoverage !== undefined && (
          <div style={{
            position: 'absolute', bottom: 12, right: 12,
            padding: '5px 12px', borderRadius: 100,
            background: 'rgba(4,8,15,0.80)',
            backdropFilter: 'blur(8px)',
            border: '1px solid rgba(245,158,11,0.3)',
            fontFamily: 'var(--font-mono)', fontSize: 11,
            color: '#f59e0b',
          }}>
            {tumorCoverage.toFixed(2)}% tumour
          </div>
        )}

        {/* Zoom hint */}
        <div style={{
          position: 'absolute', top: 12, right: 12,
          padding: '4px 8px', borderRadius: 8,
          background: 'rgba(4,8,15,0.6)',
          fontFamily: 'var(--font-mono)', fontSize: 10,
          color: 'var(--text-muted)',
        }}>
          🔍
        </div>

        {/* Heatmap colour scale legend */}
        {panelKey === 'heatmap' && (
          <div style={{
            position: 'absolute', bottom: 12, left: 12,
            display: 'flex', alignItems: 'center', gap: 6,
          }}>
            <span style={{
              fontFamily: 'var(--font-mono)', fontSize: 9.5,
              color: 'rgba(255,255,255,0.5)',
            }}>Low</span>
            <div style={{
              width: 60, height: 6, borderRadius: 3,
              background: 'linear-gradient(90deg, #000, #f00, #ff0, #fff)',
            }} />
            <span style={{
              fontFamily: 'var(--font-mono)', fontSize: 9.5,
              color: 'rgba(255,255,255,0.5)',
            }}>High</span>
          </div>
        )}
      </div>

      {/* ── Lightbox / zoom modal ──────────────────────────── */}
      {zoom && (
        <div
          onClick={() => setZoom(false)}
          style={{
            position: 'fixed', inset: 0, zIndex: 1000,
            background: 'rgba(4,8,15,0.92)',
            backdropFilter: 'blur(12px)',
            display: 'flex', alignItems: 'center', justifyContent: 'center',
            cursor: 'zoom-out', padding: 24,
          }}
        >
          <img
            src={b64ToDataUri(b64)}
            alt={label}
            style={{
              maxWidth: '90vw', maxHeight: '90vh',
              objectFit: 'contain', borderRadius: 12,
              boxShadow: `0 0 80px ${color}25`,
            }}
          />
          <button
            onClick={() => setZoom(false)}
            style={{
              position: 'fixed', top: 20, right: 24,
              width: 36, height: 36, borderRadius: '50%',
              border: '1px solid var(--border)',
              background: 'var(--bg-elevated)',
              color: 'var(--text-secondary)',
              fontSize: 16, cursor: 'pointer',
              display: 'flex', alignItems: 'center', justifyContent: 'center',
            }}
          >✕</button>
        </div>
      )}
    </>
  );
}

// ── Icons ────────────────────────────────────────────────────
function DownloadIcon() {
  return (
    <svg width="14" height="14" viewBox="0 0 24 24" fill="none"
         stroke="currentColor" strokeWidth="2" strokeLinecap="round">
      <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/>
      <polyline points="7 10 12 15 17 10"/>
      <line x1="12" y1="15" x2="12" y2="3"/>
    </svg>
  );
}

function CompareIcon() {
  return (
    <svg width="13" height="13" viewBox="0 0 24 24" fill="none"
         stroke="currentColor" strokeWidth="2" strokeLinecap="round">
      <path d="M9 3H5a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h4"/>
      <path d="M15 3h4a2 2 0 0 1 2 2v14a2 2 0 0 1-2 2h-4"/>
      <line x1="12" y1="3" x2="12" y2="21"/>
    </svg>
  );
}