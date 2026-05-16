// ═══════════════════════════════════════════════════════════
//  MetricsCard.jsx  –  Dice / IoU / Accuracy + coverage display
// ═══════════════════════════════════════════════════════════

// ── Single metric tile ───────────────────────────────────────
function MetricTile({ label, value, description, color, icon, suffix = '' }) {
  const pct   = Math.min(Math.max((value || 0), 0), 1);
  const grade =
    value === null || value === undefined ? null :
    value >= 0.85 ? { text: 'Excellent', color: '#2dd4bf' } :
    value >= 0.70 ? { text: 'Good',      color: '#38bdf8' } :
    value >= 0.55 ? { text: 'Fair',      color: '#f59e0b' } :
                    { text: 'Poor',      color: '#fb7185' };

  const displayVal = value === null || value === undefined
    ? '—'
    : suffix
    ? `${value}${suffix}`
    : value.toFixed(4);

  return (
    <div style={{
      flex: 1, minWidth: 0,
      padding: '18px 16px',
      borderRadius: 'var(--radius-lg)',
      background: 'var(--bg-card)',
      border: '1px solid var(--border)',
      display: 'flex', flexDirection: 'column', gap: 10,
      transition: 'all var(--transition)',
      position: 'relative', overflow: 'hidden',
    }}>
      {/* Subtle corner glow */}
      <div style={{
        position: 'absolute', top: 0, right: 0,
        width: 60, height: 60,
        background: `radial-gradient(circle at top right, ${color}18, transparent 70%)`,
        pointerEvents: 'none',
      }} />

      {/* Header row */}
      <div style={{ display: 'flex', alignItems: 'center',
                    justifyContent: 'space-between' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 7 }}>
          <div style={{
            width: 28, height: 28, borderRadius: 8,
            background: `${color}18`,
            border: `1px solid ${color}30`,
            display: 'flex', alignItems: 'center', justifyContent: 'center',
            flexShrink: 0,
          }}>
            {icon}
          </div>
          <span style={{
            fontFamily: 'var(--font-mono)', fontSize: 11,
            color: 'var(--text-muted)', textTransform: 'uppercase',
            letterSpacing: '0.06em',
          }}>
            {label}
          </span>
        </div>
        {grade && (
          <span style={{
            padding: '2px 8px', borderRadius: 100,
            fontFamily: 'var(--font-mono)', fontSize: 10,
            background: `${grade.color}15`,
            color: grade.color,
            border: `1px solid ${grade.color}30`,
          }}>
            {grade.text}
          </span>
        )}
      </div>

      {/* Value */}
      <div style={{
        fontFamily: 'var(--font-heading)',
        fontSize: 28, fontWeight: 800,
        color: value !== null && value !== undefined ? color : 'var(--text-muted)',
        lineHeight: 1, letterSpacing: '-0.03em',
      }}>
        {displayVal}
      </div>

      {/* Progress bar */}
      {value !== null && value !== undefined && !suffix && (
        <div style={{
          height: 3, borderRadius: 100,
          background: 'var(--bg-elevated)', overflow: 'hidden',
        }}>
          <div style={{
            height: '100%', borderRadius: 100,
            width: `${pct * 100}%`,
            background: `linear-gradient(90deg, ${color}80, ${color})`,
            transition: 'width 0.8s cubic-bezier(0.4,0,0.2,1)',
          }} />
        </div>
      )}

      {/* Description */}
      <div style={{
        fontFamily: 'var(--font-mono)', fontSize: 10.5,
        color: 'var(--text-muted)', lineHeight: 1.5,
      }}>
        {description}
      </div>
    </div>
  );
}

// ── Main MetricsCard export ──────────────────────────────────
export default function MetricsCard({ metrics, tumorCoverage, inferenceMs }) {
  /*
   * Props:
   *   metrics       : { dice, iou, accuracy } | null
   *   tumorCoverage : float (percentage 0–100) | null
   *   inferenceMs   : float (ms) | null
   */

  const hasMetics = metrics &&
    metrics.dice !== undefined &&
    metrics.iou  !== undefined;

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>

      {/* ── Section label ──────────────────────────────────── */}
      <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
        <div style={{
          height: 1, flex: 1,
          background: 'linear-gradient(90deg, var(--border), transparent)',
        }} />
        <span style={{
          fontFamily: 'var(--font-mono)', fontSize: 10.5,
          color: 'var(--text-muted)', textTransform: 'uppercase',
          letterSpacing: '0.1em',
        }}>
          {hasMetics ? 'Evaluation Metrics' : 'Prediction Info'}
        </span>
        <div style={{
          height: 1, flex: 1,
          background: 'linear-gradient(90deg, transparent, var(--border))',
        }} />
      </div>

      {/* ── Metric tiles ───────────────────────────────────── */}
      {hasMetics ? (
        <>
          <div style={{ display: 'flex', gap: 10 }}>
            <MetricTile
              label="Dice"
              value={metrics.dice}
              color="#38bdf8"
              icon={<DiceIcon />}
              description="2·|P∩T| / (|P|+|T|) — primary segmentation metric"
            />
            <MetricTile
              label="IoU"
              value={metrics.iou}
              color="#2dd4bf"
              icon={<IouIcon />}
              description="Intersection over Union — Jaccard index"
            />
            <MetricTile
              label="Accuracy"
              value={metrics.accuracy}
              color="#818cf8"
              icon={<AccIcon />}
              description="Fraction of correctly classified pixels"
            />
          </div>

          {/* Target benchmarks */}
          <div style={{
            padding: '10px 14px',
            borderRadius: 'var(--radius-md)',
            background: 'rgba(56,189,248,0.05)',
            border: '1px solid var(--accent-border)',
            display: 'flex', gap: 20, flexWrap: 'wrap',
          }}>
            {[
              { label: 'Target Dice', val: '≥ 0.80',
                pass: metrics.dice >= 0.80 },
              { label: 'Target IoU',  val: '≥ 0.70',
                pass: metrics.iou  >= 0.70 },
            ].map(({ label, val, pass }) => (
              <div key={label} style={{
                display: 'flex', alignItems: 'center', gap: 6,
              }}>
                <span style={{
                  fontFamily: 'var(--font-mono)', fontSize: 11.5,
                  color: pass ? 'var(--teal)' : 'var(--rose)',
                }}>
                  {pass ? '✓' : '✗'}
                </span>
                <span style={{
                  fontFamily: 'var(--font-mono)', fontSize: 11,
                  color: 'var(--text-secondary)',
                }}>
                  {label}:{' '}
                  <span style={{ color: 'var(--text-primary)', fontWeight: 500 }}>
                    {val}
                  </span>
                </span>
              </div>
            ))}
          </div>
        </>
      ) : (
        /* No ground-truth — show placeholder tiles */
        <div style={{
          padding: '18px',
          borderRadius: 'var(--radius-lg)',
          background: 'var(--bg-card)',
          border: '1px solid var(--border)',
          textAlign: 'center',
        }}>
          <div style={{
            fontFamily: 'var(--font-mono)', fontSize: 12,
            color: 'var(--text-muted)', lineHeight: 1.7,
          }}>
            Upload a ground-truth mask alongside the MRI to compute<br />
            <span style={{ color: 'var(--accent)' }}>Dice · IoU · Pixel Accuracy</span>
          </div>
        </div>
      )}

      {/* ── Coverage + speed row ───────────────────────────── */}
      {(tumorCoverage !== null && tumorCoverage !== undefined) && (
        <div style={{ display: 'flex', gap: 10 }}>
          <MetricTile
            label="Tumour Coverage"
            value={tumorCoverage.toFixed(2)}
            suffix="%"
            color="#f59e0b"
            icon={<ScanIcon />}
            description="% of pixels predicted as tumour"
          />
          {inferenceMs !== null && inferenceMs !== undefined && (
            <MetricTile
              label="Inference Time"
              value={inferenceMs.toFixed(1)}
              suffix=" ms"
              color="#a78bfa"
              icon={<SpeedIcon />}
              description="Model forward-pass duration on server"
            />
          )}
        </div>
      )}
    </div>
  );
}

// ── Icons ────────────────────────────────────────────────────
function DiceIcon() {
  return (
    <svg width="13" height="13" viewBox="0 0 24 24" fill="none"
         stroke="var(--accent)" strokeWidth="2" strokeLinecap="round">
      <rect x="2" y="2" width="20" height="20" rx="3"/>
      <circle cx="8"  cy="8"  r="1.2" fill="var(--accent)"/>
      <circle cx="16" cy="8"  r="1.2" fill="var(--accent)"/>
      <circle cx="8"  cy="16" r="1.2" fill="var(--accent)"/>
      <circle cx="16" cy="16" r="1.2" fill="var(--accent)"/>
    </svg>
  );
}

function IouIcon() {
  return (
    <svg width="13" height="13" viewBox="0 0 24 24" fill="none"
         stroke="var(--teal)" strokeWidth="2" strokeLinecap="round">
      <rect x="3" y="3" width="13" height="13" rx="2"/>
      <rect x="8" y="8" width="13" height="13" rx="2"/>
    </svg>
  );
}

function AccIcon() {
  return (
    <svg width="13" height="13" viewBox="0 0 24 24" fill="none"
         stroke="var(--violet)" strokeWidth="2" strokeLinecap="round">
      <polyline points="20 6 9 17 4 12"/>
    </svg>
  );
}

function ScanIcon() {
  return (
    <svg width="13" height="13" viewBox="0 0 24 24" fill="none"
         stroke="#f59e0b" strokeWidth="2" strokeLinecap="round">
      <circle cx="12" cy="12" r="3"/>
      <path d="M3 9V5a2 2 0 0 1 2-2h4M15 3h4a2 2 0 0 1 2 2v4
               M21 15v4a2 2 0 0 1-2 2h-4M9 21H5a2 2 0 0 1-2-2v-4"/>
    </svg>
  );
}

function SpeedIcon() {
  return (
    <svg width="13" height="13" viewBox="0 0 24 24" fill="none"
         stroke="#a78bfa" strokeWidth="2" strokeLinecap="round">
      <path d="M13 2L3 14h9l-1 8 10-12h-9l1-8z"/>
    </svg>
  );
}