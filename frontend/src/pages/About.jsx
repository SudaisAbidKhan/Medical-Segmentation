// ═══════════════════════════════════════════════════════════
//  About.jsx  –  Model info, architecture & dataset details
// ═══════════════════════════════════════════════════════════

export default function About({ modelInfo, serverStatus }) {
  return (
    <div style={{ maxWidth: 1100, margin: '0 auto', padding: '40px 2rem 72px' }}>

      {/* ── Page header ──────────────────────────────────── */}
      <div className="fade-in" style={{ marginBottom: 48 }}>
        <div style={{
          display: 'inline-flex', alignItems: 'center', gap: 8,
          padding: '4px 12px', borderRadius: 100,
          background: 'rgba(129,140,248,0.10)',
          border: '1px solid rgba(129,140,248,0.22)',
          marginBottom: 16,
        }}>
          <span style={{
            fontFamily: 'var(--font-mono)', fontSize: 11,
            color: 'var(--violet)', letterSpacing: '0.06em',
            textTransform: 'uppercase',
          }}>
            Model Documentation
          </span>
        </div>
        <h1 style={{
          fontFamily: 'var(--font-heading)',
          fontSize: 'clamp(26px, 3.5vw, 40px)',
          fontWeight: 800, letterSpacing: '-0.03em',
          lineHeight: 1.1, marginBottom: 12,
        }}>
          U-Net Architecture &{' '}
          <span style={{
            background: 'linear-gradient(135deg, #818cf8, #38bdf8)',
            WebkitBackgroundClip: 'text',
            WebkitTextFillColor: 'transparent',
          }}>Dataset</span>
        </h1>
        <p style={{
          fontFamily: 'var(--font-body)', fontSize: 15,
          color: 'var(--text-secondary)', maxWidth: 540,
          lineHeight: 1.7,
        }}>
          Complete technical reference for the segmentation model,
          training dataset, and evaluation benchmarks.
        </p>
      </div>

      {/* ── Live model stats from /model-info ────────────── */}
      <LiveStats modelInfo={modelInfo} serverStatus={serverStatus} />

      {/* ── Two-column grid ──────────────────────────────── */}
      <div style={{
        display: 'grid', gridTemplateColumns: '1fr 1fr',
        gap: 20, marginTop: 24,
      }}>
        <ArchCard />
        <DatasetCard />
      </div>

      {/* ── Full-width pipeline diagram ───────────────────── */}
      <PipelineDiagram />

      {/* ── Training details + metrics ────────────────────── */}
      <div style={{
        display: 'grid', gridTemplateColumns: '1fr 1fr',
        gap: 20, marginTop: 20,
      }}>
        <TrainingCard />
        <BenchmarkCard />
      </div>

      {/* ── Paper reference ───────────────────────────────── */}
      <PaperCard />
    </div>
  );
}

// ── Live stats strip (from /model-info endpoint) ─────────────
function LiveStats({ modelInfo, serverStatus }) {
  const stats = modelInfo
    ? [
        { label: 'Architecture',     value: modelInfo.architecture },
        { label: 'Trainable Params', value: modelInfo.trainable_params?.toLocaleString() },
        { label: 'Input Shape',      value: modelInfo.input_shape?.join(' × ') },
        { label: 'Mask Threshold',   value: modelInfo.mask_threshold },
        { label: 'Backend',          value: serverStatus === 'online' ? 'Online' : 'Offline' },
      ]
    : [
        { label: 'Architecture',     value: 'U-Net' },
        { label: 'Trainable Params', value: '~31,000,000' },
        { label: 'Input Shape',      value: '3 × 256 × 256' },
        { label: 'Mask Threshold',   value: '0.5' },
        { label: 'Backend',          value: serverStatus },
      ];

  return (
    <div style={{
      display: 'grid',
      gridTemplateColumns: `repeat(${stats.length}, 1fr)`,
      gap: 12, marginBottom: 4,
    }}>
      {stats.map(({ label, value }) => (
        <div key={label} style={{
          padding: '16px 14px',
          borderRadius: 'var(--radius-md)',
          background: 'var(--bg-card)',
          border: '1px solid var(--border)',
          textAlign: 'center',
        }}>
          <div style={{
            fontFamily: 'var(--font-heading)',
            fontSize: 16, fontWeight: 800,
            color: value === 'Online' ? 'var(--teal)'
                 : value === 'Offline' ? 'var(--rose)'
                 : 'var(--accent)',
            letterSpacing: '-0.02em', marginBottom: 5,
          }}>
            {value ?? '—'}
          </div>
          <div style={{
            fontFamily: 'var(--font-mono)', fontSize: 10,
            color: 'var(--text-muted)', textTransform: 'uppercase',
            letterSpacing: '0.07em',
          }}>{label}</div>
        </div>
      ))}
    </div>
  );
}

// ── Architecture card ────────────────────────────────────────
function ArchCard() {
  const layers = [
    { name: 'Input',           detail: '3 × 256 × 256',  side: 'encoder', filters: null },
    { name: 'Encoder Block 1', detail: '64 filters',      side: 'encoder', filters: 64 },
    { name: 'Encoder Block 2', detail: '128 filters',     side: 'encoder', filters: 128 },
    { name: 'Encoder Block 3', detail: '256 filters',     side: 'encoder', filters: 256 },
    { name: 'Encoder Block 4', detail: '512 filters',     side: 'encoder', filters: 512 },
    { name: 'Bottleneck',      detail: '1024 + Dropout',  side: 'bottle',  filters: 1024 },
    { name: 'Decoder Block 4', detail: '512 + skip',      side: 'decoder', filters: 512 },
    { name: 'Decoder Block 3', detail: '256 + skip',      side: 'decoder', filters: 256 },
    { name: 'Decoder Block 2', detail: '128 + skip',      side: 'decoder', filters: 128 },
    { name: 'Decoder Block 1', detail: '64 + skip',       side: 'decoder', filters: 64 },
    { name: 'Output Head',     detail: '1 × 256 × 256',   side: 'decoder', filters: null },
  ];

  const barColor = (side) =>
    side === 'encoder' ? '#38bdf8'
    : side === 'bottle' ? '#f59e0b'
    : '#2dd4bf';

  return (
    <InfoCard title="U-Net Architecture" icon={<ArchIcon />} color="var(--accent)">
      <div style={{ display: 'flex', flexDirection: 'column', gap: 3 }}>
        {layers.map(({ name, detail, side, filters }) => (
          <div key={name} style={{
            display: 'flex', alignItems: 'center', gap: 8,
          }}>
            {/* Bar proportional to filters */}
            <div style={{
              width: filters ? Math.max(12, (filters / 1024) * 56) : 8,
              height: 6, borderRadius: 3, flexShrink: 0,
              background: barColor(side),
              opacity: side === 'bottle' ? 1 : 0.75,
            }} />
            <div style={{
              fontFamily: 'var(--font-mono)', fontSize: 11,
              color: side === 'bottle' ? '#f59e0b'
                   : side === 'encoder' ? 'var(--text-secondary)'
                   : 'var(--teal)',
              flex: 1,
            }}>
              {name}
            </div>
            <div style={{
              fontFamily: 'var(--font-mono)', fontSize: 10,
              color: 'var(--text-muted)',
            }}>
              {detail}
            </div>
          </div>
        ))}
      </div>
      <div style={{
        display: 'flex', gap: 12, marginTop: 14,
        paddingTop: 12, borderTop: '1px solid var(--border)',
      }}>
        {[['Encoder', '#38bdf8'], ['Bottleneck', '#f59e0b'], ['Decoder', '#2dd4bf']].map(
          ([label, color]) => (
            <div key={label} style={{ display: 'flex', alignItems: 'center', gap: 5 }}>
              <div style={{ width: 8, height: 8, borderRadius: 2, background: color }} />
              <span style={{
                fontFamily: 'var(--font-mono)', fontSize: 10,
                color: 'var(--text-muted)',
              }}>{label}</span>
            </div>
          )
        )}
      </div>
    </InfoCard>
  );
}

// ── Dataset card ─────────────────────────────────────────────
function DatasetCard() {
  const rows = [
    ['Source',        'LGG MRI Segmentation (Kaggle)'],
    ['Images',        '3,929 paired slices'],
    ['Patients',      '110 (TCGA lower-grade glioma)'],
    ['Format',        'TIFF — 3-channel RGB'],
    ['Positive',      '~35% slices contain tumour'],
    ['Train / Val / Test', '80% / 10% / 10%'],
    ['Resize',        '256 × 256 px'],
    ['Normalise',     'ImageNet mean & std'],
    ['Augmentation',  '8 transforms (train only)'],
  ];

  return (
    <InfoCard title="LGG MRI Dataset" icon={<DataIcon />} color="var(--teal)">
      <div style={{ display: 'flex', flexDirection: 'column', gap: 0 }}>
        {rows.map(([key, val], i) => (
          <div key={key} style={{
            display: 'flex', justifyContent: 'space-between',
            alignItems: 'center',
            padding: '7px 0',
            borderBottom: i < rows.length - 1 ? '1px solid var(--border-subtle)' : 'none',
          }}>
            <span style={{
              fontFamily: 'var(--font-mono)', fontSize: 11,
              color: 'var(--text-muted)',
            }}>{key}</span>
            <span style={{
              fontFamily: 'var(--font-mono)', fontSize: 11,
              color: 'var(--text-secondary)',
              textAlign: 'right', maxWidth: '55%',
            }}>{val}</span>
          </div>
        ))}
      </div>

      <a
        href="https://www.kaggle.com/datasets/mateuszbuda/lgg-mri-segmentation"
        target="_blank" rel="noreferrer"
        style={{
          display: 'inline-flex', alignItems: 'center', gap: 6,
          marginTop: 14, padding: '7px 14px',
          borderRadius: 8,
          border: '1px solid rgba(45,212,191,0.25)',
          background: 'rgba(45,212,191,0.07)',
          color: 'var(--teal)',
          fontFamily: 'var(--font-mono)', fontSize: 11.5,
          textDecoration: 'none', transition: 'all 0.2s',
        }}
      >
        <KaggleIcon /> View on Kaggle ↗
      </a>
    </InfoCard>
  );
}

// ── Pipeline diagram ─────────────────────────────────────────
function PipelineDiagram() {
  const steps = [
    { label: 'MRI Upload',  sub: 'TIFF / PNG / JPG',   color: '#38bdf8', icon: '⬆' },
    { label: 'Preprocess',  sub: 'Resize + Normalise',  color: '#60a5fa', icon: '⚙' },
    { label: 'U-Net',       sub: 'Encoder → Decoder',   color: '#f59e0b', icon: '🧠' },
    { label: 'Sigmoid',     sub: 'Prob. map 0–1',        color: '#a78bfa', icon: '∿' },
    { label: 'Threshold',   sub: 'Binary mask @ 0.5',    color: '#818cf8', icon: '⊘' },
    { label: 'Overlay',     sub: 'Tumour highlighted',   color: '#2dd4bf', icon: '🎯' },
  ];

  return (
    <div className="card" style={{ padding: '24px 20px', marginTop: 20 }}>
      <SectionHead title="Inference Pipeline" color="var(--violet)" icon={<PipeIcon />} />
      <div style={{
        display: 'flex', alignItems: 'center',
        gap: 4, marginTop: 18, flexWrap: 'wrap',
      }}>
        {steps.map(({ label, sub, color, icon }, i) => (
          <div key={label} style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
            <div style={{
              padding: '10px 14px',
              borderRadius: 10,
              background: `${color}10`,
              border: `1px solid ${color}30`,
              textAlign: 'center', minWidth: 100,
            }}>
              <div style={{ fontSize: 18, marginBottom: 4 }}>{icon}</div>
              <div style={{
                fontFamily: 'var(--font-mono)', fontSize: 11,
                color: color, fontWeight: 600,
              }}>{label}</div>
              <div style={{
                fontFamily: 'var(--font-mono)', fontSize: 9.5,
                color: 'var(--text-muted)', marginTop: 2,
              }}>{sub}</div>
            </div>
            {i < steps.length - 1 && (
              <div style={{
                fontFamily: 'var(--font-mono)', fontSize: 14,
                color: 'var(--text-muted)',
              }}>→</div>
            )}
          </div>
        ))}
      </div>
    </div>
  );
}

// ── Training card ────────────────────────────────────────────
function TrainingCard() {
  const rows = [
    ['Epochs',          '50 (early stop @ 47)'],
    ['Batch Size',      '16'],
    ['Learning Rate',   '1e-4 (Adam)'],
    ['LR Scheduler',    'ReduceLROnPlateau ×0.5'],
    ['Loss',            '0.5×BCE + 0.5×Dice'],
    ['Dropout',         '0.3 (bottleneck)'],
    ['Grad Clipping',   'max_norm = 1.0'],
    ['Platform',        'Kaggle — NVIDIA T4 GPU'],
    ['Train Time',      '~35 minutes'],
  ];

  return (
    <InfoCard title="Training Config" icon={<TrainIcon />} color="var(--violet)">
      <div style={{ display: 'flex', flexDirection: 'column', gap: 0 }}>
        {rows.map(([key, val], i) => (
          <div key={key} style={{
            display: 'flex', justifyContent: 'space-between',
            padding: '6px 0',
            borderBottom: i < rows.length - 1 ? '1px solid var(--border-subtle)' : 'none',
          }}>
            <span style={{
              fontFamily: 'var(--font-mono)', fontSize: 11,
              color: 'var(--text-muted)',
            }}>{key}</span>
            <span style={{
              fontFamily: 'var(--font-mono)', fontSize: 11,
              color: 'var(--text-secondary)',
            }}>{val}</span>
          </div>
        ))}
      </div>
    </InfoCard>
  );
}

// ── Benchmark results card ────────────────────────────────────
function BenchmarkCard() {
  const metrics = [
    { label: 'Dice Coefficient', value: 0.8756, target: 0.80,  color: '#38bdf8' },
    { label: 'IoU / Jaccard',    value: 0.7821, target: 0.70,  color: '#2dd4bf' },
    { label: 'Pixel Accuracy',   value: 0.9731, target: 0.90,  color: '#818cf8' },
    { label: 'Precision',        value: 0.8612, target: null,  color: '#a78bfa' },
    { label: 'Recall',           value: 0.8901, target: null,  color: '#f59e0b' },
  ];

  return (
    <InfoCard title="Test Set Results" icon={<ChartIcon />} color="#38bdf8">
      <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
        {metrics.map(({ label, value, target, color }) => (
          <div key={label}>
            <div style={{
              display: 'flex', justifyContent: 'space-between',
              marginBottom: 5,
            }}>
              <span style={{
                fontFamily: 'var(--font-mono)', fontSize: 11,
                color: 'var(--text-secondary)',
              }}>{label}</span>
              <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                {target && (
                  <span style={{
                    fontFamily: 'var(--font-mono)', fontSize: 9.5,
                    color: value >= target ? 'var(--teal)' : 'var(--rose)',
                  }}>
                    {value >= target ? '✓' : '✗'} ≥{target}
                  </span>
                )}
                <span style={{
                  fontFamily: 'var(--font-heading)',
                  fontSize: 14, fontWeight: 800,
                  color: color, letterSpacing: '-0.02em',
                }}>{value.toFixed(4)}</span>
              </div>
            </div>
            {/* Progress bar */}
            <div style={{
              height: 4, borderRadius: 100,
              background: 'var(--bg-elevated)', overflow: 'hidden',
            }}>
              <div style={{
                height: '100%', borderRadius: 100,
                width: `${value * 100}%`,
                background: `linear-gradient(90deg, ${color}60, ${color})`,
              }} />
            </div>
            {/* Target marker */}
            {target && (
              <div style={{
                position: 'relative', height: 0,
              }}>
                <div style={{
                  position: 'absolute',
                  left: `${target * 100}%`,
                  top: -8, width: 1, height: 8,
                  background: 'rgba(255,255,255,0.2)',
                  transform: 'translateX(-50%)',
                }} />
              </div>
            )}
          </div>
        ))}
      </div>
      <div style={{
        marginTop: 14, padding: '10px 12px',
        borderRadius: 8,
        background: 'rgba(56,189,248,0.06)',
        border: '1px solid var(--accent-border)',
        fontFamily: 'var(--font-mono)', fontSize: 10.5,
        color: 'var(--text-muted)', lineHeight: 1.6,
      }}>
        Buda et al. 2019 ensemble: Dice ≈ 0.918<br/>
        Our single-model result: <span style={{ color: 'var(--accent)' }}>Dice = 0.8756</span>
      </div>
    </InfoCard>
  );
}

// ── Paper reference card ──────────────────────────────────────
function PaperCard() {
  return (
    <div className="card" style={{
      padding: '24px 28px', marginTop: 20,
      background: 'linear-gradient(135deg, rgba(56,189,248,0.04), rgba(45,212,191,0.03))',
      border: '1px solid var(--accent-border)',
    }}>
      <div style={{ display: 'flex', gap: 20, alignItems: 'flex-start',
                    flexWrap: 'wrap' }}>
        <div style={{
          width: 48, height: 48, borderRadius: 12, flexShrink: 0,
          background: 'var(--accent-glow)',
          border: '1px solid var(--accent-border)',
          display: 'flex', alignItems: 'center', justifyContent: 'center',
          fontSize: 22,
        }}>📄</div>
        <div style={{ flex: 1 }}>
          <div style={{
            fontFamily: 'var(--font-heading)',
            fontSize: 15, fontWeight: 700,
            color: 'var(--text-primary)', marginBottom: 6,
          }}>
            U-Net: Convolutional Networks for Biomedical Image Segmentation
          </div>
          <div style={{
            fontFamily: 'var(--font-mono)', fontSize: 11.5,
            color: 'var(--text-muted)', marginBottom: 10, lineHeight: 1.6,
          }}>
            Ronneberger, O., Fischer, P., &amp; Brox, T. · MICCAI 2015 ·
            LNCS 9351, pp. 234–241
          </div>
          <div style={{
            fontFamily: 'var(--font-body)', fontSize: 13,
            color: 'var(--text-secondary)', lineHeight: 1.65, maxWidth: 600,
          }}>
            The U-Net architecture introduces skip connections between the
            contracting (encoder) and expanding (decoder) paths, allowing
            the network to combine high-resolution spatial detail from early
            layers with deep semantic context from later layers — critical
            for precise biomedical segmentation.
          </div>
        </div>
      </div>
    </div>
  );
}

// ── Reusable card wrapper ─────────────────────────────────────
function InfoCard({ title, icon, color, children }) {
  return (
    <div className="card" style={{ padding: '22px 20px' }}>
      <SectionHead title={title} icon={icon} color={color} />
      <div style={{ marginTop: 16 }}>{children}</div>
    </div>
  );
}

function SectionHead({ title, icon, color }) {
  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: 9 }}>
      <div style={{
        width: 28, height: 28, borderRadius: 8,
        background: `${color}15`,
        border: `1px solid ${color}30`,
        display: 'flex', alignItems: 'center', justifyContent: 'center',
        flexShrink: 0,
      }}>{icon}</div>
      <span style={{
        fontFamily: 'var(--font-heading)',
        fontSize: 14, fontWeight: 700,
        color: 'var(--text-primary)',
      }}>{title}</span>
    </div>
  );
}

// ── Icons ────────────────────────────────────────────────────
const ico = (color, d) => (
  <svg width="13" height="13" viewBox="0 0 24 24" fill="none"
       stroke={color} strokeWidth="2" strokeLinecap="round">
    {d}
  </svg>
);

function ArchIcon()  { return ico('var(--accent)',  <><rect x="2" y="2" width="20" height="20" rx="2"/><path d="M7 7h10M7 12h10M7 17h10"/></>); }
function DataIcon()  { return ico('var(--teal)',    <><ellipse cx="12" cy="5" rx="9" ry="3"/><path d="M21 12c0 1.66-4 3-9 3s-9-1.34-9-3"/><path d="M3 5v14c0 1.66 4 3 9 3s9-1.34 9-3V5"/></>); }
function TrainIcon() { return ico('var(--violet)',  <><polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"/></>); }
function ChartIcon() { return ico('#38bdf8',        <><line x1="18" y1="20" x2="18" y2="10"/><line x1="12" y1="20" x2="12" y2="4"/><line x1="6" y1="20" x2="6" y2="14"/></>); }
function PipeIcon()  { return ico('var(--violet)',  <><polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/></>); }
function KaggleIcon() {
  return (
    <svg width="12" height="12" viewBox="0 0 24 24" fill="currentColor">
      <path d="M18.825 23.859c-.022.092-.117.141-.281.141h-3.139c-.187 0-.351-.082-.492-.248l-5.178-6.589-1.448 1.374v5.111c0 .235-.117.352-.351.352H5.505c-.236 0-.354-.117-.354-.352V.353c0-.233.118-.353.354-.353h2.431c.234 0 .351.12.351.353v14.343l6.203-6.272c.165-.165.33-.246.495-.246h3.239c.144 0 .236.06.285.18.046.149.034.255-.036.315l-6.555 6.344 6.836 8.507c.095.104.117.208.07.334"/>
    </svg>
  );
}