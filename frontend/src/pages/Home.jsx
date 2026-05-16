// ═══════════════════════════════════════════════════════════
//  Home.jsx  –  Main segmentation page
//
//  Layout:
//    Left col  → Upload panel + Run button + status
//    Right col → ResultViewer + MetricsCard
//
//  Modes:
//    idle      → show upload UI
//    loading   → show Loader spinner
//    result    → show ResultViewer + MetricsCard
//    error     → show error message
// ═══════════════════════════════════════════════════════════

import { useState, useCallback } from 'react';
import ImageUploader            from '../components/ImageUploader.jsx';
import ResultViewer             from '../components/ResultViewer.jsx';
import MetricsCard              from '../components/MetricsCard.jsx';
import Loader, { SpinnerInline } from '../components/Loader.jsx';
import { predictImage, predictWithMask } from '../services/api.js';

export default function Home({ serverStatus }) {
  const [imageFile,  setImageFile]  = useState(null);
  const [maskFile,   setMaskFile]   = useState(null);
  const [mode,       setMode]       = useState('idle');   // idle|loading|result|error
  const [result,     setResult]     = useState(null);
  const [errorMsg,   setErrorMsg]   = useState('');

  const isDisabled = serverStatus === 'offline' || mode === 'loading';
  const canRun     = !!imageFile && !isDisabled;

  // ── Run segmentation ───────────────────────────────────
  const handleRun = useCallback(async () => {
    if (!imageFile) return;
    setMode('loading');
    setResult(null);
    setErrorMsg('');

    try {
      const data = maskFile
        ? await predictWithMask(imageFile, maskFile)
        : await predictImage(imageFile);

      setResult(data);
      setMode('result');
    } catch (err) {
      setErrorMsg(err.message || 'Prediction failed. Is the Flask server running?');
      setMode('error');
    }
  }, [imageFile, maskFile]);

  const handleReset = () => {
    setMode('idle');
    setResult(null);
    setErrorMsg('');
    setImageFile(null);
    setMaskFile(null);
  };

  return (
    <div style={{ maxWidth: 1200, margin: '0 auto', padding: '40px 2rem 64px' }}>

      {/* ── Hero header ────────────────────────────────── */}
      <div className="fade-in" style={{ marginBottom: 40 }}>
        <div style={{
          display: 'inline-flex', alignItems: 'center', gap: 8,
          padding: '4px 12px', borderRadius: 100,
          background: 'var(--accent-glow)',
          border: '1px solid var(--accent-border)',
          marginBottom: 16,
        }}>
          <PulsingDot />
          <span style={{
            fontFamily: 'var(--font-mono)', fontSize: 11,
            color: 'var(--accent)', letterSpacing: '0.06em',
            textTransform: 'uppercase',
          }}>
            Deep Learning · Medical Imaging
          </span>
        </div>

        <h1 style={{
          fontFamily: 'var(--font-heading)',
          fontSize: 'clamp(28px, 4vw, 44px)',
          fontWeight: 800,
          letterSpacing: '-0.03em',
          lineHeight: 1.1,
          color: 'var(--text-primary)',
          marginBottom: 14,
        }}>
          Brain MRI{' '}
          <span style={{
            background: 'linear-gradient(135deg, #38bdf8, #2dd4bf)',
            WebkitBackgroundClip: 'text',
            WebkitTextFillColor: 'transparent',
          }}>
            Tumor Segmentation
          </span>
        </h1>

        <p style={{
          fontFamily: 'var(--font-body)', fontSize: 15.5,
          color: 'var(--text-secondary)', maxWidth: 560,
          lineHeight: 1.7,
        }}>
          Upload an MRI scan and the U-Net model will produce a
          pixel-wise segmentation mask, probability heatmap, and
          tumour coverage estimate in seconds.
        </p>
      </div>

      {/* ── Two-column layout ──────────────────────────── */}
      <div style={{
        display: 'grid',
        gridTemplateColumns: mode === 'result' ? '420px 1fr' : '480px 1fr',
        gap: 24,
        alignItems: 'start',
        transition: 'grid-template-columns 0.3s ease',
      }}>

        {/* ════════════════════════════════════════════════
            LEFT  –  Upload panel
        ════════════════════════════════════════════════ */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>

          {/* Upload card */}
          <div className="card" style={{ padding: 20 }}>
            <SectionLabel icon={<UploadIcon />} text="Input MRI Scan" />
            <div style={{ marginTop: 14 }}>
              <ImageUploader
                onImageSelect={setImageFile}
                onMaskSelect={setMaskFile}
                disabled={isDisabled}
              />
            </div>
          </div>

          {/* Run button */}
          <button
            onClick={handleRun}
            disabled={!canRun}
            style={{
              width: '100%', padding: '14px',
              borderRadius: 'var(--radius-md)',
              border: canRun
                ? '1px solid var(--accent-border)'
                : '1px solid var(--border)',
              background: canRun
                ? 'linear-gradient(135deg, rgba(56,189,248,0.15), rgba(45,212,191,0.10))'
                : 'var(--bg-elevated)',
              color: canRun ? 'var(--accent)' : 'var(--text-muted)',
              fontFamily: 'var(--font-heading)',
              fontSize: 15, fontWeight: 700,
              cursor: canRun ? 'pointer' : 'not-allowed',
              transition: 'all 0.2s ease',
              display: 'flex', alignItems: 'center',
              justifyContent: 'center', gap: 10,
              letterSpacing: '-0.01em',
              boxShadow: canRun ? '0 0 24px rgba(56,189,248,0.12)' : 'none',
            }}
            onMouseEnter={e => canRun && (
              e.currentTarget.style.boxShadow = '0 0 40px rgba(56,189,248,0.25)',
              e.currentTarget.style.transform = 'translateY(-1px)'
            )}
            onMouseLeave={e => (
              e.currentTarget.style.boxShadow = canRun ? '0 0 24px rgba(56,189,248,0.12)' : 'none',
              e.currentTarget.style.transform = 'translateY(0)'
            )}
          >
            {mode === 'loading'
              ? <><SpinnerInline size={17} color="var(--accent)" /> Segmenting…</>
              : <><SegmentIcon /> Run Segmentation</>
            }
          </button>

          {/* Server offline notice */}
          {serverStatus === 'offline' && (
            <div style={{
              padding: '12px 14px', borderRadius: 'var(--radius-md)',
              background: 'rgba(251,113,133,0.07)',
              border: '1px solid rgba(251,113,133,0.18)',
              fontFamily: 'var(--font-mono)', fontSize: 11.5,
              color: 'var(--rose)', lineHeight: 1.6,
            }}>
              Backend offline. Start it with:<br />
              <code style={{
                display: 'inline-block', marginTop: 4,
                background: 'rgba(251,113,133,0.10)',
                padding: '2px 8px', borderRadius: 4, fontSize: 11,
              }}>
                cd backend && python app.py
              </code>
            </div>
          )}

          {/* Reset button when showing results */}
          {mode === 'result' && (
            <button onClick={handleReset} style={{
              width: '100%', padding: '10px',
              borderRadius: 'var(--radius-md)',
              border: '1px solid var(--border)',
              background: 'transparent',
              color: 'var(--text-muted)',
              fontFamily: 'var(--font-body)', fontSize: 13,
              cursor: 'pointer', transition: 'all 0.2s',
            }}>
              ↩ New Scan
            </button>
          )}

          {/* Info chip strip */}
          <div style={{
            display: 'flex', gap: 8, flexWrap: 'wrap',
          }}>
            {[
              'TIF · PNG · JPG',
              'Max 16 MB',
              'U-Net 256×256',
            ].map(t => (
              <span key={t} style={{
                padding: '3px 10px', borderRadius: 100,
                border: '1px solid var(--border)',
                fontFamily: 'var(--font-mono)', fontSize: 10.5,
                color: 'var(--text-muted)',
              }}>{t}</span>
            ))}
          </div>
        </div>

        {/* ════════════════════════════════════════════════
            RIGHT  –  Result / idle state
        ════════════════════════════════════════════════ */}
        <div>
          {mode === 'idle' && (
            <IdlePlaceholder />
          )}

          {mode === 'loading' && (
            <div className="card" style={{ padding: 0 }}>
              <Loader />
            </div>
          )}

          {mode === 'error' && (
            <ErrorPanel message={errorMsg} onRetry={handleRun} />
          )}

          {mode === 'result' && result && (
            <div className="slide-up" style={{
              display: 'flex', flexDirection: 'column', gap: 16,
            }}>
              {/* Result images */}
              <div className="card" style={{ padding: 18 }}>
                <SectionLabel icon={<ResultIcon />} text="Segmentation Result" />
                <div style={{ marginTop: 14 }}>
                  <ResultViewer result={result} />
                </div>
              </div>

              {/* Metrics */}
              <div className="card" style={{ padding: 18 }}>
                <MetricsCard
                  metrics={result.metrics || null}
                  tumorCoverage={result.tumor_coverage}
                  inferenceMs={result.inference_ms}
                />
              </div>
            </div>
          )}
        </div>
      </div>

      {/* ── How it works strip ─────────────────────────── */}
      {mode === 'idle' && (
        <HowItWorks />
      )}
    </div>
  );
}

// ── Idle placeholder ─────────────────────────────────────────
function IdlePlaceholder() {
  return (
    <div className="card" style={{
      minHeight: 360,
      display: 'flex', flexDirection: 'column',
      alignItems: 'center', justifyContent: 'center',
      padding: '48px 32px', gap: 16, textAlign: 'center',
    }}>
      {/* Animated scan preview */}
      <div style={{ position: 'relative', marginBottom: 8 }}>
        <div style={{
          width: 120, height: 120, borderRadius: 'var(--radius-xl)',
          background: 'var(--bg-elevated)',
          border: '1px solid var(--border)',
          display: 'flex', alignItems: 'center', justifyContent: 'center',
          position: 'relative', overflow: 'hidden',
        }}>
          {/* MRI grid lines */}
          {[30, 60, 90].map(y => (
            <div key={y} style={{
              position: 'absolute', left: 0, right: 0,
              top: y, height: 1,
              background: 'rgba(56,189,248,0.08)',
            }} />
          ))}
          {[30, 60, 90].map(x => (
            <div key={x} style={{
              position: 'absolute', top: 0, bottom: 0,
              left: x, width: 1,
              background: 'rgba(56,189,248,0.08)',
            }} />
          ))}
          <MriIcon />
          {/* Scanning line animation */}
          <div style={{
            position: 'absolute', left: 0, right: 0,
            height: 2,
            background: 'linear-gradient(90deg, transparent, var(--accent), transparent)',
            animation: 'scan-line 2.5s ease-in-out infinite',
          }} />
        </div>
        {/* Orbit ring */}
        <div style={{
          position: 'absolute', inset: -12,
          borderRadius: '50%',
          border: '1px dashed var(--accent-border)',
          animation: 'spin 12s linear infinite',
        }} />
        {/* Dot on orbit */}
        <div style={{
          position: 'absolute', top: -6, left: '50%',
          width: 8, height: 8, borderRadius: '50%',
          background: 'var(--accent)',
          transform: 'translateX(-50%)',
          boxShadow: '0 0 10px var(--accent)',
        }} />
      </div>

      <div style={{
        fontFamily: 'var(--font-heading)',
        fontSize: 18, fontWeight: 700,
        color: 'var(--text-secondary)',
      }}>
        Awaiting MRI Scan
      </div>
      <div style={{
        fontFamily: 'var(--font-mono)', fontSize: 12,
        color: 'var(--text-muted)', maxWidth: 280, lineHeight: 1.7,
      }}>
        Upload a brain MRI image on the left to run U-Net segmentation
      </div>

      {/* Supported format tags */}
      <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap', justifyContent: 'center' }}>
        {['TIFF', 'PNG', 'JPG'].map(f => (
          <span key={f} style={{
            padding: '3px 10px', borderRadius: 100,
            border: '1px solid var(--border)',
            fontFamily: 'var(--font-mono)', fontSize: 10.5,
            color: 'var(--text-muted)',
          }}>{f}</span>
        ))}
      </div>
    </div>
  );
}

// ── Error panel ──────────────────────────────────────────────
function ErrorPanel({ message, onRetry }) {
  return (
    <div className="card slide-up" style={{
      padding: 28, textAlign: 'center',
      border: '1px solid rgba(251,113,133,0.2)',
      background: 'rgba(251,113,133,0.05)',
    }}>
      <div style={{ fontSize: 36, marginBottom: 14 }}>⚠️</div>
      <div style={{
        fontFamily: 'var(--font-heading)',
        fontSize: 17, fontWeight: 700,
        color: 'var(--rose)', marginBottom: 8,
      }}>Prediction Failed</div>
      <div style={{
        fontFamily: 'var(--font-mono)', fontSize: 12,
        color: 'var(--text-muted)', marginBottom: 20,
        lineHeight: 1.6,
      }}>
        {message}
      </div>
      <button onClick={onRetry} style={{
        padding: '9px 24px', borderRadius: 'var(--radius-md)',
        border: '1px solid rgba(251,113,133,0.3)',
        background: 'rgba(251,113,133,0.10)',
        color: 'var(--rose)',
        fontFamily: 'var(--font-body)', fontSize: 13, fontWeight: 500,
        cursor: 'pointer',
      }}>
        Retry
      </button>
    </div>
  );
}

// ── How it works strip ───────────────────────────────────────
function HowItWorks() {
  const steps = [
    { n: '01', icon: <UploadIcon />, title: 'Upload MRI',
      desc: 'Drop a brain MRI TIFF, PNG, or JPG from the LGG dataset or your own scans.' },
    { n: '02', icon: <NeuralIcon />, title: 'U-Net Inference',
      desc: 'The 31M-parameter U-Net runs a full encoder-decoder forward pass on the server.' },
    { n: '03', icon: <MaskIcon />,  title: 'View Results',
      desc: 'See the binary mask, red overlay, probability heatmap, and Dice / IoU metrics.' },
  ];

  return (
    <div style={{ marginTop: 64 }}>
      <div style={{ textAlign: 'center', marginBottom: 32 }}>
        <div style={{
          fontFamily: 'var(--font-mono)', fontSize: 10.5,
          color: 'var(--text-muted)', letterSpacing: '0.1em',
          textTransform: 'uppercase', marginBottom: 10,
        }}>
          How it works
        </div>
        <div style={{
          fontFamily: 'var(--font-heading)',
          fontSize: 22, fontWeight: 700,
          color: 'var(--text-primary)',
        }}>
          Three steps to segmentation
        </div>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 16 }}>
        {steps.map(({ n, icon, title, desc }) => (
          <div key={n} className="card" style={{ padding: '24px 20px' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 14 }}>
              <div style={{
                fontFamily: 'var(--font-mono)', fontSize: 10,
                color: 'var(--text-muted)',
                letterSpacing: '0.06em',
              }}>{n}</div>
              <div style={{
                width: 32, height: 32, borderRadius: 9,
                background: 'var(--accent-glow)',
                border: '1px solid var(--accent-border)',
                display: 'flex', alignItems: 'center', justifyContent: 'center',
              }}>
                {icon}
              </div>
            </div>
            <div style={{
              fontFamily: 'var(--font-heading)',
              fontSize: 15, fontWeight: 700,
              color: 'var(--text-primary)', marginBottom: 8,
            }}>{title}</div>
            <div style={{
              fontFamily: 'var(--font-body)', fontSize: 13,
              color: 'var(--text-secondary)', lineHeight: 1.65,
            }}>{desc}</div>
          </div>
        ))}
      </div>
    </div>
  );
}

// ── Section label ────────────────────────────────────────────
function SectionLabel({ icon, text }) {
  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
      <div style={{
        width: 26, height: 26, borderRadius: 7,
        background: 'var(--accent-glow)',
        border: '1px solid var(--accent-border)',
        display: 'flex', alignItems: 'center', justifyContent: 'center',
        flexShrink: 0,
      }}>{icon}</div>
      <span style={{
        fontFamily: 'var(--font-heading)',
        fontSize: 13.5, fontWeight: 700,
        color: 'var(--text-primary)',
      }}>{text}</span>
    </div>
  );
}

function PulsingDot() {
  return (
    <span style={{
      width: 7, height: 7, borderRadius: '50%',
      background: 'var(--accent)', display: 'inline-block',
      animation: 'pulse-ring 1.8s ease-in-out infinite',
    }} />
  );
}

// ── Icons ────────────────────────────────────────────────────
function UploadIcon() {
  return (
    <svg width="13" height="13" viewBox="0 0 24 24" fill="none"
         stroke="var(--accent)" strokeWidth="2" strokeLinecap="round">
      <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/>
      <polyline points="17 8 12 3 7 8"/><line x1="12" y1="3" x2="12" y2="15"/>
    </svg>
  );
}
function SegmentIcon() {
  return (
    <svg width="17" height="17" viewBox="0 0 24 24" fill="none"
         stroke="currentColor" strokeWidth="2" strokeLinecap="round">
      <polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"/>
    </svg>
  );
}
function ResultIcon() {
  return (
    <svg width="13" height="13" viewBox="0 0 24 24" fill="none"
         stroke="var(--accent)" strokeWidth="2" strokeLinecap="round">
      <rect x="3" y="3" width="18" height="18" rx="2"/>
      <path d="M3 9h18M9 21V9"/>
    </svg>
  );
}
function NeuralIcon() {
  return (
    <svg width="13" height="13" viewBox="0 0 24 24" fill="none"
         stroke="var(--accent)" strokeWidth="2" strokeLinecap="round">
      <circle cx="12" cy="5" r="2"/><circle cx="5" cy="19" r="2"/>
      <circle cx="19" cy="19" r="2"/>
      <line x1="12" y1="7" x2="5" y2="17"/>
      <line x1="12" y1="7" x2="19" y2="17"/>
      <line x1="5" y1="17" x2="19" y2="17"/>
    </svg>
  );
}
function MaskIcon() {
  return (
    <svg width="13" height="13" viewBox="0 0 24 24" fill="none"
         stroke="var(--accent)" strokeWidth="2" strokeLinecap="round">
      <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/>
    </svg>
  );
}
function MriIcon() {
  return (
    <svg width="42" height="42" viewBox="0 0 24 24" fill="none"
         stroke="var(--accent-dim)" strokeWidth="1.2" strokeLinecap="round">
      <ellipse cx="12" cy="12" rx="6" ry="8"/>
      <path d="M12 4C8 4 4 7 4 12s4 8 8 8"/>
      <path d="M12 4c4 0 8 3 8 8s-4 8-8 8"/>
      <circle cx="12" cy="12" r="2" fill="rgba(56,189,248,0.2)"
              stroke="var(--accent)"/>
    </svg>
  );
}