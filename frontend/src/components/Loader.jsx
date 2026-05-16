// ═══════════════════════════════════════════════════════════
//  Loader.jsx  –  Inference loading spinner with status text
// ═══════════════════════════════════════════════════════════

import { useState, useEffect } from 'react';

const MESSAGES = [
  'Preprocessing MRI scan…',
  'Running U-Net encoder…',
  'Applying skip connections…',
  'Decoding segmentation mask…',
  'Computing pixel probabilities…',
  'Generating overlay…',
  'Almost done…',
];

export default function Loader({ message = '' }) {
  const [msgIndex, setMsgIndex] = useState(0);

  // Cycle through messages every 1.8 s for perceived progress
  useEffect(() => {
    const id = setInterval(() => {
      setMsgIndex(i => (i + 1) % MESSAGES.length);
    }, 1800);
    return () => clearInterval(id);
  }, []);

  return (
    <div style={{
      display: 'flex',
      flexDirection: 'column',
      alignItems: 'center',
      justifyContent: 'center',
      gap: 28,
      padding: '48px 24px',
    }}>

      {/* ── Animated brain scan ring ─────────────────────── */}
      <div style={{ position: 'relative', width: 96, height: 96 }}>

        {/* Outer pulsing ring */}
        <div style={{
          position: 'absolute', inset: -8,
          borderRadius: '50%',
          border: '1px solid var(--accent-border)',
          animation: 'pulse-ring 2s ease-in-out infinite',
        }} />

        {/* Spinning arc */}
        <svg
          width="96" height="96"
          viewBox="0 0 96 96"
          style={{ animation: 'spin 1.1s linear infinite', position: 'relative' }}
        >
          <circle
            cx="48" cy="48" r="40"
            fill="none"
            stroke="var(--bg-elevated)"
            strokeWidth="4"
          />
          <circle
            cx="48" cy="48" r="40"
            fill="none"
            stroke="url(#spinGrad)"
            strokeWidth="4"
            strokeLinecap="round"
            strokeDasharray="200 52"
            strokeDashoffset="0"
          />
          <defs>
            <linearGradient id="spinGrad" x1="0%" y1="0%" x2="100%" y2="0%">
              <stop offset="0%"   stopColor="var(--accent)"     stopOpacity="0"/>
              <stop offset="100%" stopColor="var(--accent)"     stopOpacity="1"/>
            </linearGradient>
          </defs>
        </svg>

        {/* Centre brain icon */}
        <div style={{
          position: 'absolute', inset: 0,
          display: 'flex', alignItems: 'center', justifyContent: 'center',
        }}>
          <div style={{
            width: 44, height: 44,
            borderRadius: 12,
            background: 'linear-gradient(135deg, var(--accent), var(--teal))',
            display: 'flex', alignItems: 'center', justifyContent: 'center',
            boxShadow: '0 0 24px rgba(56,189,248,0.4)',
          }}>
            <NeuralIcon />
          </div>
        </div>
      </div>

      {/* ── Status text ──────────────────────────────────── */}
      <div style={{ textAlign: 'center' }}>
        <div style={{
          fontFamily: 'var(--font-heading)',
          fontSize: 17, fontWeight: 700,
          color: 'var(--text-primary)',
          marginBottom: 8,
        }}>
          Segmenting…
        </div>
        <div
          key={msgIndex}
          className="fade-in"
          style={{
            fontFamily: 'var(--font-mono)',
            fontSize: 12.5,
            color: 'var(--accent)',
            letterSpacing: '0.02em',
          }}
        >
          {message || MESSAGES[msgIndex]}
        </div>
      </div>

      {/* ── Animated progress bar ────────────────────────── */}
      <div style={{
        width: 220,
        height: 3,
        borderRadius: 100,
        background: 'var(--bg-elevated)',
        overflow: 'hidden',
      }}>
        <div style={{
          height: '100%',
          borderRadius: 100,
          background: 'linear-gradient(90deg, var(--accent), var(--teal))',
          animation: 'indeterminate 1.8s ease-in-out infinite',
        }} />
      </div>

      <style>{`
        @keyframes indeterminate {
          0%   { width: 0%;   margin-left: 0%; }
          50%  { width: 60%;  margin-left: 20%; }
          100% { width: 0%;   margin-left: 100%; }
        }
      `}</style>
    </div>
  );
}

// ── Compact inline spinner (used inside buttons) ─────────────
export function SpinnerInline({ size = 16, color = 'currentColor' }) {
  return (
    <svg
      width={size} height={size}
      viewBox="0 0 24 24"
      fill="none"
      style={{ animation: 'spin 0.75s linear infinite', flexShrink: 0 }}
    >
      <circle cx="12" cy="12" r="10"
              stroke={color} strokeWidth="3"
              strokeOpacity="0.2" />
      <path d="M12 2a10 10 0 0 1 10 10"
            stroke={color} strokeWidth="3" strokeLinecap="round" />
    </svg>
  );
}

function NeuralIcon() {
  return (
    <svg width="22" height="22" viewBox="0 0 24 24" fill="none"
         stroke="#04080f" strokeWidth="2" strokeLinecap="round">
      <circle cx="12" cy="5"  r="2"/>
      <circle cx="5"  cy="19" r="2"/>
      <circle cx="19" cy="19" r="2"/>
      <line x1="12" y1="7"  x2="5"  y2="17"/>
      <line x1="12" y1="7"  x2="19" y2="17"/>
      <line x1="5"  y1="17" x2="19" y2="17"/>
    </svg>
  );
}