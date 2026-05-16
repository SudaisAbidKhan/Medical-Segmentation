// ═══════════════════════════════════════════════════════════
//  Navbar.jsx  –  Top navigation bar (standalone component)
//  Used by App.jsx — receives serverStatus as a prop
// ═══════════════════════════════════════════════════════════

import { useState, useEffect } from 'react';
import { NavLink } from 'react-router-dom';

export default function Navbar({ serverStatus }) {
  const [scrolled, setScrolled] = useState(false);

  useEffect(() => {
    const onScroll = () => setScrolled(window.scrollY > 20);
    window.addEventListener('scroll', onScroll, { passive: true });
    return () => window.removeEventListener('scroll', onScroll);
  }, []);

  return (
    <header style={{
      position: 'sticky', top: 0, zIndex: 100,
      background: scrolled ? 'rgba(4,8,15,0.92)' : 'transparent',
      backdropFilter: scrolled ? 'blur(24px)' : 'none',
      WebkitBackdropFilter: scrolled ? 'blur(24px)' : 'none',
      borderBottom: scrolled
        ? '1px solid rgba(56,189,248,0.10)'
        : '1px solid transparent',
      transition: 'all 0.3s ease',
      padding: '0 2rem',
    }}>
      <nav style={{
        maxWidth: 1200, margin: '0 auto',
        height: 64,
        display: 'flex', alignItems: 'center',
        justifyContent: 'space-between',
      }}>

        {/* ── Logo ──────────────────────────────────────────── */}
        <NavLink to="/" style={{ textDecoration: 'none',
          display: 'flex', alignItems: 'center', gap: 11 }}>
          <div style={{
            width: 36, height: 36,
            background: 'linear-gradient(135deg, #38bdf8, #2dd4bf)',
            borderRadius: 10,
            display: 'flex', alignItems: 'center', justifyContent: 'center',
            boxShadow: '0 0 20px rgba(56,189,248,0.30)',
            flexShrink: 0,
          }}>
            <BrainSvg />
          </div>
          <div>
            <div style={{
              fontFamily: 'var(--font-heading)',
              fontWeight: 800, fontSize: 17,
              letterSpacing: '-0.02em',
              color: 'var(--text-primary)', lineHeight: 1.1,
            }}>NeuroSeg</div>
            <div style={{
              fontFamily: 'var(--font-mono)', fontSize: 9.5,
              color: 'var(--text-muted)',
              letterSpacing: '0.08em', textTransform: 'uppercase',
            }}>U-Net Segmentation</div>
          </div>
        </NavLink>

        {/* ── Nav links ─────────────────────────────────────── */}
        <div style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
          {[
            { to: '/',      label: 'Segment',    end: true,  icon: <GridIcon /> },
            { to: '/about', label: 'About Model', end: false, icon: <InfoIcon /> },
          ].map(({ to, label, end, icon }) => (
            <NavLink key={to} to={to} end={end} style={({ isActive }) => ({
              display: 'flex', alignItems: 'center', gap: 7,
              padding: '7px 16px', borderRadius: 'var(--radius-md)',
              fontFamily: 'var(--font-body)', fontSize: 13.5, fontWeight: 500,
              color: isActive ? 'var(--accent)' : 'var(--text-secondary)',
              background: isActive ? 'var(--accent-glow)' : 'transparent',
              border: `1px solid ${isActive ? 'var(--accent-border)' : 'transparent'}`,
              transition: 'all var(--transition)', textDecoration: 'none',
            })}>
              {icon}{label}
            </NavLink>
          ))}
        </div>

        {/* ── Server status badge ───────────────────────────── */}
        <StatusBadge status={serverStatus} />
      </nav>
    </header>
  );
}

function StatusBadge({ status }) {
  const map = {
    online:   { label: 'API Online',    cls: 'badge-online',  dot: '#2dd4bf' },
    offline:  { label: 'API Offline',   cls: 'badge-offline', dot: '#fb7185' },
    checking: { label: 'Connecting…',   cls: 'badge-loading', dot: '#38bdf8' },
  };
  const { label, cls, dot } = map[status] || map.checking;
  return (
    <span className={`badge ${cls}`}>
      <span style={{
        width: 6, height: 6, borderRadius: '50%',
        background: dot, display: 'inline-block', flexShrink: 0,
        animation: status === 'checking' ? 'pulse-ring 1.4s ease-in-out infinite' : 'none',
      }} />
      {label}
    </span>
  );
}

function BrainSvg() {
  return (
    <svg width="20" height="20" viewBox="0 0 24 24" fill="none"
         stroke="#04080f" strokeWidth="2" strokeLinecap="round">
      <path d="M12 5a3 3 0 1 0-5.995.142M12 5a3 3 0 1 1 5.995.142"/>
      <path d="M6.005 5.142A3 3 0 0 0 3 8a3 3 0 0 0 2.25 2.906"/>
      <path d="M17.995 5.142A3 3 0 0 1 21 8a3 3 0 0 1-2.25 2.906"/>
      <path d="M5.25 10.906A3 3 0 0 0 3 14a3 3 0 0 0 3 3"/>
      <path d="M18.75 10.906A3 3 0 0 1 21 14a3 3 0 0 1-3 3"/>
      <path d="M6 17a3 3 0 0 0 6 0M18 17a3 3 0 0 1-6 0"/>
    </svg>
  );
}

function GridIcon() {
  return (
    <svg width="13" height="13" viewBox="0 0 24 24" fill="none"
         stroke="currentColor" strokeWidth="2" strokeLinecap="round">
      <rect x="3" y="3" width="7" height="7"/><rect x="14" y="3" width="7" height="7"/>
      <rect x="3" y="14" width="7" height="7"/><rect x="14" y="14" width="7" height="7"/>
    </svg>
  );
}
function InfoIcon() {
  return (
    <svg width="13" height="13" viewBox="0 0 24 24" fill="none"
         stroke="currentColor" strokeWidth="2" strokeLinecap="round">
      <circle cx="12" cy="12" r="10"/>
      <path d="M12 16v-4M12 8h.01"/>
    </svg>
  );
}