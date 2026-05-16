// ═══════════════════════════════════════════════════════════
//  App.jsx  –  Root component with routing & global state
//
//  Routes:
//    /          → Home   (image upload + prediction)
//    /about     → About  (model info + architecture)
//
//  Global state managed here:
//    • serverStatus  – polling /health every 10s
//    • modelInfo     – fetched once from /model-info
// ═══════════════════════════════════════════════════════════

import { useState, useEffect, useCallback } from 'react';
import {
  BrowserRouter as Router,
  Routes,
  Route,
  NavLink,
  useLocation,
} from 'react-router-dom';

import Home   from './pages/Home.jsx';
import About  from './pages/About.jsx';
import Navbar from './components/Navbar.jsx';
import { checkHealth, getModelInfo } from './services/api.js';

// ── Animated route wrapper ───────────────────────────────────
function PageWrapper({ children }) {
  const location = useLocation();
  return (
    <div key={location.pathname} className="slide-up" style={{ flex: 1 }}>
      {children}
    </div>
  );
}

// ── Footer ───────────────────────────────────────────────────
function Footer() {
  return (
    <footer style={{
      borderTop: '1px solid var(--border-subtle)',
      padding: '24px 2rem',
      textAlign: 'center',
    }}>
      <div style={{
        maxWidth: 1200,
        margin: '0 auto',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        flexWrap: 'wrap',
        gap: 12,
      }}>
        <div style={{
          fontFamily: 'var(--font-mono)',
          fontSize: 11.5,
          color: 'var(--text-muted)',
          letterSpacing: '0.03em',
        }}>
          U-Net · LGG Brain MRI Segmentation · PyTorch 2.2
        </div>
        <div style={{
          fontFamily: 'var(--font-mono)',
          fontSize: 11.5,
          color: 'var(--text-muted)',
          letterSpacing: '0.03em',
        }}>
          Deep Learning Major Assignment · 2024
        </div>
        <div style={{
          fontFamily: 'var(--font-mono)',
          fontSize: 11.5,
          color: 'var(--text-muted)',
        }}>
          Ronneberger et al., MICCAI 2015
        </div>
      </div>
    </footer>
  );
}

// ── Offline banner ───────────────────────────────────────────
function OfflineBanner() {
  return (
    <div style={{
      background: 'rgba(251,113,133,0.08)',
      borderBottom: '1px solid rgba(251,113,133,0.2)',
      padding: '10px 2rem',
      textAlign: 'center',
      fontFamily: 'var(--font-mono)',
      fontSize: 12.5,
      color: 'var(--rose)',
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'center',
      gap: 8,
    }}>
      <span>⚠</span>
      Flask backend is offline — start it with{' '}
      <code style={{
        background: 'rgba(251,113,133,0.1)',
        padding: '1px 8px',
        borderRadius: 4,
        fontSize: 11.5,
      }}>
        cd backend && python app.py
      </code>
      {' '}then refresh.
    </div>
  );
}

// ── Root App ─────────────────────────────────────────────────
export default function App() {
  const [serverStatus, setServerStatus] = useState('checking');
  const [modelInfo,    setModelInfo]    = useState(null);

  // ── Poll /health every 10 seconds ──────────────────────────
  const pollHealth = useCallback(async () => {
    try {
      const data = await checkHealth();
      setServerStatus(data.model_loaded ? 'online' : 'offline');
    } catch {
      setServerStatus('offline');
    }
  }, []);

  // ── Fetch model info once ───────────────────────────────────
  const fetchModelInfo = useCallback(async () => {
    try {
      const data = await getModelInfo();
      setModelInfo(data);
    } catch {
      // non-fatal — About page handles missing info gracefully
    }
  }, []);

  useEffect(() => {
    pollHealth();
    fetchModelInfo();
    const id = setInterval(pollHealth, 10_000);
    return () => clearInterval(id);
  }, [pollHealth, fetchModelInfo]);

  return (
    <Router>
      <div style={{ display: 'flex', flexDirection: 'column', minHeight: '100vh' }}>

        {/* Offline warning banner */}
        {serverStatus === 'offline' && <OfflineBanner />}

        {/* Top navigation */}
        <Navbar serverStatus={serverStatus} />

        {/* Page content */}
        <main style={{ flex: 1, display: 'flex', flexDirection: 'column' }}>
          <Routes>
            <Route path="/" element={
              <PageWrapper>
                <Home serverStatus={serverStatus} />
              </PageWrapper>
            } />
            <Route path="/about" element={
              <PageWrapper>
                <About modelInfo={modelInfo} serverStatus={serverStatus} />
              </PageWrapper>
            } />
            {/* 404 fallback */}
            <Route path="*" element={
              <PageWrapper>
                <NotFound />
              </PageWrapper>
            } />
          </Routes>
        </main>

        <Footer />
      </div>
    </Router>
  );
}

// ── 404 page ─────────────────────────────────────────────────
function NotFound() {
  return (
    <div style={{
      flex: 1,
      display: 'flex',
      flexDirection: 'column',
      alignItems: 'center',
      justifyContent: 'center',
      padding: '6rem 2rem',
      gap: 16,
      textAlign: 'center',
    }}>
      <div style={{
        fontFamily: 'var(--font-mono)',
        fontSize: 80,
        fontWeight: 300,
        color: 'var(--border)',
        lineHeight: 1,
      }}>
        404
      </div>
      <div style={{
        fontFamily: 'var(--font-heading)',
        fontSize: 22,
        color: 'var(--text-secondary)',
      }}>
        Page not found
      </div>
      <a href="/" style={{
        marginTop: 8,
        padding: '9px 22px',
        background: 'var(--accent-glow)',
        border: '1px solid var(--accent-border)',
        borderRadius: 'var(--radius-md)',
        color: 'var(--accent)',
        fontFamily: 'var(--font-body)',
        fontSize: 14,
        fontWeight: 500,
      }}>
        ← Back to Segmentation
      </a>
    </div>
  );
}

// ── end of App.jsx ───────────────────────────────────────────