import { NavLink, Outlet, useNavigate } from "react-router-dom";
import { api, auth } from "../services/api";
import { useLive } from "../hooks/useLiveSocket";

const NAV: { to: string; label: string; role?: "admin" }[] = [
  { to: "/", label: "Overview" },
  { to: "/cameras", label: "Camera wall" },
  { to: "/events", label: "Events" },
  { to: "/health", label: "System health" },
  { to: "/config/cameras", label: "Cameras", role: "admin" },
  { to: "/config/rules", label: "Rules & settings", role: "admin" },
  { to: "/demo", label: "Demo control", role: "admin" },
];

export function Layout() {
  const live = useLive();
  const nav = useNavigate();
  const user = auth.user();
  const m = live.metrics;
  const logout = async () => {
    try { await api.logout(); } catch { /* already invalid */ }
    auth.clear();
    nav("/login");
  };
  return (
    <div className="shell">
      <a href="#main" className="skip">Skip to content</a>
      <header className="topbar">
        <div className="brand"><span className="brand-mark" aria-hidden="true">◆</span> PPE Intelligence</div>
        <nav aria-label="Main">
          {NAV.filter((n) => (!n.role || user?.role === n.role) && (n.to !== "/demo" || m?.mode === "demo")).map((n) => (
            <NavLink key={n.to} to={n.to} end={n.to === "/"}>{n.label}</NavLink>
          ))}
        </nav>
        <div className="topbar-right">
          <span className={`conn conn-${live.conn}`} role="status">Live: {live.conn === "open" ? "connected" : live.conn}</span>
          <label className="sound"><input type="checkbox" checked={live.soundOn} onChange={(e) => live.setSoundOn(e.target.checked)} /> Alert sound</label>
          <span className="who">{user?.username} <small>({user?.role})</small></span>
          <button className="btn ghost" onClick={logout}>Sign out</button>
        </div>
      </header>
      {m?.mode === "demo" && <div className="banner banner-demo" role="note">DEMO MODE — SIMULATED RTSP SOURCES (stock footage looped through a local MediaMTX "NVR")</div>}
      {live.workerUp === false && <div className="banner banner-critical" role="alert">Pipeline worker down — camera states unknown; restart in progress</div>}
      {m && !m.database.up && <div className="banner banner-critical" role="alert">Database unavailable — new events are spooled locally and will be replayed</div>}
      <main id="main"><Outlet /></main>
      {live.toast && <div className="toast" role="alert">{live.toast}</div>}
    </div>
  );
}
