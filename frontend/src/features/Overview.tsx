import { Link } from "react-router-dom";
import { useLive } from "../hooks/useLiveSocket";
import { EventStatus, StatusBadge } from "../components/StatusBadge";
import { fmtBytes, fmtNum, fmtPct, fmtSec, fmtTime } from "../components/format";

function Tile({ label, value, sub }: { label: string; value: string; sub?: string }) {
  return (
    <div className="tile">
      <div className="tile-label">{label}</div>
      <div className="tile-value">{value}</div>
      {sub && <div className="tile-sub">{sub}</div>}
    </div>
  );
}

export function Overview() {
  const { metrics: m, cameras, status, events } = useLive();
  if (!m) return <p className="muted">Loading live state…</p>;
  const a = m.aggregate;
  const r = m.resources;
  const vram = r.gpu_memory_total_bytes ? `${fmtBytes(r.gpu_memory_used_bytes)} / ${fmtBytes(r.gpu_memory_total_bytes)}` : "—";
  return (
    <div className="page">
      <h1>Overview</h1>
      <section className="tiles" aria-label="Key indicators">
        <Tile label="Cameras" value={`${a.online} / ${a.cameras_total}`} sub={`online · ${a.degraded} degraded · ${a.offline} offline · ${a.error} error · ${a.disabled} disabled`} />
        <Tile label="Active violations" value={a.active_violations === null ? "—" : String(a.active_violations)} sub={`${a.events_last_hour ?? "—"} events in the last hour`} />
        <Tile label="Processed FPS (all cameras)" value={fmtNum(a.processed_fps, 1)} sub={`target ${fmtNum(a.target_fps_total, 0)} fps`} />
        <Tile label="Frame age at inference" value={fmtSec(m.latency.frame_age_at_inference_p95_s)} sub="worst camera p95" />
        <Tile label="GPU" value={fmtPct(r.gpu_util_percent)} sub={`VRAM ${vram}`} />
        <Tile label="CPU / worker RAM" value={fmtPct(r.system_cpu_percent)} sub={`worker RSS ${fmtBytes(r.worker_rss_bytes)}`} />
        <Tile label="Event → dashboard" value={fmtSec(m.latency.persist_to_dashboard_p95_s)} sub="p95, persisted → rendered" />
        <Tile label="Worker" value={m.worker.up ? "running" : "down"} sub={`restarts ${m.worker.restarts} · DB ${m.database.up ? "up" : "DOWN"} · spool ${m.spool.pending}`} />
      </section>
      <div className="two-col">
        <section className="card" aria-labelledby="cam-h">
          <h2 id="cam-h">Camera status</h2>
          <table className="table">
            <thead><tr><th>Camera</th><th>State</th><th>Processed</th><th>Last frame</th></tr></thead>
            <tbody>
              {Object.values(cameras).map((c) => {
                const s = status[c.id];
                return (
                  <tr key={c.id}>
                    <td><Link to={`/cameras/${c.id}`}>{c.id}</Link> <span className="muted">{c.name}</span></td>
                    <td><StatusBadge state={c.enabled ? s?.state : "DISABLED"} reason={s?.state_reason} /></td>
                    <td>{fmtNum(s?.processed_fps, 1, "fps")}</td>
                    <td>{s?.last_frame_age_s === null || s === undefined ? "—" : `${fmtSec(s.last_frame_age_s)} ago`}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </section>
        <section className="card" aria-labelledby="ev-h">
          <h2 id="ev-h">Recent events <Link className="small" to="/events">all events →</Link></h2>
          {events.length === 0 ? <p className="muted">No open violations.</p> : (
            <ul className="event-list">
              {events.slice(0, 12).map((e) => (
                <li key={e.id}>
                  <Link to={`/events?open=${e.id}`}>{fmtTime(e.confirmed_at)}</Link>
                  <span>{e.camera_name}</span>
                  <span className="muted">track #{e.track_id}</span>
                  <EventStatus status={e.status} />
                </li>
              ))}
            </ul>
          )}
        </section>
      </div>
    </div>
  );
}
