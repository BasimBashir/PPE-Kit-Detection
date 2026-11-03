import { useState } from "react";
import { Link, useParams } from "react-router-dom";
import { useLive } from "../hooks/useLiveSocket";
import { LiveView } from "../components/LiveView";
import { EventStatus, StatusBadge } from "../components/StatusBadge";
import { fmtDuration, fmtNum, fmtPct, fmtSec, fmtTime } from "../components/format";

export function CameraWall() {
  const { cameras, status } = useLive();
  const [overlay, setOverlay] = useState(true);
  const list = Object.values(cameras);
  return (
    <div className="page">
      <div className="page-head">
        <h1>Camera wall</h1>
        <label className="toggle"><input type="checkbox" checked={overlay} onChange={(e) => setOverlay(e.target.checked)} /> Detection overlays</label>
      </div>
      {list.length === 0 && <p className="muted">No cameras configured.</p>}
      <div className="wall">
        {list.map((c) => {
          const s = status[c.id];
          const live = c.enabled && s && (s.state === "ONLINE" || s.state === "DEGRADED");
          return (
            <article key={c.id} className="cam-card" aria-label={`Camera ${c.id}`}>
              <Link to={`/cameras/${c.id}`} className="cam-media">
                {live ? <LiveView cameraId={c.id} overlay={overlay} intervalMs={500} /> :
                  <div className="liveview-empty">{c.enabled ? s?.state_reason ?? "waiting for worker" : "disabled"}</div>}
              </Link>
              <div className="cam-meta">
                <div className="cam-title"><strong>{c.id}</strong> <span className="muted">{c.name}</span></div>
                <StatusBadge state={c.enabled ? s?.state : "DISABLED"} reason={s?.state_reason} />
              </div>
              <dl className="cam-stats">
                <div><dt>source</dt><dd>{fmtNum(s?.source_fps, 0, "fps")}</dd></div>
                <div><dt>decoded</dt><dd>{fmtNum(s?.decoded_fps, 1, "fps")}</dd></div>
                <div><dt>processed</dt><dd>{fmtNum(s?.processed_fps, 1, "fps")}</dd></div>
                <div><dt>frame age</dt><dd>{fmtSec(s?.last_frame_age_s)}</dd></div>
                <div><dt>reconnects</dt><dd>{s?.reconnect_count ?? "—"}</dd></div>
                <div><dt>violations</dt><dd className={c.open_violations ? "crit" : ""}>{c.open_violations}</dd></div>
              </dl>
            </article>
          );
        })}
      </div>
    </div>
  );
}

export function CameraDetail() {
  const { id = "" } = useParams();
  const { cameras, status, events } = useLive();
  const [overlay, setOverlay] = useState(true);
  const c = cameras[id];
  const s = status[id];
  if (!c) return <p className="muted">Camera {id} not found.</p>;
  const camEvents = events.filter((e) => e.camera_id === id).slice(0, 20);
  const live = c.enabled && s && (s.state === "ONLINE" || s.state === "DEGRADED");
  return (
    <div className="page">
      <div className="page-head">
        <h1>{c.id} <span className="muted">{c.name}</span></h1>
        <StatusBadge state={c.enabled ? s?.state : "DISABLED"} reason={s?.state_reason} />
        <label className="toggle"><input type="checkbox" checked={overlay} onChange={(e) => setOverlay(e.target.checked)} /> Overlays</label>
      </div>
      <div className="detail">
        <div className="detail-view">
          {live ? <LiveView cameraId={id} overlay={overlay} intervalMs={250} className="large" /> :
            <div className="liveview-empty large">{s?.state_reason ?? "no live frames"}</div>}
          <p className="legend"><span className="lg ok">■ compliant</span> <span className="lg sus">■ suspected</span> <span className="lg conf">■ confirmed no hard hat</span> <span className="lg unk">■ unknown / not assessable</span> · dashed = head ROI · blue = helmet · orange = bare head</p>
        </div>
        <aside className="card">
          <h2>Stream health</h2>
          <dl className="kv">
            <dt>State reason</dt><dd>{s?.state_reason ?? "—"}</dd>
            <dt>Resolution</dt><dd>{s?.width ? `${s.width}×${s.height}` : "—"}</dd>
            <dt>Source / decoded / processed</dt><dd>{fmtNum(s?.source_fps, 0)} / {fmtNum(s?.decoded_fps, 1)} / {fmtNum(s?.processed_fps, 1)} fps (target {fmtNum(s?.target_fps, 0)})</dd>
            <dt>Frame age at inference (p95)</dt><dd>{fmtSec(s?.frame_age_at_inference_p95_s)}</dd>
            <dt>Dropped (stale) frames</dt><dd>{s?.frames_dropped ?? "—"}</dd>
            <dt>Errors / reconnects</dt><dd>{s?.decode_errors ?? "—"} / {s?.reconnect_count ?? "—"}</dd>
            <dt>Uptime (this worker run)</dt><dd>{fmtPct(s ? s.uptime_ratio * 100 : null)}</dd>
            <dt>Last recovery time</dt><dd>{fmtSec(s?.last_recovery_s)}</dd>
            <dt>Active tracks</dt><dd>{s?.active_tracks ?? "—"}</dd>
            <dt>In state since</dt><dd>{fmtTime(s?.state_since)} ({s?.state_since ? fmtDuration((Date.now() - Date.parse(s.state_since)) / 1000) : "—"})</dd>
          </dl>
          <h2>Configuration</h2>
          <dl className="kv">
            <dt>Stream URI</dt><dd><code>{c.stream_uri}</code></dd>
            <dt>Credentials</dt><dd>{c.credential_ref ? <code>{c.credential_ref}</code> : "none"} {c.credential_ref && (c.has_credentials ? "(resolved)" : "(missing!)")}</dd>
            <dt>Transport</dt><dd>{c.transport.toUpperCase()}</dd>
            <dt>Site / location</dt><dd>{c.site ?? "—"} / {c.location ?? "—"}</dd>
          </dl>
        </aside>
      </div>
      <section className="card">
        <h2>Recent events on this camera</h2>
        {camEvents.length === 0 ? <p className="muted">No open events.</p> : (
          <table className="table">
            <thead><tr><th>Confirmed</th><th>Track</th><th>Status</th><th>Time to confirm</th><th /></tr></thead>
            <tbody>{camEvents.map((e) => (
              <tr key={e.id}><td>{fmtTime(e.confirmed_at)}</td><td>#{e.track_id}</td><td><EventStatus status={e.status} /></td>
                <td>{fmtSec(e.latency.time_to_confirm_s)}</td><td><Link to={`/events?open=${e.id}`}>review</Link></td></tr>
            ))}</tbody>
          </table>
        )}
      </section>
    </div>
  );
}
