// System health: one series per chart, one axis, units on axes, crosshair tooltip (dataviz rules).
import { CartesianGrid, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { useLive } from "../hooks/useLiveSocket";
import { api } from "../services/api";
import { StatusBadge } from "../components/StatusBadge";
import { fmtBytes, fmtDuration, fmtNum, fmtPct, fmtSec, fmtTime } from "../components/format";
import { auth } from "../services/api";

const SERIES = "#3987e5"; // single-series hue (reference palette slot 1, dark step)
const GRID = "#2c2c2a";
const INK = "#c3c2b7";

function Chart({ title, data, dataKey, unit, fmt }: { title: string; data: object[]; dataKey: string; unit: string; fmt: (v: number) => string }) {
  return (
    <figure className="card chart">
      <figcaption>{title} <span className="muted">({unit})</span></figcaption>
      {data.length < 2 ? <p className="muted">Collecting data…</p> : (
        <ResponsiveContainer width="100%" height={180}>
          <LineChart data={data} margin={{ top: 8, right: 12, bottom: 0, left: 0 }}>
            <CartesianGrid stroke={GRID} vertical={false} />
            <XAxis dataKey="t" tick={{ fill: INK, fontSize: 11 }} tickLine={false} axisLine={{ stroke: GRID }} minTickGap={40} />
            <YAxis tick={{ fill: INK, fontSize: 11 }} tickLine={false} axisLine={false} width={52} tickFormatter={(v) => fmt(v)} />
            <Tooltip cursor={{ stroke: INK, strokeDasharray: "3 3" }} contentStyle={{ background: "#1a1a19", border: `1px solid ${GRID}`, color: "#f0efec" }}
              formatter={(v) => [fmt(Number(v)), title]} labelStyle={{ color: INK }} />
            <Line type="monotone" dataKey={dataKey} stroke={SERIES} strokeWidth={2} dot={false} isAnimationActive={false} connectNulls />
          </LineChart>
        </ResponsiveContainer>
      )}
    </figure>
  );
}

export function SystemHealth() {
  const { metrics: m, cameras } = useLive();
  if (!m) return <p className="muted">Loading…</p>;
  const data = m.history.map((h) => ({ ...h, t: fmtTime(h.ts), rss_mb: h.worker_rss_bytes ? h.worker_rss_bytes / 1048576 : null }));
  const r = m.resources;
  const isAdmin = auth.user()?.role === "admin";
  return (
    <div className="page">
      <div className="page-head">
        <h1>System health</h1>
        {isAdmin && <button className="btn" onClick={() => api.restartWorker()}>Restart worker</button>}
      </div>
      <section className="tiles">
        <div className="tile"><div className="tile-label">Worker</div><div className="tile-value">{m.worker.up ? "running" : "DOWN"}</div>
          <div className="tile-sub">pid {m.worker.pid ?? "—"} · up {fmtDuration(m.worker.uptime_s)} · restarts {m.worker.restarts} · {m.worker.device ?? "—"}</div></div>
        <div className="tile"><div className="tile-label">GPU</div><div className="tile-value">{fmtPct(r.gpu_util_percent)}</div>
          <div className="tile-sub">VRAM {fmtBytes(r.gpu_memory_used_bytes)} / {fmtBytes(r.gpu_memory_total_bytes)} · torch {fmtBytes(r.torch_cuda_allocated_bytes)} · {fmtNum(r.gpu_temperature_c, 0, "°C")} · {fmtNum(r.gpu_power_w, 0, "W")}</div></div>
        <div className="tile"><div className="tile-label">Memory</div><div className="tile-value">{fmtBytes(r.worker_rss_bytes)}</div>
          <div className="tile-sub">worker RSS · API {fmtBytes(r.api_rss_bytes)} · host used {fmtBytes(r.system_memory_used_bytes)}</div></div>
        <div className="tile"><div className="tile-label">CPU</div><div className="tile-value">{fmtPct(r.system_cpu_percent)}</div>
          <div className="tile-sub">host · worker process {fmtPct(r.worker_cpu_percent)} (100 % = 1 core)</div></div>
        <div className="tile"><div className="tile-label">Alert pipeline</div><div className="tile-value">{m.outbox.pending} pending</div>
          <div className="tile-sub">outbox failed {m.outbox.failed} · spool {m.spool.pending} · DB {m.database.up ? "up" : "DOWN"} · disk free {fmtBytes(r.disk_free_bytes)}</div></div>
      </section>
      <div className="charts">
        <Chart title="Aggregate processed FPS" data={data} dataKey="processed_fps" unit="frames/s" fmt={(v) => v.toFixed(0)} />
        <Chart title="GPU utilisation" data={data} dataKey="gpu_util_percent" unit="%" fmt={(v) => `${v.toFixed(0)}`} />
        <Chart title="Worker resident memory" data={data} dataKey="rss_mb" unit="MiB" fmt={(v) => v.toFixed(0)} />
        <Chart title="Frame age at inference, worst camera p95" data={data} dataKey="frame_age_p95_s" unit="s" fmt={(v) => v.toFixed(2)} />
      </div>
      <section className="card">
        <h2>Inference stages <span className="muted small">(per frame, last 60 s; median batch size {fmtNum(m.batch_size_mean, 1)})</span></h2>
        <table className="table">
          <thead><tr><th>Stage</th><th>p50</th><th>p95</th><th>p99</th><th>samples</th></tr></thead>
          <tbody>{Object.entries(m.inference).map(([k, s]) => (
            <tr key={k}><td>{k}</td><td>{fmtNum(s.p50_ms, 2, "ms")}</td><td>{fmtNum(s.p95_ms, 2, "ms")}</td><td>{fmtNum(s.p99_ms, 2, "ms")}</td><td>{s.count}</td></tr>))}</tbody>
        </table>
      </section>
      <section className="card">
        <h2>Per-camera stream health</h2>
        <table className="table">
          <thead><tr><th>Camera</th><th>State</th><th>Decoded</th><th>Processed / target</th><th>Frame age p95</th><th>Dropped</th><th>Errors</th><th>Reconnects</th><th>Last recovery</th><th>Uptime</th><th>Queue</th></tr></thead>
          <tbody>{m.cameras.map((c) => (
            <tr key={c.camera_id}><td>{c.camera_id} <span className="muted">{cameras[c.camera_id]?.name}</span></td><td><StatusBadge state={c.state} reason={c.state_reason} /></td>
              <td>{fmtNum(c.decoded_fps, 1, "fps")}</td><td>{fmtNum(c.processed_fps, 1)} / {fmtNum(c.target_fps, 0)} fps</td>
              <td>{fmtSec(c.frame_age_at_inference_p95_s)}</td><td>{c.frames_dropped}</td><td>{c.decode_errors}</td><td>{c.reconnect_count}</td>
              <td>{fmtSec(c.last_recovery_s)}</td><td>{fmtPct(c.uptime_ratio * 100)}</td><td>{c.queue_depth}</td></tr>))}</tbody>
        </table>
        <p className="muted small">Dropped = frames replaced in the 1-frame latest-frame slot before inference took them (stale-frame policy, by design).</p>
      </section>
    </div>
  );
}
