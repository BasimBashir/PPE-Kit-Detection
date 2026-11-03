// Demo control (admin, demo mode): choose how many simulated cameras stream and inject faults into the real
// pipeline. Every action hits the real system - the dashboard reacts through the normal health/metrics paths.
import { useCallback, useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { api, type SimStreams } from "../services/api";
import { useLive } from "../hooks/useLiveSocket";
import { StatusBadge } from "../components/StatusBadge";
import { fmtNum, fmtSec } from "../components/format";

export function DemoControl() {
  const { status, cameras } = useLive();
  const [sim, setSim] = useState<SimStreams | null>(null);
  const [count, setCount] = useState(10);
  const [busy, setBusy] = useState<string | null>(null);
  const [log, setLog] = useState<string[]>([]);
  const [err, setErr] = useState<string | null>(null);

  const refresh = useCallback(() => {
    api.demoStreams().then((s) => { setSim(s); setErr(null); }).catch((e) => setErr(e.message));
  }, []);
  useEffect(() => {
    refresh();
    const t = setInterval(refresh, 3000);
    return () => clearInterval(t);
  }, [refresh]);
  const simCount = sim?.count;
  useEffect(() => { if (simCount !== undefined) setCount(simCount); }, [simCount]);

  const run = async (label: string, fn: () => Promise<unknown>) => {
    setBusy(label);
    try {
      await fn();
      setLog((l) => [`${new Date().toLocaleTimeString()}  ${label}`, ...l].slice(0, 30));
      refresh();
    } catch (e) {
      setErr(`${label}: ${(e as Error).message}`);
    } finally {
      setBusy(null);
    }
  };

  return (
    <div className="page">
      <h1>Demo control</h1>
      <p className="muted">Simulated RTSP cameras (stock clips looped through a local MediaMTX "NVR" and a Toxiproxy network
        shim). Actions below act on the real ingest/inference pipeline — watch the <Link to="/cameras">camera wall</Link> and{" "}
        <Link to="/health">system health</Link> react. To use real cameras instead, add them on the <Link to="/config/cameras">Cameras</Link> page.</p>
      {err && <p className="error" role="alert">{err}</p>}

      <section className="card">
        <h2>Number of live streams</h2>
        {!sim ? <p className="muted">Waiting for the simulator…</p> : (
          <div className="row-actions">
            <label>Streams (0–{sim.available})
              <input type="range" min={0} max={sim.available} value={count} onChange={(e) => setCount(Number(e.target.value))} aria-valuetext={`${count} streams`} />
            </label>
            <strong className="big-num">{count}</strong>
            <button className="btn primary" disabled={!!busy || count === sim.count} onClick={() => run(`set ${count} streams`, () => api.setDemoStreams(count))}>Apply</button>
            <span className="muted">currently {sim.count} publishing · cameras outside the selection are disabled (kept, not deleted)</span>
          </div>
        )}
      </section>

      <section className="card">
        <h2>Per-camera faults</h2>
        <table className="table">
          <thead><tr><th>Camera</th><th>Simulator</th><th>Pipeline state</th><th>Processed</th><th>Frame age</th><th>Reconnects</th><th>Inject</th></tr></thead>
          <tbody>
            {(sim?.cameras ?? []).map((c) => {
              const s = status[c.camera_id];
              const enabled = cameras[c.camera_id]?.enabled;
              const b = (label: string, fn: () => Promise<unknown>) => (
                <button className="btn" disabled={!!busy} onClick={() => run(`${c.camera_id}: ${label}`, fn)}>{label}</button>);
              return (
                <tr key={c.camera_id}>
                  <td>{c.camera_id} <span className="muted">{cameras[c.camera_id]?.name}</span></td>
                  <td>{c.stalled ? "stalled" : c.publishing ? "publishing" : "stopped"}</td>
                  <td><StatusBadge state={enabled === false ? "DISABLED" : s?.state} reason={s?.state_reason} /></td>
                  <td>{fmtNum(s?.processed_fps, 1, "fps")}</td>
                  <td>{fmtSec(s?.last_frame_age_s)}</td>
                  <td>{s?.reconnect_count ?? "—"}</td>
                  <td className="row-actions">
                    {c.publishing || c.stalled ? b("Unplug", () => api.demoAction(c.camera_id, "stop")) : b("Plug in", () => api.demoAction(c.camera_id, "start"))}
                    {c.stalled ? b("Resume", () => api.demoAction(c.camera_id, "resume")) : b("Stall", () => api.demoAction(c.camera_id, "stall"))}
                    {b("Jitter 800±400 ms", () => api.demoNetwork(c.camera_id, { kind: "latency", latency: 800, jitter: 400 }))}
                    {b("RTSP timeout", () => api.demoNetwork(c.camera_id, { kind: "timeout", timeout: 1 }))}
                    {b("Refuse conn.", () => api.demoAction(c.camera_id, "disable_proxy"))}
                    {b("Clear network", async () => { await api.demoAction(c.camera_id, "enable_proxy"); await api.demoNetwork(c.camera_id, { kind: "clear" }); })}
                    {b("Decoder error", () => api.fault("decoder_error", c.camera_id))}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </section>

      <section className="card">
        <h2>System-level faults</h2>
        <div className="row-actions">
          <button className="btn" disabled={!!busy} onClick={() => run("NVR restart", () => api.nvrRestart())}>Restart NVR (all cameras)</button>
          <button className="btn" disabled={!!busy} onClick={() => run("inference exception", () => api.fault("inference_exception"))}>Inference exception</button>
          <button className="btn" disabled={!!busy} onClick={() => run("GPU OOM (simulated)", () => api.fault("gpu_oom"))}>GPU out-of-memory</button>
          <button className="btn danger" disabled={!!busy} onClick={() => run("worker crash", () => api.fault("crash"))}>Crash pipeline worker</button>
          <button className="btn" disabled={!!busy} onClick={() => run("disk full 60 s", () => api.fault("disk_full", undefined, 60))}>Disk full (60 s)</button>
        </div>
        <p className="muted small">Database outage: run <code>docker compose stop postgres</code> then <code>docker compose start postgres</code> — events are spooled locally and replayed.</p>
      </section>

      <section className="card">
        <h2>Action log</h2>
        {log.length === 0 ? <p className="muted">No actions yet.</p> : <ol className="history">{log.map((l, i) => <li key={i}><code>{l}</code></li>)}</ol>}
        {busy && <p role="status">running: {busy}…</p>}
      </section>
    </div>
  );
}
