import { useEffect, useState, type FormEvent } from "react";
import { api } from "../services/api";
import { StatusBadge } from "../components/StatusBadge";
import type { Camera, CameraCreate, CameraTestResult, Rule, SettingsOut, TemporalPolicy, Thresholds, Zone } from "../types/api";

const EMPTY: CameraCreate = { id: "", name: "", stream_uri: "rtsp://", credential_ref: "env:RTSP_READ", transport: "tcp", enabled: true };

export function CameraConfig() {
  const [cams, setCams] = useState<Camera[]>([]);
  const [form, setForm] = useState<CameraCreate>(EMPTY);
  const [editing, setEditing] = useState<string | null>(null);
  const [tests, setTests] = useState<Record<string, CameraTestResult | "running">>({});
  const [err, setErr] = useState<string | null>(null);
  const load = () => api.cameras().then(setCams).catch((e) => setErr(e.message));
  useEffect(() => { load(); }, []);

  const submit = async (e: FormEvent) => {
    e.preventDefault();
    try {
      const body = { ...form, credential_ref: form.credential_ref || null, target_fps: form.target_fps || null };
      if (editing) { const { id: _id, ...rest } = body; await api.updateCamera(editing, rest); }
      else await api.createCamera(body);
      setForm(EMPTY); setEditing(null); setErr(null); load();
    } catch (x) { setErr((x as Error).message); }
  };
  const edit = (c: Camera) => {
    setEditing(c.id);
    setForm({ id: c.id, name: c.name, stream_uri: c.stream_uri, credential_ref: c.credential_ref, transport: c.transport,
      enabled: c.enabled, site: c.site, location: c.location, target_fps: c.target_fps });
  };
  const test = async (id: string) => {
    setTests((t) => ({ ...t, [id]: "running" }));
    try { const r = await api.testCamera(id); setTests((t) => ({ ...t, [id]: r })); }
    catch (x) { setErr((x as Error).message); }
  };
  const toggle = async (c: Camera) => { await api.updateCamera(c.id, { enabled: !c.enabled }).catch((x) => setErr(x.message)); load(); };
  const del = async (c: Camera) => {
    if (!confirm(`Delete camera ${c.id}?`)) return;
    try { await api.deleteCamera(c.id); load(); } catch (x) { setErr((x as Error).message); }
  };

  return (
    <div className="page">
      <h1>Camera configuration</h1>
      {err && <p className="error" role="alert">{err}</p>}
      <table className="table">
        <thead><tr><th>ID</th><th>Name</th><th>Stream</th><th>Credentials</th><th>State</th><th>Connectivity test</th><th /></tr></thead>
        <tbody>{cams.map((c) => {
          const t = tests[c.id];
          return (
            <tr key={c.id}>
              <td>{c.id}</td><td>{c.name}</td><td><code>{c.stream_uri}</code> <small>{c.transport}</small></td>
              <td>{c.credential_ref ? <code>{c.credential_ref}</code> : "none"} {c.credential_ref && !c.has_credentials && <span className="error">unresolved</span>}</td>
              <td><StatusBadge state={c.enabled ? c.status?.state : "DISABLED"} reason={c.status?.state_reason} /></td>
              <td>{t === "running" ? "testing…" : t ? (t.ok ? `OK ${t.codec} ${t.width}×${t.height} @${t.fps?.toFixed(0)} fps (${t.latency_ms} ms)` : <span className="error">{t.error}</span>) : null}</td>
              <td className="row-actions">
                <button className="btn" onClick={() => test(c.id)}>Test</button>
                <button className="btn" onClick={() => edit(c)}>Edit</button>
                <button className="btn" onClick={() => toggle(c)}>{c.enabled ? "Disable" : "Enable"}</button>
                <button className="btn danger" onClick={() => del(c)}>Delete</button>
              </td>
            </tr>);
        })}</tbody>
      </table>
      <form className="card form-grid" onSubmit={submit} aria-label={editing ? `Edit ${editing}` : "Add camera"}>
        <h2>{editing ? `Edit ${editing}` : "Add camera"}</h2>
        <label>ID<input value={form.id} disabled={!!editing} pattern="[a-z0-9][a-z0-9_-]{0,31}" required onChange={(e) => setForm({ ...form, id: e.target.value })} /></label>
        <label>Name<input value={form.name} required onChange={(e) => setForm({ ...form, name: e.target.value })} /></label>
        <label>Stream URI (no credentials)<input value={form.stream_uri} required onChange={(e) => setForm({ ...form, stream_uri: e.target.value })} /></label>
        <label>Credential reference<input value={form.credential_ref ?? ""} placeholder="env:PREFIX → PREFIX_USER / PREFIX_PASS" onChange={(e) => setForm({ ...form, credential_ref: e.target.value })} /></label>
        <label>Transport<select value={form.transport} onChange={(e) => setForm({ ...form, transport: e.target.value as "tcp" | "udp" })}><option>tcp</option><option>udp</option></select></label>
        <label>Target FPS (blank = default)<input type="number" min={0.5} max={30} step={0.5} value={form.target_fps ?? ""} onChange={(e) => setForm({ ...form, target_fps: e.target.value ? Number(e.target.value) : null })} /></label>
        <label>Site<input value={form.site ?? ""} onChange={(e) => setForm({ ...form, site: e.target.value })} /></label>
        <label>Location<input value={form.location ?? ""} onChange={(e) => setForm({ ...form, location: e.target.value })} /></label>
        <p className="muted small">Passwords are never entered or shown here: they live in the host environment and are referenced by name.</p>
        <div className="row-actions">
          <button className="btn primary">{editing ? "Save" : "Add camera"}</button>
          {editing && <button type="button" className="btn ghost" onClick={() => { setEditing(null); setForm(EMPTY); }}>Cancel</button>}
        </div>
      </form>
    </div>
  );
}

const NUM = (v: string) => Number(v);

export function RulesSettings() {
  const [rules, setRules] = useState<Rule[]>([]);
  const [settings, setSettings] = useState<SettingsOut | null>(null);
  const [cams, setCams] = useState<Camera[]>([]);
  const [zoneCam, setZoneCam] = useState("");
  const [zones, setZones] = useState<Zone[]>([]);
  const [zoneForm, setZoneForm] = useState({ id: "", name: "", kind: "monitor" as Zone["kind"], polygon: "0,0 1,0 1,1 0,1" });
  const [msg, setMsg] = useState<string | null>(null);
  const [err, setErr] = useState<string | null>(null);
  useEffect(() => {
    api.rules().then(setRules).catch((e) => setErr(e.message));
    api.settings().then(setSettings).catch(() => {});
    api.cameras().then((c) => { setCams(c); setZoneCam(c[0]?.id ?? ""); }).catch(() => {});
  }, []);
  useEffect(() => { if (zoneCam) api.zones(zoneCam).then(setZones).catch(() => setZones([])); }, [zoneCam]);

  const save = async (r: Rule) => {
    try {
      const out = await api.patchRule(r.id, { thresholds: r.thresholds, temporal_policy: r.temporal_policy, enabled: r.enabled, severity: r.severity });
      setRules((rs) => rs.map((x) => (x.id === out.id ? out : x)));
      setMsg(`Saved ${r.id}: confirmation needs ≈ ${out.effective_confirm_seconds} s at the configured FPS`); setErr(null);
    } catch (e) { setErr((e as Error).message); }
  };
  const upd = (id: string, f: (r: Rule) => Rule) => setRules((rs) => rs.map((r) => (r.id === id ? f(r) : r)));
  const th = (r: Rule, k: keyof Thresholds, v: number) => upd(r.id, (x) => ({ ...x, thresholds: { ...x.thresholds, [k]: v } }));
  const tp = (r: Rule, k: keyof TemporalPolicy, v: number | string) => upd(r.id, (x) => ({ ...x, temporal_policy: { ...x.temporal_policy, [k]: v } }));
  const addZone = async (e: FormEvent) => {
    e.preventDefault();
    try {
      const polygon = zoneForm.polygon.trim().split(/\s+/).map((p) => p.split(",").map(Number) as [number, number]);
      await api.createZone(zoneCam, { id: zoneForm.id, name: zoneForm.name, kind: zoneForm.kind, polygon, enabled: true });
      setZones(await api.zones(zoneCam)); setErr(null);
    } catch (x) { setErr((x as Error).message); }
  };

  return (
    <div className="page">
      <h1>Rules & settings</h1>
      {msg && <p className="ok" role="status">{msg}</p>}
      {err && <p className="error" role="alert">{err}</p>}
      {rules.map((r) => (
        <section key={r.id} className="card">
          <h2>{r.name} <small className="muted">{r.id}</small></h2>
          <p className="muted">Effective minimum confirmation time: <strong>{r.effective_confirm_seconds} s</strong> (spec 5.1: sample counts depend on processed FPS)</p>
          <div className="form-grid">
            <label><input type="checkbox" checked={r.enabled} onChange={(e) => upd(r.id, (x) => ({ ...x, enabled: e.target.checked }))} /> enabled</label>
            {(Object.keys(r.thresholds) as (keyof Thresholds)[]).map((k) => (
              <label key={k}>{k.replace(/_/g, " ")}<input type="number" step="any" value={r.thresholds[k]} onChange={(e) => th(r, k, NUM(e.target.value))} /></label>
            ))}
            <label>mode<select value={r.temporal_policy.mode} onChange={(e) => tp(r, "mode", e.target.value)}><option>count</option><option>duration</option><option>both</option></select></label>
            {(Object.keys(r.temporal_policy) as (keyof TemporalPolicy)[]).filter((k) => k !== "mode").map((k) => (
              <label key={k}>{k.replace(/_/g, " ")}<input type="number" step="any" value={r.temporal_policy[k] as number} onChange={(e) => tp(r, k, NUM(e.target.value))} /></label>
            ))}
          </div>
          <button className="btn primary" onClick={() => save(r)}>Save rule</button>
        </section>
      ))}
      <section className="card">
        <h2>Zones</h2>
        <label>Camera <select value={zoneCam} onChange={(e) => setZoneCam(e.target.value)}>{cams.map((c) => <option key={c.id}>{c.id}</option>)}</select></label>
        <ul>{zones.map((z) => <li key={z.id}><code>{z.id}</code> {z.name} — {z.kind} — {z.polygon.map((p) => p.join(",")).join(" ")}
          <button className="btn ghost" onClick={async () => { await api.deleteZone(z.id); setZones(await api.zones(zoneCam)); }}>delete</button></li>)}</ul>
        <form className="form-grid" onSubmit={addZone}>
          <label>ID<input required value={zoneForm.id} onChange={(e) => setZoneForm({ ...zoneForm, id: e.target.value })} /></label>
          <label>Name<input required value={zoneForm.name} onChange={(e) => setZoneForm({ ...zoneForm, name: e.target.value })} /></label>
          <label>Kind<select value={zoneForm.kind} onChange={(e) => setZoneForm({ ...zoneForm, kind: e.target.value as Zone["kind"] })}><option>monitor</option><option>ignore</option><option>privacy</option></select></label>
          <label>Polygon (normalised x,y pairs)<input value={zoneForm.polygon} onChange={(e) => setZoneForm({ ...zoneForm, polygon: e.target.value })} /></label>
          <button className="btn">Add zone</button>
        </form>
      </section>
      {settings && (
        <section className="card">
          <h2>Evidence retention</h2>
          <label>Retention (days)<input type="number" min={1} max={3650} value={settings.evidence_retention_days}
            onChange={(e) => setSettings({ ...settings, evidence_retention_days: NUM(e.target.value) })} /></label>
          <div className="row-actions">
            <button className="btn primary" onClick={async () => { setSettings(await api.patchSettings({ evidence_retention_days: settings.evidence_retention_days })); setMsg("Retention saved"); }}>Save</button>
            <button className="btn" onClick={async () => { const r = await api.runRetention(); setMsg(`Retention run: ${r.deleted_files} files deleted`); }}>Run retention now</button>
          </div>
        </section>
      )}
    </div>
  );
}
