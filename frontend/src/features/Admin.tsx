import { useEffect, useState, type FormEvent } from "react";
import { api, auth } from "../services/api";
import { StatusBadge } from "../components/StatusBadge";
import { ZoneDrawer } from "../components/ZoneDrawer";
import { fmtDateTime } from "../components/format";
import type { Camera, CameraCreate, CameraTestResult, PPEItemInfo, Preset, Role, Rule, Schedule, SettingsOut, TemporalPolicy, Thresholds, User, Zone } from "../types/api";

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
const EMPTY_RULE = { id: "", name: "", ppe_class: "hard_hat", camera_id: null as string | null, zone_id: null as string | null, schedule: null as Schedule | null, severity: "medium" as Rule["severity"], enabled: true, mandatory: true };

const DAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"];
export const DEFAULT_SCHEDULE: Schedule = { tz: "Asia/Dubai", days: [0, 1, 2, 3, 4], windows: ["07:00-12:00", "13:00-17:00"] };

/** Shift hours a rule is enforced in (site-local time). Breaks = the gaps between windows. */
export function ScheduleEditor({ value, onChange }: { value: Schedule | null; onChange: (s: Schedule | null) => void }) {
  return (
    <fieldset className="form-grid">
      <legend>Schedule</legend>
      <label><input type="checkbox" checked={value !== null} onChange={(e) => onChange(e.target.checked ? DEFAULT_SCHEDULE : null)} /> only during shift hours <small className="muted">unchecked = always enforced</small></label>
      {value && (
        <>
          <label>Time zone (IANA, e.g. Asia/Dubai, Asia/Riyadh)<input value={value.tz} onChange={(e) => onChange({ ...value, tz: e.target.value })} /></label>
          {/* committed on blur: parsing on every keystroke would swallow the space between windows */}
          <label>Shift windows, breaks left out (e.g. 07:00-12:00 13:00-17:00)<input key={value.windows.join(" ")} defaultValue={value.windows.join(" ")} onBlur={(e) => onChange({ ...value, windows: e.target.value.trim().split(/\s+/).filter(Boolean) })} /></label>
          <div role="group" aria-label="Days">
            {DAYS.map((d, i) => (
              <label key={d}><input type="checkbox" checked={value.days.includes(i)} onChange={(e) => onChange({ ...value, days: e.target.checked ? [...value.days, i].sort((a, b) => a - b) : value.days.filter((x) => x !== i) })} />{d}</label>
            ))}
          </div>
        </>
      )}
    </fieldset>
  );
}

export function RulesSettings() {
  const [rules, setRules] = useState<Rule[]>([]);
  const [settings, setSettings] = useState<SettingsOut | null>(null);
  const [cams, setCams] = useState<Camera[]>([]);
  const [zoneCam, setZoneCam] = useState("");
  const [zones, setZones] = useState<Zone[]>([]);
  const [zoneForm, setZoneForm] = useState({ id: "", name: "", kind: "monitor" as Zone["kind"], polygon: "0,0 1,0 1,1 0,1" });
  const [items, setItems] = useState<PPEItemInfo[]>([]);
  const [presets, setPresets] = useState<Preset[]>([]);
  const [presetForm, setPresetForm] = useState<{ name: string; items: Preset["items"] }>({ name: "", items: {} });
  const [applyTo, setApplyTo] = useState<Record<string, { camera_id: string | null; zone_id: string | null }>>({});
  const [ruleForm, setRuleForm] = useState(EMPTY_RULE);
  const [msg, setMsg] = useState<string | null>(null);
  const [err, setErr] = useState<string | null>(null);
  useEffect(() => {
    api.rules().then(setRules).catch((e) => setErr(e.message));
    api.settings().then(setSettings).catch(() => {});
    api.cameras().then((c) => { setCams(c); setZoneCam(c[0]?.id ?? ""); }).catch(() => {});
    api.ppeItems().then(setItems).catch(() => {});
    api.presets().then(setPresets).catch((e) => setErr(e.message));
  }, []);
  useEffect(() => { if (zoneCam) api.zones(zoneCam).then(setZones).catch(() => setZones([])); }, [zoneCam]);

  const save = async (r: Rule) => {
    try {
      const out = await api.patchRule(r.id, { thresholds: r.thresholds, temporal_policy: r.temporal_policy, enabled: r.enabled, mandatory: r.mandatory, severity: r.severity, schedule: r.schedule });
      setRules((rs) => rs.map((x) => (x.id === out.id ? out : x)));
      setMsg(`Saved ${r.id}: confirmation needs ≈ ${out.effective_confirm_seconds} s at the configured FPS`); setErr(null);
    } catch (e) { setErr((e as Error).message); }
  };
  const upd = (id: string, f: (r: Rule) => Rule) => setRules((rs) => rs.map((r) => (r.id === id ? f(r) : r)));
  const th = (r: Rule, k: keyof Thresholds, v: number) => upd(r.id, (x) => ({ ...x, thresholds: { ...x.thresholds, [k]: v } }));
  const tp = (r: Rule, k: keyof TemporalPolicy, v: number | string) => upd(r.id, (x) => ({ ...x, temporal_policy: { ...x.temporal_policy, [k]: v } }));
  const addRule = async (e: FormEvent) => {
    e.preventDefault();
    try {
      const out = await api.createRule(ruleForm);
      setRules((rs) => [...rs, out]); setRuleForm(EMPTY_RULE);
      setMsg(`Created ${out.id}`); setErr(null);
    } catch (x) { setErr((x as Error).message); }
  };
  const itemLabel = (id: string) => items.find((i) => i.id === id)?.label ?? id;
  const applyPreset = async (e: FormEvent, name: string) => {
    e.preventDefault();
    try {
      const out = await api.applyPreset(name, applyTo[name] ?? { camera_id: null, zone_id: null });
      setRules((rs) => { const m = new Map(rs.map((r) => [r.id, r])); for (const r of out) m.set(r.id, r); return [...m.values()]; });
      setMsg(`Applied ${name}: ${out.length} rules`); setErr(null);
    } catch (x) { setErr((x as Error).message); }
  };
  const savePreset = async (e: FormEvent) => {
    e.preventDefault();
    try {
      await api.putPreset(presetForm.name, presetForm.items);
      setPresets(await api.presets()); setMsg(`Saved preset ${presetForm.name}`); setErr(null);
    } catch (x) { setErr((x as Error).message); }
  };
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
          <h2>{r.name} <small className="muted">{r.id} · {itemLabel(r.ppe_class)} · {r.camera_id ?? "all cameras"} / {r.zone_id ?? "full frame"}</small></h2>
          <p className="muted">Effective minimum confirmation time: <strong>{r.effective_confirm_seconds} s</strong> (spec 5.1: sample counts depend on processed FPS)</p>
          <div className="form-grid">
            <label><input type="checkbox" checked={r.enabled} onChange={(e) => upd(r.id, (x) => ({ ...x, enabled: e.target.checked }))} /> enabled</label>
            <label><input type="checkbox" checked={r.mandatory} onChange={(e) => upd(r.id, (x) => ({ ...x, mandatory: e.target.checked }))} /> mandatory (alerts) <small className="muted">unchecked = secondary: recorded, no live alert</small></label>
            {(Object.keys(r.thresholds) as (keyof Thresholds)[]).map((k) => (
              <label key={k}>{k.replace(/_/g, " ")}<input type="number" step="any" value={r.thresholds[k]} onChange={(e) => th(r, k, NUM(e.target.value))} /></label>
            ))}
            <label>mode<select value={r.temporal_policy.mode} onChange={(e) => tp(r, "mode", e.target.value)}><option>count</option><option>duration</option><option>both</option></select></label>
            {(Object.keys(r.temporal_policy) as (keyof TemporalPolicy)[]).filter((k) => k !== "mode").map((k) => (
              <label key={k}>{k.replace(/_/g, " ")}<input type="number" step="any" value={r.temporal_policy[k] as number} onChange={(e) => tp(r, k, NUM(e.target.value))} /></label>
            ))}
          </div>
          <ScheduleEditor value={r.schedule} onChange={(s) => upd(r.id, (x) => ({ ...x, schedule: s }))} />
          <button className="btn primary" onClick={() => save(r)}>Save rule</button>
        </section>
      ))}
      <form className="card form-grid" onSubmit={addRule} aria-label="New rule">
        <h2>New rule</h2>
        <label>ID<input value={ruleForm.id} pattern="[a-z0-9][a-z0-9_-]{0,63}" required onChange={(e) => setRuleForm({ ...ruleForm, id: e.target.value })} /></label>
        <label>Name<input value={ruleForm.name} required onChange={(e) => setRuleForm({ ...ruleForm, name: e.target.value })} /></label>
        <label>Check<select value={ruleForm.ppe_class} onChange={(e) => setRuleForm({ ...ruleForm, ppe_class: e.target.value })}>
          <optgroup label="PPE">{items.filter((i) => i.kind !== "behaviour").map((i) => <option key={i.id} value={i.id}>{i.label}</option>)}</optgroup>
          <optgroup label="Behaviour">{items.filter((i) => i.kind === "behaviour").map((i) => <option key={i.id} value={i.id}>{i.label}</option>)}</optgroup>
        </select></label>
        <label>Camera<select value={ruleForm.camera_id ?? ""} onChange={(e) => setRuleForm({ ...ruleForm, camera_id: e.target.value || null })}><option value="">All cameras</option>{cams.map((c) => <option key={c.id} value={c.id}>{c.id}</option>)}</select></label>
        <label>Zone ID (blank = full frame; see Zones below)<input value={ruleForm.zone_id ?? ""} onChange={(e) => setRuleForm({ ...ruleForm, zone_id: e.target.value || null })} /></label>
        <label>Severity<select value={ruleForm.severity} onChange={(e) => setRuleForm({ ...ruleForm, severity: e.target.value as Rule["severity"] })}><option>low</option><option>medium</option><option>high</option></select></label>
        <label><input type="checkbox" checked={ruleForm.mandatory} onChange={(e) => setRuleForm({ ...ruleForm, mandatory: e.target.checked })} /> mandatory (alerts) <small className="muted">unchecked = secondary: recorded, no live alert</small></label>
        <div className="row-actions"><button className="btn primary">Create rule</button></div>
      </form>
      <section className="card">
        <h2>Presets</h2>
        <p className="muted small">Mandatory = live alert · Secondary = recorded, no live alert · Off = not checked</p>
        <ul>{presets.map((p) => {
          const t = applyTo[p.name] ?? { camera_id: null, zone_id: null };
          return (
            <li key={p.name}>
              <code>{p.name}</code> — {Object.entries(p.items).map(([id, m]) => `${itemLabel(id)}: ${m}`).join(" · ") || "empty"}
              <button className="btn ghost" onClick={() => setPresetForm({ name: p.name, items: { ...p.items } })}>edit</button>
              <button className="btn ghost" onClick={async () => { await api.deletePreset(p.name); setPresets(await api.presets()); }}>delete</button>
              <form className="form-grid" onSubmit={(e) => applyPreset(e, p.name)} aria-label={`Apply ${p.name}`}>
                <label>Camera<select value={t.camera_id ?? ""} onChange={(e) => setApplyTo({ ...applyTo, [p.name]: { ...t, camera_id: e.target.value || null } })}><option value="">All cameras</option>{cams.map((c) => <option key={c.id} value={c.id}>{c.id}</option>)}</select></label>
                <label>Zone ID (blank = full frame)<input value={t.zone_id ?? ""} onChange={(e) => setApplyTo({ ...applyTo, [p.name]: { ...t, zone_id: e.target.value || null } })} /></label>
                <button className="btn primary">Apply</button>
              </form>
            </li>);
        })}</ul>
        <form className="form-grid" onSubmit={savePreset} aria-label="New or edit preset">
          <h2>New / edit preset</h2>
          <label>Name<input value={presetForm.name} pattern="[a-z0-9][a-z0-9_-]{0,23}" required onChange={(e) => setPresetForm({ ...presetForm, name: e.target.value })} /></label>
          {items.map((i) => (
            <label key={i.id}>{i.label}<select value={presetForm.items[i.id] ?? "off"} onChange={(e) => {
              const { [i.id]: _drop, ...rest } = presetForm.items;
              setPresetForm({ ...presetForm, items: e.target.value === "off" ? rest : { ...rest, [i.id]: e.target.value as "mandatory" | "secondary" } });
            }}><option value="off">Off</option><option value="mandatory">Mandatory</option><option value="secondary">Secondary</option></select></label>
          ))}
          <div className="row-actions"><button className="btn primary">Save preset</button></div>
        </form>
      </section>
      <section className="card">
        <h2>Zones</h2>
        <label>Camera <select value={zoneCam} onChange={(e) => setZoneCam(e.target.value)}>{cams.map((c) => <option key={c.id}>{c.id}</option>)}</select></label>
        <ZoneDrawer cameraId={zoneCam} polygon={zoneForm.polygon} onChange={(polygon) => setZoneForm({ ...zoneForm, polygon })} existing={zones} draftKind={zoneForm.kind} />
        <ul>{zones.map((z) => <li key={z.id}><code>{z.id}</code> {z.name} — {z.kind} — {z.polygon.map((p) => p.join(",")).join(" ")}
          <button className="btn ghost" onClick={async () => { await api.deleteZone(z.id); setZones(await api.zones(zoneCam)); }}>delete</button></li>)}</ul>
        <form className="form-grid" onSubmit={addZone}>
          <label>ID<input required value={zoneForm.id} onChange={(e) => setZoneForm({ ...zoneForm, id: e.target.value })} /></label>
          <label>Name<input required value={zoneForm.name} onChange={(e) => setZoneForm({ ...zoneForm, name: e.target.value })} /></label>
          <label>Kind<select value={zoneForm.kind} onChange={(e) => {
            const kind = e.target.value as Zone["kind"];
            // a door starts from an editable default line (IN side shown); 2 clicks redraw it
            setZoneForm({ ...zoneForm, kind, polygon: kind === "door" ? "0.2,0.5 0.8,0.5" : "" });
          }}><option>monitor</option><option>ignore</option><option>privacy</option><option>door</option></select></label>
          <label>Polygon (normalised x,y pairs)<input value={zoneForm.polygon} placeholder={zoneForm.kind === "door" ? "line start then finish, IN side shown: click to redraw" : ""} onChange={(e) => setZoneForm({ ...zoneForm, polygon: e.target.value })} /></label>
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

const ROLES: Role[] = ["viewer", "operator", "admin"];
const EMPTY_USER = { username: "", password: "", role: "viewer" as Role };

export function UsersAdmin() {
  const [users, setUsers] = useState<User[]>([]);
  const [form, setForm] = useState(EMPTY_USER);
  const [err, setErr] = useState<string | null>(null);
  const me = auth.user();
  const load = () => api.users().then(setUsers).catch((e) => setErr(e.message));
  useEffect(() => { load(); }, []);

  const submit = async (e: FormEvent) => {
    e.preventDefault();
    try { await api.createUser(form); setForm(EMPTY_USER); setErr(null); load(); }
    catch (x) { setErr((x as Error).message); }
  };
  const update = async (u: User, b: { role?: Role; enabled?: boolean }) => {
    try { await api.updateUser(u.id, b); setErr(null); load(); } catch (x) { setErr((x as Error).message); }
  };

  return (
    <div className="page">
      <h1>Users</h1>
      {err && <p className="error" role="alert">{err}</p>}
      <table className="table">
        <thead><tr><th>Username</th><th>Role</th><th>State</th><th>Created</th><th /></tr></thead>
        <tbody>{users.map((u) => (
          <tr key={u.id}>
            <td>{u.username}</td>
            <td><select aria-label={`Role for ${u.username}`} value={u.role} disabled={u.id === me?.id}
              onChange={(e) => update(u, { role: e.target.value as Role })}>{ROLES.map((r) => <option key={r}>{r}</option>)}</select></td>
            <td>{u.enabled ? "enabled" : "disabled"}</td>
            <td>{fmtDateTime(u.created_at)}</td>
            <td className="row-actions">
              {/* self-lockout guard: an admin cannot disable or demote their own account here */}
              <button className="btn" disabled={u.id === me?.id} onClick={() => update(u, { enabled: !u.enabled })}>{u.enabled ? "Disable" : "Enable"}</button>
            </td>
          </tr>))}</tbody>
      </table>
      <form className="card form-grid" onSubmit={submit} aria-label="Add user">
        <h2>Add user</h2>
        <label>Username<input value={form.username} pattern="[a-zA-Z0-9_.\-]{3,64}" required onChange={(e) => setForm({ ...form, username: e.target.value })} /></label>
        <label>Password<input type="password" autoComplete="new-password" minLength={8} maxLength={256} value={form.password} required onChange={(e) => setForm({ ...form, password: e.target.value })} /></label>
        <label>Role<select value={form.role} onChange={(e) => setForm({ ...form, role: e.target.value as Role })}>{ROLES.map((r) => <option key={r}>{r}</option>)}</select></label>
        <div className="row-actions"><button className="btn primary">Add user</button></div>
      </form>
    </div>
  );
}
