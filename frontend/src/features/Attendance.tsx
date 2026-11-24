// Attendance (Phase 5, ADR-016): enrol people, review door crossings, report time outside against a daily budget.
// Identity is a face embedding (torso appearance only when the face is away); names never leak into PPE events.
import { useEffect, useState, type FormEvent, type MouseEvent } from "react";
import { api, auth, imageUrl } from "../services/api";
import type { AttendanceReport, Camera, Crossing, FrameOut, Person } from "../types/api";

const iso = (d: Date) => d.toISOString().slice(0, 10);
const today = () => iso(new Date());

function Thumb({ id }: { id: string }) {
  const [src, setSrc] = useState<string | null>(null);
  useEffect(() => {
    let alive = true, url: string | null = null;
    imageUrl(`/api/v1/attendance/crossings/${id}/image`)
      .then((u) => { if (alive) { url = u; setSrc(u); } else URL.revokeObjectURL(u); })
      .catch(() => setSrc(null));
    return () => { alive = false; if (url) URL.revokeObjectURL(url); };
  }, [id]);
  return src ? <img src={src} alt={`Crossing ${id}`} width={56} height={56} style={{ objectFit: "cover", borderRadius: 4 }} /> : <span className="muted">—</span>;
}

export function Attendance() {
  const canWrite = auth.user()?.role !== "viewer";
  const [persons, setPersons] = useState<Person[]>([]);
  const [crossings, setCrossings] = useState<Crossing[]>([]);
  const [unknown, setUnknown] = useState(false);
  const [picked, setPicked] = useState<Record<string, string>>({});
  const [form, setForm] = useState({ name: "", allowed_minutes_per_day: 60 });
  const [report, setReport] = useState<{ personId: string; start: string; end: string; out: AttendanceReport | null }>(
    { personId: "", start: iso(new Date(Date.now() - 6 * 864e5)), end: today(), out: null });
  const [msg, setMsg] = useState<string | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [cams, setCams] = useState<Camera[]>([]);
  const [cap, setCap] = useState<{ person: Person; cameraId: string; frame: FrameOut | null; box: number[] | null } | null>(null);

  const loadPersons = () => api.persons().then(setPersons).catch((e) => setErr(e.message));
  const loadCrossings = () => api.crossings({ unknown, limit: 50 })
    .then((p) => setCrossings(p.items)).catch((e) => setErr(e.message));

  useEffect(() => { loadPersons(); }, []);
  useEffect(() => { loadCrossings(); }, [unknown]);
  useEffect(() => { if (canWrite) api.cameras().then(setCams).catch(() => setCams([])); }, [canWrite]);

  const run = async (fn: () => Promise<unknown>, note: string) => {
    try { await fn(); setMsg(note); setErr(null); }
    catch (e) { setErr((e as Error).message); }
  };

  const addPerson = (e: FormEvent) => {
    e.preventDefault();
    run(async () => { await api.createPerson(form); setForm({ ...form, name: "" }); await loadPersons(); }, "Person enrolled — assign or enrol crossings below.");
  };

  const saveBudget = (p: Person) => run(async () => { await api.updatePerson(p.id, { allowed_minutes_per_day: p.allowed_minutes_per_day }); await loadPersons(); }, "Budget saved");

  const assign = (x: Crossing) => run(async () => {
    await api.assignCrossing(x.id, picked[x.id] || null);
    await Promise.all([loadCrossings(), loadPersons()]);
  }, "Crossing assigned — intervals rebuilt");

  const enrol = (x: Crossing) => run(async () => {
    await api.enrol(picked[x.id], { crossing_id: x.id });
    await Promise.all([loadCrossings(), loadPersons()]);
  }, "Face and torso added to the person's gallery");

  const loadFrame = (cameraId: string) => {
    setCap((s) => s && { ...s, cameraId, frame: null, box: null });
    api.frame(cameraId).then((frame) => setCap((s) => s && s.cameraId === cameraId ? { ...s, frame } : s))
      .catch((e) => setErr(e.message));
  };

  const openCapture = (p: Person) => {
    const first = cams[0]?.id ?? "";
    setCap({ person: p, cameraId: first, frame: null, box: null });
    if (first) loadFrame(first);
  };

  // click a face: snap to the nearest track box containing the point, else a small box around the click
  const pickBox = (e: MouseEvent<HTMLImageElement>) => {
    if (!cap?.frame) return;
    const r = e.currentTarget.getBoundingClientRect();
    const x = ((e.clientX - r.left) / r.width) * cap.frame.width;
    const y = ((e.clientY - r.top) / r.height) * cap.frame.height;
    const hit = cap.frame.tracks.find((t) => x >= t.box[0] && x <= t.box[2] && y >= t.box[1] && y <= t.box[3]);
    const half = cap.frame.width * 0.12;
    const box = hit ? hit.box : [x - half, y - half, x + half, y + half];
    setCap((s) => s && { ...s, box });
  };

  const capture = () => cap && run(async () => {
    await api.enrol(cap.person.id, { camera_id: cap.cameraId, box: cap.box as [number, number, number, number] });
    await loadPersons();
  }, `Reference added for ${cap.person.name} — re-enrol from another angle while they face the camera`);

  const runReport = (e: FormEvent) => {
    e.preventDefault();
    run(async () => setReport({ ...report, out: await api.attendanceReport(report.personId, report.start, report.end) }), "Report generated");
  };

  return (
    <div className="page">
      <h1>Attendance</h1>
      <p className="muted">
        A door zone is two points drawn on the camera snapshot in the walking direction of an entry (config → Rules &amp;
        settings → Zones); the camera must see people clearly at the door, from a side wall or a roof mount. Crossings
        match enrolled people by face (torso appearance only when the face is turned away, track continuity within a
        pass); unmatched ones stay Unknown until you assign or enrol them. Enrol several references per person — a
        frontal and a profile — because a profile or overhead view is a weaker match. Minutes count as{" "}
        <em>outside</em> between an EXIT and the next ENTER (or now).
      </p>
      {msg && <p className="ok" role="status">{msg}</p>}
      {err && <p className="error" role="alert">{err}</p>}

      <section className="card">
        <h2>People</h2>
        <table className="table">
          <thead><tr><th>Name</th><th>Allowed outside / day</th><th>State</th><th>Crossings</th><th /></tr></thead>
          <tbody>
            {persons.map((p) => (
              <tr key={p.id}>
                <td>{p.name}</td>
                <td>
                  <input type="number" min={0} max={1440} value={p.allowed_minutes_per_day} disabled={!canWrite}
                    aria-label={`Allowed minutes per day for ${p.name}`}
                    onChange={(e) => setPersons((ps) => ps.map((x) => x.id === p.id ? { ...x, allowed_minutes_per_day: Number(e.target.value) } : x))} />
                  {" min"}
                  {canWrite && <button className="btn ghost" onClick={() => saveBudget(p)}>save</button>}
                </td>
                <td>{p.inside ? "inside" : "outside / absent"}{" "}
                  <small className="muted">{p.face_refs} face / {p.torso_refs} torso refs</small></td>
                <td>{p.crossings}</td>
                <td>{canWrite && (
                  <div className="row-actions">
                    <button className="btn ghost" onClick={() => openCapture(p)}>capture</button>
                    <button className="btn ghost" onClick={() => run(async () => { await api.deletePerson(p.id); await loadPersons(); }, "Person deleted; their crossings moved to Unknown")}>delete</button>
                  </div>
                )}</td>
              </tr>
            ))}
            {!persons.length && <tr><td colSpan={5} className="muted">No people enrolled yet.</td></tr>}
          </tbody>
        </table>
        {canWrite && (
          <form className="form-grid" onSubmit={addPerson}>
            <label>Name<input required value={form.name} onChange={(e) => setForm({ ...form, name: e.target.value })} /></label>
            <label>Allowed outside / day (min)<input type="number" min={0} max={1440} value={form.allowed_minutes_per_day}
              onChange={(e) => setForm({ ...form, allowed_minutes_per_day: Number(e.target.value) })} /></label>
            <button className="btn primary">Add person</button>
          </form>
        )}
        {canWrite && cap && (
          <div className="card">
            <h3>Capture references for {cap.person.name}</h3>
            <div className="form-grid">
              <label>Camera
                <select value={cap.cameraId} onChange={(e) => loadFrame(e.target.value)}
                  aria-label={`Capture camera for ${cap.person.name}`}>
                  {cams.map((cm) => <option key={cm.id} value={cm.id}>{cm.name}</option>)}
                </select>
              </label>
              <button className="btn ghost" disabled={!cap.cameraId} onClick={() => loadFrame(cap.cameraId)}>refresh frame</button>
              <button className="btn ghost" onClick={() => setCap(null)}>close</button>
            </div>
            {cap.frame ? (
              <>
                <p className="muted">
                  Click the person&apos;s face (a tracked person snaps to their box). Add a frontal shot first, then a
                  three-quarter and a profile while they turn — each is kept as a separate reference.
                </p>
                <img src={`data:image/jpeg;base64,${cap.frame.image_b64}`} onClick={pickBox} alt="Live frame"
                  style={{ width: "100%", maxWidth: 640, borderRadius: 6, cursor: "crosshair" }} />
                <p className="muted">
                  {cap.box ? `box ${cap.box.map(Math.round).join(", ")} px` : "no face picked yet"}
                  {" "}<button className="btn primary" disabled={!cap.box} onClick={capture}>enrol face</button>
                </p>
              </>
            ) : <p className="muted">{cap.cameraId ? "loading frame…" : "no cameras configured"}</p>}
          </div>
        )}
      </section>

      <section className="card">
        <h2>Crossings <label className="toggle"><input type="checkbox" checked={unknown} onChange={(e) => setUnknown(e.target.checked)} /> Unknown only</label></h2>
        <table className="table">
          <thead><tr><th>Crop</th><th>When</th><th>Camera / door</th><th>Dir</th><th>Person</th><th>Assign to</th><th /></tr></thead>
          <tbody>
            {crossings.map((x) => (
              <tr key={x.id}>
                <td>{x.has_image ? <Thumb id={x.id} /> : <span className="muted">—</span>}</td>
                <td>{new Date(x.occurred_at).toLocaleString()}</td>
                <td>{x.camera_name ?? x.camera_id} / {x.zone_id}</td>
                <td>{x.direction}</td>
                <td>{x.person_name ?? <em className="muted">Unknown</em>}{" "}
                  {x.match_score != null && <small className="muted">{x.match_score.toFixed(2)} {x.matched_by}</small>}
                  {x.matched_by === "track" && <small className="muted">track</small>}</td>
                <td>{canWrite && (
                  <select aria-label={`Assign crossing ${x.id}`} value={picked[x.id] ?? ""}
                    onChange={(e) => setPicked({ ...picked, [x.id]: e.target.value })}>
                    <option value="">—</option>
                    {persons.map((p) => <option key={p.id} value={p.id}>{p.name}</option>)}
                  </select>
                )}</td>
                <td>{canWrite && (
                  <div className="row-actions">
                    <button className="btn ghost" disabled={!picked[x.id]} onClick={() => assign(x)}>assign</button>
                    <button className="btn ghost" disabled={!picked[x.id] || !x.has_image} onClick={() => enrol(x)} title="Add this crop's appearance to the chosen person">enrol</button>
                  </div>
                )}</td>
              </tr>
            ))}
            {!crossings.length && <tr><td colSpan={7} className="muted">No crossings recorded.</td></tr>}
          </tbody>
        </table>
      </section>

      <section className="card">
        <h2>Report</h2>
        <form className="form-grid" onSubmit={runReport}>
          <label>Person<select required value={report.personId} onChange={(e) => setReport({ ...report, personId: e.target.value, out: null })}>
            <option value="">—</option>{persons.map((p) => <option key={p.id} value={p.id}>{p.name}</option>)}
          </select></label>
          <label>From<input type="date" required value={report.start} onChange={(e) => setReport({ ...report, start: e.target.value, out: null })} /></label>
          <label>To<input type="date" required value={report.end} onChange={(e) => setReport({ ...report, end: e.target.value, out: null })} /></label>
          <button className="btn primary">Generate</button>
        </form>
        {report.out && (
          <>
            <p className="muted">{report.out.person.name} · {report.out.from.slice(0, 10)} → {report.out.to.slice(0, 10)} · total outside {report.out.total.outside_minutes} min, over budget {report.out.total.over_minutes} min</p>
            <table className="table">
              <thead><tr><th>Day</th><th>Outside (min)</th><th>Allowed (min)</th><th>Over budget (min)</th></tr></thead>
              <tbody>
                {report.out.days.map((d) => (
                  <tr key={d.date} className={d.over_minutes > 0 ? "selected" : ""}>
                    <td>{d.date}</td><td>{d.outside_minutes}</td><td>{d.allowed_minutes}</td><td>{d.over_minutes}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </>
        )}
      </section>
    </div>
  );
}
