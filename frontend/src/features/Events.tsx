import { useCallback, useEffect, useState } from "react";
import { useSearchParams } from "react-router-dom";
import { api, auth, imageUrl, type EventQuery } from "../services/api";
import { useLive } from "../hooks/useLiveSocket";
import { EventStatus } from "../components/StatusBadge";
import { fmtDateTime, fmtNum, fmtSec } from "../components/format";
import type { EventDetail, Page, PPEEvent } from "../types/api";

function Evidence({ id, has, className }: { id: string; has: boolean; className?: string }) {
  const [src, setSrc] = useState<string | null>(null);
  useEffect(() => {
    if (!has) return;
    let url: string | null = null;
    let alive = true;
    imageUrl(`/api/v1/events/${id}/evidence`).then((u) => { if (alive) { url = u; setSrc(u); } else URL.revokeObjectURL(u); }).catch(() => setSrc(null));
    return () => { alive = false; if (url) URL.revokeObjectURL(url); };
  }, [id, has]);
  if (!has) return <div className={`evidence-missing ${className ?? ""}`}>no evidence</div>;
  return src ? <img className={className} src={src} alt={`Evidence snapshot for event ${id}`} /> : <div className={`evidence-missing ${className ?? ""}`}>loading…</div>;
}

export function Events() {
  const live = useLive();
  const [params, setParams] = useSearchParams();
  const [q, setQ] = useState<EventQuery>({ limit: 25, offset: 0 });
  const [page, setPage] = useState<Page<PPEEvent> | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [detail, setDetail] = useState<EventDetail | null>(null);
  const [note, setNote] = useState("");
  const role = auth.user()?.role ?? "viewer";
  const canAct = role === "operator" || role === "admin";
  const openId = params.get("open");

  const load = useCallback(() => {
    api.events(q).then((p) => { setPage(p); setErr(null); }).catch((e) => setErr(e.message));
  }, [q]);
  useEffect(load, [load]);
  // live: refetch when a new or updated event arrives
  const newest = live.events[0]?.id;
  useEffect(() => { load(); }, [newest, live.events.length, load]);
  useEffect(() => {
    if (openId) api.event(openId).then(setDetail).catch(() => setDetail(null));
    else setDetail(null);
  }, [openId]);

  const act = async (fn: () => Promise<unknown>) => {
    try {
      await fn();
      setNote("");
      if (openId) setDetail(await api.event(openId));
      load();
    } catch (e) { setErr((e as Error).message); }
  };
  const set = (k: keyof EventQuery, v: string) => setQ((x) => ({ ...x, [k]: v || undefined, offset: 0 }));

  return (
    <div className="page">
      <h1>Events</h1>
      <form className="filters" onSubmit={(e) => e.preventDefault()} aria-label="Event filters">
        <label>Camera<select value={q.camera_id ?? ""} onChange={(e) => set("camera_id", e.target.value)}>
          <option value="">all</option>{Object.keys(live.cameras).map((c) => <option key={c}>{c}</option>)}</select></label>
        <label>Status<select value={q.status ?? ""} onChange={(e) => set("status", e.target.value)}>
          <option value="">all</option><option>OPEN</option><option>ACKNOWLEDGED</option><option>RESOLVED</option></select></label>
        <label>Feedback<select value={q.feedback ?? ""} onChange={(e) => set("feedback", e.target.value)}>
          <option value="">any</option><option value="NONE">unlabelled</option><option>TRUE_POSITIVE</option><option>FALSE_POSITIVE</option></select></label>
        <label>Rule<input value={q.rule_id ?? ""} onChange={(e) => set("rule_id", e.target.value)} placeholder="hardhat-required" /></label>
        <label>Since<input type="datetime-local" onChange={(e) => set("since", e.target.value ? new Date(e.target.value).toISOString() : "")} /></label>
        <label>Until<input type="datetime-local" onChange={(e) => set("until", e.target.value ? new Date(e.target.value).toISOString() : "")} /></label>
      </form>
      {err && <p className="error" role="alert">{err}</p>}
      {!page ? <p className="muted">Loading…</p> : page.items.length === 0 ? <p className="muted">No events match.</p> : (
        <table className="table events">
          <thead><tr><th>Evidence</th><th>Confirmed</th><th>Camera</th><th>Track</th><th>Status</th><th>Feedback</th>
            <th title="first violating observation → confirmation">Time to confirm</th><th title="confirmation → persisted">Persist</th><th title="persisted → rendered in a dashboard">To dashboard</th></tr></thead>
          <tbody>{page.items.map((e) => (
            <tr key={e.id} className={e.id === openId ? "selected" : ""} onClick={() => setParams({ open: e.id })} tabIndex={0}
              onKeyDown={(k) => k.key === "Enter" && setParams({ open: e.id })}>
              <td><Evidence id={e.id} has={e.has_evidence} className="thumb" /></td>
              <td>{fmtDateTime(e.confirmed_at)}</td><td>{e.camera_name}</td><td>#{e.track_id}{e.reconfirm_count ? ` (+${e.reconfirm_count})` : ""}</td>
              <td><EventStatus status={e.status} /></td><td>{e.feedback?.replace("_", " ").toLowerCase() ?? "—"}</td>
              <td>{fmtSec(e.latency.time_to_confirm_s)}</td><td>{fmtSec(e.latency.confirm_to_persist_s)}</td><td>{fmtSec(e.latency.persist_to_dashboard_s)}</td>
            </tr>))}</tbody>
        </table>
      )}
      {page && page.total > (q.limit ?? 25) && (
        <div className="pager">
          <button className="btn" disabled={!q.offset} onClick={() => setQ((x) => ({ ...x, offset: Math.max(0, (x.offset ?? 0) - (x.limit ?? 25)) }))}>← newer</button>
          <span>{(q.offset ?? 0) + 1}–{Math.min(page.total, (q.offset ?? 0) + (q.limit ?? 25))} of {page.total}</span>
          <button className="btn" disabled={(q.offset ?? 0) + (q.limit ?? 25) >= page.total} onClick={() => setQ((x) => ({ ...x, offset: (x.offset ?? 0) + (x.limit ?? 25) }))}>older →</button>
        </div>
      )}
      {detail && (
        <aside className="drawer" aria-label="Event detail">
          <div className="drawer-head"><h2>Event {detail.camera_name} · track #{detail.track_id}</h2>
            <button className="btn ghost" onClick={() => setParams({})} aria-label="Close detail">✕</button></div>
          <Evidence id={detail.id} has={detail.has_evidence} className="evidence-large" />
          {detail.evidence_error && <p className="error">Evidence not stored: {detail.evidence_error}</p>}
          <dl className="kv">
            <dt>Status</dt><dd><EventStatus status={detail.status} /></dd>
            <dt>Reason</dt><dd>{detail.reason}</dd>
            <dt>Votes</dt><dd>{detail.confidence_summary.no_helmet_votes} of {detail.confidence_summary.decisive_votes} decisive observations without hard hat</dd>
            <dt>Mean confidence</dt><dd>{fmtNum(detail.confidence_summary.mean_confidence, 2)} · head ROI from {detail.confidence_summary.roi_source}{detail.confidence_summary.classifier_used ? " · classifier used" : ""}</dd>
            <dt>First seen → confirmed</dt><dd>{fmtDateTime(detail.first_seen_at)} → {fmtDateTime(detail.confirmed_at)} ({fmtSec(detail.latency.time_to_confirm_s)})</dd>
            <dt>Frame → persisted</dt><dd>{fmtSec(detail.latency.frame_to_persist_s ?? null)}</dd>
            <dt>Episode ended</dt><dd>{fmtDateTime(detail.episode_ended_at)}</dd>
            <dt>Models</dt><dd>{Object.entries(detail.model_versions).map(([k, v]) => <code key={k}>{k}: {v} </code>)}</dd>
          </dl>
          {canAct && (
            <div className="actions">
              <label>Note<input value={note} onChange={(e) => setNote(e.target.value)} maxLength={1000} /></label>
              <button className="btn" disabled={detail.status !== "OPEN"} onClick={() => act(() => api.acknowledge(detail.id, note || undefined))}>Acknowledge</button>
              <button className="btn primary" disabled={detail.status === "RESOLVED"} onClick={() => act(() => api.resolve(detail.id, note || undefined))}>Resolve</button>
              <button className="btn" onClick={() => act(() => api.feedback(detail.id, "TRUE_POSITIVE", note || undefined))}>True alert</button>
              <button className="btn" onClick={() => act(() => api.feedback(detail.id, "FALSE_POSITIVE", note || undefined))}>False alert</button>
            </div>
          )}
          <h3>History</h3>
          <ol className="history">{detail.actions.map((a) => <li key={a.id}>{fmtDateTime(a.created_at)} — {a.action.toLowerCase()} by {a.actor}{a.note ? `: ${a.note}` : ""}</li>)}</ol>
          <p className="muted small">Model output is an assistive signal; confirm visually before acting. Feedback is evaluation metadata, not training data.</p>
        </aside>
      )}
    </div>
  );
}
