import { act, fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { moduleSummary } from "./components/LiveView";
import { StatusBadge } from "./components/StatusBadge";
import { DEFAULT_SCHEDULE, ScheduleEditor, UsersAdmin } from "./features/Admin";
import { TeacherRow } from "./features/Events";
import { initialLive, liveReducer } from "./hooks/liveReducer";
import { LiveProvider, useLive, wsUrl } from "./hooks/useLiveSocket";
import { eventQueryString, parseError } from "./services/api";
import { CAMERA_STATES, type Camera, type MetricsSummary, type PPEEvent } from "./types/api";

const metrics = (up = true) => ({ worker: { up }, mode: "demo" }) as unknown as MetricsSummary;
const cam = (id: string) => ({ id, name: id, enabled: true, status: null }) as unknown as Camera;
const ev = (id: string, t: string) => ({ id, confirmed_at: t, camera_name: "cam01", track_id: 1 }) as unknown as PPEEvent;

describe("moduleSummary (per-stream checks, ADR-017)", () => {
  const tr = (id: number, cls: string, st: string) => ({ track_id: id, ppe_class: cls, ppe_state: st });
  const frame = (tracks: object[], persons = 0) =>
    ({ tracks, detections: Array.from({ length: persons }, () => ({ cls: "person" })) }) as never;
  it("names each selected check with its live verdicts, never raw model classes", () => {
    const f = frame([tr(1, "hard_hat", "CONFIRMED_NON_COMPLIANT"), tr(2, "hard_hat", "COMPLIANT"),
      tr(1, "safety_vest", "SUSPECTED_NON_COMPLIANT")]);
    expect(moduleSummary(f, ["hard_hat", "safety_vest"])).toBe("Hard hat: 1 NO HAT · 1 OK   |   Safety vest: 1 checking");
  });
  it("reports attendance as people in view and an idle check as nobody in view", () => {
    expect(moduleSummary(frame([], 2), ["person"])).toBe("Attendance: 2 in view");
    expect(moduleSummary(frame([]), ["fall"])).toBe("Fall: no one in view");
  });
  it("falls back to the checks present when the stream runs the full cascade", () => {
    expect(moduleSummary(frame([tr(3, "phone", "CONFIRMED_NON_COMPLIANT")]), [])).toBe("Phone use: 1 PHONE");
  });
});

describe("StatusBadge", () => {
  it.each(CAMERA_STATES)("renders text (not color alone) for %s", (s) => {
    render(<StatusBadge state={s} />);
    expect(screen.getByText(new RegExp(s === "ONLINE" ? "Online" : s.charAt(0) + s.slice(1).toLowerCase()))).toBeInTheDocument();
  });
});

describe("liveReducer", () => {
  it("applies snapshot, camera_status, event_created and acks", () => {
    let s = liveReducer(initialLive, { type: "snapshot", cameras: [cam("cam01"), cam("cam02")], open_events: [], metrics: metrics() });
    expect(Object.keys(s.cameras)).toEqual(["cam01", "cam02"]);
    s = liveReducer(s, { type: "camera_status", cameras: [{ camera_id: "cam01", state: "ONLINE" } as never] });
    expect(s.status.cam01.state).toBe("ONLINE");
    s = liveReducer(s, { type: "event_created", event: ev("e1", "2026-11-02T10:00:00Z") });
    s = liveReducer(s, { type: "event_created", event: ev("e2", "2026-11-02T11:00:00Z") });
    expect(s.events.map((e) => e.id)).toEqual(["e2", "e1"]);
    expect(s.newEventIds).toEqual(["e1", "e2"]);
    s = liveReducer(s, { type: "acked", ids: ["e1", "e2"] });
    expect(s.newEventIds).toEqual([]);
  });
  it("worker down clears live camera status (never show stale ONLINE)", () => {
    let s = liveReducer(initialLive, { type: "camera_status", cameras: [{ camera_id: "cam01", state: "ONLINE" } as never] });
    s = liveReducer(s, { type: "worker", up: false, pid: null, run_id: null });
    expect(s.status).toEqual({}) ;
    expect(s.workerUp).toBe(false);
  });
});

describe("api helpers", () => {
  it("builds event query strings without empty values", () => {
    expect(eventQueryString({ camera_id: "cam03", status: "", limit: 25, offset: 0 })).toBe("?camera_id=cam03&limit=25&offset=0");
    expect(eventQueryString({})).toBe("");
  });
  it("parses the structured error format", async () => {
    const res = new Response(JSON.stringify({ error: { code: "conflict", message: "already resolved" } }), { status: 409 });
    const e = await parseError(res);
    expect([e.status, e.code, e.message]).toEqual([409, "conflict", "already resolved"]);
    const e2 = await parseError(new Response("oops", { status: 500, statusText: "Server Error" }));
    expect(e2.code).toBe("http_error");
  });
  it("builds the websocket URL", () => {
    expect(wsUrl("a b", { protocol: "http:", host: "x:8000" } as Location)).toBe("ws://x:8000/ws/live?token=a%20b");
  });
});

class FakeWS {
  static last: FakeWS;
  sent: string[] = [];
  onopen?: () => void; onmessage?: (e: { data: string }) => void; onclose?: () => void;
  constructor() { FakeWS.last = this; }
  send(d: string) { this.sent.push(d); }
  close() {}
  emit(m: unknown) { this.onmessage?.({ data: JSON.stringify(m) }); }
}

function Probe() {
  const l = useLive();
  return <ul>{l.events.map((e) => <li key={e.id}>{e.id}</li>)}</ul>;
}

describe("LiveProvider connection stability", () => {
  it("opens exactly one socket across re-renders with the DEFAULT factory (no reconnect loop)", async () => {
    sessionStorage.setItem("ppe.token", "tok");
    let created = 0;
    class CountingWS extends FakeWS { constructor() { super(); created++; } }
    vi.stubGlobal("WebSocket", CountingWS);
    function Host() { return <LiveProvider><Probe /></LiveProvider>; }
    const { rerender } = render(<Host />);
    await act(async () => { FakeWS.last.onopen?.(); });  // state change => re-render
    rerender(<Host />);
    rerender(<Host />);
    expect(created).toBe(1);
    vi.unstubAllGlobals();
  });
});

describe("LiveProvider render acks", () => {
  it("sends exactly one event_rendered per created event", async () => {
    sessionStorage.setItem("ppe.token", "tok");
    vi.stubGlobal("requestAnimationFrame", (cb: FrameRequestCallback) => { cb(0); return 1; });
    render(<LiveProvider socketFactory={() => new FakeWS() as unknown as WebSocket}><Probe /></LiveProvider>);
    const ws = FakeWS.last;
    await act(async () => { ws.onopen?.(); ws.emit({ type: "event_created", event: ev("e9", "2026-11-02T10:00:00Z") }); });
    await act(async () => { ws.emit({ type: "event_updated", event: ev("e9", "2026-11-02T10:00:00Z") }); });
    expect(screen.getByText("e9")).toBeInTheDocument();
    const acks = ws.sent.map((s) => JSON.parse(s)).filter((m) => m.type === "event_rendered");
    expect(acks).toHaveLength(1);
    expect(acks[0].event_id).toBe("e9");
  });
});

describe("TeacherRow", () => {
  it("shows the teacher verdict and probabilities, and nothing without a review", () => {
    const { container, rerender } = render(<dl><TeacherRow review={null} /></dl>);
    expect(container.querySelector("dt")).toBeNull();
    rerender(<dl><TeacherRow review={{ kind: "violation", verdict: "LIKELY_FALSE_ALERT", p_teacher: 0.05, p_model: 0.81, fused: 0.1, provider: "gemini", latency_ms: 212, created_at: "2026-11-18T10:00:00Z" }} /></dl>);
    expect(screen.getByText("Teacher review")).toBeInTheDocument();
    expect(screen.getByText(/likely false alert — for human review \(alert unchanged\)/)).toBeInTheDocument();
    expect(screen.getByText(/teacher 0.05 \/ model 0.81 \/ fused 0.10/)).toBeInTheDocument();
  });
});

describe("UsersAdmin", () => {
  it("lists users, creates one, and guards the signed-in admin's own row", async () => {
    const admin = { id: "u1", username: "admin", role: "admin", enabled: true, created_at: "2026-11-01T10:00:00Z" };
    const op = { id: "u2", username: "op1", role: "operator", enabled: true, created_at: "2026-11-02T10:00:00Z" };
    sessionStorage.setItem("ppe.token", "tok");
    sessionStorage.setItem("ppe.user", JSON.stringify(admin));
    const calls: [string, RequestInit | undefined][] = [];
    vi.stubGlobal("fetch", vi.fn(async (url: string, init?: RequestInit) => {
      calls.push([url, init]);
      const body = init?.method === "POST" ? { ...op, id: "u3", username: "new1" } : [admin, op];
      return new Response(JSON.stringify(body), { status: init?.method === "POST" ? 201 : 200 });
    }));
    render(<UsersAdmin />);
    expect(await screen.findByText("op1")).toBeInTheDocument();
    const [selfBtn, opBtn] = screen.getAllByRole("button", { name: "Disable" });
    expect(selfBtn).toBeDisabled();
    expect(opBtn).toBeEnabled();
    fireEvent.change(screen.getByLabelText("Username"), { target: { value: "new1" } });
    fireEvent.change(screen.getByLabelText("Password"), { target: { value: "long-enough-1" } });
    fireEvent.change(screen.getByLabelText("Role"), { target: { value: "operator" } });
    await act(async () => { fireEvent.submit(screen.getByRole("form", { name: "Add user" })); });
    const post = calls.find(([, i]) => i?.method === "POST")!;
    expect(post[0]).toBe("/api/v1/users");
    expect(JSON.parse(post[1]!.body as string)).toEqual({ username: "new1", password: "long-enough-1", role: "operator" });
    vi.unstubAllGlobals();
  });
});

describe("ScheduleEditor", () => {
  it("enables a default shift, toggles days and parses windows on blur", () => {
    const onChange = vi.fn();
    const { rerender } = render(<ScheduleEditor value={null} onChange={onChange} />);
    fireEvent.click(screen.getByLabelText(/only during shift hours/));
    expect(onChange).toHaveBeenLastCalledWith(DEFAULT_SCHEDULE);
    rerender(<ScheduleEditor value={DEFAULT_SCHEDULE} onChange={onChange} />);
    fireEvent.click(screen.getByLabelText("Mon"));
    expect(onChange).toHaveBeenLastCalledWith({ ...DEFAULT_SCHEDULE, days: DEFAULT_SCHEDULE.days.filter((d) => d !== 0) });
    const windows = screen.getByLabelText(/Shift windows/);
    fireEvent.change(windows, { target: { value: "06:00-14:00  22:00-06:00 " } });
    fireEvent.blur(windows);
    expect(onChange).toHaveBeenLastCalledWith({ ...DEFAULT_SCHEDULE, windows: ["06:00-14:00", "22:00-06:00"] });
    fireEvent.click(screen.getByLabelText(/only during shift hours/));
    expect(onChange).toHaveBeenLastCalledWith(null);
  });
});
