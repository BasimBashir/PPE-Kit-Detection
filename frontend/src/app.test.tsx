import { act, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { StatusBadge } from "./components/StatusBadge";
import { initialLive, liveReducer } from "./hooks/liveReducer";
import { LiveProvider, useLive, wsUrl } from "./hooks/useLiveSocket";
import { eventQueryString, parseError } from "./services/api";
import { CAMERA_STATES, type Camera, type MetricsSummary, type PPEEvent } from "./types/api";

const metrics = (up = true) => ({ worker: { up }, mode: "demo" }) as unknown as MetricsSummary;
const cam = (id: string) => ({ id, name: id, enabled: true, status: null }) as unknown as Camera;
const ev = (id: string, t: string) => ({ id, confirmed_at: t, camera_name: "cam01", track_id: 1 }) as unknown as PPEEvent;

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
