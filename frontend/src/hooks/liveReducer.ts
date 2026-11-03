// Pure reducer for /ws/live messages (unit-tested). The server sends a full snapshot on every (re)connect.
import type { Camera, CameraStatusMsgItem, MetricsSummary, PPEEvent, WsMessage } from "../types/api";

export interface LiveState {
  cameras: Record<string, Camera>;
  status: Record<string, CameraStatusMsgItem>;
  events: PPEEvent[]; // open + recent, newest first
  metrics: MetricsSummary | null;
  workerUp: boolean | null;
  newEventIds: string[]; // event_created ids awaiting the render ack
}

export const initialLive: LiveState = { cameras: {}, status: {}, events: [], metrics: null, workerUp: null, newEventIds: [] };
const MAX_EVENTS = 200;

function upsertEvent(list: PPEEvent[], ev: PPEEvent): PPEEvent[] {
  const rest = list.filter((e) => e.id !== ev.id);
  return [ev, ...rest].sort((a, b) => b.confirmed_at.localeCompare(a.confirmed_at)).slice(0, MAX_EVENTS);
}

export function liveReducer(s: LiveState, m: WsMessage | { type: "acked"; ids: string[] }): LiveState {
  switch (m.type) {
    case "snapshot": {
      const cameras: Record<string, Camera> = {};
      const status: Record<string, CameraStatusMsgItem> = {};
      for (const c of m.cameras) {
        cameras[c.id] = c;
        if (c.status) status[c.id] = { ...c.status, camera_id: c.id };
      }
      return { ...s, cameras, status, events: m.open_events, metrics: m.metrics, workerUp: m.metrics.worker.up };
    }
    case "camera_status": {
      const status: Record<string, CameraStatusMsgItem> = {};
      for (const c of m.cameras) status[c.camera_id] = c;
      return { ...s, status };
    }
    case "event_created":
      return { ...s, events: upsertEvent(s.events, m.event), newEventIds: [...s.newEventIds, m.event.id] };
    case "event_updated":
      return { ...s, events: upsertEvent(s.events, m.event) };
    case "metrics":
      return { ...s, metrics: m.metrics, workerUp: m.metrics.worker.up };
    case "worker":
      return { ...s, workerUp: m.up, status: m.up ? s.status : {} };
    case "acked":
      return { ...s, newEventIds: s.newEventIds.filter((id) => !m.ids.includes(id)) };
    default:
      return s;
  }
}
