// Typed REST client: bearer auth from sessionStorage, structured error parsing (FR-API-004).
import type {
  Camera, CameraCreate, CameraTestResult, CameraUpdate, EventDetail, FrameOut, MetricsSummary, Page, PPEEvent,
  Rule, SettingsOut, TemporalPolicy, Thresholds, User, Zone,
} from "../types/api";

const TOKEN_KEY = "ppe.token";
const USER_KEY = "ppe.user";

export class ApiError extends Error {
  constructor(public status: number, public code: string, message: string, public details?: unknown) {
    super(message);
  }
}

export const auth = {
  token: () => sessionStorage.getItem(TOKEN_KEY),
  user: (): User | null => {
    const raw = sessionStorage.getItem(USER_KEY);
    return raw ? (JSON.parse(raw) as User) : null;
  },
  set(token: string, user: User) {
    sessionStorage.setItem(TOKEN_KEY, token);
    sessionStorage.setItem(USER_KEY, JSON.stringify(user));
  },
  clear() {
    sessionStorage.removeItem(TOKEN_KEY);
    sessionStorage.removeItem(USER_KEY);
  },
};

let onUnauthorized: () => void = () => {};
export const setUnauthorizedHandler = (fn: () => void) => { onUnauthorized = fn; };

export async function parseError(res: Response): Promise<ApiError> {
  try {
    const body = await res.json();
    const e = body?.error;
    if (e && typeof e.code === "string") return new ApiError(res.status, e.code, e.message ?? e.code, e.details);
  } catch { /* not JSON */ }
  return new ApiError(res.status, "http_error", `${res.status} ${res.statusText}`);
}

async function raw(path: string, init: RequestInit = {}): Promise<Response> {
  const headers = new Headers(init.headers);
  const t = auth.token();
  if (t) headers.set("Authorization", `Bearer ${t}`);
  if (init.body && !headers.has("Content-Type")) headers.set("Content-Type", "application/json");
  const res = await fetch(path, { ...init, headers });
  if (res.status === 401 && !path.endsWith("/auth/login")) {
    auth.clear();
    onUnauthorized();
  }
  if (!res.ok) throw await parseError(res);
  return res;
}

async function json<T>(path: string, init: RequestInit = {}): Promise<T> {
  const res = await raw(path, init);
  return res.status === 204 ? (undefined as T) : ((await res.json()) as T);
}

const post = <T,>(path: string, body?: unknown) => json<T>(path, { method: "POST", body: body === undefined ? undefined : JSON.stringify(body) });
const patch = <T,>(path: string, body: unknown) => json<T>(path, { method: "PATCH", body: JSON.stringify(body) });

export interface EventQuery { camera_id?: string; status?: string; rule_id?: string; feedback?: string; since?: string; until?: string; limit?: number; offset?: number }

export function eventQueryString(q: EventQuery): string {
  const p = new URLSearchParams();
  for (const [k, v] of Object.entries(q)) if (v !== undefined && v !== null && v !== "") p.set(k, String(v));
  const s = p.toString();
  return s ? `?${s}` : "";
}

/** Authenticated image fetch -> object URL (img tags cannot send bearer headers). Caller must revoke. */
export async function imageUrl(path: string): Promise<string> {
  const res = await raw(path);
  return URL.createObjectURL(await res.blob());
}

export const api = {
  login: (username: string, password: string) =>
    post<{ access_token: string; expires_at: string; user: User }>("/api/v1/auth/login", { username, password }),
  logout: () => post<void>("/api/v1/auth/logout"),
  cameras: () => json<Camera[]>("/api/v1/cameras"),
  camera: (id: string) => json<Camera>(`/api/v1/cameras/${id}`),
  createCamera: (b: CameraCreate) => post<Camera>("/api/v1/cameras", b),
  updateCamera: (id: string, b: CameraUpdate) => patch<Camera>(`/api/v1/cameras/${id}`, b),
  deleteCamera: (id: string) => json<void>(`/api/v1/cameras/${id}`, { method: "DELETE" }),
  testCamera: (id: string) => post<CameraTestResult>(`/api/v1/cameras/${id}/test`),
  frame: (id: string) => json<FrameOut>(`/api/v1/cameras/${id}/frame`),
  zones: (id: string) => json<Zone[]>(`/api/v1/cameras/${id}/zones`),
  createZone: (cam: string, z: Omit<Zone, "camera_id">) => post<Zone>(`/api/v1/cameras/${cam}/zones`, z),
  deleteZone: (id: string) => json<void>(`/api/v1/zones/${id}`, { method: "DELETE" }),
  rules: () => json<Rule[]>("/api/v1/rules"),
  patchRule: (id: string, b: { thresholds?: Thresholds; temporal_policy?: TemporalPolicy; enabled?: boolean; severity?: string }) =>
    patch<Rule>(`/api/v1/rules/${id}`, b),
  events: (q: EventQuery) => json<Page<PPEEvent>>(`/api/v1/events${eventQueryString(q)}`),
  event: (id: string) => json<EventDetail>(`/api/v1/events/${id}`),
  acknowledge: (id: string, note?: string) => post<PPEEvent>(`/api/v1/events/${id}/acknowledge`, { note }),
  resolve: (id: string, note?: string) => post<PPEEvent>(`/api/v1/events/${id}/resolve`, { note }),
  feedback: (id: string, label: "TRUE_POSITIVE" | "FALSE_POSITIVE", note?: string) =>
    post<PPEEvent>(`/api/v1/events/${id}/feedback`, { label, note }),
  metrics: () => json<MetricsSummary>("/api/v1/metrics"),
  settings: () => json<SettingsOut>("/api/v1/settings"),
  patchSettings: (b: { evidence_retention_days: number }) => patch<SettingsOut>("/api/v1/settings", b),
  runRetention: () => post<{ deleted_files: number; freed_bytes: number }>("/api/v1/system/retention/run"),
  restartWorker: () => post<{ accepted: boolean }>("/api/v1/system/worker/restart"),
  // demo mode: simulated cameras + fault injection
  demoStreams: () => json<SimStreams>("/api/v1/demo/streams"),
  setDemoStreams: (count: number) => post<SimStreams>("/api/v1/demo/streams", { count }),
  demoAction: (cam: string, action: "stop" | "start" | "stall" | "resume" | "disable_proxy" | "enable_proxy") =>
    post<unknown>(`/api/v1/demo/streams/${cam}/${action}`),
  demoNetwork: (cam: string, body: { kind: "latency" | "timeout" | "clear"; latency?: number; jitter?: number; timeout?: number }) =>
    post<unknown>(`/api/v1/demo/streams/${cam}/network`, body),
  nvrRestart: () => post<unknown>("/api/v1/demo/nvr/restart"),
  fault: (kind: string, camera_id?: string, duration_s?: number) =>
    post<unknown>("/api/v1/system/faults", { kind, camera_id, duration_s }),
};

export interface SimCamera { camera_id: string; publishing: boolean; stalled: boolean; pid: number | null; restarts: number; wanted: boolean }
export interface SimStreams { count: number; available: number; cameras: SimCamera[] }
