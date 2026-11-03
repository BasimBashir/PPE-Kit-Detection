import type { CameraState } from "../types/api";

// Status never relies on color alone (spec 15.2): every state has an icon + text label.
const MAP: Record<CameraState | "UNKNOWN", { icon: string; tone: string; label: string }> = {
  ONLINE: { icon: "●", tone: "good", label: "Online" },
  DEGRADED: { icon: "▲", tone: "warning", label: "Degraded" },
  RECOVERING: { icon: "↻", tone: "serious", label: "Recovering" },
  CONNECTING: { icon: "…", tone: "neutral", label: "Connecting" },
  OFFLINE: { icon: "✕", tone: "critical", label: "Offline" },
  ERROR: { icon: "!", tone: "critical", label: "Error" },
  DISABLED: { icon: "–", tone: "muted", label: "Disabled" },
  UNKNOWN: { icon: "?", tone: "muted", label: "Unknown" },
};

export function StatusBadge({ state, reason }: { state: CameraState | null | undefined; reason?: string }) {
  const s = MAP[state ?? "UNKNOWN"];
  return (
    <span className={`badge tone-${s.tone}`} title={reason || s.label} data-state={state ?? "UNKNOWN"}>
      <span aria-hidden="true" className="badge-icon">{s.icon}</span>
      {s.label}
    </span>
  );
}

const EV_TONE: Record<string, string> = { OPEN: "critical", ACKNOWLEDGED: "warning", RESOLVED: "good" };
export function EventStatus({ status }: { status: string }) {
  return <span className={`badge tone-${EV_TONE[status] ?? "muted"}`}>{status.toLowerCase()}</span>;
}
