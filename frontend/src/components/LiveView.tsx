// Live preview with optional client-side overlays (FR-UI-002/003): thin strokes, labels kept off the person.
import { useEffect, useRef, useState } from "react";
import { api, imageUrl } from "../services/api";
import type { FrameOut, OverlayTrack } from "../types/api";

const STATE_COLOR: Record<string, string> = {
  COMPLIANT: "#0ca30c",
  CLEARED: "#0ca30c",
  SUSPECTED_NON_COMPLIANT: "#fab219",
  CONFIRMED_NON_COMPLIANT: "#d03b3b",
  UNKNOWN: "#9a9993",
};
const STATE_SHORT: Record<string, string> = {
  COMPLIANT: "OK", CLEARED: "CLEARED", SUSPECTED_NON_COMPLIANT: "SUSPECT", UNKNOWN: "?",
};
const ITEM_SHORT: Record<string, string> = { hard_hat: "HAT", safety_vest: "VEST", gloves: "GLOVES", goggles: "GOGGLES", safety_shoes: "SHOES" };
const BEHAVIOUR_SHORT: Record<string, string> = { fall: "PERSON DOWN", fight: "FIGHT", phone: "PHONE", stalling: "STALLING" };
/** A stream's selectable checks (ADR-017), as the operator names them. "person" = tracking / attendance only. */
export const MODULE_LABEL: Record<string, string> = {
  hard_hat: "Hard hat", safety_vest: "Safety vest", gloves: "Gloves", goggles: "Goggles", safety_shoes: "Safety shoes",
  phone: "Phone use", fall: "Fall", fight: "Fight", stalling: "Stalling", person: "Attendance",
};
const alertOf = (cls: string) => BEHAVIOUR_SHORT[cls] ?? `NO ${ITEM_SHORT[cls] ?? cls.toUpperCase()}`;

/** Per selected check: live verdicts of the people in view, e.g. "Hard hat: 1 NO HAT · 2 OK". */
export function moduleSummary(f: FrameOut, modules: string[]): string {
  const mods = modules.length ? modules : [...new Set(f.tracks.map((t) => t.ppe_class))];
  return mods.map((m) => {
    if (m === "person") return `${MODULE_LABEL.person}: ${f.detections.filter((d) => d.cls === "person").length} in view`;
    const ts = f.tracks.filter((t) => t.ppe_class === m);
    const bad = ts.filter((t) => t.ppe_state === "CONFIRMED_NON_COMPLIANT").length;
    const ok = ts.filter((t) => t.ppe_state === "COMPLIANT" || t.ppe_state === "CLEARED").length;
    const checking = ts.length - bad - ok;
    const parts = [bad && `${bad} ${alertOf(m)}`, ok && `${ok} OK`, checking && `${checking} checking`].filter(Boolean);
    return `${MODULE_LABEL[m] ?? m}: ${parts.length ? parts.join(" · ") : "no one in view"}`;
  }).join("   |   ");
}

function draw(ctx: CanvasRenderingContext2D, f: FrameOut) {
  ctx.clearRect(0, 0, f.width, f.height);
  ctx.lineWidth = 1.5;
  ctx.font = "11px system-ui, sans-serif";
  // one entry per (person, check): draw each person once, worst verdict colours the box, all alerts in the label
  const byTrack = new Map<number, OverlayTrack[]>();
  for (const t of f.tracks as OverlayTrack[]) byTrack.set(t.track_id, [...(byTrack.get(t.track_id) ?? []), t]);
  for (const group of byTrack.values()) {
    const alerts = group.filter((x) => x.ppe_state === "CONFIRMED_NON_COMPLIANT").map((x) => alertOf(x.ppe_class));
    const t = group.find((x) => x.ppe_state === "CONFIRMED_NON_COMPLIANT") ?? group[0];
    const c = STATE_COLOR[t.ppe_state] ?? "#9a9993";
    const [x1, y1, x2, y2] = t.box;
    ctx.strokeStyle = c;
    ctx.setLineDash([]);
    ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
    if (t.item_roi) {
      ctx.setLineDash([3, 3]);
      ctx.strokeRect(t.item_roi[0], t.item_roi[1], t.item_roi[2] - t.item_roi[0], t.item_roi[3] - t.item_roi[1]);
    }
    ctx.setLineDash([]);
    for (const [b, col] of [[t.present_box, "#3987e5"], [t.absent_box, "#ec835a"]] as const) {
      if (b) { ctx.strokeStyle = col; ctx.lineWidth = 1; ctx.strokeRect(b[0], b[1], b[2] - b[0], b[3] - b[1]); ctx.lineWidth = 1.5; }
    }
    const label = `#${t.track_id} ${alerts.length ? alerts.join(" · ") : STATE_SHORT[t.ppe_state] ?? t.ppe_state}`;
    const w = ctx.measureText(label).width + 6;
    const ly = y1 - 15 >= 0 ? y1 - 15 : y2 + 1; // above the box, or below if clipped - never over the head
    ctx.fillStyle = "rgba(10,10,10,0.75)";
    ctx.fillRect(x1, ly, w, 14);
    ctx.fillStyle = c;
    ctx.fillText(label, x1 + 3, ly + 11);
  }
}

export function LiveView({ cameraId, overlay, intervalMs, className, modules = [] }: { cameraId: string; overlay: boolean; intervalMs: number; className?: string; modules?: string[] }) {
  const [src, setSrc] = useState<string | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [frame, setFrame] = useState<FrameOut | null>(null);
  const canvas = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    let alive = true;
    let url: string | null = null;
    const tick = async () => {
      try {
        if (overlay) {
          const f = await api.frame(cameraId);
          if (!alive) return;
          setFrame(f);
          setSrc(`data:image/jpeg;base64,${f.image_b64}`);
        } else {
          const u = await imageUrl(`/api/v1/cameras/${cameraId}/snapshot`);
          if (!alive) { URL.revokeObjectURL(u); return; }
          if (url) URL.revokeObjectURL(url);
          url = u;
          setFrame(null);
          setSrc(u);
        }
        setErr(null);
      } catch (e) {
        if (alive) setErr((e as Error).message);
      }
    };
    tick();
    const t = setInterval(tick, intervalMs);
    return () => { alive = false; clearInterval(t); if (url) URL.revokeObjectURL(url); };
  }, [cameraId, overlay, intervalMs]);

  useEffect(() => {
    const ctx = canvas.current?.getContext("2d");
    if (ctx && frame && overlay) draw(ctx, frame);
    else if (ctx && canvas.current) ctx.clearRect(0, 0, canvas.current.width, canvas.current.height);
  }, [frame, overlay]);

  const dets = frame ? moduleSummary(frame, modules) : "";

  return (
    <div className={`liveview ${className ?? ""}`}>
      {src ? <img src={src} alt={`Live preview of camera ${cameraId}`} /> : <div className="liveview-empty">{err ? "No live frame" : "Loading…"}</div>}
      {overlay && frame && <canvas ref={canvas} width={frame.width} height={frame.height} aria-hidden="true" />}
      {overlay && frame && dets && <div className="liveview-dets" role="status">{dets}</div>}
      {err && src && <div className="liveview-stale" role="status">stale — {err}</div>}
    </div>
  );
}
