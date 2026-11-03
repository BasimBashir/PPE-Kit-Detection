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
  COMPLIANT: "OK", CLEARED: "CLEARED", SUSPECTED_NON_COMPLIANT: "SUSPECT", CONFIRMED_NON_COMPLIANT: "NO HAT", UNKNOWN: "?",
};

function draw(ctx: CanvasRenderingContext2D, f: FrameOut) {
  ctx.clearRect(0, 0, f.width, f.height);
  ctx.lineWidth = 1.5;
  ctx.font = "11px system-ui, sans-serif";
  for (const t of f.tracks as OverlayTrack[]) {
    const c = STATE_COLOR[t.ppe_state] ?? "#9a9993";
    const [x1, y1, x2, y2] = t.box;
    ctx.strokeStyle = c;
    ctx.setLineDash([]);
    ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
    if (t.head_roi) {
      ctx.setLineDash([3, 3]);
      ctx.strokeRect(t.head_roi[0], t.head_roi[1], t.head_roi[2] - t.head_roi[0], t.head_roi[3] - t.head_roi[1]);
    }
    ctx.setLineDash([]);
    for (const [b, col] of [[t.helmet_box, "#3987e5"], [t.head_box, "#ec835a"]] as const) {
      if (b) { ctx.strokeStyle = col; ctx.lineWidth = 1; ctx.strokeRect(b[0], b[1], b[2] - b[0], b[3] - b[1]); ctx.lineWidth = 1.5; }
    }
    const label = `#${t.track_id} ${STATE_SHORT[t.ppe_state] ?? t.ppe_state}`;
    const w = ctx.measureText(label).width + 6;
    const ly = y1 - 15 >= 0 ? y1 - 15 : y2 + 1; // above the box, or below if clipped - never over the head
    ctx.fillStyle = "rgba(10,10,10,0.75)";
    ctx.fillRect(x1, ly, w, 14);
    ctx.fillStyle = c;
    ctx.fillText(label, x1 + 3, ly + 11);
  }
}

export function LiveView({ cameraId, overlay, intervalMs, className }: { cameraId: string; overlay: boolean; intervalMs: number; className?: string }) {
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

  return (
    <div className={`liveview ${className ?? ""}`}>
      {src ? <img src={src} alt={`Live preview of camera ${cameraId}`} /> : <div className="liveview-empty">{err ? "No live frame" : "Loading…"}</div>}
      {overlay && frame && <canvas ref={canvas} width={frame.width} height={frame.height} aria-hidden="true" />}
      {err && src && <div className="liveview-stale" role="status">stale — {err}</div>}
    </div>
  );
}
