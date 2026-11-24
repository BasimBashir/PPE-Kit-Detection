// Draw-on-snapshot zone editor: a pointer aid writing into the polygon text input, which stays the source of truth.
import { useEffect, useRef, useState } from "react";
import { imageUrl } from "../services/api";
import type { Zone } from "../types/api";

const W = 640;
const KIND_COLOR: Record<Zone["kind"], string> = { monitor: "#3987e5", ignore: "#9a9993", privacy: "#8e44ad", door: "#e5a239" };
const parse = (p: string) => p.trim().split(/\s+/).filter(Boolean).map((s) => s.split(",").map(Number) as [number, number]);
const fmt = (pts: [number, number][]) => pts.map((p) => p.join(",")).join(" ");

export function ZoneDrawer({ cameraId, polygon, onChange, existing, draftKind }: {
  cameraId: string; polygon: string; onChange: (polygon: string) => void; existing: Zone[]; draftKind: Zone["kind"];
}) {
  const [snapshot, setSnapshot] = useState<string | null>(null);
  const [img, setImg] = useState<HTMLImageElement | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const canvas = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    let url: string | null = null;
    const el = new Image();
    imageUrl(`/api/v1/cameras/${cameraId}/snapshot`)
      .then((u) => { url = u; el.onload = () => setImg(el); el.src = u; setSnapshot(u); })
      .catch((e) => setErr((e as Error).message));
    return () => { if (url) URL.revokeObjectURL(url); };
  }, [cameraId]);

  const drawDraft = (ctx: CanvasRenderingContext2D, pts: [number, number][], h: number) => {
    if (!pts.length) return;
    const c = KIND_COLOR[draftKind];
    if (pts.length >= 3) {
      ctx.fillStyle = `${c}44`;
      ctx.beginPath();
      pts.forEach(([x, y], i) => (i ? ctx.lineTo(x * W, y * h) : ctx.moveTo(x * W, y * h)));
      ctx.closePath();
      ctx.fill();
    }
    ctx.strokeStyle = c;
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    pts.forEach(([x, y], i) => (i ? ctx.lineTo(x * W, y * h) : ctx.moveTo(x * W, y * h)));
    if (pts.length >= 3) ctx.closePath();
    ctx.stroke();
    ctx.fillStyle = c;
    for (const [x, y] of pts) { ctx.beginPath(); ctx.arc(x * W, y * h, 3, 0, Math.PI * 2); ctx.fill(); }
    if (draftKind === "door" && pts.length === 2) {
      // second point is the walking direction when entering: positive side = inside
      const [a, b] = pts;
      const dx = b[0] - a[0], dy = b[1] - a[1];
      const n = Math.hypot(dx, dy) || 1;
      const mx = (a[0] + b[0]) / 2, my = (a[1] + b[1]) / 2;
      ctx.font = "bold 12px system-ui, sans-serif";
      ctx.fillText("IN", (mx - (dy / n) * 0.03) * W, (my + (dx / n) * 0.03) * h);
      ctx.fillText("OUT", (mx + (dy / n) * 0.03) * W, (my - (dx / n) * 0.03) * h);
    }
  };

  useEffect(() => {
    const ctx = canvas.current?.getContext("2d");
    if (!ctx) return;
    const ar = img ? img.height / img.width : 9 / 16;
    canvas.current!.width = W;
    canvas.current!.height = Math.round(W * ar);
    ctx.clearRect(0, 0, W, canvas.current!.height);
    if (img) ctx.drawImage(img, 0, 0, W, canvas.current!.height);
    ctx.font = "11px system-ui, sans-serif";
    for (const z of existing) {
      const pts = z.polygon;
      if (pts.length < 2) continue;
      ctx.setLineDash([4, 4]);
      ctx.lineWidth = 1;
      ctx.strokeStyle = "rgba(255,255,255,0.6)";
      ctx.beginPath();
      pts.forEach(([x, y], i) => (i ? ctx.lineTo(x * W, y * canvas.current!.height) : ctx.moveTo(x * W, y * canvas.current!.height)));
      ctx.closePath();
      ctx.stroke();
      ctx.setLineDash([]);
      ctx.fillStyle = "rgba(255,255,255,0.6)";
      ctx.fillText(z.id, pts[0][0] * W + 3, pts[0][1] * canvas.current!.height - 3);
    }
    // always draw the draft from the shared state: clicks and typing in the text input both land here
    const pts = parse(polygon).filter(([x, y]) => Number.isFinite(x) && Number.isFinite(y) && x >= 0 && x <= 1 && y >= 0 && y <= 1);
    drawDraft(ctx, pts, canvas.current!.height);
  }, [img, polygon, existing, draftKind]);

  const click = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const r = e.currentTarget.getBoundingClientRect();
    const x = Math.round(((e.clientX - r.left) / r.width) * 1000) / 1000;
    const y = Math.round(((e.clientY - r.top) / r.height) * 1000) / 1000;
    const nx = Math.min(Math.max(x, 0), 1), ny = Math.min(Math.max(y, 0), 1);
    const pts = parse(polygon);
    onChange(draftKind === "door" && pts.length >= 2 ? fmt([[nx, ny]]) : fmt([...pts, [nx, ny]]));
  };

  const n = parse(polygon).length;

  return (
    <div>
      {snapshot && img ? <canvas ref={canvas} style={{ maxWidth: "100%" }} role="img" aria-label={`Zone drawing canvas for camera ${cameraId}, ${n} points placed`} onClick={click} /> :
        <div className="liveview-empty" style={{ width: W, maxWidth: "100%" }}>{err ? `No snapshot: ${err}` : "Loading snapshot…"}</div>}
      <div className="row-actions">
        <button type="button" className="btn" disabled={!n} onClick={() => onChange(fmt(parse(polygon).slice(0, -1)))}>Undo point</button>
        <button type="button" className="btn" disabled={!n} onClick={() => onChange("")}>Clear</button>
      </div>
    </div>
  );
}
