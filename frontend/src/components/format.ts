export const fmtTime = (iso: string | null | undefined) =>
  iso ? new Date(iso).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" }) : "—";
export const fmtDateTime = (iso: string | null | undefined) =>
  iso ? new Date(iso).toLocaleString([], { dateStyle: "short", timeStyle: "medium" }) : "—";
export const fmtNum = (v: number | null | undefined, digits = 1, unit = "") =>
  v === null || v === undefined || Number.isNaN(v) ? "—" : `${v.toFixed(digits)}${unit ? ` ${unit}` : ""}`;
export const fmtSec = (v: number | null | undefined) =>
  v === null || v === undefined ? "—" : v < 1 ? `${Math.round(v * 1000)} ms` : `${v.toFixed(2)} s`;
export const fmtBytes = (v: number | null | undefined) => {
  if (v === null || v === undefined) return "—";
  const u = ["B", "KB", "MB", "GB", "TB"];
  let i = 0;
  let x = v;
  while (x >= 1024 && i < u.length - 1) { x /= 1024; i++; }
  return `${x.toFixed(i >= 3 ? 1 : 0)} ${u[i]}`;
};
export const fmtPct = (v: number | null | undefined) => (v === null || v === undefined ? "—" : `${Math.round(v)} %`);
export const fmtDuration = (s: number | null | undefined) => {
  if (s === null || s === undefined) return "—";
  const h = Math.floor(s / 3600), m = Math.floor((s % 3600) / 60), sec = Math.floor(s % 60);
  return h ? `${h}h ${m}m` : m ? `${m}m ${sec}s` : `${sec}s`;
};
