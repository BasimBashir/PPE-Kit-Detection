// /ws/live connection: exponential reconnect, snapshot-based resync, render acks for latency (NFR-PERF-005).
import { createContext, useContext, useEffect, useLayoutEffect, useReducer, useRef, useState, type ReactNode } from "react";
import { auth } from "../services/api";
import { initialLive, liveReducer, type LiveState } from "./liveReducer";
import type { WsMessage } from "../types/api";

export type ConnState = "connecting" | "open" | "closed";
interface LiveCtx extends LiveState { conn: ConnState; soundOn: boolean; setSoundOn: (v: boolean) => void; toast: string | null }

const Ctx = createContext<LiveCtx | null>(null);

export function wsUrl(token: string, loc: Location = window.location): string {
  const proto = loc.protocol === "https:" ? "wss:" : "ws:";
  return `${proto}//${loc.host}/ws/live?token=${encodeURIComponent(token)}`;
}

function beep() {
  try {
    const ctx = new AudioContext();
    const o = ctx.createOscillator();
    o.frequency.value = 1100;
    o.connect(ctx.destination);
    o.start();
    o.stop(ctx.currentTime + 0.25);
  } catch { /* audio unavailable */ }
}

// Module-level default: an inline default would be a new function every render and, being an effect dependency,
// would close and reopen the socket on every render (found by the headless-browser check).
const defaultSocketFactory = (u: string) => new WebSocket(u);

export function LiveProvider({ children, socketFactory = defaultSocketFactory }: { children: ReactNode; socketFactory?: (url: string) => WebSocket }) {
  const [state, dispatch] = useReducer(liveReducer, initialLive);
  const [conn, setConn] = useState<ConnState>("connecting");
  const [soundOn, setSoundOn] = useState(false);
  const [toast, setToast] = useState<string | null>(null);
  const wsRef = useRef<WebSocket | null>(null);
  const soundRef = useRef(soundOn);
  soundRef.current = soundOn;

  useEffect(() => {
    let stopped = false;
    let delay = 1000;
    let timer: ReturnType<typeof setTimeout> | undefined;
    const connect = () => {
      const token = auth.token();
      if (!token || stopped) return;
      setConn("connecting");
      const ws = socketFactory(wsUrl(token));
      wsRef.current = ws;
      ws.onopen = () => { setConn("open"); delay = 1000; };
      ws.onmessage = (e) => {
        const msg = JSON.parse(e.data) as WsMessage;
        dispatch(msg);
        if (msg.type === "event_created") {
          setToast(`New violation: ${msg.event.camera_name} - track #${msg.event.track_id}`);
          if (soundRef.current) beep();
        }
      };
      ws.onclose = () => {
        setConn("closed");
        if (!stopped) timer = setTimeout(connect, delay);
        delay = Math.min(delay * 2, 15000);
      };
    };
    connect();
    return () => { stopped = true; clearTimeout(timer); wsRef.current?.close(); };
  }, [socketFactory]);

  // Ack each new event once, after the frame that paints it (layout effect + rAF) - measures persist->render.
  useLayoutEffect(() => {
    if (!state.newEventIds.length) return;
    const ids = [...state.newEventIds];
    const raf = requestAnimationFrame(() => {
      const rendered_at = new Date().toISOString();
      for (const id of ids) wsRef.current?.send(JSON.stringify({ type: "event_rendered", event_id: id, rendered_at }));
      dispatch({ type: "acked", ids });
    });
    return () => cancelAnimationFrame(raf);
  }, [state.newEventIds]);

  useEffect(() => {
    if (!toast) return;
    const t = setTimeout(() => setToast(null), 6000);
    return () => clearTimeout(t);
  }, [toast]);

  return <Ctx.Provider value={{ ...state, conn, soundOn, setSoundOn, toast }}>{children}</Ctx.Provider>;
}

export function useLive(): LiveCtx {
  const c = useContext(Ctx);
  if (!c) throw new Error("useLive outside LiveProvider");
  return c;
}
