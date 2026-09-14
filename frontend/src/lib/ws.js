import { useEffect, useRef } from "react";

/** Reconnecting websocket hook. onMessage receives parsed JSON; onStatus gets "open" | "closed". */
export function useSocket(url, onMessage, { enabled = true, onStatus } = {}) {
  const handler = useRef(onMessage);
  const status = useRef(onStatus);
  handler.current = onMessage;
  status.current = onStatus;
  useEffect(() => {
    if (!enabled || !url) return undefined;
    let ws = null;
    let timer = null;
    let ping = null;
    let closed = false;
    let delay = 1000;
    const connect = () => {
      ws = new WebSocket(url);
      ws.onopen = () => {
        delay = 1000;
        status.current && status.current("open");
        ping = setInterval(() => ws && ws.readyState === 1 && ws.send("ping"), 20000);
      };
      ws.onmessage = (ev) => {
        try {
          handler.current(JSON.parse(ev.data));
        } catch (e) {
          console.warn("bad ws message", e);
        }
      };
      ws.onclose = () => {
        status.current && status.current("closed");
        clearInterval(ping);
        if (!closed) {
          timer = setTimeout(connect, delay);
          delay = Math.min(delay * 2, 15000);
        }
      };
      ws.onerror = () => ws && ws.close();
    };
    connect();
    return () => {
      closed = true;
      clearTimeout(timer);
      clearInterval(ping);
      ws && ws.close();
    };
  }, [url, enabled]);
}
