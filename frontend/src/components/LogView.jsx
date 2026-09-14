import React, { useEffect, useRef } from "react";

export default function LogView({ lines, tall = false }) {
  const ref = useRef(null);
  const stick = useRef(true);
  useEffect(() => {
    const el = ref.current;
    if (el && stick.current) el.scrollTop = el.scrollHeight;
  }, [lines]);
  const onScroll = () => {
    const el = ref.current;
    if (!el) return;
    stick.current = el.scrollHeight - el.scrollTop - el.clientHeight < 40;
  };
  if (!lines || !lines.length) return <div className="log">{"(no output yet)"}</div>;
  return (
    <div className={`log ${tall ? "tall" : ""}`} ref={ref} onScroll={onScroll}>
      {lines.map((l, i) => (
        <div key={i} className={l.startsWith("[atelier]") ? "sys" : /error|traceback|exception/i.test(l) ? "err" : ""}>
          {l}
        </div>
      ))}
    </div>
  );
}
