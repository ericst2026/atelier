import React, { useEffect, useRef } from "react";
import { useT } from "../i18n";

export default function LogView({ lines, tall = false }) {
  const t = useT();
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
  if (!lines || !lines.length) return <div className="log">{t("run.noOutput")}</div>;
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
