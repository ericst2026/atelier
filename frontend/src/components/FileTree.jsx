import React, { useState } from "react";
import { ChevronDown, ChevronRight, File, Folder } from "lucide-react";
import { fmtBytes } from "../lib/format";

/** entries: [{path,type,bytes}] flat; renders a collapsible tree. */
export default function FileTree({ entries, selected, onSelect }) {
  const [closed, setClosed] = useState({});
  const byParent = {};
  for (const e of entries) {
    const parent = e.path.includes("/") ? e.path.slice(0, e.path.lastIndexOf("/")) : "";
    (byParent[parent] = byParent[parent] || []).push(e);
  }
  const render = (parent, depth) =>
    (byParent[parent] || []).map((e) => {
      const name = e.path.slice(e.path.lastIndexOf("/") + 1);
      if (e.type === "dir") {
        const isClosed = closed[e.path];
        return (
          <div key={e.path}>
            <div className="node dir" style={{ paddingLeft: 6 + depth * 14 }} onClick={() => setClosed({ ...closed, [e.path]: !isClosed })}>
              {isClosed ? <ChevronRight size={13} /> : <ChevronDown size={13} />}
              <Folder size={13} /> {name}
            </div>
            {!isClosed && render(e.path, depth + 1)}
          </div>
        );
      }
      return (
        <div key={e.path} className={`node ${selected === e.path ? "active" : ""}`} style={{ paddingLeft: 24 + depth * 14 }} onClick={() => onSelect && onSelect(e.path)} title={`${e.path} · ${fmtBytes(e.bytes)}`}>
          <File size={13} /> {name}
        </div>
      );
    });
  if (!entries || !entries.length) return <div className="faint small">empty</div>;
  return <div className="tree">{render("", 0)}</div>;
}
