import React from "react";

/** Small markdown renderer (headings, paragraphs, lists, fenced code, inline code/bold/italic/links). No dependency, offline-safe. */
function inline(text, key) {
  const parts = [];
  const re = /(`[^`]+`)|(\*\*[^*]+\*\*)|(\*[^*]+\*)|(\[[^\]]+\]\([^)]+\))/g;
  let last = 0;
  let m;
  let i = 0;
  while ((m = re.exec(text))) {
    if (m.index > last) parts.push(text.slice(last, m.index));
    const s = m[0];
    if (s.startsWith("`")) parts.push(<code key={`${key}-${i}`}>{s.slice(1, -1)}</code>);
    else if (s.startsWith("**")) parts.push(<strong key={`${key}-${i}`}>{s.slice(2, -2)}</strong>);
    else if (s.startsWith("*")) parts.push(<em key={`${key}-${i}`}>{s.slice(1, -1)}</em>);
    else {
      const mm = /\[([^\]]+)\]\(([^)]+)\)/.exec(s);
      parts.push(
        <a key={`${key}-${i}`} href={mm[2]} target="_blank" rel="noreferrer">
          {mm[1]}
        </a>
      );
    }
    last = m.index + s.length;
    i += 1;
  }
  if (last < text.length) parts.push(text.slice(last));
  return parts;
}

export default function Markdown({ text, className = "" }) {
  if (!text) return null;
  const lines = text.replace(/\r/g, "").split("\n");
  const out = [];
  let i = 0;
  let k = 0;
  while (i < lines.length) {
    const line = lines[i];
    if (line.startsWith("```")) {
      const buf = [];
      i += 1;
      while (i < lines.length && !lines[i].startsWith("```")) buf.push(lines[i++]);
      i += 1;
      out.push(
        <pre key={k++}>
          <code>{buf.join("\n")}</code>
        </pre>
      );
      continue;
    }
    const h = /^(#{1,3})\s+(.*)$/.exec(line);
    if (h) {
      const Tag = `h${h[1].length}`;
      out.push(<Tag key={k++}>{inline(h[2], k)}</Tag>);
      i += 1;
      continue;
    }
    if (/^\s*[-*]\s+/.test(line) || /^\s*\d+\.\s+/.test(line)) {
      const ordered = /^\s*\d+\./.test(line);
      const items = [];
      while (i < lines.length && (/^\s*[-*]\s+/.test(lines[i]) || /^\s*\d+\.\s+/.test(lines[i]))) {
        items.push(lines[i].replace(/^\s*([-*]|\d+\.)\s+/, ""));
        i += 1;
      }
      const L = ordered ? "ol" : "ul";
      out.push(<L key={k++}>{items.map((it, j) => <li key={j}>{inline(it, `${k}-${j}`)}</li>)}</L>);
      continue;
    }
    if (!line.trim()) {
      i += 1;
      continue;
    }
    const buf = [line];
    i += 1;
    while (i < lines.length && lines[i].trim() && !/^(#{1,3}\s|```|\s*[-*]\s|\s*\d+\.\s)/.test(lines[i])) buf.push(lines[i++]);
    out.push(<p key={k++}>{inline(buf.join(" "), k)}</p>);
  }
  return <div className={`md ${className}`}>{out}</div>;
}
