import React from "react";
import Editor, { loader } from "@monaco-editor/react";
import * as monaco from "monaco-editor";
import editorWorker from "monaco-editor/esm/vs/editor/editor.worker?worker";
import jsonWorker from "monaco-editor/esm/vs/language/json/json.worker?worker";

// Bundle Monaco locally: the default loader fetches it from a CDN, which does not exist on an offline LAN.
self.MonacoEnvironment = { getWorker: (_id, label) => (label === "json" ? new jsonWorker() : new editorWorker()) };
loader.config({ monaco });
monaco.editor.defineTheme("atelier", {
  base: "vs-dark",
  inherit: true,
  rules: [
    { token: "comment", foreground: "5f7192", fontStyle: "italic" },
    { token: "keyword", foreground: "b79cff" },
    { token: "string", foreground: "5fd3b8" },
    { token: "number", foreground: "ffd166" },
    { token: "type", foreground: "7cc4ff" },
  ],
  colors: { "editor.background": "#0b1526", "editor.lineHighlightBackground": "#15243b", "editorLineNumber.foreground": "#5f7192", "editor.selectionBackground": "#2a3d5f" },
});

const LANG = { py: "python", js: "javascript", jsx: "javascript", ts: "typescript", json: "json", md: "markdown", yaml: "yaml", yml: "yaml", sh: "shell", txt: "plaintext", jsonl: "plaintext", toml: "ini", cfg: "ini", html: "html", css: "css" };
export const languageFor = (path) => LANG[(path || "").split(".").pop().toLowerCase()] || "plaintext";

export default function CodeEditor({ path, value, onChange, readOnly = false, onSave }) {
  return (
    <Editor
      height="100%"
      theme="atelier"
      language={languageFor(path)}
      value={value ?? ""}
      onChange={(v) => onChange && onChange(v ?? "")}
      onMount={(editor) => {
        editor.addCommand(monaco.KeyMod.CtrlCmd | monaco.KeyCode.KeyS, () => onSave && onSave());
      }}
      options={{ readOnly, minimap: { enabled: false }, fontSize: 13, fontFamily: "ui-monospace, JetBrains Mono, Menlo, Consolas, monospace", scrollBeyondLastLine: false, wordWrap: "on", tabSize: 4, automaticLayout: true }}
    />
  );
}
