// Checks the translations: every t("…") key used in src/ exists in English, every
// language has the same keys as English, and JSX text that was left untranslated
// is listed. Run: node scripts/check-i18n.mjs [file-or-folder …]
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const SRC = path.join(ROOT, "src");
const LOCALES = path.join(SRC, "i18n", "locales");

const flat = (obj, prefix = "", out = {}) => {
  for (const [k, v] of Object.entries(obj)) {
    if (v && typeof v === "object") flat(v, `${prefix}${k}.`, out);
    else out[`${prefix}${k}`] = v;
  }
  return out;
};
const langs = {};
for (const lang of fs.readdirSync(LOCALES)) {
  langs[lang] = {};
  for (const f of fs.readdirSync(path.join(LOCALES, lang))) {
    if (!f.endsWith(".json")) continue;
    const ns = f.replace(/\.json$/, "");
    try {
      Object.assign(langs[lang], flat(JSON.parse(fs.readFileSync(path.join(LOCALES, lang, f), "utf8")), `${ns}.`));
    } catch (e) {
      console.log(`BAD JSON ${lang}/${f}: ${e.message}`);
      process.exitCode = 1;
    }
  }
}
const en = langs.en || {};
let problems = 0;
const report = (msg) => {
  problems++;
  console.log(msg);
};
for (const [lang, keys] of Object.entries(langs)) {
  if (lang === "en") continue;
  for (const k of Object.keys(en)) if (!(k in keys)) report(`missing in ${lang}: ${k}`);
  for (const k of Object.keys(keys)) if (!(k in en)) report(`only in ${lang}: ${k}`);
}

const walk = (p) => (fs.statSync(p).isDirectory() ? fs.readdirSync(p).flatMap((f) => walk(path.join(p, f))) : /\.(jsx?|mjs)$/.test(p) ? [p] : []);
const targets = process.argv.slice(2).map((a) => path.resolve(a));
const files = (targets.length ? targets : [SRC]).flatMap(walk).filter((f) => !f.includes(`${path.sep}i18n${path.sep}`));
// a key is known when it, or a plural form of it, is in English
const known = (k) => k in en || Object.keys(en).some((e) => e.startsWith(`${k}_`));
for (const f of files) {
  const src = fs.readFileSync(f, "utf8");
  const rel = path.relative(ROOT, f);
  for (const m of src.matchAll(/\bt\(\s*["']([\w.-]+)["']/g)) if (!known(m[1])) report(`${rel}: unknown key ${m[1]}`);
  if (!f.endsWith(".jsx")) continue;
  // text between tags that has letters in it and is not an expression
  src.split("\n").forEach((line, i) => {
    const s = line.trim();
    if (!s || s.startsWith("//") || s.startsWith("*") || s.startsWith("/*") || s.startsWith("import ")) return;
    const texts = [...s.matchAll(/>([^<>{}]*[A-Za-z]{2,}[^<>{}]*)</g)].map((m) => m[1].trim());
    // a line of words between tags; a lone lowercase identifier is a boolean prop (connectNulls)
    if (/^[A-Za-z][^<>{}=;()]*[A-Za-z.?!:…]$/.test(s) && !/^(return|const|let|if|else|export|function|case|default)\b/.test(s) && !/^[a-z][A-Za-z0-9]*$/.test(s)) texts.push(s);
    for (const t of texts) if (t) report(`${rel}:${i + 1}: text? ${t.slice(0, 80)}`);
    for (const m of s.matchAll(/\b(title|placeholder|aria-label|alt|label)="([^"]*[A-Za-z]{2,}[^"]*)"/g)) report(`${rel}:${i + 1}: ${m[1]}? ${m[2].slice(0, 80)}`);
  });
}
console.log(problems ? `${problems} problem(s)` : "i18n: all good");
if (problems) process.exitCode = 1;
