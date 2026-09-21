/** Translations for every label in the app.
 *
 *  Words live in src/i18n/locales/<language>/<namespace>.json — one folder per
 *  language, one file per part of the app — and are looked up as
 *  `t("namespace.key")`. To add a language, copy the `en` folder to a new code
 *  (`ko`, `fr`…), translate the values and set `language.name` in its
 *  common.json; it appears in the language menu without any code change. A key
 *  missing from a language falls back to English, then to the key itself.
 *
 *  Values can carry {placeholders}: t("home.runs", { n: 3 }) with "{n} runs".
 *  A count picks a plural form when the file has one: with { count: 1 } the key
 *  `x` is looked up as `x_one`, `x_other` (and `x_zero` for none) before `x`. */
import React, { createContext, useCallback, useContext, useEffect, useMemo, useState } from "react";

const FILES = import.meta.glob("./locales/*/*.json", { eager: true, import: "default" });
const FALLBACK = "en";
const STORE = "lang";

// { en: { common: {...}, topbar: {...} }, ja: {...} }
const DICT = {};
for (const [path, messages] of Object.entries(FILES)) {
  const [, lang, ns] = path.match(/\.\/locales\/([^/]+)\/([^/]+)\.json$/);
  (DICT[lang] ||= {})[ns] = messages;
}

/** Every language there is a folder for, English first, named in its own words. */
export const LANGUAGES = Object.keys(DICT)
  .sort((a, b) => (a === FALLBACK ? -1 : b === FALLBACK ? 1 : a.localeCompare(b)))
  .map((code) => ({ code, name: DICT[code]?.common?.language?.name || code }));

const lookup = (lang, key) => {
  let node = DICT[lang];
  for (const part of key.split(".")) {
    if (node == null || typeof node !== "object") return undefined;
    node = node[part];
  }
  return typeof node === "string" ? node : undefined;
};

const warned = new Set();
function translate(lang, key, vars) {
  const plural = vars && typeof vars.count === "number" ? pluralKeys(lang, key, vars.count) : [];
  let text;
  for (const k of [...plural, key]) {
    text = lookup(lang, k) ?? lookup(FALLBACK, k);
    if (text !== undefined) break;
  }
  if (text === undefined) {
    if (import.meta.env.DEV && !warned.has(key)) {
      warned.add(key);
      console.warn(`[i18n] no text for "${key}"`);
    }
    text = key;
  }
  return vars ? text.replace(/\{(\w+)\}/g, (m, name) => (vars[name] !== undefined && vars[name] !== null ? String(vars[name]) : m)) : text;
}

function pluralKeys(lang, key, count) {
  const keys = count === 0 ? [`${key}_zero`] : [];
  let form = "other";
  try {
    form = new Intl.PluralRules(lang).select(count);
  } catch {
    /* an unknown language code: "other" */
  }
  return [...keys, `${key}_${form}`, `${key}_other`];
}

function initialLanguage() {
  const known = (code) => code && DICT[code] && code;
  // a wall display is set by its URL (?lang=ja), so each screen can differ
  try {
    const url = new URLSearchParams(window.location.search).get("lang");
    if (known(url)) return url;
  } catch {
    /* no location */
  }
  try {
    const saved = window.localStorage.getItem(STORE);
    if (known(saved)) return saved;
  } catch {
    /* storage blocked */
  }
  for (const nav of navigator.languages || [navigator.language]) {
    const base = (nav || "").toLowerCase().split("-")[0];
    if (known(base)) return base;
  }
  return FALLBACK;
}

// the language in use, for code outside React (formatters, chart titles); it is
// always read during a render of a component that uses useT, so it is current
let current = initialLanguage();

/** Translate outside a component. Inside one, use useT so it re-renders. */
export const t = (key, vars) => translate(current, key, vars);
export const currentLanguage = () => current;

const I18nContext = createContext({ lang: current, setLang: () => {}, t });

export function I18nProvider({ children }) {
  const [lang, setLangState] = useState(current);
  const setLang = useCallback((code) => {
    if (!DICT[code]) return;
    current = code;
    try {
      window.localStorage.setItem(STORE, code);
    } catch {
      /* storage blocked: the choice lasts until reload */
    }
    setLangState(code);
  }, []);
  useEffect(() => {
    document.documentElement.lang = lang;
  }, [lang]);
  current = lang;
  const value = useMemo(() => ({ lang, setLang, t: (key, vars) => translate(lang, key, vars), languages: LANGUAGES }), [lang, setLang]);
  return <I18nContext.Provider value={value}>{children}</I18nContext.Provider>;
}

/** { t, lang, setLang, languages } */
export const useI18n = () => useContext(I18nContext);

/** The translate function; the component re-renders when the language changes. */
export const useT = () => useContext(I18nContext).t;

/** A menu of the languages there are folders for. */
export function LanguageSelect({ className = "langselect" }) {
  const { lang, setLang, t: tr } = useI18n();
  if (LANGUAGES.length < 2) return null;
  return (
    <select className={className} value={lang} onChange={(e) => setLang(e.target.value)} aria-label={tr("common.language.choose")} title={tr("common.language.choose")}>
      {LANGUAGES.map((l) => (
        <option key={l.code} value={l.code}>
          {l.name}
        </option>
      ))}
    </select>
  );
}
