# Languages and branding

## Changing the name or the logo

- **Logo:** replace `frontend/public/brand/logo.svg` with your own image. Any format works. If the file name or extension changes, update `logo` in `frontend/src/brand.js`.
- **Name:** edit `name` in `frontend/src/brand.js` (currently "LLM Course"). To use a different name in one language, add it to `names`, e.g. `names: { ja: "LLM講座" }`.

The logo appears in the top bar, on the sign-in and sign-up pages, and as the browser-tab icon. The name is also the page title.

## Editing the words

Every label lives in `locales/<language>/<part>.json`. There is one folder per language and one file per part of the app:

| file | where it shows |
|---|---|
| `common.json` | words used everywhere: Save, Loading…, run states, roles, the language's own name |
| `topbar.json`, `auth.json` | the top bar; sign-in and sign-up |
| `home.json`, `experiment.json`, `params.json`, `run.json`, `results.json` | the experiment list and an experiment's page |
| `class.json`, `teacher.json`, `review.json` | the class panel, the wall settings, marking |
| `display.json`, `charts.json` | the wall screens and the charts |
| `admin.json`, `account.json`, `format.json`, `widgets.json` | accounts, dates and small parts |

Change a value, save, and the dev server (`npm run dev`) shows it at once. Rules:

- Keep the keys (left side) as they are, and change only the text (right side).
- `{name}` in a value is filled in by the app. Keep it, anywhere in the sentence.
- Keys ending in `_one` / `_other` (and `_zero`) are singular/plural forms of one text.
- A key missing from a language shows the English text instead.

After editing, check that every language still has every key:

```
node scripts/check-i18n.mjs
```

## Adding a language

1. Copy `locales/en` to `locales/<code>`, e.g. `locales/ko`.
2. Translate the values, and set `language.name` in its `common.json` to the language's own name (e.g. "한국어").

It appears in the language menu straight away, with no code change. People choose a language from the menu in the top bar or on the sign-in page, and the choice is remembered in that browser. A wall screen can be fixed to a language with its URL, e.g. `/display/3?lang=ja`.

## In code

```jsx
import { useT } from "../i18n";

function Thing({ n }) {
  const t = useT();                        // re-renders when the language changes
  return <p>{t("home.runs", { count: n })}</p>;
}
```

Outside a component, such as in a formatter called while rendering, import `{ t }` from `../i18n` instead.
