/** The course's name and logo, in one place.
 *
 *  To change the logo, replace frontend/public/brand/logo.svg with any image —
 *  keep the name, or point `logo` at the new file. To show a different name in
 *  one language, add it to `names` (the language code as the key). */
export const BRAND = {
  name: "LLM Course",
  names: {},
  logo: "/brand/logo.svg",
};

/** The course name as it reads in this language. */
export const brandName = (lang) => BRAND.names[lang] || BRAND.name;
