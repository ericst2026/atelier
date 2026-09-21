import React from "react";
import { BRAND, brandName } from "../brand";
import { useI18n } from "../i18n";

/** The course logo and name. Both come from src/brand.js. */
export default function Brand({ size = 24, name = true, className = "" }) {
  const { lang } = useI18n();
  return (
    <span className={`brand ${className}`}>
      <img src={BRAND.logo} alt={name ? "" : brandName(lang)} width={size} height={size} className="brandlogo" />
      {name && <span className="brandname">{brandName(lang)}</span>}
    </span>
  );
}
