import React from "react";
import { Link, NavLink } from "react-router-dom";
import { LogOut } from "lucide-react";
import { LanguageSelect, useT } from "../i18n";
import { useAuth } from "../lib/auth";
import Brand from "./Brand";
import GpuStrip from "./GpuStrip";

export default function TopBar() {
  const { user, logout, isTeacher, isAdmin } = useAuth();
  const t = useT();
  return (
    <header className="topbar">
      <Link to="/" className="brandlink">
        <Brand />
      </Link>
      <nav className="topnav">
        <NavLink to="/" end>
          {t("topbar.experiments")}
        </NavLink>
        {isTeacher && <NavLink to="/teacher">{t("topbar.class")}</NavLink>}
        {isAdmin && <NavLink to="/admin">{t("topbar.accounts")}</NavLink>}
      </nav>
      <span className="spacer" />
      <GpuStrip />
      <LanguageSelect />
      {user && (
        <div className="userchip">
          <Link to="/account" title={t("topbar.accountTitle")}>
            {user.name || user.username}
          </Link>
          <span className={`role ${user.role}`}>{t(`common.role.${user.role}`)}</span>
          <button className="btn sm ghost" onClick={logout} title={t("topbar.signOut")}>
            <LogOut size={14} />
          </button>
        </div>
      )}
    </header>
  );
}
