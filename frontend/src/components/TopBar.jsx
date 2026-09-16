import React from "react";
import { Link, NavLink } from "react-router-dom";
import { LogOut } from "lucide-react";
import { useAuth } from "../lib/auth";
import GpuStrip from "./GpuStrip";

export default function TopBar() {
  const { user, logout, isTeacher, isAdmin } = useAuth();
  return (
    <header className="topbar">
      <Link to="/" className="brand">
        Ate<b>lier</b>
      </Link>
      <nav className="topnav">
        <NavLink to="/" end>
          Experiments
        </NavLink>
        {isTeacher && <NavLink to="/teacher">Class</NavLink>}
        {isAdmin && <NavLink to="/admin">Accounts</NavLink>}
        <NavLink to="/display/1" target="_blank">
          Displays
        </NavLink>
      </nav>
      <span className="spacer" />
      <GpuStrip />
      {user && (
        <div className="userchip">
          <Link to="/account" title="Your account and password">
            {user.name || user.username}
          </Link>
          <span className={`role ${user.role}`}>{user.role}</span>
          <button className="btn sm ghost" onClick={logout} title="Sign out">
            <LogOut size={14} />
          </button>
        </div>
      )}
    </header>
  );
}
