import React, { createContext, useCallback, useContext, useEffect, useState } from "react";
import { api, getToken, setToken } from "./api";

const Ctx = createContext(null);

export function AuthProvider({ children }) {
  const [user, setUser] = useState(null);
  const [ready, setReady] = useState(false);
  // what an admin grants you can change while you are signed in, so this is worth
  // re-reading rather than trusting what the session started with
  const refresh = useCallback(
    () =>
      api("/auth/me")
        .then(setUser)
        .catch(() => {}),
    []
  );
  useEffect(() => {
    if (!getToken()) {
      setReady(true);
      return;
    }
    api("/auth/me")
      .then(setUser)
      .catch(() => setToken(null))
      .finally(() => setReady(true));
  }, []);
  const login = useCallback(async (username, password) => {
    const data = await api("/auth/login", { method: "POST", body: { username, password } });
    setToken(data.token);
    setUser(data.user);
    return data.user;
  }, []);
  const logout = useCallback(() => {
    setToken(null);
    setUser(null);
  }, []);
  return <Ctx.Provider value={{ user, ready, login, logout, refresh, isTeacher: user?.role === "teacher" || user?.role === "admin", isAdmin: user?.role === "admin" }}>{children}</Ctx.Provider>;
}

export const useAuth = () => useContext(Ctx);
