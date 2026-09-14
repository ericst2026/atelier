import React, { createContext, useCallback, useContext, useState } from "react";

const Ctx = createContext(() => {});

export function ToastProvider({ children }) {
  const [toast, setToast] = useState(null);
  const show = useCallback((message, ok = false) => {
    setToast({ message, ok });
    setTimeout(() => setToast(null), 4500);
  }, []);
  return (
    <Ctx.Provider value={show}>
      {children}
      {toast && <div className={`toast ${toast.ok ? "ok" : ""}`}>{toast.message}</div>}
    </Ctx.Provider>
  );
}

export const useToast = () => useContext(Ctx);
