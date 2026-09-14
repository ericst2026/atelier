const TOKEN_KEY = "atelier.token";

export class ApiError extends Error {
  constructor(message, status) {
    super(message);
    this.status = status;
  }
}

export const getToken = () => localStorage.getItem(TOKEN_KEY);
export const setToken = (t) => (t ? localStorage.setItem(TOKEN_KEY, t) : localStorage.removeItem(TOKEN_KEY));

export async function api(path, { method = "GET", body, form, signal } = {}) {
  const headers = {};
  const token = getToken();
  if (token) headers.Authorization = `Bearer ${token}`;
  let payload;
  if (form) payload = form;
  else if (body !== undefined) {
    headers["Content-Type"] = "application/json";
    payload = JSON.stringify(body);
  }
  const res = await fetch(`/api${path}`, { method, headers, body: payload, signal });
  if (res.status === 204) return null;
  const text = await res.text();
  let data = null;
  try {
    data = text ? JSON.parse(text) : null;
  } catch {
    data = text;
  }
  if (!res.ok) {
    const detail = data && data.detail ? (typeof data.detail === "string" ? data.detail : JSON.stringify(data.detail)) : res.statusText;
    throw new ApiError(detail, res.status);
  }
  return data;
}

/** URL for links the browser opens directly (downloads, iframes). */
export const fileUrl = (path) => `/api${path}${path.includes("?") ? "&" : "?"}token=${encodeURIComponent(getToken() || "")}`;
export const wsUrl = (path, withToken = true) => {
  const proto = location.protocol === "https:" ? "wss" : "ws";
  const q = withToken ? `?token=${encodeURIComponent(getToken() || "")}` : "";
  return `${proto}://${location.host}${path}${q}`;
};
