import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Dev server proxies API, websockets and Grafana to the local backend so the app
// runs without nginx while developing. Production is served by nginx (deploy/nginx).
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      "/api": { target: "http://localhost", changeOrigin: true },
      "/ws": { target: "ws://localhost", ws: true },
      "/metrics": { target: "http://localhost" },
      "/grafana": { target: "http://localhost", changeOrigin: true },
    },
  },
  build: { outDir: "dist", sourcemap: false, chunkSizeWarningLimit: 4000 },
});
