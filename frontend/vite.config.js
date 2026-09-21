import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Dev server proxies API, websockets and Grafana to the local backend so the app
// runs without nginx while developing. Production is served by nginx (deploy/nginx).
export default defineConfig({
  plugins: [react()],
  server: {
    // bind to this machine's LAN address only, so the dev server is reachable from
    // the network but not offered on the VPN or the docker/WSL interfaces. Set
    // VITE_DEV_HOST when the address changes (vite exits if it cannot bind it);
    // note that with a fixed address, localhost is no longer served.
    host: process.env.VITE_DEV_HOST || "192.168.170.101",
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
