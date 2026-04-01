import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      "/api": {
        target: "http://localhost:8000",
        changeOrigin: true,
        timeout: 600_000, // 10 min – model inference can be slow
        configure: (proxy) => {
          // Silence EPIPE / ECONNRESET errors on large uploads
          proxy.on("error", (err, _req, res) => {
            console.warn("Proxy error:", err.message);
            if (!res.headersSent) {
              (res as any).writeHead?.(502, { "Content-Type": "application/json" });
            }
            (res as any).end?.(JSON.stringify({ detail: `Proxy error: ${err.message}` }));
          });
          // Remove default max payload limit
          proxy.on("proxyReq", (proxyReq) => {
            proxyReq.setHeader("Connection", "keep-alive");
          });
        },
      },
    },
  },
});
