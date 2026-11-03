/// <reference types="vitest/config" />
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

const api = "http://127.0.0.1:8000";

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      "/api": api,
      "/metrics": api,
      "/ws": { target: api, ws: true },
    },
  },
  build: { outDir: "dist", sourcemap: false },
  test: { environment: "jsdom", setupFiles: ["./src/test-setup.ts"], globals: true },
});
