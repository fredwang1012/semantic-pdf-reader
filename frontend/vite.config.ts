import { defineConfig } from "vitest/config";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";

// READER_API overrides the backend address when port 8000 is taken (e.g. READER_API=http://localhost:8001)
const api = process.env.READER_API ?? "http://localhost:8000";

export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: {
    port: 5173,
    proxy: {
      "/api": { target: api, changeOrigin: true },
    },
  },
  test: { environment: "node", include: ["test/**/*.test.{ts,tsx}"] },
});
