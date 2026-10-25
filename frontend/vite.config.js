import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      "/results": "http://localhost:8000",
      "/runs": "http://localhost:8000",
      "/reset": "http://localhost:8000",
      "/step": "http://localhost:8000",
      "/state": "http://localhost:8000",
    },
  },
});
