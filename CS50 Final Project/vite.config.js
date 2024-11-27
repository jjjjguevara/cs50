import { defineConfig, loadEnv } from "vite";
import react from "@vitejs/plugin-react";
import path from "path";

export default defineConfig(({ command, mode }) => {
  // Load env file based on `mode` in the current working directory.
  const env = loadEnv(mode, process.cwd());

  return {
    plugins: [react()],
    root: path.resolve(__dirname, "app/static"),
    base: "/",
    server: {
      port: 5173,
      proxy:
        command === "serve"
          ? {
              "/api": {
                target: env.VITE_API_BASE_URL,
                changeOrigin: true,
                secure: false,
              },
            }
          : undefined,
    },
    build: {
      outDir: "../dist",
      assetsDir: "static",
    },
  };
});
