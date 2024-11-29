import { defineConfig, loadEnv } from "vite";
import react from "@vitejs/plugin-react";
import path from "path";

export default defineConfig(({ command, mode }) => {
  const env = loadEnv(mode, process.cwd());

  return {
    plugins: [react()],
    root: path.resolve(__dirname, "app/static"),
    base: "/static/dist/",

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
      outDir: path.resolve(__dirname, "app/static/dist"),
      emptyOutDir: true,
      assetsDir: "",
      rollupOptions: {
        input: {
          academic: path.resolve(__dirname, "app/static/js/src/academic.jsx"),
          app: path.resolve(__dirname, "app/static/js/src/app.jsx"),
        },
        output: {
          entryFileNames: `[name].js`,
          chunkFileNames: `chunks/[name].[hash].js`,
          assetFileNames: `assets/[name].[ext]`,
        },
      },
    },

    resolve: {
      alias: {
        "@": path.resolve(__dirname, "app/static/js"),
        "@components": path.resolve(__dirname, "app/static/js/components"),
        "@utils": path.resolve(__dirname, "app/static/js/utils"),
        "@src": path.resolve(__dirname, "app/static/js/src"),
      },
    },
  };
});
