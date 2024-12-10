import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import path from "path";

export default defineConfig({
  plugins: [react()],
  root: path.resolve(__dirname, "app/static"),
  base: "/static/dist/",

  resolve: {
    alias: {
      "@": path.resolve(__dirname, "./app/static/js"),
      "@components": path.resolve(__dirname, "./app/static/js/components"),
      "@utils": path.resolve(__dirname, "./app/static/js/utils"),
      "@src": path.resolve(__dirname, "./app/static/js/src"),
      "@artifacts": path.resolve(__dirname, "./app/dita/artifacts"),
      react: path.resolve(__dirname, "node_modules/react"),
      "react-dom": path.resolve(__dirname, "node_modules/react-dom"),
    },
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
        manualChunks: (id) => {
          if (id.includes("node_modules")) {
            return "vendor";
          }
          if (id.includes("/artifacts/components/")) {
            return "artifacts";
          }
          if (id.includes("/components/")) {
            return "components";
          }
          if (id.includes("/utils/")) {
            return "utils";
          }
        },
        entryFileNames: "[name].js",
        chunkFileNames: "js/[name]-[hash].js",
        assetFileNames: "assets/[name]-[hash][extname]",
      },
    },
  },
});
