import { defineConfig, loadEnv } from "vite";
import react from "@vitejs/plugin-react";
import path from "path";

export default defineConfig(({ command, mode }) => {
  const env = loadEnv(mode, process.cwd());

  return {
    plugins: [react()],
    root: path.resolve(__dirname, "app/static"),
    base: "/static/dist/",

    build: {
      outDir: path.resolve(__dirname, "app/static/dist"),
      emptyOutDir: true,
      assetsDir: "",
      rollupOptions: {
        input: {
          academic: path.resolve(__dirname, "app/static/js/src/academic.jsx"),
          app: path.resolve(__dirname, "app/static/js/src/app.jsx"),
          components: path.resolve(
            __dirname,
            "app/static/js/utils/componentRegistry.js",
          ),
          webComponents: path.resolve(
            __dirname,
            "app/static/js/utils/ReactWebComponentWrapper.js",
          ),
        },
        external: [
          "react",
          "react-dom",
          "recharts",
          "lucide-react",
          "@radix-ui/react-switch",
          "@radix-ui/react-label",
        ],
        output: {
          entryFileNames: `[name].js`,
          chunkFileNames: `chunks/[name].[hash].js`,
          assetFileNames: `assets/[name].[ext]`,
          globals: {
            react: "React",
            "react-dom": "ReactDOM",
            recharts: "Recharts",
            "lucide-react": "Lucide",
          },
        },
      },
    },

    resolve: {
      alias: {
        "@": path.resolve(__dirname, "app/static/js"),
        "@components": path.resolve(__dirname, "app/static/js/components"),
        "@utils": path.resolve(__dirname, "app/static/js/utils"),
        "@src": path.resolve(__dirname, "app/static/js/src"),
        "@artifacts": path.resolve(__dirname, "app/dita/artifacts/components"),
      },
    },

    optimizeDeps: {
      include: ["react", "react-dom", "recharts", "lucide-react"],
    },

    server: {
      watch: {
        ignored: ["!**/app/dita/artifacts/**"],
      },
    },
  };
});
