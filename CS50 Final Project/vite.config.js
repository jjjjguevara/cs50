import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import path from "path";
import fs from "fs";

// Verify paths before configuration
const verifyPath = (pathToVerify, alias) => {
  const exists = fs.existsSync(pathToVerify);
  console.log(
    `Verifying ${alias}: ${pathToVerify} - ${exists ? "EXISTS" : "MISSING"}`,
  );
  return exists;
};

export default defineConfig(({ command, mode }) => {
  // Verify paths
  const paths = {
    "@": path.resolve(__dirname, "app/static/js"),
    "@components": path.resolve(__dirname, "app/static/js/components"),
    "@utils": path.resolve(__dirname, "app/static/js/utils"),
    "@src": path.resolve(__dirname, "app/static/js/src"),
    "@artifacts": path.resolve(__dirname, "app/dita/artifacts"),
  };

  console.log("\nVerifying paths during Vite configuration:");
  Object.entries(paths).forEach(([alias, fullPath]) => {
    verifyPath(fullPath, alias);
  });

  const brownianPath = path.resolve(
    __dirname,
    "app/dita/artifacts/components/brownian.jsx",
  );
  verifyPath(brownianPath, "brownian.jsx");

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
        },
        external: ["react", "react-dom", "recharts"],
        output: {
          manualChunks: (id) => {
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
          globals: {
            react: "React",
            "react-dom": "ReactDOM",
            recharts: "Recharts",
            "lucide-react": "LucideIcons",
          },
          format: "es",
        },
      },
    },

    resolve: {
      alias: [
        {
          find: "@",
          replacement: paths["@"],
        },
        {
          find: "@components",
          replacement: paths["@components"],
        },
        {
          find: "@utils",
          replacement: paths["@utils"],
        },
        {
          find: "@src",
          replacement: paths["@src"],
        },
        {
          find: "@artifacts",
          replacement: paths["@artifacts"],
        },
      ],
      extensions: [".js", ".jsx", ".ts", ".tsx"],
    },

    optimizeDeps: {
      include: ["react", "react-dom", "recharts"],
      exclude: [],
    },
  };
});
