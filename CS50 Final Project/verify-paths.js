// verify-paths.js
import path from "path";
import { fileURLToPath } from "url";
import { dirname } from "path";
import fs from "fs";

// Get current file's directory
const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

// Use absolute paths
const basePath = process.cwd();

const paths = {
  "@": path.resolve(basePath, "app/static/js"),
  "@components": path.resolve(basePath, "app/static/js/components"),
  "@utils": path.resolve(basePath, "app/static/js/utils"),
  "@src": path.resolve(basePath, "app/static/js/src"),
  "@artifacts": path.resolve(basePath, "app/dita/artifacts"),
};

console.log("Verifying paths...\n");

Object.entries(paths).forEach(([alias, fullPath]) => {
  console.log(`Checking ${alias}...`);
  console.log(`Full path: ${fullPath}`);
  console.log(`Exists: ${fs.existsSync(fullPath)}`);
  if (fs.existsSync(fullPath)) {
    console.log("Contents:", fs.readdirSync(fullPath));
  }
  console.log("\n");
});

// Also check specific artifact path
const brownianPath = path.resolve(
  basePath,
  "app/dita/artifacts/components/brownian.jsx",
);
console.log("Checking Brownian component...");
console.log(`Path: ${brownianPath}`);
console.log(`Exists: ${fs.existsSync(brownianPath)}`);
if (fs.existsSync(brownianPath)) {
  console.log("File size:", fs.statSync(brownianPath).size, "bytes");
  // Try to read the file
  try {
    const content = fs.readFileSync(brownianPath, "utf8");
    console.log("File can be read successfully");
    console.log("First 100 characters:", content.substring(0, 100));
  } catch (e) {
    console.error("Error reading file:", e);
  }
}
