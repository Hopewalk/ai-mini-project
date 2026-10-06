// Copy the browser files of pnpm-installed packages into app/static/vendor (served by Flask).
import { cpSync, existsSync, mkdirSync, rmSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const root = join(dirname(fileURLToPath(import.meta.url)), "..");
const modules = join(root, "node_modules");
const vendor = join(root, "app", "static", "vendor");

// [package, source path inside the package, destination under vendor/]
const FILES = [
  ["bootstrap", "dist/css/bootstrap.min.css", "bootstrap/bootstrap.min.css"],
  ["bootstrap", "dist/css/bootstrap.min.css.map", "bootstrap/bootstrap.min.css.map"],
  ["bootstrap", "dist/js/bootstrap.bundle.min.js", "bootstrap/bootstrap.bundle.min.js"],
  ["bootstrap", "dist/js/bootstrap.bundle.min.js.map", "bootstrap/bootstrap.bundle.min.js.map"],
  ["leaflet", "dist", "leaflet"], // leaflet.css references images/ relative to itself
  ...["400", "500", "600", "700"].map((w) => ["@fontsource/ibm-plex-sans-thai", `${w}.css`, `ibm-plex-sans-thai/${w}.css`]),
  ["@fontsource/ibm-plex-sans-thai", "files", "ibm-plex-sans-thai/files"],
];

rmSync(vendor, { recursive: true, force: true });
for (const [pkg, src, dest] of FILES) {
  const from = join(modules, pkg, src);
  if (!existsSync(from)) {
    console.error(`missing ${from} — run \`pnpm install\` first`);
    process.exit(1);
  }
  const to = join(vendor, dest);
  mkdirSync(dirname(to), { recursive: true });
  cpSync(from, to, { recursive: true });
}
console.log(`vendor assets -> ${vendor}`);
