import { execFileSync } from "node:child_process";
import { existsSync, readFileSync } from "node:fs";
import { resolve } from "node:path";
import { fileURLToPath } from "node:url";

const repo = resolve(fileURLToPath(new URL("..", import.meta.url)));
const read = (path) => readFileSync(resolve(repo, path), "utf8");
const json = (path) => JSON.parse(read(path));
const failures = [];

const rootPackage = json("package.json");
const expectedVersion = rootPackage.version;
const versions = new Map([
  ["package.json", rootPackage.version],
  ["app/package.json", json("app/package.json").version],
  ["tauri/package.json", json("tauri/package.json").version],
  ["tauri/src-tauri/tauri.conf.json", json("tauri/src-tauri/tauri.conf.json").version],
  [
    "tauri/src-tauri/Cargo.toml",
    read("tauri/src-tauri/Cargo.toml").match(/^version = "([^"]+)"/m)?.[1],
  ],
  ["backend/__init__.py", read("backend/__init__.py").match(/__version__ = "([^"]+)"/)?.[1]],
  ["backend/pyproject.toml", read("backend/pyproject.toml").match(/^version = "([^"]+)"/m)?.[1]],
]);

for (const [source, version] of versions) {
  if (version !== expectedVersion) {
    failures.push(`${source} reports ${version ?? "no version"}; expected ${expectedVersion}`);
  }
}

if (!expectedVersion.includes("-alpha.")) {
  failures.push(`release version ${expectedVersion} is not an alpha version`);
}

const workspaces = JSON.stringify(rootPackage.workspaces);
if (workspaces !== JSON.stringify(["app", "tauri"])) {
  failures.push(`active workspaces must be app + tauri; found ${workspaces}`);
}

if (rootPackage.scripts["dev:server"] !== "uvicorn backend.main:app --reload --port 17494") {
  failures.push("dev:server must use Diarix's isolated port 17494");
}

const tauriConfig = json("tauri/src-tauri/tauri.conf.json");
if (tauriConfig.productName !== "Diarix" || tauriConfig.identifier !== "com.diarix.app") {
  failures.push("Tauri product name or bundle identifier is not the Diarix identity");
}
if (tauriConfig.app?.windows?.some((window) => window.devtools !== false)) {
  failures.push("production Tauri windows must ship with devtools disabled");
}

for (const required of [
  "README.md",
  "CHANGELOG.md",
  "LICENSE",
  "THIRD_PARTY_NOTICES.md",
  "docs/ALPHA_RELEASE.md",
  "docs/ARCHITECTURE.md",
  "docs/DEVELOPMENT.md",
  "installer/build-diarix-setup.ps1",
]) {
  try {
    read(required);
  } catch {
    failures.push(`required release file is missing: ${required}`);
  }
}

const tracked = execFileSync("git", ["ls-files", "-z"], { cwd: repo })
  .toString("utf8")
  .split("\0")
  .filter((path) => path && existsSync(resolve(repo, path)));
const forbiddenTracked = tracked.filter(
  (path) =>
    /(^|\/)(node_modules|target|dist|build|__pycache__|\.pytest_cache|\.ruff_cache)(\/|$)/.test(path) ||
    /^installer\/build-logs\//.test(path) ||
    (/^data\//.test(path) && path !== "data/.gitkeep") ||
    /^(landing|web)\//.test(path) ||
    /^backend\/(mcp_server|mcp_shim)\//.test(path) ||
    path === ".mcp.json",
);
if (forbiddenTracked.length > 0) {
  failures.push(`generated or user data is tracked:\n  ${forbiddenTracked.join("\n  ")}`);
}

for (const workflow of [".github/workflows/build-windows.yml", ".github/workflows/release.yml"]) {
  if (/voicebox v__VERSION__/i.test(read(workflow))) {
    failures.push(`${workflow} still publishes releases under the Voicebox name`);
  }
}

for (const service of ["backend/services/cuda.py", "backend/services/rocm.py"]) {
  const source = read(service);
  if (!source.includes("https://github.com/Yannam-Builds/Diarix/releases/download")) {
    failures.push(`${service} must download optional runtimes from Diarix releases`);
  }
  if (source.includes("https://github.com/jamiepine/voicebox/releases/download")) {
    failures.push(`${service} still downloads optional runtimes from Voicebox releases`);
  }
}

const installerScript = read("installer/build-diarix-setup.ps1");
if (!/\[int\]\$BuildCpuPercent = 25/.test(installerScript)) {
  failures.push("installer builds must default to 25% of logical CPU workers");
}
if (!/-Destination \(Join-Path \$PayloadDir 'diarix-server\.exe'\)/.test(installerScript)) {
  failures.push("custom installer payloads must use Tauri's runtime sidecar filename");
}

if (failures.length > 0) {
  console.error("Diarix alpha verification failed:\n");
  for (const failure of failures) console.error(`- ${failure}`);
  process.exit(1);
}

console.log(`Diarix ${expectedVersion} alpha contracts verified (${tracked.length} tracked files).`);
