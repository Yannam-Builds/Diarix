// Mirrors app/src/global.d.ts — the tauri workspace's tsconfig only includes
// tauri/src, so the Window augmentation must be visible here too for
// platform/lifecycle.ts.
interface Window {
  __diarixServerStartedByApp?: boolean;
}
