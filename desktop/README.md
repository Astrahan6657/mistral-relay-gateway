# Optional Desktop Tray Shell (Tauri v2)

This thin Tauri v2 wrapper turns `mistral-relay-gateway` into a self-contained desktop tray app:

- Bundles the Python core proxy as a standalone PyInstaller Sidecar (`externalBin`) — no Python/uv required on end-user machines
- Runs the core proxy automatically on `http://127.0.0.1:8790`
- Persists settings & accounts to `~/.config/mistral-relay-gateway/{config,accounts}.json`
- Logs core stdout/stderr to `~/.config/mistral-relay-gateway/core.log` (accessible from the tray menu: **查看核心日志**)
- Closing the window minimizes to the system tray while keeping the proxy alive

## Prerequisites

- Python 3.11+ & `uv`
- Node.js 20+ & `npm`
- Rust toolchain (`rustup` / `cargo`)
- On Linux: `libwebkit2gtk-4.1-dev libappindicator3-dev librsvg2-dev patchelf`

## Step 1: Build the Standalone Sidecar Binary

`desktop/src-tauri/binaries/` is gitignored; generate the sidecar binary for your platform first:

- **macOS / Linux**:
  ```bash
  bash scripts/build_sidecar.sh
  ```
- **Windows (PowerShell)**:
  ```powershell
  .\scripts\build_sidecar.ps1
  ```

## Step 2: Build or Run the Desktop App

```bash
cd desktop
npm install
npm run dev     # development mode
npm run build   # produces .dmg/.app (macOS), .msi/.exe (Windows), .AppImage/.deb (Linux)
```

*(Note: Pushing a Git tag like `v0.1.0` automatically triggers `.github/workflows/release.yml` to build and upload cross-platform installers to GitHub Releases.)*
