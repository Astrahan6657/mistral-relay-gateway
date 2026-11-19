param(
    [string]$TargetTriple = ""
)

$ErrorActionPreference = "Stop"
$RootDir = Split-Path -Parent $PSScriptRoot

if (-not $TargetTriple) {
    $hostLine = (& rustc -vV | Select-String "^host:").ToString()
    $TargetTriple = ($hostLine -split "\s+")[1].Trim()
}

$OutDir = Join-Path $RootDir "desktop\src-tauri\binaries"
$WorkDir = Join-Path ([System.IO.Path]::GetTempPath()) ("mrg-pyi-" + [System.Guid]::NewGuid().ToString("N"))
New-Item -ItemType Directory -Force -Path $OutDir | Out-Null
New-Item -ItemType Directory -Force -Path $WorkDir | Out-Null

try {
    Write-Host "==> Building standalone Sidecar for target: $TargetTriple"
    Push-Location $RootDir
    uv sync
    $staticSrc = Join-Path $RootDir "src\mistral_relay_gateway\static"
    $entry = Join-Path $RootDir "scripts\sidecar_entry.py"

    uv run --with pyinstaller pyinstaller `
      --onefile `
      --clean `
      --noconfirm `
      --name "mistral-relay-gateway-$TargetTriple" `
      --distpath $OutDir `
      --workpath $WorkDir `
      --specpath $WorkDir `
      --add-data "$staticSrc;mistral_relay_gateway/static" `
      --hidden-import uvicorn.logging `
      --hidden-import uvicorn.loops.auto `
      --hidden-import uvicorn.loops.asyncio `
      --hidden-import uvicorn.protocols.http.auto `
      --hidden-import uvicorn.protocols.http.h11_impl `
      --hidden-import uvicorn.protocols.http.httptools_impl `
      --hidden-import uvicorn.protocols.websockets.auto `
      --hidden-import uvicorn.lifespan.on `
      --hidden-import uvicorn.lifespan.off `
      $entry
    Pop-Location
    Get-ChildItem $OutDir
    Write-Host "==> Sidecar ready."
} finally {
    Remove-Item -Recurse -Force $WorkDir -ErrorAction SilentlyContinue
}
