param([string]$Python = "python")

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
Push-Location $ProjectRoot
try {
    & $Python -m pip install -e ".[dev]"
    & $Python -m PyInstaller --noconfirm --clean --onedir `
        --name RIDFusionBackend `
        --console `
        --collect-submodules rid_fusion `
        packaging/backend_entry.py
    Write-Host "Backend created at dist/RIDFusionBackend/"
}
finally {
    Pop-Location
}
