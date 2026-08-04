param([string]$Python = "python")

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
Push-Location $ProjectRoot
try {
    & $Python -m compileall -q rid_fusion
    & $Python -m pytest
    & $Python -m rid_fusion.desktop_api selftest
    powershell.exe -NoProfile -ExecutionPolicy Bypass -File desktop/Run-RIDFusion.ps1 -ValidateOnly
}
finally {
    Pop-Location
}
