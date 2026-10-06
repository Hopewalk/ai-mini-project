<#
.SYNOPSIS
  Train all land-appraisal models with live progress (Windows PowerShell).
  Keep this file ASCII-only: Windows PowerShell 5.1 reads BOM-less scripts as ANSI.

.EXAMPLE
  .\scripts\train.ps1                          # train with current data (.env: MODEL_VERSION / LAND_PROVINCES)
  .\scripts\train.ps1 --prepare                # rebuild cells + OSM features, then train
  .\scripts\train.ps1 --provinces 10,90 --version v-test --prepare
  .\scripts\train.ps1 --only regression,bn
  .\scripts\train.ps1 -Load                    # also load results into MongoDB afterwards
#>
param([switch]$Load)

# not "Stop": progress bars are written to stderr, which PowerShell 5.1 would treat as a terminating
# error when output is redirected; failures are detected with $LASTEXITCODE instead
$ErrorActionPreference = "Continue"
Set-Location (Split-Path $PSScriptRoot -Parent)

# terminals opened before Poetry was installed don't see it yet: reload PATH from the user environment
if (-not (Get-Command poetry -ErrorAction SilentlyContinue)) {
    $env:PNPM_HOME = [Environment]::GetEnvironmentVariable("PNPM_HOME", "User")
    $env:Path = [Environment]::ExpandEnvironmentVariables(
        [Environment]::GetEnvironmentVariable("Path", "Machine") + ";" + [Environment]::GetEnvironmentVariable("Path", "User"))
}
if (-not (Get-Command poetry -ErrorAction SilentlyContinue)) {
    throw "poetry not found - install it first (see README.md, section Tooling)"
}

$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONWARNINGS = "ignore"
# PowerShell parses `--only regression,bn` as an array; turn it back into "regression,bn"
$pyArgs = @($args | ForEach-Object { if ($_ -is [array]) { $_ -join "," } else { "$_" } })
poetry run python -m ml.train_all @pyArgs
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

if ($Load) {
    poetry run python -m scripts.load_mongo
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    if (docker compose ps --services --status running 2>$null | Select-String -Quiet "^web$") {
        docker compose restart web | Out-Null
        Write-Host "web container restarted -> http://localhost:8000"
    }
}
