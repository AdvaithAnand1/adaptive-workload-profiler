param(
    [switch]$KillProcess
)

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

$metaPath = Join-Path $PSScriptRoot "run_logs\openclaw_codex_loop_current.json"
$defaultStopFile = Join-Path $PSScriptRoot ".openclaw.stop"

$meta = $null
if (Test-Path -LiteralPath $metaPath) {
    try {
        $meta = Get-Content -LiteralPath $metaPath -Raw | ConvertFrom-Json
    } catch {
        Write-Host "Warning: could not parse metadata file at $metaPath"
    }
}

$stopFile = if ($meta -and $meta.stop_file) { [string]$meta.stop_file } else { $defaultStopFile }
New-Item -ItemType File -Path $stopFile -Force | Out-Null
Write-Host "Stop file created: $stopFile"

if ($KillProcess -and $meta -and $meta.pid) {
    $targetPid = [int]$meta.pid
    try {
        Stop-Process -Id $targetPid -Force -ErrorAction Stop
        Write-Host "Loop process terminated (PID $targetPid)."
    } catch {
        Write-Host "Could not terminate PID ${targetPid}: $($_.Exception.Message)"
    }
} elseif ($meta -and $meta.pid) {
    $targetPid = [int]$meta.pid
    $proc = Get-Process -Id $targetPid -ErrorAction SilentlyContinue
    if ($proc) {
        Write-Host "Loop process still running (PID $targetPid); it should stop after current pass."
    } else {
        Write-Host "No active loop process found for PID $targetPid."
    }
}
