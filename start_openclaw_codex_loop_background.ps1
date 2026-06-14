param(
    [string]$Goal = "",
    [string]$GoalFile = "",
    [int]$Hours = 3,
    [int]$Iterations = 0,
    [int]$PauseSeconds = 180,
    [string]$Agent = "main",
    [ValidateSet("off", "minimal", "low", "medium", "high", "xhigh")]
    [string]$Thinking = "xhigh",
    [string]$StopFile = ".openclaw.stop",
    [switch]$SkipGitSnapshot,
    [switch]$NoJson
)

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

if ($Hours -lt 1) {
    throw "Hours must be >= 1."
}
if ($PauseSeconds -lt 0) {
    throw "PauseSeconds must be >= 0."
}

if ($Iterations -le 0) {
    $estimatedPassSec = [Math]::Max(120, ($PauseSeconds + 120))
    $Iterations = [int][Math]::Max(1, [Math]::Ceiling(($Hours * 3600.0) / $estimatedPassSec))
}

$sessionId = [guid]::NewGuid().ToString()
$stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$loopLogPath = "run_logs\openclaw_codex_loop_$stamp.log"
$runnerMetaPath = Join-Path $PSScriptRoot "run_logs\openclaw_codex_loop_current.json"

New-Item -ItemType Directory -Force -Path (Join-Path $PSScriptRoot "run_logs") | Out-Null

$loopScript = Join-Path $PSScriptRoot "run_openclaw_codex_loop.ps1"
if (-not (Test-Path -LiteralPath $loopScript)) {
    throw "Loop script not found: $loopScript"
}

$argList = @(
    "-NoProfile",
    "-ExecutionPolicy", "Bypass",
    "-File", $loopScript,
    "-Iterations", $Iterations,
    "-PauseSeconds", $PauseSeconds,
    "-Agent", $Agent,
    "-SessionId", $sessionId,
    "-Thinking", $Thinking,
    "-LogPath", $loopLogPath,
    "-StopFile", $StopFile,
    "-NoConfirm"
)

if (-not [string]::IsNullOrWhiteSpace($GoalFile)) {
    $argList += @("-GoalFile", $GoalFile)
} else {
    if ([string]::IsNullOrWhiteSpace($Goal)) {
        throw "Provide -Goal or -GoalFile."
    }
    $argList += @("-Goal", $Goal)
}

if ($SkipGitSnapshot) {
    $argList += "-SkipGitSnapshot"
}
if ($NoJson) {
    $argList += "-NoJson"
}

$proc = Start-Process -FilePath "powershell.exe" -ArgumentList $argList -WindowStyle Hidden -WorkingDirectory $PSScriptRoot -PassThru

$meta = [ordered]@{
    started_at = (Get-Date -Format "yyyy-MM-ddTHH:mm:ssK")
    pid = $proc.Id
    session_id = $sessionId
    iterations = $Iterations
    pause_seconds = $PauseSeconds
    agent = $Agent
    thinking = $Thinking
    log_path = (Join-Path $PSScriptRoot $loopLogPath)
    stop_file = (Join-Path $PSScriptRoot $StopFile)
    goal = if (-not [string]::IsNullOrWhiteSpace($GoalFile)) { "GoalFile: $GoalFile" } else { $Goal }
}

$meta | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath $runnerMetaPath -Encoding UTF8

Write-Host "OpenClaw background loop started."
Write-Host "  PID        : $($proc.Id)"
Write-Host "  SessionId  : $sessionId"
Write-Host "  Iterations : $Iterations"
Write-Host "  Pause      : ${PauseSeconds}s"
Write-Host "  Log        : $(Join-Path $PSScriptRoot $loopLogPath)"
Write-Host "  Meta       : $runnerMetaPath"
Write-Host "  Stop file  : $(Join-Path $PSScriptRoot $StopFile)"
Write-Host ""
Write-Host "To stop gracefully:"
Write-Host "  New-Item -ItemType File -Path $(Join-Path $PSScriptRoot $StopFile) -Force"
Write-Host "or:"
Write-Host "  .\\stop_openclaw_codex_loop.ps1"
