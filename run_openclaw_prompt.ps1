param(
    [string]$Prompt = "",
    [string]$PromptFile = "",
    [string]$Agent = "main",
    [string]$SessionId = "",
    [string]$LogPath = "",
    [switch]$NoConfirm,
    [switch]$NoJson
)

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

$env:Path = "$env:USERPROFILE\AppData\Roaming\npm;C:\Program Files\nodejs;$env:Path"
$userToken = [Environment]::GetEnvironmentVariable("OPENCLAW_GATEWAY_TOKEN", "User")
if (-not [string]::IsNullOrWhiteSpace($userToken)) {
    $env:OPENCLAW_GATEWAY_TOKEN = $userToken
}

$openclawCmd = "openclaw"
if (-not (Get-Command $openclawCmd -ErrorAction SilentlyContinue)) {
    $candidate = Join-Path $env:USERPROFILE "AppData\Roaming\npm\openclaw.cmd"
    if (Test-Path -LiteralPath $candidate) {
        $openclawCmd = $candidate
    } else {
        throw "OpenClaw not found in PATH or at $candidate."
    }
}

if (-not [string]::IsNullOrWhiteSpace($PromptFile)) {
    if (-not (Test-Path -LiteralPath $PromptFile)) {
        throw "PromptFile not found: $PromptFile"
    }
    $Prompt = Get-Content -LiteralPath $PromptFile -Raw
}

if ([string]::IsNullOrWhiteSpace($Prompt)) {
    Write-Host "No prompt provided. Enter one now (leave blank to cancel)."
    $Prompt = Read-Host "OpenClaw prompt"
}

if ([string]::IsNullOrWhiteSpace($Prompt)) {
    Write-Host "Cancelled: no prompt provided."
    exit 0
}

if ([string]::IsNullOrWhiteSpace($SessionId)) {
    $SessionId = [guid]::NewGuid().ToString()
}

if ([string]::IsNullOrWhiteSpace($LogPath)) {
    $stamp = Get-Date -Format "yyyyMMdd_HHmmss"
    $LogPath = "run_logs\openclaw_prompt_$stamp.log"
}

$resolvedLogPath = Join-Path $PSScriptRoot $LogPath
New-Item -ItemType Directory -Force -Path (Split-Path $resolvedLogPath -Parent) | Out-Null

Write-Host "OpenClaw launcher ready:"
Write-Host "  Agent     : $Agent"
Write-Host "  SessionId : $SessionId"
Write-Host "  Log       : $resolvedLogPath"
Write-Host ""
Write-Host "Prompt preview:"
Write-Host $Prompt
Write-Host ""

if (-not $NoConfirm) {
    $answer = Read-Host "Start OpenClaw now? [y/N]"
    if ($answer -notin @("y", "Y", "yes", "YES")) {
        Write-Host "Cancelled."
        exit 0
    }
}

$args = @(
    "agent",
    "--local",
    "--agent",
    $Agent,
    "--session-id",
    $SessionId,
    "-m",
    $Prompt
)

if (-not $NoJson) {
    $args += "--json"
}

$startedAt = Get-Date -Format "yyyy-MM-ddTHH:mm:ssK"
"[$startedAt] Starting OpenClaw prompt run (session-id=$SessionId, agent=$Agent)" | Tee-Object -FilePath $resolvedLogPath -Append
& $openclawCmd @args | Tee-Object -FilePath $resolvedLogPath -Append
$exitCode = $LASTEXITCODE
$endedAt = Get-Date -Format "yyyy-MM-ddTHH:mm:ssK"
"[$endedAt] Completed OpenClaw prompt run (session-id=$SessionId, exit=$exitCode)" | Tee-Object -FilePath $resolvedLogPath -Append
exit $exitCode
