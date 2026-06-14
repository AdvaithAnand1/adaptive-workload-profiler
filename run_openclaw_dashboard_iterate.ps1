param(
    [string]$ExtraPrompt = "",
    [string]$LogPath = "run_logs\\openclaw_247.log"
)

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

$resolvedLogPath = Join-Path $PSScriptRoot $LogPath
New-Item -ItemType Directory -Force -Path (Split-Path $resolvedLogPath -Parent) | Out-Null

$basePrompt = @"
Proceed automatically and iterate on the dashboard/demo GUI changes in this repository.

Goals:
1) Keep console/log output hidden from primary UX by preserving/improving the tabbed layout.
2) Improve organization and clarity of tabs and sections for demo use.
3) Address configuration oversights so users have maximum intended customizability (config path selection, save/reload/reset behavior, mapping/hotkey flexibility, validation messaging).
4) Make real file edits (not suggestions), then run validation commands and fix any issues.

Validation required:
- python -m unittest tests.test_config
- python -m py_compile demo_gui.py config.py controller.py oracle_client.py

Then provide a concise summary of exactly what changed and why.
"@

$prompt = if ([string]::IsNullOrWhiteSpace($ExtraPrompt)) {
    $basePrompt
} else {
    "$basePrompt`n`nAdditional request:`n$ExtraPrompt"
}

$env:Path = "$env:USERPROFILE\AppData\Roaming\npm;C:\Program Files\nodejs;$env:Path"
$userToken = [Environment]::GetEnvironmentVariable("OPENCLAW_GATEWAY_TOKEN", "User")
if (-not [string]::IsNullOrWhiteSpace($userToken)) {
    $env:OPENCLAW_GATEWAY_TOKEN = $userToken
}

$sid = [guid]::NewGuid().ToString()
$startedAt = Get-Date -Format "yyyy-MM-ddTHH:mm:ssK"
"[$startedAt] Starting OpenClaw dashboard iteration (session-id=$sid)" | Tee-Object -FilePath $resolvedLogPath -Append
openclaw agent --local --agent main --session-id $sid -m $prompt --json | Tee-Object -FilePath $resolvedLogPath -Append
$endedAt = Get-Date -Format "yyyy-MM-ddTHH:mm:ssK"
"[$endedAt] Completed OpenClaw dashboard iteration (session-id=$sid)" | Tee-Object -FilePath $resolvedLogPath -Append
