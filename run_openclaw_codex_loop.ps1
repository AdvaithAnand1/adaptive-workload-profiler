param(
    [string]$Goal = "",
    [string]$GoalFile = "",
    [int]$Iterations = 6,
    [int]$PauseSeconds = 20,
    [string]$Agent = "main",
    [string]$SessionId = "",
    [ValidateSet("off", "minimal", "low", "medium", "high", "xhigh")]
    [string]$Thinking = "high",
    [string]$LogPath = "",
    [string]$StopFile = ".openclaw.stop",
    [switch]$SkipGitSnapshot,
    [switch]$NoConfirm,
    [switch]$NoJson
)

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

if ($Iterations -lt 1) {
    throw "Iterations must be >= 1."
}
if ($PauseSeconds -lt 0) {
    throw "PauseSeconds must be >= 0."
}

$env:Path = "$env:USERPROFILE\AppData\Roaming\npm;C:\Program Files\nodejs;$env:Path"
$userToken = [Environment]::GetEnvironmentVariable("OPENCLAW_GATEWAY_TOKEN", "User")
if (-not [string]::IsNullOrWhiteSpace($userToken)) {
    $env:OPENCLAW_GATEWAY_TOKEN = $userToken
}
$userOllamaKey = [Environment]::GetEnvironmentVariable("OLLAMA_API_KEY", "User")
if (-not [string]::IsNullOrWhiteSpace($userOllamaKey)) {
    $env:OLLAMA_API_KEY = $userOllamaKey
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

if (-not [string]::IsNullOrWhiteSpace($GoalFile)) {
    if (-not (Test-Path -LiteralPath $GoalFile)) {
        throw "GoalFile not found: $GoalFile"
    }
    $Goal = Get-Content -LiteralPath $GoalFile -Raw
}

if ([string]::IsNullOrWhiteSpace($Goal)) {
    Write-Host "No goal provided. Enter one now (leave blank to cancel)."
    $Goal = Read-Host "Loop goal"
}

if ([string]::IsNullOrWhiteSpace($Goal)) {
    Write-Host "Cancelled: no goal provided."
    exit 0
}

if ([string]::IsNullOrWhiteSpace($SessionId)) {
    $SessionId = [guid]::NewGuid().ToString()
}

if ([string]::IsNullOrWhiteSpace($LogPath)) {
    $stamp = Get-Date -Format "yyyyMMdd_HHmmss"
    $LogPath = "run_logs\openclaw_codex_loop_$stamp.log"
}

$resolvedLogPath = Join-Path $PSScriptRoot $LogPath
New-Item -ItemType Directory -Force -Path (Split-Path $resolvedLogPath -Parent) | Out-Null

$resolvedStopFile = Join-Path $PSScriptRoot $StopFile
if (Test-Path -LiteralPath $resolvedStopFile) {
    Remove-Item -LiteralPath $resolvedStopFile -Force
}

function Resolve-EffectiveThinking([string]$requestedThinking) {
    $effective = $requestedThinking
    try {
        $primaryModel = (& $openclawCmd config get agents.defaults.model.primary 2>$null)
        if ($requestedThinking -eq "xhigh" -and $primaryModel -like "ollama/*") {
            $effective = "medium"
        }
    } catch {
        $effective = $requestedThinking
    }
    return $effective
}

$effectiveThinking = Resolve-EffectiveThinking $Thinking

Write-Host "OpenClaw Codex loop configured:"
Write-Host "  Agent       : $Agent"
Write-Host "  SessionId   : $SessionId"
Write-Host "  Thinking    : $Thinking (effective: $effectiveThinking)"
Write-Host "  Iterations  : $Iterations"
Write-Host "  Pause       : ${PauseSeconds}s"
Write-Host "  Log         : $resolvedLogPath"
Write-Host "  Stop file   : $resolvedStopFile"
Write-Host ""
Write-Host "Goal:"
Write-Host $Goal
Write-Host ""
Write-Host "To stop early while running, create this file:"
Write-Host "  $resolvedStopFile"
Write-Host ""

$focusTracks = @(
    "Model quality: improve prediction reliability using data quality, feature signal quality, calibration, threshold tuning, and validation loops.",
    "UI quality: improve dashboard clarity, visual hierarchy, layout ergonomics, and reduced cognitive load while preserving controls.",
    "Practical profiling behavior: make predictions actually drive useful parameter/profile changes with safer switching logic and measurable outcomes.",
    "Integration path: explore G-Helper-compatible/mock flow and generic Windows fallback profiling mechanisms when G-Helper is unavailable."
)

function Get-GitSnapshot {
    if ($SkipGitSnapshot) {
        return "Git snapshot disabled for this run."
    }

    if (-not (Get-Command git -ErrorAction SilentlyContinue)) {
        return "Git not available in PATH."
    }

    $trackedChanged = @()
    $untracked = @()
    $recentCommits = @()
    try {
        $trackedChanged = @(git diff --name-only 2>$null)
    } catch {}
    try {
        $untracked = @(git ls-files --others --exclude-standard 2>$null)
    } catch {}
    try {
        $recentCommits = @(git log -n 3 --pretty=format:"%h %s" 2>$null)
    } catch {}

    $trackedPreview = if ($trackedChanged.Count -gt 0) {
        ($trackedChanged | Select-Object -First 12) -join ", "
    } else {
        "(none)"
    }
    $untrackedPreview = if ($untracked.Count -gt 0) {
        ($untracked | Select-Object -First 12) -join ", "
    } else {
        "(none)"
    }
    $commitPreview = if ($recentCommits.Count -gt 0) {
        ($recentCommits | Select-Object -First 3) -join " | "
    } else {
        "(none)"
    }

    return @"
Tracked changes: $($trackedChanged.Count) file(s). Preview: $trackedPreview
Untracked files: $($untracked.Count) file(s). Preview: $untrackedPreview
Recent commits: $commitPreview
"@
}

if (-not $NoConfirm) {
    $answer = Read-Host "Start loop now? [y/N]"
    if ($answer -notin @("y", "Y", "yes", "YES")) {
        Write-Host "Cancelled."
        exit 0
    }
}

$runStartedAt = Get-Date -Format "yyyy-MM-ddTHH:mm:ssK"
"[$runStartedAt] Starting OpenClaw Codex loop (session-id=$SessionId, agent=$Agent, passes=$Iterations)" |
    Tee-Object -FilePath $resolvedLogPath -Append
if ($effectiveThinking -ne $Thinking) {
    "[$runStartedAt] Thinking auto-adjusted from $Thinking to $effectiveThinking for current model compatibility." |
        Tee-Object -FilePath $resolvedLogPath -Append
}

$lastExitCode = 0
for ($i = 1; $i -le $Iterations; $i++) {
    if (Test-Path -LiteralPath $resolvedStopFile) {
        $stopAt = Get-Date -Format "yyyy-MM-ddTHH:mm:ssK"
        "[$stopAt] Stop file detected. Halting loop before pass $i." |
            Tee-Object -FilePath $resolvedLogPath -Append
        break
    }

    $trackIdx = ($i - 1) % $focusTracks.Count
    $primaryTrack = $focusTracks[$trackIdx]
    $secondaryTrack = $focusTracks[($trackIdx + 1) % $focusTracks.Count]
    $gitSnapshot = Get-GitSnapshot

    $iterationPrompt = @"
Workspace: $PSScriptRoot
Goal:
$Goal

Current pass: $i of $Iterations.
Primary focus this pass:
$primaryTrack
Secondary focus this pass:
$secondaryTrack

Current repo snapshot:
$gitSnapshot

Instructions:
1) Think at high level first: identify the highest-leverage next move for this pass and explain why.
2) Propose 2-3 options with tradeoffs, choose one, and execute concrete repo changes (not only suggestions).
3) Improve both model and product quality over time:
   - model prediction quality (data collection strategy, features, confidence/stability logic, calibration, thresholds)
   - UI quality (dashboard usability, formatting, visual clarity)
   - practical profiling outcomes (changes that actually alter useful behavior)
4) Explore practical integrations:
   - if useful, inspect/align with open-source G-Helper behavior or create a robust mock path
   - propose and/or implement generic Windows profiling fallbacks where appropriate
5) Run relevant lightweight validation commands for your edits.
6) End with:
   - what changed this pass
   - validation run/results
   - top next move for the following pass
7) Keep moving unless blocked by a hard external dependency.
"@

    $passStart = Get-Date -Format "yyyy-MM-ddTHH:mm:ssK"
    "[$passStart] Pass $i/$Iterations starting." |
        Tee-Object -FilePath $resolvedLogPath -Append

    $args = @(
        "agent",
        "--local",
        "--agent",
        $Agent,
        "--session-id",
        $SessionId,
        "--thinking",
        $effectiveThinking,
        "-m",
        $iterationPrompt
    )
    if (-not $NoJson) {
        $args += "--json"
    }

    $passOutput = & $openclawCmd @args 2>&1
    if ($passOutput) {
        $passOutput | Tee-Object -FilePath $resolvedLogPath -Append
    }
    $lastExitCode = $LASTEXITCODE

    if ($lastExitCode -ne 0 -and $effectiveThinking -eq "xhigh") {
        $outputText = ($passOutput | Out-String)
        if ($outputText -match 'Thinking level "xhigh" is only supported') {
            $retryThinking = "medium"
            $retryAt = Get-Date -Format "yyyy-MM-ddTHH:mm:ssK"
            "[$retryAt] Retrying pass $i with thinking=$retryThinking due xhigh incompatibility." |
                Tee-Object -FilePath $resolvedLogPath -Append

            $argsRetry = @(
                "agent",
                "--local",
                "--agent",
                $Agent,
                "--session-id",
                $SessionId,
                "--thinking",
                $retryThinking,
                "-m",
                $iterationPrompt
            )
            if (-not $NoJson) {
                $argsRetry += "--json"
            }

            $retryOutput = & $openclawCmd @argsRetry 2>&1
            if ($retryOutput) {
                $retryOutput | Tee-Object -FilePath $resolvedLogPath -Append
            }
            $lastExitCode = $LASTEXITCODE
            if ($lastExitCode -eq 0) {
                $effectiveThinking = $retryThinking
            }
        }
    }

    $passEnd = Get-Date -Format "yyyy-MM-ddTHH:mm:ssK"
    "[$passEnd] Pass $i/$Iterations finished (exit=$lastExitCode)." |
        Tee-Object -FilePath $resolvedLogPath -Append

    if ($lastExitCode -ne 0) {
        "[$passEnd] Non-zero exit. Stopping loop." |
            Tee-Object -FilePath $resolvedLogPath -Append
        break
    }

    if ($i -lt $Iterations -and $PauseSeconds -gt 0) {
        Start-Sleep -Seconds $PauseSeconds
    }
}

$runEndedAt = Get-Date -Format "yyyy-MM-ddTHH:mm:ssK"
"[$runEndedAt] Loop completed (final-exit=$lastExitCode)." |
    Tee-Object -FilePath $resolvedLogPath -Append
exit $lastExitCode
