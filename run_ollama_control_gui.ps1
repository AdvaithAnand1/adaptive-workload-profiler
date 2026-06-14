param(
    [string]$PythonPath = ""
)

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

if (-not [string]::IsNullOrWhiteSpace($PythonPath)) {
    if (-not (Test-Path -LiteralPath $PythonPath)) {
        throw "PythonPath not found: $PythonPath"
    }
    Write-Host "[ollama-gui] Using Python: $PythonPath"
    & $PythonPath ".\ollama_control_gui.py"
    exit $LASTEXITCODE
}

$venvCandidates = @(
    ".\.venv\Scripts\python.exe",
    ".\venv\Scripts\python.exe",
    ".\env\Scripts\python.exe"
)

foreach ($candidate in $venvCandidates) {
    if (Test-Path -LiteralPath $candidate) {
        Write-Host "[ollama-gui] Using virtualenv Python: $candidate"
        & $candidate ".\ollama_control_gui.py"
        exit $LASTEXITCODE
    }
}

if (Get-Command py -ErrorAction SilentlyContinue) {
    Write-Host "[ollama-gui] Using launcher: py -3.12"
    py -3.12 ".\ollama_control_gui.py"
    exit $LASTEXITCODE
}

if (Get-Command python -ErrorAction SilentlyContinue) {
    Write-Host "[ollama-gui] Using system Python: python"
    python ".\ollama_control_gui.py"
    exit $LASTEXITCODE
}

throw "No Python interpreter found. Install Python or create .venv first."
