param(
    [switch]$Mock,
    [string]$PythonPath = ""
)

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

if ($Mock) {
    $env:PERFANALYZE_MOCK = "1"
    Write-Host "[demo] PERFANALYZE_MOCK=1"
} else {
    Remove-Item Env:PERFANALYZE_MOCK -ErrorAction SilentlyContinue
}

if (-not [string]::IsNullOrWhiteSpace($PythonPath)) {
    if (-not (Test-Path -LiteralPath $PythonPath)) {
        throw "PythonPath not found: $PythonPath"
    }
    Write-Host "[demo] Using Python: $PythonPath"
    & $PythonPath ".\demo_gui.py"
    exit $LASTEXITCODE
}

$venvCandidates = @(
    ".\.venv\Scripts\python.exe",
    ".\venv\Scripts\python.exe",
    ".\env\Scripts\python.exe"
)

foreach ($candidate in $venvCandidates) {
    if (Test-Path -LiteralPath $candidate) {
        Write-Host "[demo] Using virtualenv Python: $candidate"
        & $candidate ".\demo_gui.py"
        exit $LASTEXITCODE
    }
}

if (Get-Command py -ErrorAction SilentlyContinue) {
    Write-Host "[demo] Using launcher: py -3.12"
    py -3.12 ".\demo_gui.py"
    exit $LASTEXITCODE
}

if (Get-Command python -ErrorAction SilentlyContinue) {
    Write-Host "[demo] Using system Python: python"
    python ".\demo_gui.py"
    exit $LASTEXITCODE
}

throw "No Python interpreter found. Install Python or create .venv first."
