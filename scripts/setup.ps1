param(
    [string]$PythonExe = "python",
    [string]$VenvDir = ".venv"
)

$ErrorActionPreference = "Stop"

function Require-Command {
    param([string]$Name)
    if (-not (Get-Command $Name -ErrorAction SilentlyContinue)) {
        throw "Required command '$Name' was not found on PATH."
    }
}

Require-Command ollama
Require-Command $PythonExe

$pythonVersion = & $PythonExe -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"
if ([version]$pythonVersion -lt [version]"3.10") {
    throw "Python 3.10+ is required. Current version: $pythonVersion"
}

if (-not (Test-Path $VenvDir)) {
    & $PythonExe -m venv $VenvDir
}

$venvPython = Join-Path $VenvDir "Scripts\python.exe"
& $venvPython -m pip install --upgrade pip
& $venvPython -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
& $venvPython -m pip install -r requirements.txt

$env:COQUI_TOS_AGREED = "1"

$models = @(
    "wmb/llamasupport:latest",
    "nemotron-mini:4b-instruct-q8_0",
    "nomic-embed-text:latest"
)

foreach ($model in $models) {
    ollama pull $model
}

Write-Host "Setup complete. Activate the venv with: .\\$VenvDir\\Scripts\\Activate.ps1"