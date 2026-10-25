param([string]$Python = '3.12')
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $projectRoot
$env:UV_CACHE_DIR = Join-Path $projectRoot '.cache\uv'
$env:HF_HOME = Join-Path $projectRoot '.cache\huggingface'
$env:PYTHONNOUSERSITE = '1'
if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    throw 'Install uv from https://docs.astral.sh/uv/getting-started/installation/ first.'
}
if (-not (Test-Path -LiteralPath '.venv\Scripts\python.exe')) {
    uv venv --python $Python .venv
    if ($LASTEXITCODE -ne 0) { throw 'Environment creation failed.' }
}
uv pip install --python .venv\Scripts\python.exe torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu
if ($LASTEXITCODE -ne 0) { throw 'CPU PyTorch installation failed.' }
if (Test-Path -LiteralPath 'requirements-lock.txt') {
    uv pip install --python .venv\Scripts\python.exe -r requirements-lock.txt --extra-index-url https://download.pytorch.org/whl/cpu --index-strategy unsafe-best-match
    if ($LASTEXITCODE -ne 0) { throw 'Locked dependency installation failed.' }
}
uv pip install --python .venv\Scripts\python.exe -e '.[test,deploy]'
if ($LASTEXITCODE -ne 0) { throw 'Project installation failed.' }
& .venv\Scripts\python.exe -c "import torch; assert torch.version.cuda is None and not torch.cuda.is_available(); print('CPU-only PyTorch:', torch.__version__)"
if ($LASTEXITCODE -ne 0) { throw 'CPU-only verification failed.' }
