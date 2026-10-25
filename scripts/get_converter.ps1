$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$revision = 'b3fa69955ea81c3b60e77e3c1e2224c4a826ffb9'
$expectedArchiveHash = 'BBCDEDBE0B20DD1AA6F7DB40F690AF5DD70B766828CF9FF4A2188676A5D703CE'
$cacheRoot = Join-Path $projectRoot '.cache\converter'
$archive = Join-Path $cacheRoot 'source.zip'
$sourceRoot = Join-Path $cacheRoot 'source'
$converter = Join-Path $sourceRoot "llama.cpp-$revision\convert_hf_to_gguf.py"
New-Item -ItemType Directory -Path $cacheRoot -Force | Out-Null
if (-not (Test-Path -LiteralPath $archive)) {
    Invoke-WebRequest -Uri "https://codeload.github.com/ggml-org/llama.cpp/zip/$revision" -OutFile $archive
}
if ((Get-FileHash -LiteralPath $archive -Algorithm SHA256).Hash -ne $expectedArchiveHash) {
    throw 'Converter archive checksum differs from the tested archive. Inspect the download before proceeding.'
}
if (-not (Test-Path -LiteralPath $converter)) {
    Expand-Archive -LiteralPath $archive -DestinationPath $sourceRoot
}
Write-Output $converter
