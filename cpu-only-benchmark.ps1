param(
    [string]$Model = 'qwen2.5-coder:7b',
    [ValidateRange(1, 128)][int]$Threads = 12,
    [ValidateRange(1, 10)][int]$Runs = 2,
    [ValidateRange(16, 512)][int]$Tokens = 128
)
$ErrorActionPreference = 'Stop'
$baseUri = 'http://127.0.0.1:11434'
$initial = Invoke-RestMethod -Uri "$baseUri/api/ps" -TimeoutSec 10
if (@($initial.models).Count -gt 0) {
    throw 'A model is already loaded. Run this experiment when Ollama is idle.'
}
$installed = Invoke-RestMethod -Uri "$baseUri/api/tags" -TimeoutSec 10
if (-not ($installed.models.name -contains $Model)) {
    throw "Model '$Model' is not installed. This script does not download models."
}
$version = Invoke-RestMethod -Uri "$baseUri/api/version" -TimeoutSec 10
$processor = Get-CimInstance Win32_Processor
$machine = Get-CimInstance Win32_ComputerSystem
$prompt = 'Explain step by step how merge sort works, then provide a Python implementation.'
$results = @()
$reportPath = Join-Path $PSScriptRoot 'cpu-only-benchmark-results.json'
try {
    for ($run = 1; $run -le $Runs; $run++) {
        $payload = @{
            model = $Model
            prompt = $prompt
            stream = $false
            keep_alive = '2m'
            options = @{
                num_gpu = 0
                num_thread = $Threads
                num_ctx = 2048
                num_predict = $Tokens
                temperature = 0
                seed = 42
            }
        } | ConvertTo-Json -Depth 5
        $response = Invoke-RestMethod -Uri "$baseUri/api/generate" -Method Post -ContentType 'application/json' -Body $payload -TimeoutSec 300
        $status = Invoke-RestMethod -Uri "$baseUri/api/ps" -TimeoutSec 10
        $loaded = @($status.models | Where-Object { $_.name -eq $Model })
        if ($loaded.Count -ne 1 -or $null -eq $loaded[0].size_vram -or $loaded[0].size_vram -ne 0) {
            throw 'Ollama did not confirm zero VRAM allocation for this model.'
        }
        if ($response.eval_duration -le 0) { throw 'Missing token timing.' }
        $result = [pscustomobject]@{
            run = $run
            load_state = $(if ($run -eq 1) { 'initial model load; filesystem cache uncontrolled' } else { 'warm model' })
            output_tokens = $response.eval_count
            decoding_tokens_per_second = [math]::Round($response.eval_count * 1e9 / $response.eval_duration, 3)
            total_seconds = [math]::Round($response.total_duration / 1e9, 3)
            load_seconds = [math]::Round($response.load_duration / 1e9, 3)
            prompt_tokens = $response.prompt_eval_count
            prompt_seconds = [math]::Round($response.prompt_eval_duration / 1e9, 3)
            ollama_reported_model_bytes = $loaded[0].size
            vram_bytes = $loaded[0].size_vram
            model_digest = $loaded[0].digest
            model_details = $loaded[0].details
            response = $response.response
            raw_response = $response
            running_model = $loaded[0]
        }
        $results += $result
        $result | Select-Object run,output_tokens,decoding_tokens_per_second,total_seconds,load_seconds,vram_bytes | ConvertTo-Json -Compress
    }
    [pscustomobject]@{
        recorded_at = (Get-Date).ToString('o')
        purpose = 'Short local CPU-only inference demonstration, not a quality or general performance benchmark.'
        cpu = $processor.Name
        physical_cores = $processor.NumberOfCores
        logical_processors = $processor.NumberOfLogicalProcessors
        physical_memory_bytes = $machine.TotalPhysicalMemory
        ollama_version = $version.version
        model = $Model
        prompt = $prompt
        options = ($payload | ConvertFrom-Json).options
        limitations = @('Two short runs by default; no control of other system load or filesystem cache.', 'Decoding rate excludes prompt processing and model loading.', 'Ollama size is not measured peak process RAM.', 'Zero VRAM is verified using Ollama model status, not hardware tracing.', 'No training or task-quality evaluation is performed.')
        runs = $results
    } | ConvertTo-Json -Depth 12 | Set-Content -LiteralPath $reportPath -Encoding utf8
    Write-Output "Saved: $reportPath"
} finally {
    $unload = @{ model = $Model; keep_alive = 0; stream = $false } | ConvertTo-Json
    try { Invoke-RestMethod -Uri "$baseUri/api/generate" -Method Post -ContentType 'application/json' -Body $unload -TimeoutSec 30 | Out-Null }
    catch { Write-Warning 'Could not unload the test model; its keep-alive timer will expire.' }
}
