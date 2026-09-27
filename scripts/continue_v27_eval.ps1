param([int]$FirstPid)

$ErrorActionPreference = 'Stop'
$repo = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
$firstSummary = Join-Path $repo 'benchmarks\outputs\v25-9b-v27-fresh\summary.json'
$secondSummary = Join-Path $repo 'benchmarks\outputs\v24-9b-v27-fresh\summary.json'
$python = Join-Path $repo '.venv-nvidia\Scripts\python.exe'
$runner = Join-Path $repo 'evaluation\run_semantic_v1_3_9b.py'

# The first benchmark is already running. Never load two 9B models on the GPU.
$deadline = (Get-Date).AddHours(2)
while (-not (Test-Path -LiteralPath $firstSummary)) {
    if ((Get-Date) -ge $deadline) { throw 'V25 V27 evaluation timed out' }
    if (-not (Get-Process -Id $FirstPid -ErrorAction SilentlyContinue)) {
        throw 'V25 V27 evaluator exited without a complete summary'
    }
    Start-Sleep -Seconds 10
}
if (Test-Path -LiteralPath $secondSummary) { exit 0 }
Set-Location -LiteralPath $repo
& $python $runner --run 'training/runs/qwen35-9b-v24/20260926T180548Z-28e20ea2' `
    --panel 'benchmarks/v27/fresh.jsonl' --output 'benchmarks/outputs/v24-9b-v27-fresh'
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
