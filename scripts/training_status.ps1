param(
    [switch]$Watch,
    [switch]$IncludeSmoke,
    [ValidateRange(1, 3600)][int]$Seconds = 5
)

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$runsRoot = Join-Path $repoRoot 'training\runs'
$launchDir = Join-Path $repoRoot 'training'

function Show-TrainingStatus {
    foreach ($launchFile in @(Get-ChildItem -LiteralPath $launchDir -Filter 'v*_9b_launch.json' -File | Sort-Object Name)) {
        try {
            $launch = Get-Content -LiteralPath $launchFile.FullName -Raw | ConvertFrom-Json
            $version = $launchFile.BaseName.Split('_')[0]
            $launchState = switch ($launch.stage) {
                'training_complete' { 'OK' }
                'complete' { 'OK' }
                'failed' { 'FAILED' }
                default {
                    if ($launchFile.LastWriteTime -gt (Get-Date).AddMinutes(-10)) { 'RUNNING' }
                    else { 'INTERRUPTED' }
                }
            }
            Write-Host "train qwen35-9b-$version pipeline: $launchState ($($launch.stage))"
        } catch {
            Write-Host "train $($launchFile.BaseName) pipeline: INVALID STATUS"
        }
    }
    if (-not (Test-Path -LiteralPath $runsRoot)) {
        Write-Host 'No training runs found.'
        return
    }

    $manifests = @(Get-ChildItem -LiteralPath $runsRoot -Filter manifest.json -Recurse -File |
        Sort-Object FullName)
    if ($manifests.Count -eq 0) {
        Write-Host 'No training runs found.'
        return
    }

    foreach ($file in $manifests) {
        try {
            $run = Get-Content -LiteralPath $file.FullName -Raw | ConvertFrom-Json
            if ($run.mode -eq 'smoke' -and -not $IncludeSmoke) { continue }
            $runDir = Split-Path -Parent $file.FullName
            if ($run.mode -eq 'smoke') {
                $lastStep = [int]$run.steps
                $maxSteps = [int]$run.config.smoke.max_steps
            } else {
                $lastStep = [int]$run.steps
                if ($lastStep -eq 0) { $lastStep = [int]$run.last_checkpoint_step }
                if ($lastStep -eq 0) {
                    $stepLog = Join-Path $runDir 'steps.jsonl'
                    if (Test-Path -LiteralPath $stepLog) {
                        try { $lastStep = [int]((Get-Content -LiteralPath $stepLog -Tail 1 | ConvertFrom-Json).step) }
                        catch { $lastStep = 0 }
                    }
                }
                $maxSteps = [int]$run.config.training.max_steps
            }
            $state = switch ($run.status) {
                'complete' { 'OK' }
                'failed' { 'FAILED' }
                'running' {
                    $stepLog = Join-Path $runDir 'steps.jsonl'
                    $activity = if (Test-Path -LiteralPath $stepLog) {
                        (Get-Item -LiteralPath $stepLog).LastWriteTime
                    } else { $file.LastWriteTime }
                    if ($activity -gt (Get-Date).AddMinutes(-5)) { 'RUNNING' }
                    else { 'INTERRUPTED' }
                }
                default { $run.status.ToString().ToUpperInvariant() }
            }
            $progress = if ($maxSteps -gt 0) { " ($lastStep/$maxSteps)" } else { '' }
            Write-Host "train $($run.model) $($run.mode) $($run.run_id): $state$progress"
        } catch {
            Write-Host "train $($file.Directory.Name): INVALID MANIFEST"
        }
    }
}

do {
    if ($Watch) { Clear-Host }
    Show-TrainingStatus
    if ($Watch) {
        Write-Host "`nRefresh: ${Seconds}s. Press Ctrl+C to stop."
        Start-Sleep -Seconds $Seconds
    }
} while ($Watch)
