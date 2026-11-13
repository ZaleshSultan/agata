param([string]$RunName = ('smoke-' + (Get-Date -Format 'yyyyMMdd-HHmmss')), [int]$Steps = 5)
$ErrorActionPreference = 'Stop'
if ($RunName -notmatch '^[a-zA-Z0-9_-]+$') { throw 'RunName must be a simple folder name.' }
$labRoot = $PSScriptRoot
$condaInfo = (& conda info --json | ConvertFrom-Json)
$labPython = Join-Path $condaInfo.root_prefix 'envs\agata-ft\python.exe'
if (-not (Test-Path -LiteralPath $labPython)) { throw 'Run setup_windows.ps1 first.' }
function Invoke-LabStep([string]$ScriptName, [string[]]$Parameters) {
    & $labPython (Join-Path $labRoot $ScriptName) @Parameters
    if ($LASTEXITCODE -ne 0) { throw "Failed: $ScriptName. See console and preserved reports." }
}
$runPath = "outputs/$RunName"
$reportPath = "evaluation/reports/$RunName"
if (Test-Path -LiteralPath (Join-Path $labRoot $runPath)) { throw 'Run directory exists. Choose a new RunName.' }
$previousUtf8 = $env:PYTHONUTF8
try {
    $env:PYTHONUTF8 = '1'
    Invoke-LabStep 'diagnose.py' @('--cuda')
    Invoke-LabStep 'download_model.py' @()
    if (-not (Test-Path -LiteralPath (Join-Path $labRoot 'datasets/train.jsonl'))) {
        Invoke-LabStep 'prepare_dataset.py' @()
    }
    Invoke-LabStep 'inference.py' @('--output', "$reportPath-base-generation.json")
    Invoke-LabStep 'evaluate.py' @('--output', "$reportPath-before", '--max-new-tokens', '96')
    Invoke-LabStep 'train.py' @('--smoke', '--max-steps', "$Steps", '--output-dir', $runPath)
    Invoke-LabStep 'inference.py' @('--adapter', "$runPath/adapter", '--output', "$reportPath-reload.json")
    Invoke-LabStep 'evaluate.py' @('--adapter', "$runPath/adapter", '--output', "$reportPath-after", '--max-new-tokens', '96')
    Invoke-LabStep 'compare_models.py' @("$reportPath-before.json", "$reportPath-after.json", '--output', "$reportPath-comparison")
    Write-Output "Smoke test complete. Full-training gate: --smoke-report $runPath/training_report.json"
} finally {
    $env:PYTHONUTF8 = $previousUtf8
}
