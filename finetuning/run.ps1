param(
    [Parameter(Mandatory=$true,Position=0)][string]$Script,
    [Parameter(ValueFromRemainingArguments=$true)][string[]]$ScriptArguments
)
$ErrorActionPreference = 'Stop'
$condaInfo = (& conda info --json | ConvertFrom-Json)
if ($LASTEXITCODE -ne 0) { throw 'Cannot locate Conda.' }
$labPython = Join-Path $condaInfo.root_prefix 'envs\agata-ft\python.exe'
if (-not (Test-Path -LiteralPath $labPython)) { throw 'Run finetuning/setup_windows.ps1 first.' }
$scriptPath = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot $Script))
if (-not $scriptPath.StartsWith($PSScriptRoot + [IO.Path]::DirectorySeparatorChar, [StringComparison]::OrdinalIgnoreCase)) {
    throw 'Script must be inside finetuning.'
}
$previousUtf8 = $env:PYTHONUTF8
try {
    $env:PYTHONUTF8 = '1'
    & $labPython $scriptPath @ScriptArguments
    $result = $LASTEXITCODE
} finally {
    $env:PYTHONUTF8 = $previousUtf8
}
exit $result
