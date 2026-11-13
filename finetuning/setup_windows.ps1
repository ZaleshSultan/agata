param([string]$EnvironmentName = 'agata-ft')
$ErrorActionPreference = 'Stop'
$labRoot = $PSScriptRoot
$condaInfo = (& conda info --json | ConvertFrom-Json)
if ($LASTEXITCODE -ne 0) { throw 'Conda is required. See README.md.' }
$labPython = Join-Path $condaInfo.root_prefix "envs\$EnvironmentName\python.exe"
if (-not (Test-Path -LiteralPath $labPython)) {
    $condaLock = Join-Path $labRoot 'environment-win64.lock.txt'
    if (Test-Path -LiteralPath $condaLock) {
        & conda create --name $EnvironmentName --file $condaLock -y
    } else {
        & conda create --name $EnvironmentName python=3.12.13 pip=26.1.2 -y
    }
    if ($LASTEXITCODE -ne 0) { throw 'Conda environment creation failed.' }
}
& $labPython -c "import sys; assert sys.version_info[:2] == (3,12), 'This lock requires Python 3.12'"
if ($LASTEXITCODE -ne 0) { throw 'Wrong Python version in the target environment.' }
& $labPython -m pip install torch==2.14.1 torchvision==0.29.1 --index-url https://download.pytorch.org/whl/cu130
if ($LASTEXITCODE -ne 0) { throw 'PyTorch installation failed.' }
$requirements = Join-Path $labRoot 'requirements.lock.txt'
if (-not (Test-Path -LiteralPath $requirements)) { $requirements = Join-Path $labRoot 'requirements.txt' }
& $labPython -m pip install -r $requirements --extra-index-url https://download.pytorch.org/whl/cu130
if ($LASTEXITCODE -ne 0) { throw 'Fine-tuning dependency installation failed.' }
& $labPython -m pip check
if ($LASTEXITCODE -ne 0) { throw 'Dependency compatibility check failed.' }
& $labPython (Join-Path $labRoot 'diagnose.py') --cuda
if ($LASTEXITCODE -ne 0) { throw 'Environment diagnostics failed.' }
Write-Output "Ready: $labPython"
