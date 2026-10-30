$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot
if (Test-Path '.\venv\Scripts\python.exe') {
    $python = '.\venv\Scripts\python.exe'
} else {
    $python = 'python'
}
& $python -m pip install -r requirements.txt
& $python .\run_revised_benchmark.py --seeds 42 43 44 45 46
Write-Host "Completed. Results are in results_revised\"
