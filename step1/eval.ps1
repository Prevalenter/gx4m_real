$ErrorActionPreference = "Stop"
$RepoRoot = Resolve-Path (Join-Path $PSScriptRoot "..\..\..")
Push-Location -LiteralPath $RepoRoot
try {
    & "C:\env_isaaclab\python.exe" -u scripts/run_dexlong_gear.py eval --task gear3_split_step1 @args
    $Code = $LASTEXITCODE
} finally { Pop-Location }
exit $Code
