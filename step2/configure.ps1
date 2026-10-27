$ErrorActionPreference = "Stop"
$RepoRoot = Resolve-Path (Join-Path $PSScriptRoot "..\..\..")
Push-Location -LiteralPath $RepoRoot
try {
    & "C:\env_isaaclab\python.exe" -u scripts/configure_dexlong_gear_viser.py --task gear3_split --stage step2 @args
    $Code = $LASTEXITCODE
} finally { Pop-Location }
exit $Code
