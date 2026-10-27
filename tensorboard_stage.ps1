# Shared TensorBoard launcher for the three split steps; settings live in tensorboard.json.
param([Parameter(Mandatory = $true)][ValidateSet("step1", "step2", "step3")][string]$Stage)
$ErrorActionPreference = "Stop"
$RepoRoot = Resolve-Path (Join-Path $PSScriptRoot "..\..")
$Config = Get-Content -Raw -LiteralPath (Join-Path $PSScriptRoot "tensorboard.json") | ConvertFrom-Json
$LogDir = Join-Path $RepoRoot (Join-Path $Config.log_root ($Config.experiment_prefix + $Stage))
$Port = $Config.ports.$Stage
$ExtraArgs = @($args)
if ($args.Count -gt 0 -and -not ([string]$args[0]).StartsWith("-")) {
    $LogDir = [string]$args[0]
    $ExtraArgs = @()
    if ($args.Count -gt 1) { $ExtraArgs = @($args[1..($args.Count - 1)]) }
}
Write-Host "TensorBoard $Stage logs: $LogDir (URL: http://$($Config.host):$Port)"
& $Config.python -m tensorboard.main --logdir $LogDir --host $Config.host --port $Port `
    --reload_interval $Config.reload_interval --samples_per_plugin $Config.samples_per_plugin @ExtraArgs
exit $LASTEXITCODE
