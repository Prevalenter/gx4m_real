# Uses the shared settings in ..\tensorboard.json.
& (Join-Path $PSScriptRoot "..\tensorboard_stage.ps1") -Stage step1 @args
exit $LASTEXITCODE
