param(
    [string]$Python = "python"
)

$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Resolve-Path (Join-Path $ScriptDir "..")
$ModelsRoot = Join-Path $ProjectRoot "models"
$EvalResRoot = Join-Path $ProjectRoot "eval-res"

if (-not (Test-Path $ModelsRoot)) {
    throw "Models directory not found: $ModelsRoot"
}

$ModelSets = Get-ChildItem $ModelsRoot -Directory |
    Where-Object { $_.Name -ne "pretrain" } |
    Sort-Object Name

if ($ModelSets.Count -eq 0) {
    throw "No model sets found under $ModelsRoot. Expected folders like models\encoder-mlm\final."
}

Write-Host ""
Write-Host "Select model set:"
for ($Index = 0; $Index -lt $ModelSets.Count; $Index++) {
    Write-Host "[$($Index + 1)] $($ModelSets[$Index].Name)"
}
$ModelChoice = [int](Read-Host "Enter number")
if ($ModelChoice -lt 1 -or $ModelChoice -gt $ModelSets.Count) {
    throw "Invalid model set choice."
}
$ModelSet = $ModelSets[$ModelChoice - 1]

$Checkpoints = Get-ChildItem $ModelSet.FullName -Directory |
    Where-Object { $_.Name -eq "final" -or $_.Name -match '^checkpoint-\d+$' } |
    Sort-Object @{ Expression = { if ($_.Name -eq "final") { [int]::MaxValue } else { [int]($_.Name -replace '^checkpoint-', '') } } }

if ($Checkpoints.Count -eq 0) {
    throw "No checkpoint-* or final folders found under: $($ModelSet.FullName)"
}

Write-Host ""
Write-Host "Select checkpoint/final:"
Write-Host "[1] best-loss"
for ($Index = 0; $Index -lt $Checkpoints.Count; $Index++) {
    Write-Host "[$($Index + 2)] $($Checkpoints[$Index].Name)"
}
$CheckpointChoice = [int](Read-Host "Enter number")
if ($CheckpointChoice -lt 1 -or $CheckpointChoice -gt ($Checkpoints.Count + 1)) {
    throw "Invalid checkpoint choice."
}
if ($CheckpointChoice -eq 1) {
    $Scored = @()
    foreach ($Candidate in $Checkpoints) {
        $StatePath = Join-Path $Candidate.FullName "trainer_state.json"
        if (-not (Test-Path $StatePath)) {
            continue
        }
        $State = Get-Content -Raw $StatePath | ConvertFrom-Json
        $Loss = $null
        if ($null -ne $State.avg_loss) {
            $Loss = [double]$State.avg_loss
        }
        elseif ($null -ne $State.loss) {
            $Loss = [double]$State.loss
        }
        if ($null -ne $Loss) {
            $Scored += [pscustomobject]@{
                Item = $Candidate
                Loss = $Loss
            }
        }
    }
    if ($Scored.Count -eq 0) {
        throw "No checkpoint/final with trainer_state loss found under: $($ModelSet.FullName)"
    }
    $Best = $Scored | Sort-Object Loss | Select-Object -First 1
    $Checkpoint = $Best.Item
    Write-Host "Selected best-loss checkpoint: $($Checkpoint.Name) (loss=$($Best.Loss))"
}
else {
    $Checkpoint = $Checkpoints[$CheckpointChoice - 2]
}

Write-Host ""
Write-Host "Select evaluation:"
Write-Host "[1] CLUE"
Write-Host "[2] ZhoBLiMP"
$EvalChoice = [int](Read-Host "Enter number")

New-Item -ItemType Directory -Force -Path $EvalResRoot | Out-Null

if ($EvalChoice -eq 1) {
    $OutputDir = Join-Path $EvalResRoot "$($ModelSet.Name)\$($Checkpoint.Name)\clue"
    & (Join-Path $ScriptDir "evaluate_clue.ps1") -Python $Python -ModelDir $Checkpoint.FullName -OutputDir $OutputDir
}
elseif ($EvalChoice -eq 2) {
    $OutputDir = Join-Path $EvalResRoot "$($ModelSet.Name)\$($Checkpoint.Name)\zhoblimp"
    & (Join-Path $ScriptDir "evaluate_zhoblimp.ps1") -Python $Python -ModelDir $Checkpoint.FullName -OutputDir $OutputDir
}
else {
    throw "Invalid evaluation choice."
}
