param(
    [string]$Python = "E:\Miniconda3\envs\babylmm\python.exe",
    [string]$ModelDir = "",
    [string]$DataDir = ""
)

$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Resolve-Path (Join-Path $ScriptDir "..\..")
$EvalRoot = Join-Path $ProjectRoot "eval-pipeline"
$OutputDir = Join-Path $ScriptDir "eval-results"

if ($ModelDir -eq "") {
    $ModelDir = Join-Path $ProjectRoot "models\pretrain\decoder\final"
}

if ($DataDir -eq "") {
    $PreferredDataDir = Join-Path $EvalRoot "evaluation_data\full_eval\zhoblimp"
    $NestedDataDir = Join-Path $EvalRoot "evaluation_data\full_eval\full_eval\zhoblimp"

    if (Test-Path $PreferredDataDir) {
        $DataDir = $PreferredDataDir
    }
    elseif (Test-Path $NestedDataDir) {
        $DataDir = $NestedDataDir
    }
    else {
        throw "ZhoBLiMP data not found. Run prepare_chinese_data.py first, then re-run this script."
    }
}

if (-not (Test-Path $Python)) {
    throw "Python not found: $Python"
}

if (-not (Test-Path $ModelDir)) {
    throw "Model directory not found: $ModelDir"
}

if (-not (Test-Path (Join-Path $ModelDir "tokenizer_config.json"))) {
    & $Python (Join-Path $ProjectRoot "scripts\export_tokenizer.py") --model_dir $ModelDir
}

New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null

Push-Location $EvalRoot
try {
    & $Python -m evaluation_pipeline.sentence_zero_shot.run `
        --model_path_or_name $ModelDir `
        --backend causal `
        --task zhoblimp `
        --data_path $DataDir `
        --output_dir $OutputDir `
        --save_predictions
}
finally {
    Pop-Location
}

Write-Host ""
Write-Host "Evaluation complete."
Write-Host "Results saved under: $OutputDir"
