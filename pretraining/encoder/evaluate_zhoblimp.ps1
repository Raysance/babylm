param(
    [string]$Python = "python",
    [string]$ModelDir = "",
    [string]$DataDir = "",
    [string]$OutputDir = "",
    [string]$BatchSize = "64",
    [string]$NonCausalBatchSize = "64"
)

$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Resolve-Path (Join-Path $ScriptDir "..\..")
$EvalRoot = Join-Path $ProjectRoot "eval-pipeline"

if ($ModelDir -eq "") {
    $ModelDir = Join-Path $ProjectRoot "models\pretrain\encoder\final"
}

if ($OutputDir -eq "") {
    $OutputDir = Join-Path $ScriptDir "eval-results-zhoblimp"
}

if ($DataDir -eq "") {
    $PreferredDataDir = Join-Path $ProjectRoot "evaluation_data\full_eval\zhoblimp"
    $EvalPipelineDataDir = Join-Path $EvalRoot "evaluation_data\full_eval\zhoblimp"
    $NestedDataDir = Join-Path $EvalRoot "evaluation_data\full_eval\full_eval\zhoblimp"

    if (Test-Path $PreferredDataDir) {
        $DataDir = $PreferredDataDir
    }
    elseif (Test-Path $EvalPipelineDataDir) {
        $DataDir = $EvalPipelineDataDir
    }
    elseif (Test-Path $NestedDataDir) {
        $DataDir = $NestedDataDir
    }
    else {
        throw "ZhoBLiMP data not found. Run eval-pipeline\prepare_chinese_data.py first, then re-run this script."
    }
}

if ($Python -match '^[A-Za-z]:[\\/]' -or $Python -match '[\\/]') {
    if (-not (Test-Path $Python)) {
        throw "Python not found: $Python"
    }
}
elseif (-not (Get-Command $Python -ErrorAction SilentlyContinue)) {
    throw "Python command not found: $Python"
}

if (-not (Test-Path $ModelDir)) {
    throw "Model directory not found: $ModelDir"
}
$ModelDir = (Resolve-Path $ModelDir).Path

if (-not (Test-Path $DataDir)) {
    throw "ZhoBLiMP data directory not found: $DataDir"
}
$DataDir = (Resolve-Path $DataDir).Path

if (-not (Test-Path (Join-Path $ModelDir "tokenizer_config.json"))) {
    & $Python (Join-Path $ProjectRoot "scripts\export_tokenizer.py") --model_dir $ModelDir
    if ($LASTEXITCODE -ne 0) {
        throw "Tokenizer export failed for: $ModelDir"
    }
}

New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null
$OutputDir = (Resolve-Path $OutputDir).Path

Push-Location $EvalRoot
try {
    & $Python -m evaluation_pipeline.sentence_zero_shot.run `
        --model_path_or_name $ModelDir `
        --backend mlm `
        --task zhoblimp `
        --data_path $DataDir `
        --output_dir $OutputDir `
        --batch_size $BatchSize `
        --non_causal_batch_size $NonCausalBatchSize `
        --save_predictions
    if ($LASTEXITCODE -ne 0) {
        throw "Encoder ZhoBLiMP evaluation failed."
    }
}
finally {
    Pop-Location
}

$ModelName = Split-Path -Leaf $ModelDir
$ReportPath = Join-Path $OutputDir "$ModelName\main\zero_shot\mlm\zhoblimp\zhoblimp\best_temperature_report.txt"
$AverageAccuracy = ""
$Temperature = ""

if (Test-Path $ReportPath) {
    $ReportLines = Get-Content $ReportPath
    for ($Index = 0; $Index -lt $ReportLines.Count; $Index++) {
        if ($ReportLines[$Index] -match '^TEMPERATURE:\s+(.+)$') {
            $Temperature = $Matches[1].Trim()
        }
        if ($ReportLines[$Index] -match '^### AVERAGE ACCURACY$' -and ($Index + 1) -lt $ReportLines.Count) {
            $AverageAccuracy = $ReportLines[$Index + 1].Trim()
        }
    }
}

[pscustomobject]@{
    model = $ModelName
    average_accuracy = $AverageAccuracy
    temperature = $Temperature
    report_path = $ReportPath
} | Export-Csv -NoTypeInformation -Encoding UTF8 (Join-Path $OutputDir "summary.csv")

Write-Host ""
Write-Host "Encoder ZhoBLiMP evaluation complete."
Write-Host "Results saved under: $OutputDir"
Write-Host "Summary saved to: $(Join-Path $OutputDir "summary.csv")"
