param(
    [Parameter(Mandatory = $true)]
    [string]$ModelDir,
    [string]$Python = "python",
    [string]$DataDir = "",
    [string]$OutputDir = "",
    [string]$Backend = "auto",
    [string]$BatchSize = "64",
    [string]$NonCausalBatchSize = "64",
    [switch]$NoNormalize
)

$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Resolve-Path (Join-Path $ScriptDir "..")
$EvalRoot = Join-Path $ProjectRoot "eval-pipeline"

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
        throw "ZhoBLiMP data not found. Run eval-pipeline\prepare_chinese_data.py first."
    }
}

if ($OutputDir -eq "") {
    $OutputDir = Join-Path $ProjectRoot "eval-res\manual\zhoblimp"
}

if (-not (Test-Path $ModelDir)) {
    throw "Model directory not found: $ModelDir"
}
$ModelDir = (Resolve-Path $ModelDir).Path
$DataDir = (Resolve-Path $DataDir).Path

if ($Backend -eq "auto") {
    $ConfigPath = Join-Path $ModelDir "config.json"
    if (-not (Test-Path $ConfigPath)) {
        throw "Cannot infer backend because config.json was not found: $ConfigPath"
    }
    $Config = Get-Content -Raw $ConfigPath | ConvertFrom-Json
    $ModelType = [string]$Config.model_type
    if ($ModelDir -match "mntp-bert") {
        $Backend = "mntp"
    }
    elseif ($ModelType -match "gpt") {
        $Backend = "causal"
    }
    else {
        $Backend = "mlm"
    }
}

if ($Backend -notin @("causal", "mlm", "mntp")) {
    throw "Unsupported ZhoBLiMP backend: $Backend. Use causal, mlm, mntp, or auto."
}

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
    $CommandArgs = @(
        "-m", "evaluation_pipeline.sentence_zero_shot.run",
        "--model_path_or_name", $ModelDir,
        "--backend", $Backend,
        "--task", "zhoblimp",
        "--data_path", $DataDir,
        "--output_dir", $OutputDir,
        "--batch_size", $BatchSize,
        "--non_causal_batch_size", $NonCausalBatchSize,
        "--save_predictions"
    )
    if (-not $NoNormalize) {
        $CommandArgs += "--normalize_scores"
    }

    & $Python @CommandArgs
    if ($LASTEXITCODE -ne 0) {
        throw "ZhoBLiMP evaluation failed."
    }
}
finally {
    Pop-Location
}

$ModelName = Split-Path -Leaf $ModelDir
$ReportPath = Join-Path $OutputDir "$ModelName\main\zero_shot\$Backend\zhoblimp\zhoblimp\best_temperature_report.txt"
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
    backend = $Backend
    normalized = -not $NoNormalize
    average_accuracy = $AverageAccuracy
    temperature = $Temperature
    report_path = $ReportPath
} | Export-Csv -NoTypeInformation -Encoding UTF8 (Join-Path $OutputDir "summary.csv")

Write-Host ""
Write-Host "ZhoBLiMP evaluation complete."
Write-Host "Results saved under: $OutputDir"
