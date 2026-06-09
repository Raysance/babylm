param(
    [string]$Python = "E:\Miniconda3\envs\babylmm\python.exe",
    [string]$ModelRoot = "",
    [string]$DataDir = ""
)

$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Resolve-Path (Join-Path $ScriptDir "..\..")
$EvalRoot = Join-Path $ProjectRoot "eval-pipeline"
$OutputDir = Join-Path $ScriptDir "eval-results-all"
$SummaryPath = Join-Path $OutputDir "summary.csv"

if ($ModelRoot -eq "") {
    $ModelRoot = Join-Path $ProjectRoot "models\pretrain\decoder"
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

if (-not (Test-Path $ModelRoot)) {
    throw "Model root not found: $ModelRoot"
}

New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null

$CheckpointDirs = Get-ChildItem $ModelRoot -Directory |
    Where-Object { $_.Name -match '^checkpoint-\d+$' } |
    Sort-Object { [int]($_.Name -replace '^checkpoint-', '') }

$ModelDirs = @($CheckpointDirs)
$FinalDir = Join-Path $ModelRoot "final"
if (Test-Path $FinalDir) {
    $ModelDirs += Get-Item $FinalDir
}

if ($ModelDirs.Count -eq 0) {
    throw "No checkpoint-* or final directories found under: $ModelRoot"
}

$Rows = @()

foreach ($ModelDirItem in $ModelDirs) {
    $ModelDir = $ModelDirItem.FullName
    $ModelName = $ModelDirItem.Name

    Write-Host ""
    Write-Host "=== Evaluating $ModelName ==="

    if (-not (Test-Path (Join-Path $ModelDir "tokenizer_config.json"))) {
        & $Python (Join-Path $ProjectRoot "scripts\export_tokenizer.py") --model_dir $ModelDir
    }

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

    $ReportPath = Join-Path $OutputDir "$ModelName\main\zero_shot\causal\zhoblimp\zhoblimp\best_temperature_report.txt"
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

    $Rows += [pscustomobject]@{
        model = $ModelName
        average_accuracy = $AverageAccuracy
        temperature = $Temperature
        report_path = $ReportPath
    }
}

$Rows | Export-Csv -NoTypeInformation -Encoding UTF8 $SummaryPath

Write-Host ""
Write-Host "All evaluations complete."
Write-Host "Summary saved to: $SummaryPath"
