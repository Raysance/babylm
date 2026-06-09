param(
    [string]$Python = "python",
    [string]$ModelRoot = "",
    [string]$DataRoot = "",
    [string]$LearningRate = "3e-5",
    [string]$BatchSize = "64",
    [string]$MaxEpochs = "5",
    [string]$WscEpochs = "5",
    [string]$Seed = "42",
    [string]$SequenceLength = "128"
)

$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Resolve-Path (Join-Path $ScriptDir "..\..")
$EvalRoot = Join-Path $ProjectRoot "eval-pipeline"
$OutputDir = Join-Path $ScriptDir "eval-results-all"
$SaveDir = Join-Path $OutputDir "finetuned-models"
$SummaryPath = Join-Path $OutputDir "summary.csv"

if ($ModelRoot -eq "") {
    $ModelRoot = Join-Path $ProjectRoot "models\pretrain\encoder"
}

if ($DataRoot -eq "") {
    $PreferredDataRoot = Join-Path $EvalRoot "evaluation_data\full_eval\clue"
    $NestedDataRoot = Join-Path $EvalRoot "evaluation_data\full_eval\full_eval\clue"

    if (Test-Path $PreferredDataRoot) {
        $DataRoot = $PreferredDataRoot
    }
    elseif (Test-Path $NestedDataRoot) {
        $DataRoot = $NestedDataRoot
    }
    else {
        throw "CLUE data not found. Run eval-pipeline\prepare_chinese_data.py first, then re-run this script."
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

if (-not (Test-Path $ModelRoot)) {
    throw "Model root not found: $ModelRoot"
}

New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null
New-Item -ItemType Directory -Force -Path $SaveDir | Out-Null

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

$Tasks = @(
    [pscustomobject]@{
        Name = "afqmc"
        NumLabels = "2"
        Epochs = $MaxEpochs
        Metrics = @("accuracy", "f1", "mcc")
        MetricForValid = "accuracy"
    },
    [pscustomobject]@{
        Name = "ocnli"
        NumLabels = "3"
        Epochs = $MaxEpochs
        Metrics = @("accuracy")
        MetricForValid = "accuracy"
    },
    [pscustomobject]@{
        Name = "tnews"
        NumLabels = "15"
        Epochs = $MaxEpochs
        Metrics = @("accuracy")
        MetricForValid = "accuracy"
    },
    [pscustomobject]@{
        Name = "cluewsc2020"
        NumLabels = "2"
        Epochs = $WscEpochs
        Metrics = @("accuracy", "f1", "mcc")
        MetricForValid = "accuracy"
    }
)

$Rows = @()

foreach ($ModelDirItem in $ModelDirs) {
    $ModelDir = $ModelDirItem.FullName
    $ModelName = $ModelDirItem.Name

    Write-Host ""
    Write-Host "=== Evaluating $ModelName ==="

    if (-not (Test-Path (Join-Path $ModelDir "tokenizer_config.json"))) {
        & $Python (Join-Path $ProjectRoot "scripts\export_tokenizer.py") --model_dir $ModelDir
        if ($LASTEXITCODE -ne 0) {
            throw "Tokenizer export failed for: $ModelDir"
        }
    }

    Push-Location $EvalRoot
    try {
        foreach ($Task in $Tasks) {
            Write-Host ""
            Write-Host "=== Evaluating $ModelName on $($Task.Name) ==="

            $TrainData = Join-Path $DataRoot "$($Task.Name).train.jsonl"
            $ValidData = Join-Path $DataRoot "$($Task.Name).valid.jsonl"

            if (-not (Test-Path $TrainData)) {
                throw "Train data not found: $TrainData"
            }
            if (-not (Test-Path $ValidData)) {
                throw "Valid data not found: $ValidData"
            }

            & $Python -m evaluation_pipeline.finetune.run `
                --model_name_or_path $ModelDir `
                --train_data $TrainData `
                --valid_data $ValidData `
                --predict_data $ValidData `
                --task $Task.Name `
                --num_labels $Task.NumLabels `
                --batch_size $BatchSize `
                --learning_rate $LearningRate `
                --num_epochs $Task.Epochs `
                --sequence_length $SequenceLength `
                --results_dir $OutputDir `
                --save `
                --save_dir $SaveDir `
                --metrics $Task.Metrics `
                --metric_for_valid $Task.MetricForValid `
                --seed $Seed `
                --verbose
            if ($LASTEXITCODE -ne 0) {
                throw "Evaluation failed for model $ModelName task $($Task.Name)"
            }

            $ResultsPath = Join-Path $OutputDir "$ModelName\main\finetune\$($Task.Name)\results.txt"
            $Metrics = @{}

            if (Test-Path $ResultsPath) {
                foreach ($Line in Get-Content $ResultsPath) {
                    if ($Line -match '^([^:]+):\s+(.+)$') {
                        $Metrics[$Matches[1].Trim()] = $Matches[2].Trim()
                    }
                }
            }

            $Rows += [pscustomobject]@{
                model = $ModelName
                task = $Task.Name
                accuracy = $Metrics["accuracy"]
                f1 = $Metrics["f1"]
                mcc = $Metrics["mcc"]
                results_path = $ResultsPath
            }
        }
    }
    finally {
        Pop-Location
    }
}

$Rows | Export-Csv -NoTypeInformation -Encoding UTF8 $SummaryPath

Write-Host ""
Write-Host "All encoder evaluations complete."
Write-Host "Summary saved to: $SummaryPath"
