param(
    [Parameter(Mandatory = $true)]
    [string]$ModelDir,
    [string]$Python = "python",
    [string]$DataRoot = "",
    [string]$OutputDir = "",
    [string]$LearningRate = "3e-5",
    [string]$BatchSize = "64",
    [string]$MaxEpochs = "5",
    [string]$WscEpochs = "5",
    [string]$Seed = "42",
    [string]$SequenceLength = "128"
)

$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Resolve-Path (Join-Path $ScriptDir "..")
$EvalRoot = Join-Path $ProjectRoot "eval-pipeline"

if ($DataRoot -eq "") {
    $PreferredDataRoot = Join-Path $ProjectRoot "evaluation_data\full_eval\clue"
    $EvalPipelineDataRoot = Join-Path $EvalRoot "evaluation_data\full_eval\clue"
    $NestedDataRoot = Join-Path $EvalRoot "evaluation_data\full_eval\full_eval\clue"

    if (Test-Path $PreferredDataRoot) {
        $DataRoot = $PreferredDataRoot
    }
    elseif (Test-Path $EvalPipelineDataRoot) {
        $DataRoot = $EvalPipelineDataRoot
    }
    elseif (Test-Path $NestedDataRoot) {
        $DataRoot = $NestedDataRoot
    }
    else {
        throw "CLUE data not found. Run eval-pipeline\prepare_chinese_data.py first."
    }
}

if ($OutputDir -eq "") {
    $OutputDir = Join-Path $ProjectRoot "eval-res\manual\clue"
}

if (-not (Test-Path $ModelDir)) {
    throw "Model directory not found: $ModelDir"
}
$ModelDir = (Resolve-Path $ModelDir).Path
$DataRoot = (Resolve-Path $DataRoot).Path

$ConfigPath = Join-Path $ModelDir "config.json"
if (-not (Test-Path $ConfigPath)) {
    throw "Cannot infer model type because config.json was not found: $ConfigPath"
}
$Config = Get-Content -Raw $ConfigPath | ConvertFrom-Json
$ModelType = [string]$Config.model_type
$IsCausal = $ModelType -match "gpt"

if (-not (Test-Path (Join-Path $ModelDir "tokenizer_config.json"))) {
    & $Python (Join-Path $ProjectRoot "scripts\export_tokenizer.py") --model_dir $ModelDir
    if ($LASTEXITCODE -ne 0) {
        throw "Tokenizer export failed for: $ModelDir"
    }
}

New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null
$OutputDir = (Resolve-Path $OutputDir).Path
$SaveDir = Join-Path $OutputDir "finetuned-models"
New-Item -ItemType Directory -Force -Path $SaveDir | Out-Null
$SaveDir = (Resolve-Path $SaveDir).Path
$SummaryPath = Join-Path $OutputDir "summary.csv"

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
$ModelName = Split-Path -Leaf $ModelDir

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

        $CommandArgs = @(
            "-m", "evaluation_pipeline.finetune.run",
            "--model_name_or_path", $ModelDir,
            "--train_data", $TrainData,
            "--valid_data", $ValidData,
            "--predict_data", $ValidData,
            "--task", $Task.Name,
            "--num_labels", $Task.NumLabels,
            "--batch_size", $BatchSize,
            "--learning_rate", $LearningRate,
            "--num_epochs", $Task.Epochs,
            "--sequence_length", $SequenceLength,
            "--results_dir", $OutputDir,
            "--save",
            "--save_dir", $SaveDir,
            "--metrics"
        )
        $CommandArgs += $Task.Metrics
        $CommandArgs += @(
            "--metric_for_valid", $Task.MetricForValid,
            "--seed", $Seed,
            "--verbose"
        )
        if ($IsCausal) {
            $CommandArgs += @("--causal", "--take_final")
        }

        & $Python @CommandArgs
        if ($LASTEXITCODE -ne 0) {
            throw "Evaluation failed for task: $($Task.Name)"
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
            model_type = $ModelType
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

$Rows | Export-Csv -NoTypeInformation -Encoding UTF8 $SummaryPath

Write-Host ""
Write-Host "CLUE evaluation complete."
Write-Host "Results saved under: $OutputDir"
