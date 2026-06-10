param(
    [string]$Python = "python",
    [string]$ExperimentName = "",
    [ValidateSet("", "decoder-causal", "encoder-mlm", "elc-bert", "gpt-bert", "cwt-bert", "mntp-bert")]
    [string]$Method = "",
    [ValidateSet("", "clue", "zhoblimp", "both")]
    [string]$Eval = "",
    [string]$Checkpoint = "final",
    [string]$BatchSize = ""
)

$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Resolve-Path (Join-Path $ScriptDir "..")
$ExperimentsRoot = Join-Path $ProjectRoot "experiments"

$Methods = @(
    [pscustomobject]@{
        Name = "decoder-causal"
        Script = Join-Path $ProjectRoot "pretraining\train_decoder_causal.py"
    },
    [pscustomobject]@{
        Name = "encoder-mlm"
        Script = Join-Path $ProjectRoot "pretraining\train_encoder_mlm.py"
    },
    [pscustomobject]@{
        Name = "elc-bert"
        Script = Join-Path $ProjectRoot "pretraining\train_encoder_elc_bert.py"
    },
    [pscustomobject]@{
        Name = "gpt-bert"
        Script = Join-Path $ProjectRoot "pretraining\train_encoder_gpt_bert.py"
    },
    [pscustomobject]@{
        Name = "cwt-bert"
        Script = Join-Path $ProjectRoot "pretraining\train_encoder_cwt_bert.py"
    },
    [pscustomobject]@{
        Name = "mntp-bert"
        Script = Join-Path $ProjectRoot "pretraining\train_encoder_mntp_bert.py"
    }
)

function Select-FromList {
    param(
        [string]$Title,
        [object[]]$Items,
        [scriptblock]$Label
    )

    Write-Host ""
    Write-Host $Title
    for ($Index = 0; $Index -lt $Items.Count; $Index++) {
        Write-Host "[$($Index + 1)] $(& $Label $Items[$Index])"
    }
    $Choice = [int](Read-Host "Enter number")
    if ($Choice -lt 1 -or $Choice -gt $Items.Count) {
        throw "Invalid choice."
    }
    return $Items[$Choice - 1]
}

function Read-ExperimentName {
    $DefaultName = "exp-" + (Get-Date -Format "yyyyMMdd-HHmmss")
    $Name = Read-Host "Experiment name [$DefaultName]"
    if ([string]::IsNullOrWhiteSpace($Name)) {
        return $DefaultName
    }
    return $Name.Trim()
}

function Find-CheckpointDir {
    param(
        [string]$ModelRoot,
        [string]$CheckpointName
    )

    if ($CheckpointName -eq "best-loss") {
        $Candidates = Get-ChildItem $ModelRoot -Directory -ErrorAction SilentlyContinue |
            Where-Object { $_.Name -eq "final" -or $_.Name -match '^checkpoint-\d+$' }
        $Scored = @()
        foreach ($Candidate in $Candidates) {
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
                    Path = $Candidate.FullName
                    Name = $Candidate.Name
                    Loss = $Loss
                }
            }
        }
        if ($Scored.Count -eq 0) {
            throw "No checkpoint/final with trainer_state loss found under: $ModelRoot"
        }
        $Best = $Scored | Sort-Object Loss, Name | Select-Object -First 1
        Write-Host "Selected best-loss checkpoint: $($Best.Name) (loss=$($Best.Loss))"
        return $Best.Path
    }

    if ($CheckpointName -eq "latest") {
        $Checkpoints = Get-ChildItem $ModelRoot -Directory -ErrorAction SilentlyContinue |
            Where-Object { $_.Name -match '^checkpoint-\d+$' } |
            Sort-Object { [int]($_.Name -replace '^checkpoint-', '') }
        if ($Checkpoints.Count -gt 0) {
            return $Checkpoints[-1].FullName
        }
        $FinalDir = Join-Path $ModelRoot "final"
        if (Test-Path $FinalDir) {
            return (Resolve-Path $FinalDir).Path
        }
        throw "No checkpoint-* or final directory found under: $ModelRoot"
    }

    $Candidate = Join-Path $ModelRoot $CheckpointName
    if (-not (Test-Path $Candidate)) {
        throw "Checkpoint directory not found: $Candidate"
    }
    return (Resolve-Path $Candidate).Path
}

function Import-ResultRows {
    param(
        [string]$Path,
        [string]$EvaluationName
    )

    if (-not (Test-Path $Path)) {
        return @()
    }
    $Rows = Import-Csv $Path
    foreach ($Row in $Rows) {
        $Row | Add-Member -NotePropertyName evaluation -NotePropertyValue $EvaluationName -Force
    }
    return $Rows
}

function Write-SummaryMarkdown {
    param(
        [string]$Path,
        [string]$ExperimentName,
        [string]$MethodName,
        [string]$ModelDir,
        [object[]]$Rows
    )

    $Lines = @()
    $Lines += "# Experiment Summary"
    $Lines += ""
    $CreatedAt = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    $Lines += "- Experiment: ``$ExperimentName``"
    $Lines += "- Method: ``$MethodName``"
    $Lines += "- Model: ``$ModelDir``"
    $Lines += "- Created: ``$CreatedAt``"
    $Lines += ""

    if ($Rows.Count -eq 0) {
        $Lines += "No evaluation rows were found."
    }
    else {
        $Lines += "| evaluation | task/model | accuracy | f1 | mcc | avg_accuracy | report |"
        $Lines += "| --- | --- | ---: | ---: | ---: | ---: | --- |"
        foreach ($Row in $Rows) {
            $TaskOrModel = if ($Row.task) { $Row.task } else { $Row.model }
            $Accuracy = if ($Row.accuracy) { $Row.accuracy } else { "" }
            $F1 = if ($Row.f1) { $Row.f1 } else { "" }
            $Mcc = if ($Row.mcc) { $Row.mcc } else { "" }
            $AverageAccuracy = if ($Row.average_accuracy) { $Row.average_accuracy } else { "" }
            $Report = if ($Row.report_path) { $Row.report_path } elseif ($Row.results_path) { $Row.results_path } else { "" }
            $Lines += "| $($Row.evaluation) | $TaskOrModel | $Accuracy | $F1 | $Mcc | $AverageAccuracy | `$Report` |"
        }
    }

    Set-Content -Path $Path -Value $Lines -Encoding UTF8
}

if ($ExperimentName -eq "") {
    $ExperimentName = Read-ExperimentName
}

if ($Method -eq "") {
    $SelectedMethod = Select-FromList -Title "Select training method:" -Items $Methods -Label { param($Item) $Item.Name }
}
else {
    $SelectedMethod = $Methods | Where-Object { $_.Name -eq $Method } | Select-Object -First 1
    if ($null -eq $SelectedMethod) {
        throw "Unknown method: $Method"
    }
}

if ($Eval -eq "") {
    $EvalOptions = @("clue", "zhoblimp", "both")
    $SelectedEval = Select-FromList -Title "Select evaluation:" -Items $EvalOptions -Label { param($Item) $Item }
    $Eval = $SelectedEval
}

$ExperimentRoot = Join-Path $ExperimentsRoot $ExperimentName
$ModelRoot = Join-Path $ExperimentRoot "models\$($SelectedMethod.Name)"
$LogsRoot = Join-Path $ExperimentRoot "logs\$($SelectedMethod.Name)"
$EvalRoot = Join-Path $ExperimentRoot "eval-res\$($SelectedMethod.Name)"
New-Item -ItemType Directory -Force -Path $ModelRoot | Out-Null
New-Item -ItemType Directory -Force -Path $LogsRoot | Out-Null
New-Item -ItemType Directory -Force -Path $EvalRoot | Out-Null

$Metadata = [ordered]@{
    experiment = $ExperimentName
    method = $SelectedMethod.Name
    model_root = $ModelRoot
    logs_root = $LogsRoot
    eval_root = $EvalRoot
    checkpoint_for_eval = $Checkpoint
    evaluation = $Eval
    batch_size = $BatchSize
    started_at = Get-Date -Format "o"
}
$MetadataPath = Join-Path $ExperimentRoot "experiment.json"
$Metadata | ConvertTo-Json -Depth 4 | Set-Content -Path $MetadataPath -Encoding UTF8

Write-Host ""
Write-Host "=== Experiment ==="
Write-Host "Name: $ExperimentName"
Write-Host "Method: $($SelectedMethod.Name)"
Write-Host "Model root: $ModelRoot"
Write-Host "Eval root: $EvalRoot"
Write-Host ""
Write-Host "Training starts now. Existing checkpoint-* or final folders under model root will be reused by the training script."

$PreviousModelOutputDir = $env:BABYLM_MODEL_OUTPUT_DIR
$PreviousLogsDir = $env:BABYLM_LOGS_DIR
$PreviousBatchSize = $env:BABYLM_BATCH_SIZE
$env:BABYLM_MODEL_OUTPUT_DIR = $ModelRoot
$env:BABYLM_LOGS_DIR = $LogsRoot
if ($BatchSize -ne "") {
    $env:BABYLM_BATCH_SIZE = $BatchSize
}
try {
    & $Python $SelectedMethod.Script
    if ($LASTEXITCODE -ne 0) {
        throw "Training failed for method: $($SelectedMethod.Name)"
    }
}
finally {
    $env:BABYLM_MODEL_OUTPUT_DIR = $PreviousModelOutputDir
    $env:BABYLM_LOGS_DIR = $PreviousLogsDir
    $env:BABYLM_BATCH_SIZE = $PreviousBatchSize
}

$ModelForEval = Find-CheckpointDir -ModelRoot $ModelRoot -CheckpointName $Checkpoint
$SelectedCheckpointName = Split-Path -Leaf $ModelForEval
$AllRows = @()

if ($Eval -eq "clue" -or $Eval -eq "both") {
    $ClueOutput = Join-Path $EvalRoot "$SelectedCheckpointName\clue"
    & (Join-Path $ScriptDir "evaluate_clue.ps1") -Python $Python -ModelDir $ModelForEval -OutputDir $ClueOutput
    if ($LASTEXITCODE -ne 0) {
        throw "CLUE evaluation failed."
    }
    $AllRows += Import-ResultRows -Path (Join-Path $ClueOutput "summary.csv") -EvaluationName "clue"
}

if ($Eval -eq "zhoblimp" -or $Eval -eq "both") {
    $ZhoOutput = Join-Path $EvalRoot "$SelectedCheckpointName\zhoblimp"
    & (Join-Path $ScriptDir "evaluate_zhoblimp.ps1") -Python $Python -ModelDir $ModelForEval -OutputDir $ZhoOutput
    if ($LASTEXITCODE -ne 0) {
        throw "ZhoBLiMP evaluation failed."
    }
    $AllRows += Import-ResultRows -Path (Join-Path $ZhoOutput "summary.csv") -EvaluationName "zhoblimp"
}

$SummaryCsv = Join-Path $ExperimentRoot "summary.csv"
$SummaryMd = Join-Path $ExperimentRoot "summary.md"
if ($AllRows.Count -gt 0) {
    $AllRows | Export-Csv -NoTypeInformation -Encoding UTF8 $SummaryCsv
}
else {
    "" | Set-Content -Path $SummaryCsv -Encoding UTF8
}
Write-SummaryMarkdown -Path $SummaryMd -ExperimentName $ExperimentName -MethodName $SelectedMethod.Name -ModelDir $ModelForEval -Rows $AllRows

$Metadata["completed_at"] = Get-Date -Format "o"
$Metadata["model_for_eval"] = $ModelForEval
$Metadata["selected_checkpoint"] = $SelectedCheckpointName
$Metadata["summary_csv"] = $SummaryCsv
$Metadata["summary_md"] = $SummaryMd
$Metadata | ConvertTo-Json -Depth 4 | Set-Content -Path $MetadataPath -Encoding UTF8

Write-Host ""
Write-Host "Experiment pipeline complete."
Write-Host "Summary CSV: $SummaryCsv"
Write-Host "Summary Markdown: $SummaryMd"
