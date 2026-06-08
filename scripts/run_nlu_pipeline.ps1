param(
    [string]$Python = "E:\Miniconda3\envs\babylmm\python.exe",
    [switch]$SkipCache,
    [switch]$SkipDecoder,
    [switch]$SkipEncoder,
    [switch]$SkipEval
)

$ErrorActionPreference = "Stop"
$Root = Resolve-Path (Join-Path $PSScriptRoot "..")

Set-Location $Root

if (-not $SkipCache) {
    & $Python ".\preprocess\cache_dataset.py"
}

if (-not $SkipDecoder) {
    & $Python ".\pretraining\decoder\train.py"
}

if (-not $SkipEncoder) {
    & $Python ".\pretraining\encoder\train.py"
}

if (-not $SkipEval) {
    bash ".\scripts\eval_nlu_decoder.sh"
    bash ".\scripts\eval_nlu_encoder.sh"
}
