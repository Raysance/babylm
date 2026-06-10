@echo off
setlocal

echo.
echo Select training method:
echo [1] decoder-causal
echo [2] encoder-mlm
echo [3] elc-bert
echo [4] gpt-bert
echo [5] cwt-bert
echo [6] mntp-bert
echo.
set /p choice=Enter number: 

if "%choice%"=="1" (
    python "%~dp0pretraining\train_decoder_causal.py"
    goto :eof
)
if "%choice%"=="2" (
    python "%~dp0pretraining\train_encoder_mlm.py"
    goto :eof
)
if "%choice%"=="3" (
    python "%~dp0pretraining\train_encoder_elc_bert.py"
    goto :eof
)
if "%choice%"=="4" (
    python "%~dp0pretraining\train_encoder_gpt_bert.py"
    goto :eof
)
if "%choice%"=="5" (
    python "%~dp0pretraining\train_encoder_cwt_bert.py"
    goto :eof
)
if "%choice%"=="6" (
    python "%~dp0pretraining\train_encoder_mntp_bert.py"
    goto :eof
)

echo Invalid choice.
exit /b 1
