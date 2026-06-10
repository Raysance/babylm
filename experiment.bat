@echo off
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0evaluation\run_experiment.ps1" %*
