@echo off
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0evaluation\select_and_evaluate.ps1" %*
