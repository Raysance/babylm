@echo off
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0evaluate_all_checkpoints.ps1" %*
