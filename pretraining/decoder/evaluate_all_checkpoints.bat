@echo off
powershell -ExecutionPolicy Bypass -File "%~dp0evaluate_all_checkpoints.ps1" %*
