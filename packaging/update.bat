@echo off
rem QUANT LAB updater: downloads the latest release and replaces this app folder's program files.
rem The work folder (research data and records) lives elsewhere and is never touched.
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0update.ps1" & exit /b
