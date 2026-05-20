@echo off
REM Execute system ablation experiment - Dry Run
REM This runs with limited samples (100 DPI + 100 IPI) for quick testing

cd /d D:\PromptInjectionDetection
echo ================================================================================
echo SYSTEM ABLATION EXPERIMENT - DRY RUN
echo ================================================================================
echo.
echo Running with 100 DPI samples and 100 IPI samples...
echo.

D:\AnaConda\envs\prompt-injection\python.exe experiments\execute_system_ablation.py --dry-run

echo.
echo ================================================================================
echo DRY RUN COMPLETED
echo ================================================================================
pause
