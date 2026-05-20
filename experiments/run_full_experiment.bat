@echo off
REM Execute system ablation experiment - Full Run
REM This runs with all samples in the dataset

cd /d D:\PromptInjectionDetection
echo ================================================================================
echo SYSTEM ABLATION EXPERIMENT - FULL RUN
echo ================================================================================
echo.
echo Running with full dataset (all DPI and IPI samples)...
echo This may take several minutes...
echo.

D:\AnaConda\envs\prompt-injection\python.exe experiments\execute_system_ablation.py

echo.
echo ================================================================================
echo FULL EXPERIMENT COMPLETED
echo ================================================================================
pause
