@echo off
echo ========================================
echo   Food APIs - Interface Launcher
echo ========================================
echo.

echo Available Interfaces:
echo 1. GUI Interface (Visual)
echo 2. CLI Interface (Command Line)
echo 3. Python Client Test
echo 4. Start Both APIs
echo 5. Exit
echo.

set /p choice="Enter your choice (1-5): "

if "%choice%"=="1" (
    echo.
    echo Starting GUI Interface...
    echo Make sure both APIs are running first!
    echo.
    python api_testing_interface.py
) else if "%choice%"=="2" (
    echo.
    echo Starting CLI Interface...
    echo Make sure both APIs are running first!
    echo.
    python cli_interface.py
) else if "%choice%"=="3" (
    echo.
    echo Testing Python Client Library...
    echo Make sure both APIs are running first!
    echo.
    python api_client.py
) else if "%choice%"=="4" (
    echo.
    echo Starting both APIs...
    echo.
    echo Starting Prompt API on port 8000...
    start "Prompt API" cmd /k "python prompt_api_server.py"
    timeout /t 3 /nobreak >nul
    echo.
    echo Starting Content API on port 8001...
    start "Content API" cmd /k "python content_api_server.py"
    echo.
    echo Both APIs started! You can now use the interfaces.
    echo.
    pause
) else if "%choice%"=="5" (
    echo.
    echo Goodbye!
    exit /b 0
) else (
    echo.
    echo Invalid choice. Please enter a number between 1-5.
    echo.
    pause
    goto :eof
)

echo.
pause 