# Food APIs - Interface Launcher (PowerShell)
# Run this script to easily start different interfaces

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  Food APIs - Interface Launcher" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

function Show-Menu {
    Write-Host "Available Interfaces:" -ForegroundColor Yellow
    Write-Host "1. GUI Interface (Visual)" -ForegroundColor White
    Write-Host "2. CLI Interface (Command Line)" -ForegroundColor White
    Write-Host "3. Python Client Test" -ForegroundColor White
    Write-Host "4. Start Both APIs" -ForegroundColor White
    Write-Host "5. Health Check" -ForegroundColor White
    Write-Host "6. Exit" -ForegroundColor White
    Write-Host ""
}

function Start-GUIInterface {
    Write-Host "Starting GUI Interface..." -ForegroundColor Green
    Write-Host "Make sure both APIs are running first!" -ForegroundColor Yellow
    Write-Host ""
    
    try {
        python api_testing_interface.py
    }
    catch {
        Write-Host "Error starting GUI interface: $_" -ForegroundColor Red
    }
}

function Start-CLIInterface {
    Write-Host "Starting CLI Interface..." -ForegroundColor Green
    Write-Host "Make sure both APIs are running first!" -ForegroundColor Yellow
    Write-Host ""
    
    try {
        python cli_interface.py
    }
    catch {
        Write-Host "Error starting CLI interface: $_" -ForegroundColor Red
    }
}

function Test-PythonClient {
    Write-Host "Testing Python Client Library..." -ForegroundColor Green
    Write-Host "Make sure both APIs are running first!" -ForegroundColor Yellow
    Write-Host ""
    
    try {
        python api_client.py
    }
    catch {
        Write-Host "Error testing Python client: $_" -ForegroundColor Red
    }
}

function Start-APIs {
    Write-Host "Starting both APIs..." -ForegroundColor Green
    Write-Host ""
    
    # Check if Python is available
    try {
        $pythonVersion = python --version 2>&1
        Write-Host "Python version: $pythonVersion" -ForegroundColor Gray
    }
    catch {
        Write-Host "Error: Python not found. Please install Python and try again." -ForegroundColor Red
        return
    }
    
    # Start Prompt API
    Write-Host "Starting Prompt API on port 8000..." -ForegroundColor Yellow
    Start-Process powershell -ArgumentList "-NoExit", "-Command", "python prompt_api_server.py" -WindowStyle Normal
    
    # Wait a moment
    Start-Sleep -Seconds 3
    
    # Start Content API
    Write-Host "Starting Content API on port 8001..." -ForegroundColor Yellow
    Start-Process powershell -ArgumentList "-NoExit", "-Command", "python content_api_server.py" -WindowStyle Normal
    
    Write-Host ""
    Write-Host "Both APIs started! You can now use the interfaces." -ForegroundColor Green
    Write-Host "Prompt API: http://localhost:8000" -ForegroundColor Gray
    Write-Host "Content API: http://localhost:8001" -ForegroundColor Gray
    Write-Host ""
}

function Test-APIHealth {
    Write-Host "Testing API Health..." -ForegroundColor Green
    Write-Host ""
    
    $apis = @(
        @{Name="Prompt API"; URL="http://localhost:8000"},
        @{Name="Content API"; URL="http://localhost:8001"}
    )
    
    foreach ($api in $apis) {
        Write-Host "Testing $($api.Name)..." -ForegroundColor Yellow
        try {
            $response = Invoke-RestMethod -Uri "$($api.URL)/health" -Method Get -TimeoutSec 10
            Write-Host "✅ $($api.Name): HEALTHY" -ForegroundColor Green
            Write-Host "   Status: $($response.status)" -ForegroundColor Gray
            Write-Host "   Model Loaded: $($response.is_model_loaded)" -ForegroundColor Gray
            Write-Host "   Uptime: $([math]::Round($response.uptime, 2))s" -ForegroundColor Gray
        }
        catch {
            Write-Host "❌ $($api.Name): ERROR - $_" -ForegroundColor Red
        }
        Write-Host ""
    }
}

# Main loop
do {
    Show-Menu
    $choice = Read-Host "Enter your choice (1-6)"
    
    switch ($choice) {
        "1" { Start-GUIInterface }
        "2" { Start-CLIInterface }
        "3" { Test-PythonClient }
        "4" { Start-APIs }
        "5" { Test-APIHealth }
        "6" { 
            Write-Host "Goodbye!" -ForegroundColor Green
            exit 0
        }
        default { 
            Write-Host "Invalid choice. Please enter a number between 1-6." -ForegroundColor Red
            Write-Host ""
        }
    }
    
    if ($choice -ne "6") {
        Write-Host ""
        Read-Host "Press Enter to continue"
        Clear-Host
    }
} while ($choice -ne "6") 