@echo off

REM Twitter Content Generation API Startup Script
REM ===========================================

echo 🚀 Starting Twitter Content Generation API...

REM Check if virtual environment exists
if not exist "venv" (
    echo 📦 Creating virtual environment...
    python -m venv venv
)

REM Activate virtual environment
echo 🔧 Activating virtual environment...
call venv\Scripts\activate.bat

REM Install dependencies if requirements.txt exists
if exist "requirements.txt" (
    echo 📥 Installing dependencies...
    pip install -r requirements.txt
)

REM Download required NLP models
echo 🤖 Downloading NLP models...
python -c "import spacy; spacy.cli.download('en_core_web_sm')" 2>nul || echo ⚠️  SpaCy model download failed, continuing...
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')" 2>nul || echo ⚠️  NLTK download failed, continuing...

REM Start the server
echo 🌐 Starting server on http://localhost:8000
echo 📚 API Documentation: http://localhost:8000/docs
echo 🏥 Health Check: http://localhost:8000/health
echo.
echo Press Ctrl+C to stop the server
echo.

python run_server.py

pause 