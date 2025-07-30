#!/bin/bash

# Twitter Content Generation API Startup Script
# ===========================================

echo "🚀 Starting Twitter Content Generation API..."

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python -m venv venv
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Install dependencies if requirements.txt exists
if [ -f "requirements.txt" ]; then
    echo "📥 Installing dependencies..."
    pip install -r requirements.txt
fi

# Download required NLP models
echo "🤖 Downloading NLP models..."
python -c "import spacy; spacy.cli.download('en_core_web_sm')" 2>/dev/null || echo "⚠️  SpaCy model download failed, continuing..."
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')" 2>/dev/null || echo "⚠️  NLTK download failed, continuing..."

# Start the server
echo "🌐 Starting server on http://localhost:8000"
echo "📚 API Documentation: http://localhost:8000/docs"
echo "🏥 Health Check: http://localhost:8000/health"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

python run_server.py 