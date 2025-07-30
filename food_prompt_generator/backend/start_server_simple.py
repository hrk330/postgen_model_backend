#!/usr/bin/env python3
"""
Simple Server Startup Script
============================

A simplified server startup script that handles compatibility issues gracefully.
"""

import os
import sys
import logging
from pathlib import Path

# Add parent directory to path
current_dir = Path(__file__).parent
parent_dir = current_dir.parent
sys.path.insert(0, str(parent_dir))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def setup_compatibility():
    """Setup compatibility layers"""
    try:
        # Import compatibility modules
        import transformers_compatibility
        logger.info("✅ Transformers compatibility loaded")
    except ImportError:
        logger.warning("⚠️  Transformers compatibility module not found")
    
    try:
        import huggingface_compatibility
        logger.info("✅ HuggingFace compatibility loaded")
    except ImportError:
        logger.warning("⚠️  HuggingFace compatibility module not found")

def check_dependencies():
    """Check if all required dependencies are available"""
    try:
        import fastapi
        import uvicorn
        logger.info("✅ Core dependencies available")
        return True
    except ImportError as e:
        logger.error(f"❌ Missing dependency: {e}")
        return False

def main():
    """Main startup function"""
    logger.info("🚀 Starting Twitter Content Generation API Server...")
    
    # Setup compatibility layers
    setup_compatibility()
    
    # Check dependencies
    if not check_dependencies():
        logger.error("❌ Dependencies check failed. Please install requirements.")
        return 1
    
    # Import and start the server
    try:
        # Import the main app
        from main import app
        logger.info("✅ Server components loaded successfully")
        
        import uvicorn
        
        logger.info("🌐 Starting server on http://localhost:8000")
        logger.info("📚 API Documentation: http://localhost:8000/docs")
        logger.info("🏥 Health Check: http://localhost:8000/health")
        
        uvicorn.run(
            app,
            host="0.0.0.0",
            port=8000,
            log_level="info"
        )
        
    except Exception as e:
        logger.error(f"❌ Failed to start server: {e}")
        logger.error("Try running the compatibility test first: python test_compatibility.py")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main()) 