#!/usr/bin/env python3
"""
Server Startup Script
=====================

This script starts the Twitter Content Generation API server
with proper configuration and error handling.
"""

import uvicorn
import logging
import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('api_server.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

def main():
    """Start the FastAPI server"""
    try:
        logger.info("🚀 Starting Twitter Content Generation API Server...")
        
        # Server configuration
        host = os.getenv("HOST", "0.0.0.0")
        port = int(os.getenv("PORT", 8000))
        reload = os.getenv("RELOAD", "false").lower() == "true"
        
        logger.info(f"📍 Server will run on {host}:{port}")
        logger.info(f"🔄 Auto-reload: {reload}")
        
        # Start server
        uvicorn.run(
            "main:app",
            host=host,
            port=port,
            reload=reload,
            log_level="info",
            access_log=True
        )
        
    except KeyboardInterrupt:
        logger.info("⏹️  Server stopped by user")
    except Exception as e:
        logger.error(f"❌ Server failed to start: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 