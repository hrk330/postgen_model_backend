#!/usr/bin/env python3
"""
Fixed Server Startup Script
==========================

This script handles the torch version conflict and starts the server properly.
"""

import os
import sys
import logging
from pathlib import Path

# Add parent directory to path
current_dir = Path(__file__).parent
parent_dir = current_dir.parent
sys.path.insert(0, str(parent_dir))

# Import compatibility modules early to fix huggingface_hub issues
try:
    import transformers_compatibility
    print("✅ Transformers compatibility loaded")
except ImportError:
    print("⚠️  Transformers compatibility module not found")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def check_torch_version():
    """Check and handle torch version conflicts"""
    try:
        import torch
        logger.info(f"PyTorch version: {torch.__version__}")
        
        # Try to import torchaudio, but don't fail if it's missing
        try:
            import torchaudio
            logger.info(f"TorchAudio version: {torchaudio.__version__}")
            
            # Check if there's a version mismatch
            if torch.__version__ != torchaudio.__version__.split('+')[0]:
                logger.warning("⚠️  TorchAudio version mismatch detected")
                logger.warning("This is usually not a problem for our use case")
                logger.info("Continuing with server startup...")
        except ImportError:
            logger.warning("⚠️  TorchAudio not available, but this is optional")
            logger.info("Continuing with server startup...")
        
        return True
    except Exception as e:
        logger.error(f"❌ Torch version check failed: {e}")
        return False

def check_dependencies():
    """Check if all required dependencies are available"""
    try:
        import fastapi
        import uvicorn
        import pandas
        import numpy
        import torch
        logger.info("✅ All core dependencies are available")
        
        # Check transformers separately to handle huggingface_hub issues
        try:
            import transformers
            logger.info("✅ Transformers available")
            
            # Test if cached_download is available (common issue)
            try:
                from huggingface_compatibility import cached_download
                logger.info("✅ HuggingFace Hub compatibility OK")
            except ImportError:
                logger.warning("⚠️  HuggingFace Hub compatibility issue detected")
                logger.warning("Run 'python fix_compatibility.py' to fix this")
                logger.warning("Continuing anyway...")
                
        except ImportError as e:
            logger.warning(f"⚠️  Transformers import issue: {e}")
            logger.warning("This might cause issues with model loading, but continuing...")
        
        return True
    except ImportError as e:
        logger.error(f"❌ Missing dependency: {e}")
        logger.info("Please install dependencies with: pip install -r requirements_combined_2.txt")
        return False

def check_models():
    """Check if required model files exist"""
    parent_dir = Path(__file__).parent.parent
    
    # Check keywords model
    keywords_file = parent_dir / "keywords_model" / "enhanced_keywords_extraction.py"
    if not keywords_file.exists():
        logger.error(f"❌ Keywords model not found: {keywords_file}")
        return False
    
    # Check prompt generator
    prompt_file = parent_dir / "prompt_generator.py"
    if not prompt_file.exists():
        logger.error(f"❌ Prompt generator not found: {prompt_file}")
        return False
    
    # Check content generator
    content_file = parent_dir / "content_generator" / "content_generator_optimized.py"
    if not content_file.exists():
        logger.error(f"❌ Content generator not found: {content_file}")
        return False
    
    logger.info("✅ All model files are available")
    return True

def main():
    """Main startup function"""
    logger.info("🚀 Starting Twitter Content Generation API Server...")
    
    # Check torch version
    if not check_torch_version():
        logger.warning("⚠️  Torch version issue detected, but continuing...")
    
    # Check dependencies
    if not check_dependencies():
        logger.error("❌ Dependencies check failed. Please install requirements.")
        return 1
    
    # Check models
    if not check_models():
        logger.error("❌ Model files check failed.")
        return 1
    
    # Import and start the server
    try:
        # Try to import with error handling
        try:
            # Import from the current directory (backend folder)
            import os
            import sys
            
            # Make sure we're importing from the backend directory
            backend_dir = os.path.dirname(os.path.abspath(__file__))
            if backend_dir not in sys.path:
                sys.path.insert(0, backend_dir)
            
            from main import app
            logger.info("✅ Server components loaded successfully")
        except ImportError as e:
            logger.error(f"❌ Failed to import main app: {e}")
            logger.info("Trying alternative import method...")
            
            # Alternative import method - be more explicit
            try:
                import importlib.util
                main_path = os.path.join(backend_dir, "main.py")
                spec = importlib.util.spec_from_file_location("main", main_path)
                if spec is None:
                    raise ImportError(f"Could not create spec for {main_path}")
                main_module = importlib.util.module_from_spec(spec)
                if spec.loader is None:
                    raise ImportError(f"Could not get loader for {main_path}")
                spec.loader.exec_module(main_module)
                app = main_module.app
                logger.info("✅ Server components loaded successfully (alternative method)")
            except Exception as e2:
                logger.error(f"❌ Alternative import also failed: {e2}")
                logger.error(f"Backend directory: {backend_dir}")
                logger.error(f"Main.py exists: {os.path.exists(os.path.join(backend_dir, 'main.py'))}")
                return 1
        
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
        logger.error("Try using 'python run_server.py' instead for a simpler startup")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main()) 