#!/usr/bin/env python3
"""
Test Compatibility Script
========================

This script tests the compatibility fixes before running the full server.
"""

import sys
import os
from pathlib import Path

# Add current directory to path
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

# Import compatibility modules FIRST to patch huggingface_hub early
def setup_compatibility():
    """Setup compatibility layers early"""
    try:
        # Import huggingface compatibility first
        import huggingface_compatibility
        print("✅ HuggingFace compatibility loaded early")
        
        # Import transformers compatibility
        import transformers_compatibility
        print("✅ Transformers compatibility loaded early")
        
        return True
    except Exception as e:
        print(f"❌ Early compatibility setup failed: {e}")
        return False

def test_huggingface_compatibility():
    """Test huggingface_hub compatibility"""
    print("🧪 Testing HuggingFace Hub compatibility...")
    
    try:
        # Test the compatibility module
        import huggingface_compatibility
        print("✅ HuggingFace compatibility module imported successfully")
        
        # Test cached_download function
        from huggingface_compatibility import cached_download
        print("✅ cached_download function available")
        
        return True
    except Exception as e:
        print(f"❌ HuggingFace compatibility test failed: {e}")
        return False

def test_transformers_compatibility():
    """Test transformers compatibility"""
    print("🧪 Testing Transformers compatibility...")
    
    try:
        # Test the compatibility module
        import transformers_compatibility
        print("✅ Transformers compatibility module imported successfully")
        
        return True
    except Exception as e:
        print(f"❌ Transformers compatibility test failed: {e}")
        return False

def test_model_imports():
    """Test model imports"""
    print("🧪 Testing model imports...")
    
    try:
        # Add parent directory to path
        parent_dir = current_dir.parent
        sys.path.insert(0, str(parent_dir))
        
        # Test importing the main modules
        from keywords_model.enhanced_keywords_extraction import EnhancedKeywordsExtraction
        print("✅ Keywords model imported successfully")
        
        from prompt_generator import PromptGenerator
        print("✅ Prompt generator imported successfully")
        
        from content_generator.content_generator_optimized import OptimizedContentGenerator
        print("✅ Content generator imported successfully")
        
        return True
    except Exception as e:
        print(f"❌ Model import test failed: {e}")
        return False

def test_fastapi_import():
    """Test FastAPI import"""
    print("🧪 Testing FastAPI import...")
    
    try:
        from fastapi import FastAPI
        print("✅ FastAPI imported successfully")
        return True
    except Exception as e:
        print(f"❌ FastAPI import test failed: {e}")
        return False

def main():
    """Run all compatibility tests"""
    print("🚀 Starting compatibility tests...")
    print("=" * 50)
    
    # Setup compatibility first
    if not setup_compatibility():
        print("❌ Early compatibility setup failed")
        return False
    
    tests = [
        test_huggingface_compatibility,
        test_transformers_compatibility,
        test_fastapi_import,
        test_model_imports
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"❌ Test failed with exception: {e}")
    
    print("=" * 50)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("✅ All compatibility tests passed! Server should start successfully.")
        print("You can now run: python start_server_fixed.py")
        return True
    else:
        print("❌ Some compatibility tests failed. Please check the errors above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 