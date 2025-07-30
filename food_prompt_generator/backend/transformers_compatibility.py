"""
Transformers Compatibility Module
================================

This module provides compatibility for transformers and huggingface_hub
import issues that may arise from version mismatches.
"""

import sys
import os
from pathlib import Path
from typing import Optional, Union, Dict, Any

# Try to patch any potential huggingface_hub issues
try:
    # Import transformers first
    import transformers
    
    # Import our compatibility module
    try:
        from huggingface_compatibility import cached_download, hf_hub_download, snapshot_download
        
        # Patch transformers if it tries to import cached_download
        try:
            import transformers.utils.hub
            # Use setattr to avoid type checker issues
            setattr(transformers.utils.hub, 'cached_download', cached_download)
        except (ImportError, AttributeError):
            pass
        
        print("✅ Transformers and HuggingFace Hub compatibility layer created successfully")
        
    except ImportError as e:
        print(f"❌ Could not import huggingface_compatibility: {e}")
        print("Please ensure huggingface_compatibility.py is available")
        
        # Create dummy functions
        def cached_download(*args: Any, **kwargs: Any) -> str:
            raise ImportError("huggingface_compatibility not available")
        
        def hf_hub_download(*args: Any, **kwargs: Any) -> str:
            raise ImportError("huggingface_compatibility not available")
        
        def snapshot_download(*args: Any, **kwargs: Any) -> str:
            raise ImportError("huggingface_compatibility not available")
    
    print("✅ Transformers compatibility check completed")
    
except ImportError as e:
    print(f"❌ Could not import transformers: {e}")
    print("Please install with: pip install transformers")

# Make functions available at module level
__all__ = ['cached_download', 'hf_hub_download', 'snapshot_download'] 