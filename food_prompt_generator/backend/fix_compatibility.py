#!/usr/bin/env python3
"""
HuggingFace Hub Compatibility Fix
=================================

This script fixes the compatibility issue with huggingface_hub where
cached_download has been removed in newer versions.
"""

import sys
import os
from pathlib import Path

def fix_huggingface_compatibility():
    """Fix huggingface_hub compatibility issues"""
    
    # Add the current directory to Python path
    current_dir = Path(__file__).parent
    sys.path.insert(0, str(current_dir))
    
    # Create a compatibility module
    compatibility_code = '''
"""
HuggingFace Hub Compatibility Module
====================================

This module provides compatibility for older huggingface_hub APIs
that have been removed in newer versions.
"""

import os
import sys
from pathlib import Path
from typing import Optional, Union, Dict, Any

# Try to import from huggingface_hub
try:
    from huggingface_hub import hf_hub_download, snapshot_download
    from huggingface_hub.utils import HfHubHTTPError
    
    # Create a compatibility function for cached_download
    def cached_download(
        repo_id: str,
        filename: str,
        cache_dir: Optional[Union[str, Path]] = None,
        force_download: bool = False,
        resume_download: bool = True,
        proxies: Optional[Dict[str, str]] = None,
        local_files_only: bool = False,
        token: Optional[Union[bool, str]] = None,
        **kwargs
    ) -> str:
        """
        Compatibility function for cached_download.
        Uses hf_hub_download instead.
        """
        return hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            cache_dir=cache_dir,
            force_download=force_download,
            resume_download=resume_download,
            proxies=proxies,
            local_files_only=local_files_only,
            token=token,
            **kwargs
        )
    
    print("✅ HuggingFace Hub compatibility layer created successfully")
    
except ImportError as e:
    print(f"❌ Could not import huggingface_hub: {e}")
    print("Please install with: pip install huggingface_hub")
    
    # Create a dummy function that raises an error
    def cached_download(*args, **kwargs):
        raise ImportError("huggingface_hub not available. Install with: pip install huggingface_hub")
    
    def hf_hub_download(*args, **kwargs):
        raise ImportError("huggingface_hub not available. Install with: pip install huggingface_hub")
    
    def snapshot_download(*args, **kwargs):
        raise ImportError("huggingface_hub not available. Install with: pip install huggingface_hub")

# Make functions available at module level
__all__ = ['cached_download', 'hf_hub_download', 'snapshot_download']
'''
    
    # Write the compatibility module
    compatibility_file = current_dir / "huggingface_compatibility.py"
    with open(compatibility_file, 'w') as f:
        f.write(compatibility_code)
    
    print(f"✅ Created compatibility module: {compatibility_file}")
    
    # Update the start_server_fixed.py to use the compatibility layer
    server_file = current_dir / "start_server_fixed.py"
    if server_file.exists():
        with open(server_file, 'r') as f:
            content = f.read()
        
        # Replace the direct import with our compatibility layer
        old_import = "from huggingface_hub import cached_download"
        new_import = "from huggingface_compatibility import cached_download"
        
        if old_import in content:
            content = content.replace(old_import, new_import)
            
            with open(server_file, 'w') as f:
                f.write(content)
            
            print("✅ Updated start_server_fixed.py to use compatibility layer")
        else:
            print("ℹ️  No direct cached_download import found in start_server_fixed.py")
    
    print("✅ HuggingFace Hub compatibility fix completed!")
    print("You can now run: python start_server_fixed.py")

if __name__ == "__main__":
    fix_huggingface_compatibility() 