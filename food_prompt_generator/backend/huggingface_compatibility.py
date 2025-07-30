"""
HuggingFace Hub Compatibility Module
====================================

This module provides compatibility for older huggingface_hub APIs
that have been removed in newer versions.
"""

import os
import sys
from pathlib import Path
from typing import Optional, Union, Dict, Any, Callable

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
        **kwargs: Any
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

    # Patch the huggingface_hub module itself
    try:
        import huggingface_hub
        # Use setattr to avoid type checker issues
        setattr(huggingface_hub, 'cached_download', cached_download)
        print("✅ Patched huggingface_hub module with cached_download")
    except Exception as e:
        print(f"⚠️  Could not patch huggingface_hub module: {e}")

    print("✅ HuggingFace Hub compatibility layer created successfully")

except ImportError as e:
    print(f"❌ Could not import huggingface_hub: {e}")
    print("Please install with: pip install huggingface_hub")

    # Create dummy functions that raise errors
    def _dummy_cached_download(*args: Any, **kwargs: Any) -> str:
        raise ImportError("huggingface_hub not available. Install with: pip install huggingface_hub")

    def _dummy_hf_hub_download(*args: Any, **kwargs: Any) -> str:
        raise ImportError("huggingface_hub not available. Install with: pip install huggingface_hub")

    def _dummy_snapshot_download(*args: Any, **kwargs: Any) -> str:
        raise ImportError("huggingface_hub not available. Install with: pip install huggingface_hub")
    
    # Assign to the expected names
    cached_download = _dummy_cached_download
    hf_hub_download = _dummy_hf_hub_download
    snapshot_download = _dummy_snapshot_download

# Make functions available at module level
__all__ = ['cached_download', 'hf_hub_download', 'snapshot_download'] 