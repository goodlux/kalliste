"""Temporary script to trace model loading by patching open_clip."""

import sys
import traceback

# Store the original import function
_original_import = __builtins__.__import__

def _patched_import(name, *args, **kwargs):
    """Patched import that traces open_clip loading."""
    module = _original_import(name, *args, **kwargs)
    
    # If this is open_clip being imported, patch its functions
    if name == 'open_clip' or name.startswith('open_clip.'):
        print(f"\n🔍 IMPORT DETECTED: {name}")
        print("📍 Import stack trace:")
        for frame in traceback.extract_stack()[-10:-1]:
            print(f"  {frame.filename}:{frame.lineno} in {frame.name}")
    
    return module

# Replace the import function
__builtins__.__import__ = _patched_import

print("🐛 Import tracer installed - will trace open_clip imports")
