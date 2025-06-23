"""Temporary debug script to trace ViT-B-32 loading."""

import sys
import logging
import traceback

# Store the original logger class
_original_logger_class = logging.getLoggerClass()

class DebugLogger(logging.Logger):
    """Custom logger that captures stack traces for specific messages."""
    
    def _log(self, level, msg, args, **kwargs):
        # Check if this is the message we're looking for
        if "ViT-B-32" in str(msg) and ("Loaded" in str(msg) or "Loading" in str(msg)):
            # Capture and log the stack trace
            stack = traceback.extract_stack()
            print("\n" + "="*80)
            print(f"🔍 CAUGHT ViT-B-32 LOADING: {msg}")
            print("📍 Stack trace:")
            for frame in stack[-10:]:  # Show last 10 frames
                print(f"  {frame.filename}:{frame.lineno} in {frame.name}")
                if frame.line:
                    print(f"    {frame.line.strip()}")
            print("="*80 + "\n")
        
        # Call the original logging method
        super()._log(level, msg, args, **kwargs)

# Replace the logger class
logging.setLoggerClass(DebugLogger)

# Import this at the very beginning of your main script
print("🐛 Debug logger installed - will trace ViT-B-32 loading")
