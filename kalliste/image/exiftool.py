"""Async wrapper for exiftool commands."""
import asyncio
import logging
from typing import List, Tuple

logger = logging.getLogger(__name__)

async def run_exiftool(cmd: List[str], timeout: int = 30) -> Tuple[bool, str, str]:
    """
    Async wrapper for exiftool commands with timeout and interrupt handling.
    Returns (success, stdout, stderr)
    """
    try:
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        
        stdout, stderr = await asyncio.wait_for(
            process.communicate(),
            timeout=timeout
        )
        
        # Decode bytes to strings
        stdout = stdout.decode('utf-8') if stdout else ''
        stderr = stderr.decode('utf-8') if stderr else ''
        
        return process.returncode == 0, stdout, stderr
        
    except asyncio.TimeoutError:
        logger.error("Exiftool command timed out")
        await _cleanup_process(process)
        return False, "", "Timeout"
        
    except asyncio.CancelledError:
        await _cleanup_process(process)
        raise
        
    except Exception as e:
        logger.error(f"Exiftool error: {e}")
        return False, "", str(e)

async def _cleanup_process(process):
    """Helper to clean up subprocess"""
    try:
        process.terminate()
        try:
            await asyncio.wait_for(process.wait(), timeout=2)
        except asyncio.TimeoutError:
            process.kill()
    except Exception as e:
        logger.error(f"Error cleaning up process: {e}")