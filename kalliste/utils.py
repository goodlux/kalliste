"""Utility functions for error handling and logging."""
from typing import Optional, Type
import traceback
from rich.console import Console
from rich.traceback import Traceback
from rich.panel import Panel
from rich.text import Text
from rich.logging import RichHandler as BaseRichHandler
import logging
from functools import wraps

# Create a rich console for error display
console = Console(stderr=True)

class RichHandler(BaseRichHandler):
    """A logging handler that uses rich for output."""
    
    def __init__(self, level: int = logging.NOTSET):
        super().__init__(level)
        self.console = Console(stderr=True)
        
    def emit(self, record: logging.LogRecord):
        try:
            # Format with file location but keep it concise
            location = f"[dim][{record.filename}:{record.lineno}][/]"
            
            # Choose style based on level
            if record.levelno >= logging.ERROR:
                level_style = "bold red"
                # For errors, only show the error message here
                # The traceback will be handled by format_error
                msg = record.getMessage()
            elif record.levelno >= logging.WARNING:
                level_style = "yellow"
                msg = record.getMessage()
            elif record.levelno >= logging.INFO:
                level_style = "green"
                msg = record.getMessage()
            else:
                level_style = "blue"
                msg = record.getMessage()
            
            # Format and print the message
            formatted_msg = f"{location} [{level_style}]{record.levelname}[/] {msg}"
            self.console.print(formatted_msg)
                
        except Exception:
            self.handleError(record)

def format_error(e: Exception, title: Optional[str] = None, show_locals: bool = False) -> None:
    """Format an exception with rich styling.
    
    Args:
        e: The exception to format
        title: Optional custom title for the error panel
        show_locals: Whether to show local variables in the traceback
    """
    panel_title = title or f"[red bold]{e.__class__.__name__}[/]"
    
    # Get the most relevant parts of the traceback
    tb = Traceback.from_exception(
        type(e),
        e,
        e.__traceback__,
        show_locals=show_locals,  # Only show locals when explicitly requested
        width=100,
        extra_lines=1,  # Reduced from 3
        theme="monokai",
        word_wrap=True
    )
    
    # Create a panel with the error message
    error_text = Text(str(e))
    error_panel = Panel(
        error_text,
        title=panel_title,
        border_style="red",
        padding=(1, 2)
    )
    
    # Print both the error message and traceback
    console.print(error_panel)
    console.print(tb)

def handle_exceptions(logger=None, show_locals=False):
    """Decorator for consistent exception handling.
    
    Args:
        logger: Optional logger instance to use
        show_locals: Whether to show local variables in traceback
    """
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                # Log the error if we have a logger
                if logger:
                    logger.error(str(e))
                
                # Format and display the error
                format_error(e, show_locals=show_locals)
                
                # Re-raise the exception
                raise
        return wrapper
    return decorator

# Configure root logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
    handlers=[RichHandler()]
)

# Enable detailed DEBUG logging for taggers
tagger_logger = logging.getLogger('kalliste.taggers')
tagger_logger.setLevel(logging.DEBUG)

# Create a special debug handler for taggers
debug_handler = RichHandler(level=logging.DEBUG)
debug_formatter = logging.Formatter(
    '%(asctime)s - %(name)s - [%(filename)s:%(lineno)d] - %(levelname)s - %(message)s'
)
debug_handler.setFormatter(debug_formatter)

# Remove any existing handlers and add our new debug handler
tagger_logger.handlers = []
tagger_logger.addHandler(debug_handler)

# Prevent debug messages from propagating up to parent loggers
tagger_logger.propagate = False