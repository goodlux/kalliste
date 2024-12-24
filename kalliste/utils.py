"""Utility functions for error handling and logging."""
from typing import Optional, Type
import traceback
from rich.console import Console
from rich.traceback import Traceback
from rich.panel import Panel
from rich.text import Text
import logging

# Create a rich console for error display
console = Console(stderr=True)

def format_error(e: Exception, title: Optional[str] = None) -> None:
    """Format an exception with rich styling.
    
    Args:
        e: The exception to format
        title: Optional title for the error panel
    """
    # Create panel title
    panel_title = title or f"[red bold]{e.__class__.__name__}[/]"
    
    # Get styled traceback
    tb = Traceback.from_exception(
        type(e),
        e,
        e.__traceback__,
        show_locals=True,
        width=100,
        extra_lines=3,
        theme="monokai",  # or "solarized-dark", "native", etc.
        word_wrap=True
    )
    
    # Create error message panel
    error_text = Text(str(e))
    error_panel = Panel(
        error_text,
        title=panel_title,
        border_style="red",
        padding=(1, 2)
    )
    
    # Print error summary and traceback
    console.print(error_panel)
    console.print(tb)

class RichHandler(logging.Handler):
    """A logging handler that uses rich for output."""
    
    def __init__(self, level: int = logging.NOTSET):
        super().__init__(level)
        self.console = Console(stderr=True)
        
    def emit(self, record: logging.LogRecord):
        try:
            # Get the message
            msg = self.format(record)
            
            # Choose style based on level
            if record.levelno >= logging.ERROR:
                style = "bold red"
            elif record.levelno >= logging.WARNING:
                style = "yellow"
            elif record.levelno >= logging.INFO:
                style = "green"
            else:
                style = "blue"
                
            # Add exception info if present
            if record.exc_info:
                exc_type, exc_value, exc_tb = record.exc_info
                tb = Traceback.from_exception(
                    exc_type,
                    exc_value,
                    exc_tb,
                    show_locals=True,
                    width=100,
                    extra_lines=3,
                    theme="monokai",
                    word_wrap=True
                )
                self.console.print(msg, style=style)
                self.console.print(tb)
            else:
                self.console.print(msg, style=style)
                
        except Exception:
            self.handleError(record)