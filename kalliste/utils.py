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

# Configure root logger with file and line numbers
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
    handlers=[RichHandler()]  # Add our handler by default
)

def format_error(e: Exception, title: Optional[str] = None) -> None:
    """Format an exception with rich styling."""
    panel_title = title or f"[red bold]{e.__class__.__name__}[/]"
    tb = Traceback.from_exception(
        type(e),
        e,
        e.__traceback__,
        show_locals=True,
        width=100,
        extra_lines=3,
        theme="monokai",
        word_wrap=True
    )
    error_text = Text(str(e))
    error_panel = Panel(
        error_text,
        title=panel_title,
        border_style="red",
        padding=(1, 2)
    )
    console.print(error_panel)
    console.print(tb)

class RichHandler(logging.Handler):
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
                # For errors, use the full error formatter
                if record.exc_info:
                    format_error(record.exc_info[1])
                    return
            elif record.levelno >= logging.WARNING:
                level_style = "yellow"
            elif record.levelno >= logging.INFO:
                level_style = "green"
            else:
                level_style = "blue"
            
            # Format the message
            msg = f"{location} [{level_style}]{record.levelname}[/] {record.getMessage()}"
            
            # Print message
            self.console.print(msg)
                
        except Exception:
            self.handleError(record)