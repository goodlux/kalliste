#!/usr/bin/env python3
"""
Kalliste CLI - Simple command line interface for the image processing pipeline.
"""
# TEMPORARY DEBUG IMPORT - Remove after fixing ViT-B-32 loading issue
from kalliste.utils.debug_vit_loading import *

import asyncio
import click
import logging
import sys
from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.logging import RichHandler

console = Console()

# Set up logging with Rich handler for pretty output
def setup_logging(verbose: bool = False):
    """Configure logging with Rich handler for beautiful console output."""
    level = logging.DEBUG if verbose else logging.INFO
    
    # Configure root logger
    logging.basicConfig(
        level=level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[
            RichHandler(
                rich_tracebacks=True,
                console=console,
                show_time=False,  # We'll use rich progress instead
                show_path=False   # Keep it clean
            )
        ]
    )
    
    # Silence noisy libraries
    logging.getLogger("ultralytics").setLevel(logging.WARNING)
    logging.getLogger("torch").setLevel(logging.WARNING)
    logging.getLogger("PIL").setLevel(logging.WARNING)
    logging.getLogger("tensorflow").setLevel(logging.ERROR)

@click.group()
@click.version_option()
def main():
    """Kalliste - Image tagging and cropping for Stable Diffusion training."""
    console.print(Panel.fit("🎨 Kalliste Image Processing", style="bold blue"))

@main.command()
@click.option('--input', '-i', type=click.Path(exists=True, path_type=Path), 
              default=Path('/Volumes/g2/kalliste_photos/kalliste_input'),
              help='Input directory with images to process')
@click.option('--output', '-o', type=click.Path(path_type=Path),
              default=Path('/Volumes/m01/kalliste_data/images'),
              help='Output directory for processed images')
@click.option('--processed', '-p', type=click.Path(path_type=Path),
              default=Path('/Volumes/g2/kalliste_photos/kalliste_processed'),
              help='Directory to move processed input images')
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose logging')
def process(input, output, processed, verbose):
    """Process images through the full Kalliste pipeline."""
    # Set up logging first
    setup_logging(verbose)
    
    console.print(f"📂 Input: {input}")
    console.print(f"📁 Output: {output}")
    console.print(f"📋 Processed: {processed}")
    
    # Create directories if they don't exist
    output.mkdir(exist_ok=True, parents=True)
    processed.mkdir(exist_ok=True, parents=True)
    
    # Run the async pipeline
    asyncio.run(_run_pipeline(input, output, processed))

async def _run_pipeline(input_path: Path, output_path: Path, processed_path: Path):
    """Run the async processing pipeline."""
    try:
        from kalliste.image.batch_processor import BatchProcessor
        
        # Don't use Progress spinner - it conflicts with logging output
        console.print("[yellow]🎯 Setting up processor...[/yellow]")
        
        processor = BatchProcessor(
            input_path=str(input_path),
            output_path=str(output_path),
            processed_path=str(processed_path)
        )
        
        console.print("[yellow]🔌 Initializing models...[/yellow]")
        await processor.setup()
        
        console.print("[green]🚀 Processing images...[/green]")
        console.print("[dim]Watch for accept/reject status below:[/dim]")
        console.print("")
        
        await processor.process_all()
        
        console.print("")
        console.print("[bold green]✅ Complete![/bold green]")
        
    except Exception as e:
        console.print(f"❌ Error: {e}", style="bold red")
        raise

@main.command()
def deps():
    """Check if all dependencies are properly installed."""
    console.print("🔍 Checking dependencies...")
    
    checks = [
        ("torch", "PyTorch"),
        ("torchvision", "TorchVision"), 
        ("ultralytics", "YOLO"),
        ("cv2", "OpenCV"),
        ("PIL", "Pillow"),
        ("numpy", "NumPy"),
        ("chromadb", "ChromaDB"),
        ("pymilvus", "Milvus"),
    ]
    
    all_good = True
    for module, name in checks:
        try:
            __import__(module)
            console.print(f"✅ {name}")
        except ImportError:
            console.print(f"❌ {name} - not installed")
            all_good = False
    
    if all_good:
        console.print("\n🎉 All dependencies are installed!", style="bold green")
    else:
        console.print("\n💡 Run: uv sync", style="bold yellow")

if __name__ == "__main__":
    main()
