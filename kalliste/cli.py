#!/usr/bin/env python3
"""
Kalliste CLI - Simple command line interface for the image processing pipeline.
"""
import asyncio
import click
from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn

console = Console()

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
def process(input, output, processed):
    """Process images through the full Kalliste pipeline."""
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
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Setting up processor...", total=None)
            
            processor = BatchProcessor(
                input_path=str(input_path),
                output_path=str(output_path),
                processed_path=str(processed_path)
            )
            
            progress.update(task, description="Initializing models...")
            await processor.setup()
            
            progress.update(task, description="Processing images...")
            await processor.process_all()
            
            progress.update(task, description="✅ Complete!", completed=100)
            
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
