# Kalliste - Quick Start

## Prerequisites

1. **Install uv** (if not already installed):
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

2. **Make sure Docker is running** (for Milvus database)

## Super Simple Startup

1. Navigate to the project directory:
   ```bash
   cd /Users/rob/repos/kalliste
   ```

2. Make the start script executable (first time only):
   ```bash
   chmod +x start.sh
   ```

3. Run everything:
   ```bash
   ./start.sh
   ```

That's it! The script will:
- Install all dependencies with uv
- Start the Milvus database
- Run the image processing pipeline

## Manual Commands (if you prefer)

```bash
# Install dependencies
uv sync

# Start database
docker compose up -d

# Check that everything is working
uv run kalliste deps

# Run the pipeline
uv run kalliste process

# Or run with custom paths
uv run kalliste process --input /path/to/images --output /path/to/output
```

## What This Does

Kalliste processes images for Stable Diffusion training by:
1. Detecting objects/faces in images
2. Creating smart crops around subjects
3. Generating captions and tags
4. Storing embeddings in a vector database
5. Organizing everything for training

## Troubleshooting

- **"uv not found"**: Install uv first (see prerequisites)
- **Permission denied**: Run `chmod +x start.sh`
- **Docker errors**: Make sure Docker Desktop is running
- **Path errors**: Update paths in the CLI commands to match your setup
