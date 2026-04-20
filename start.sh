#!/bin/bash
set -e

echo "🎨 Starting Kalliste..."

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo "❌ uv is not installed. Please install it first:"
    echo "   curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi

# Check if we're in the right directory
if [ ! -f "pyproject.toml" ]; then
    echo "❌ Please run this script from the kalliste project root directory"
    exit 1
fi

# Clear stale env vars from other package managers
unset VIRTUAL_ENV CONDA_DEFAULT_ENV CONDA_PREFIX 2>/dev/null

# Install/sync dependencies
echo "📦 Installing dependencies with uv..."
uv sync --python 3.11

# Start milvus if needed
if [ -f "docker-compose.yml" ]; then
    echo "🐳 Starting Milvus database..."
    docker compose up -d
fi

# Run the pipeline
echo "🚀 Running Kalliste pipeline..."
uv run kalliste process

echo "✅ Done!"
