#!/bin/bash

echo "=========================================="
echo "🔬 Univisual Segmentation Server Setup"
echo "=========================================="

cd "$(dirname "$0")"

# Create virtual environment
echo ""
echo "Step 1: Creating virtual environment..."
python3 -m venv venv

# Activate virtual environment
echo ""
echo "Step 2: Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo ""
echo "Step 3: Upgrading pip..."
pip install --upgrade pip --quiet

# Install core dependencies first
echo ""
echo "Step 4: Installing core dependencies..."
pip install fastapi uvicorn python-multipart numpy pillow scipy scikit-image --quiet

# Install Cellpose
echo ""
echo "Step 5: Installing Cellpose (this may take a while)..."
pip install cellpose --quiet

# Install StarDist and TensorFlow
echo ""
echo "Step 6: Installing StarDist and TensorFlow..."
pip install tensorflow stardist csbdeep --quiet

echo ""
echo "=========================================="
echo "✅ Setup complete!"
echo ""
echo "To start the server, run:"
echo "  cd $(pwd)"
echo "  source venv/bin/activate"
echo "  python main.py"
echo ""
echo "Server will be at: http://localhost:8000"
echo "API docs at: http://localhost:8000/docs"
echo "=========================================="
