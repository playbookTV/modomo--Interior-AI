#!/bin/bash

# Post-deployment script to install Depth Anything V2
# This avoids dependency conflicts during initial Railway deployment

set -e

echo "🔧 Installing Depth Anything V2 (post-deployment)"
echo "=================================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if we're in the right environment
if [ ! -f "main_refactored.py" ]; then
    print_error "This script should be run from the modomo-scraper directory"
    exit 1
fi

print_status "Checking current depth estimation setup..."

# Check if Depth Anything V2 is already installed
if python3 -c "from depth_anything_v2.dpt import DepthAnythingV2; print('✅ Depth Anything V2 already available')" 2>/dev/null; then
    print_success "Depth Anything V2 is already installed"
    exit 0
fi

# Check if ZoeDepth is available as fallback
if python3 -c "import torch.hub; torch.hub.load('isl-org/ZoeDepth', 'ZoeD_N', pretrained=False); print('✅ ZoeDepth available as fallback')" 2>/dev/null; then
    print_status "ZoeDepth is available as fallback"
else
    print_warning "ZoeDepth not available either"
fi

print_status "Installing Depth Anything V2..."

# Method 1: Try the patched version first (should work with newer huggingface-hub)
print_status "Attempting installation of patched Depth Anything V2..."
if pip install git+https://github.com/badayvedat/Depth-Anything-V2.git@badayvedat-patch-1; then
    print_success "Patched Depth Anything V2 installed successfully"
    
    # Test the installation
    if python3 -c "from depth_anything_v2.dpt import DepthAnythingV2; print('✅ Installation verified')" 2>/dev/null; then
        print_success "Depth Anything V2 installation verified"
        exit 0
    else
        print_warning "Installation succeeded but import failed"
    fi
else
    print_warning "Patched version installation failed"
fi

# Method 2: Try original repo with force reinstall of dependencies
print_status "Attempting original Depth Anything V2 with dependency override..."
if pip install git+https://github.com/DepthAnything/Depth-Anything-V2.git --force-reinstall --no-deps; then
    print_status "Installing minimal dependencies for Depth Anything V2..."
    pip install torch torchvision opencv-python pillow numpy
    
    # Test the installation
    if python3 -c "from depth_anything_v2.dpt import DepthAnythingV2; print('✅ Installation verified')" 2>/dev/null; then
        print_success "Original Depth Anything V2 installed successfully"
        exit 0
    else
        print_warning "Original version import failed"
    fi
else
    print_warning "Original version installation failed"
fi

# Method 3: Manual installation from source
print_status "Attempting manual installation from source..."
if [ ! -d "Depth-Anything-V2" ]; then
    print_status "Cloning Depth Anything V2 repository..."
    git clone https://github.com/DepthAnything/Depth-Anything-V2.git
fi

cd Depth-Anything-V2

# Try to install manually
if pip install -e . --no-deps; then
    cd ..
    
    # Test the installation
    if python3 -c "from depth_anything_v2.dpt import DepthAnythingV2; print('✅ Installation verified')" 2>/dev/null; then
        print_success "Manual Depth Anything V2 installation successful"
        exit 0
    else
        print_warning "Manual installation import failed"
    fi
else
    cd ..
    print_warning "Manual installation failed"
fi

# If all methods fail, provide guidance
print_error "❌ Could not install Depth Anything V2"
print_status "The system will fall back to ZoeDepth for depth estimation"
print_status ""
print_status "To manually install Depth Anything V2 after deployment:"
print_status "1. SSH into your Railway deployment"
print_status "2. Run this script: ./install-depth-anything-v2.sh"
print_status "3. Or install manually:"
print_status "   pip install git+https://github.com/badayvedat/Depth-Anything-V2.git@badayvedat-patch-1"
print_status ""
print_status "The application will continue to work with ZoeDepth as fallback"

exit 1