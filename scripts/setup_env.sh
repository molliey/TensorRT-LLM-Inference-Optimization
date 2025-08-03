#!/bin/bash

# TensorRT-LLM Inference Optimization Environment Setup Script
# This script sets up the complete environment for TensorRT-LLM inference

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
CUDA_VERSION=${CUDA_VERSION:-11.8}
TENSORRT_VERSION=${TENSORRT_VERSION:-8.6.1}
PYTHON_VERSION=${PYTHON_VERSION:-3.8}
WORKSPACE_DIR=${WORKSPACE_DIR:-$(pwd)}

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

check_system() {
    print_status "Checking system requirements..."
    
    # Check OS
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        print_success "Linux OS detected"
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        print_warning "macOS detected - GPU acceleration may not be available"
    else
        print_error "Unsupported OS: $OSTYPE"
        exit 1
    fi
    
    # Check Python version
    if command -v python3 &> /dev/null; then
        python_version=$(python3 --version | cut -d' ' -f2 | cut -d'.' -f1,2)
        print_success "Python $python_version found"
    else
        print_error "Python 3 not found. Please install Python 3.8 or later."
        exit 1
    fi
    
    # Check NVIDIA GPU
    if command -v nvidia-smi &> /dev/null; then
        gpu_info=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits | head -1)
        print_success "NVIDIA GPU detected: $gpu_info"
    else
        print_warning "nvidia-smi not found. GPU acceleration may not be available."
    fi
    
    # Check CUDA
    if command -v nvcc &> /dev/null; then
        cuda_version=$(nvcc --version | grep "release" | awk '{print $6}' | cut -c2-)
        print_success "CUDA $cuda_version found"
    else
        print_warning "CUDA not found in PATH. Will attempt to install."
    fi
}

install_system_dependencies() {
    print_status "Installing system dependencies..."
    
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        # Ubuntu/Debian
        if command -v apt-get &> /dev/null; then
            sudo apt-get update
            sudo apt-get install -y \
                build-essential \
                cmake \
                wget \
                curl \
                git \
                python3-pip \
                python3-dev \
                libssl-dev \
                libffi-dev \
                libxml2-dev \
                libxslt1-dev \
                zlib1g-dev \
                libjpeg-dev \
                libpng-dev
        # CentOS/RHEL
        elif command -v yum &> /dev/null; then
            sudo yum groupinstall -y "Development Tools"
            sudo yum install -y \
                cmake \
                wget \
                curl \
                git \
                python3-pip \
                python3-devel \
                openssl-devel \
                libffi-devel \
                libxml2-devel \
                libxslt-devel \
                zlib-devel \
                libjpeg-devel \
                libpng-devel
        fi
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        # macOS
        if command -v brew &> /dev/null; then
            brew install cmake wget curl git
        else
            print_error "Homebrew not found. Please install Homebrew first."
            exit 1
        fi
    fi
    
    print_success "System dependencies installed"
}

setup_cuda() {
    print_status "Setting up CUDA $CUDA_VERSION..."
    
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        # Check if CUDA is already installed
        if [ -d "/usr/local/cuda-$CUDA_VERSION" ]; then
            print_success "CUDA $CUDA_VERSION already installed"
        else
            print_status "Installing CUDA $CUDA_VERSION..."
            
            # Download and install CUDA toolkit
            cuda_url="https://developer.download.nvidia.com/compute/cuda/$CUDA_VERSION/local_installers/cuda_${CUDA_VERSION}_520.61.05_linux.run"
            
            wget -q $cuda_url -O cuda_installer.run
            sudo sh cuda_installer.run --silent --toolkit --no-opengl-libs
            rm cuda_installer.run
        fi
        
        # Set up environment variables
        echo "export PATH=/usr/local/cuda-$CUDA_VERSION/bin:\$PATH" >> ~/.bashrc
        echo "export LD_LIBRARY_PATH=/usr/local/cuda-$CUDA_VERSION/lib64:\$LD_LIBRARY_PATH" >> ~/.bashrc
        
        export PATH=/usr/local/cuda-$CUDA_VERSION/bin:$PATH
        export LD_LIBRARY_PATH=/usr/local/cuda-$CUDA_VERSION/lib64:$LD_LIBRARY_PATH
        
        print_success "CUDA environment configured"
    else
        print_warning "CUDA installation skipped on non-Linux systems"
    fi
}

setup_python_environment() {
    print_status "Setting up Python environment..."
    
    # Create virtual environment
    if [ ! -d "venv" ]; then
        python3 -m venv venv
        print_success "Virtual environment created"
    else
        print_status "Virtual environment already exists"
    fi
    
    # Activate virtual environment
    source venv/bin/activate
    
    # Upgrade pip
    pip install --upgrade pip setuptools wheel
    
    print_success "Python environment ready"
}

install_tensorrt() {
    print_status "Installing TensorRT $TENSORRT_VERSION..."
    
    # Install TensorRT via pip
    pip install nvidia-tensorrt==$TENSORRT_VERSION
    
    # Verify installation
    python -c "import tensorrt; print(f'TensorRT version: {tensorrt.__version__}')"
    
    print_success "TensorRT installed successfully"
}

install_python_dependencies() {
    print_status "Installing Python dependencies..."
    
    # Install PyTorch with CUDA support
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    else
        pip install torch torchvision torchaudio
    fi
    
    # Install other dependencies
    pip install -r requirements.txt
    
    # Install development dependencies
    cat > requirements-dev.txt << EOF
pytest>=7.4.0
pytest-asyncio>=0.21.0
black>=23.0.0
isort>=5.12.0
flake8>=6.0.0
mypy>=1.4.0
pre-commit>=3.3.0
jupyter>=1.0.0
jupyterlab>=4.0.0
EOF
    
    pip install -r requirements-dev.txt
    
    print_success "Python dependencies installed"
}

setup_directories() {
    print_status "Setting up project directories..."
    
    # Create necessary directories
    mkdir -p models/cache
    mkdir -p models/onnx
    mkdir -p engines
    mkdir -p logs
    mkdir -p benchmark/results
    mkdir -p data
    mkdir -p checkpoints
    
    # Set permissions
    chmod 755 models engines logs benchmark data checkpoints
    chmod 644 scripts/*.sh
    chmod +x scripts/*.sh
    
    print_success "Directory structure created"
}

setup_git_hooks() {
    print_status "Setting up Git hooks..."
    
    if [ -d ".git" ]; then
        # Install pre-commit hooks
        pre-commit install
        
        # Create pre-commit configuration
        cat > .pre-commit-config.yaml << EOF
repos:
  - repo: https://github.com/psf/black
    rev: 23.7.0
    hooks:
      - id: black
        language_version: python3.8
  
  - repo: https://github.com/pycqa/isort
    rev: 5.12.0
    hooks:
      - id: isort
        args: [--profile, black]
  
  - repo: https://github.com/pycqa/flake8
    rev: 6.0.0
    hooks:
      - id: flake8
        args: [--max-line-length=88, --extend-ignore=E203,W503]
EOF
        
        print_success "Git hooks configured"
    else
        print_warning "Not a Git repository, skipping Git hooks setup"
    fi
}

create_env_file() {
    print_status "Creating environment configuration..."
    
    cat > .env << EOF
# TensorRT-LLM Environment Configuration
CUDA_VERSION=$CUDA_VERSION
TENSORRT_VERSION=$TENSORRT_VERSION
PYTHON_VERSION=$PYTHON_VERSION

# Model Configuration
MODEL_NAME=gpt2
MODEL_CACHE_DIR=./models/cache
ONNX_OUTPUT_DIR=./models/onnx
ENGINE_OUTPUT_DIR=./engines

# Server Configuration
SERVER_HOST=0.0.0.0
SERVER_PORT=8000
LOG_LEVEL=INFO
MAX_BATCH_SIZE=8
MAX_SEQUENCE_LENGTH=1024

# GPU Configuration
CUDA_VISIBLE_DEVICES=0
GPU_MEMORY_FRACTION=0.9

# Optimization Configuration
ENABLE_KV_CACHE=true
ENABLE_FLASH_ATTENTION=true
TENSORRT_PRECISION=fp16
MAX_WORKSPACE_SIZE=1073741824

# Benchmark Configuration
BENCHMARK_OUTPUT_DIR=./benchmark/results
NUM_WARMUP_RUNS=10
NUM_BENCHMARK_RUNS=100
EOF
    
    print_success "Environment file created"
}

verify_installation() {
    print_status "Verifying installation..."
    
    # Test imports
    python -c "
import torch
import transformers
import tensorrt
import onnx
import onnxruntime
import fastapi
import uvicorn
print('✓ All core packages imported successfully')
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU device: {torch.cuda.get_device_name(0)}')
print(f'TensorRT version: {tensorrt.__version__}')
print(f'Transformers version: {transformers.__version__}')
"
    
    # Test CUDA functionality
    if command -v nvidia-smi &> /dev/null; then
        nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv
    fi
    
    print_success "Installation verification completed"
}

print_next_steps() {
    print_success "Environment setup completed successfully!"
    echo ""
    echo -e "${BLUE}Next steps:${NC}"
    echo "1. Activate the virtual environment: source venv/bin/activate"
    echo "2. Load environment variables: source .env"
    echo "3. Build TensorRT engine: ./scripts/build_engine.sh"
    echo "4. Start the server: ./scripts/run_server.sh"
    echo "5. Run benchmarks: ./scripts/benchmark.sh"
    echo ""
    echo -e "${BLUE}Useful commands:${NC}"
    echo "- Check GPU status: nvidia-smi"
    echo "- Test model loading: python -c 'from models import GPT2Loader; loader = GPT2Loader(); loader.load_model()'"
    echo "- Run tests: pytest tests/"
    echo "- Format code: black . && isort ."
    echo ""
}

main() {
    echo -e "${GREEN}=== TensorRT-LLM Inference Optimization Setup ===${NC}"
    echo ""
    
    check_system
    install_system_dependencies
    setup_cuda
    setup_python_environment
    install_tensorrt
    install_python_dependencies
    setup_directories
    setup_git_hooks
    create_env_file
    verify_installation
    print_next_steps
}

# Handle command line arguments
case "${1:-all}" in
    "system")
        check_system
        install_system_dependencies
        ;;
    "cuda")
        setup_cuda
        ;;
    "python")
        setup_python_environment
        install_python_dependencies
        ;;
    "tensorrt")
        install_tensorrt
        ;;
    "verify")
        verify_installation
        ;;
    "all"|*)
        main
        ;;
esac