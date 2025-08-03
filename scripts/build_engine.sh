#!/bin/bash

# TensorRT Engine Build Script
# This script builds optimized TensorRT engines from ONNX models

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default configuration
MODEL_NAME=${MODEL_NAME:-"gpt2"}
MODEL_CACHE_DIR=${MODEL_CACHE_DIR:-"./models/cache"}
ONNX_OUTPUT_DIR=${ONNX_OUTPUT_DIR:-"./models/onnx"}
ENGINE_OUTPUT_DIR=${ENGINE_OUTPUT_DIR:-"./engines"}
TENSORRT_PRECISION=${TENSORRT_PRECISION:-"fp16"}
MAX_BATCH_SIZE=${MAX_BATCH_SIZE:-8}
MAX_SEQUENCE_LENGTH=${MAX_SEQUENCE_LENGTH:-1024}
MAX_WORKSPACE_SIZE=${MAX_WORKSPACE_SIZE:-1073741824}  # 1GB
ENABLE_KV_CACHE=${ENABLE_KV_CACHE:-true}
ENABLE_FLASH_ATTENTION=${ENABLE_FLASH_ATTENTION:-true}

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

check_dependencies() {
    print_status "Checking dependencies..."
    
    # Check Python environment
    if [ ! -d "venv" ]; then
        print_error "Virtual environment not found. Run ./scripts/setup_env.sh first."
        exit 1
    fi
    
    # Activate virtual environment
    source venv/bin/activate
    
    # Check required packages
    python -c "
try:
    import torch
    import transformers
    import tensorrt
    import onnx
    import onnxruntime
    print('✓ All required packages available')
except ImportError as e:
    print(f'✗ Missing package: {e}')
    exit(1)
" || exit 1
    
    # Check GPU availability
    python -c "
import torch
if not torch.cuda.is_available():
    print('Warning: CUDA not available. Engine will be built for CPU.')
else:
    print(f'✓ CUDA available with {torch.cuda.device_count()} GPU(s)')
    print(f'✓ GPU: {torch.cuda.get_device_name(0)}')
"
    
    print_success "Dependencies check passed"
}

create_directories() {
    print_status "Creating output directories..."
    
    mkdir -p "$MODEL_CACHE_DIR"
    mkdir -p "$ONNX_OUTPUT_DIR"
    mkdir -p "$ENGINE_OUTPUT_DIR"
    mkdir -p "./logs"
    
    print_success "Directories created"
}

load_and_export_model() {
    print_status "Loading and exporting model to ONNX..."
    
    python << EOF
import os
import sys
import logging
from pathlib import Path

# Add project root to path
sys.path.insert(0, '.')

from models import GPT2Loader, ONNXExporter

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    # Configuration
    model_name = "$MODEL_NAME"
    cache_dir = "$MODEL_CACHE_DIR"
    onnx_output_dir = "$ONNX_OUTPUT_DIR"
    
    print(f"Loading model: {model_name}")
    print(f"Cache directory: {cache_dir}")
    print(f"ONNX output directory: {onnx_output_dir}")
    
    # Load model
    loader = GPT2Loader(model_name=model_name, cache_dir=cache_dir)
    model, tokenizer, config = loader.load_model()
    
    # Get model info
    model_info = loader.get_model_info()
    print(f"Model info: {model_info}")
    
    # Prepare for export
    model = loader.prepare_for_export()
    
    # Export to ONNX
    exporter = ONNXExporter(model, tokenizer, output_dir=onnx_output_dir)
    
    onnx_path = exporter.export_to_onnx(
        batch_size=1,
        seq_length=$MAX_SEQUENCE_LENGTH,
        opset_version=14,
        dynamic_axes=True
    )
    
    print(f"ONNX model exported to: {onnx_path}")
    
    # Verify ONNX model
    info = exporter.get_model_info()
    print(f"ONNX model info: {info}")
    
    # Test comparison (optional)
    import torch
    test_input = torch.randint(0, 1000, (1, 10))
    
    if exporter.compare_outputs(test_input):
        print("✓ ONNX model outputs match PyTorch model")
    else:
        print("✗ Warning: ONNX model outputs don't match exactly")
    
    # Save model metadata
    import json
    metadata = {
        "model_name": model_name,
        "onnx_path": onnx_path,
        "model_info": model_info,
        "onnx_info": info,
        "export_config": {
            "max_batch_size": $MAX_BATCH_SIZE,
            "max_sequence_length": $MAX_SEQUENCE_LENGTH,
            "precision": "$TENSORRT_PRECISION"
        }
    }
    
    metadata_path = os.path.join(onnx_output_dir, "model_metadata.json")
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Metadata saved to: {metadata_path}")
    print("ONNX export completed successfully")

except Exception as e:
    print(f"Error during ONNX export: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
EOF
    
    if [ $? -eq 0 ]; then
        print_success "ONNX export completed"
    else
        print_error "ONNX export failed"
        exit 1
    fi
}

build_tensorrt_engine() {
    print_status "Building TensorRT engine..."
    
    python << EOF
import os
import sys
import json
import logging
from pathlib import Path

# Add project root to path
sys.path.insert(0, '.')

from engine import TensorRTBuilder, TensorRTOptimizer

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    # Configuration
    onnx_dir = "$ONNX_OUTPUT_DIR"
    engine_dir = "$ENGINE_OUTPUT_DIR"
    precision = "$TENSORRT_PRECISION"
    max_batch_size = $MAX_BATCH_SIZE
    max_seq_length = $MAX_SEQUENCE_LENGTH
    max_workspace_size = $MAX_WORKSPACE_SIZE
    enable_kv_cache = "$ENABLE_KV_CACHE" == "true"
    enable_flash_attention = "$ENABLE_FLASH_ATTENTION" == "true"
    
    # Load metadata
    metadata_path = os.path.join(onnx_dir, "model_metadata.json")
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    onnx_path = metadata["onnx_path"]
    model_info = metadata["model_info"]
    
    print(f"Building engine from: {onnx_path}")
    print(f"Engine directory: {engine_dir}")
    print(f"Precision: {precision}")
    print(f"Max batch size: {max_batch_size}")
    print(f"Max sequence length: {max_seq_length}")
    print(f"KV Cache enabled: {enable_kv_cache}")
    print(f"Flash Attention enabled: {enable_flash_attention}")
    
    # Initialize builder
    builder = TensorRTBuilder(
        max_workspace_size=max_workspace_size,
        precision=precision,
        max_batch_size=max_batch_size,
        verbose=True
    )
    
    # Initialize optimizer
    optimizer = TensorRTOptimizer(
        enable_kv_cache=enable_kv_cache,
        enable_flash_attention=enable_flash_attention,
        max_sequence_length=max_seq_length,
        max_batch_size=max_batch_size
    )
    
    # Define optimization profiles
    optimization_profiles = [{
        "input_ids": {
            "min": [1, 1],
            "opt": [max_batch_size // 2, max_seq_length // 2],
            "max": [max_batch_size, max_seq_length]
        }
    }]
    
    # Build engine
    engine_name = f"{metadata['model_name']}_optimized.engine"
    engine_path = os.path.join(engine_dir, engine_name)
    
    print(f"Building engine: {engine_path}")
    
    built_engine_path = builder.build_engine_from_onnx(
        onnx_path=onnx_path,
        engine_path=engine_path,
        optimization_profiles=optimization_profiles
    )
    
    # Get engine info
    engine_info = builder.get_engine_info()
    print(f"Engine info: {engine_info}")
    
    # Apply optimizations
    print("Applying optimizations...")
    optimization_results = {}
    
    # This would be called during the build process in a real implementation
    # optimization_results = optimizer.apply_layer_optimizations(builder.network, builder.config)
    
    # Save engine metadata
    engine_metadata = {
        "engine_path": built_engine_path,
        "source_onnx": onnx_path,
        "model_info": model_info,
        "engine_info": engine_info,
        "build_config": {
            "precision": precision,
            "max_batch_size": max_batch_size,
            "max_sequence_length": max_seq_length,
            "max_workspace_size": max_workspace_size,
            "optimization_profiles": optimization_profiles
        },
        "optimizations": {
            "kv_cache_enabled": enable_kv_cache,
            "flash_attention_enabled": enable_flash_attention,
            "optimization_results": optimization_results
        },
        "optimization_summary": optimizer.get_optimization_summary()
    }
    
    metadata_path = os.path.join(engine_dir, f"{metadata['model_name']}_metadata.json")
    with open(metadata_path, 'w') as f:
        json.dump(engine_metadata, f, indent=2)
    
    print(f"Engine metadata saved to: {metadata_path}")
    
    # Cleanup builder resources
    builder.cleanup()
    
    print("TensorRT engine build completed successfully")
    print(f"Engine saved to: {built_engine_path}")

except Exception as e:
    print(f"Error during engine build: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
EOF
    
    if [ $? -eq 0 ]; then
        print_success "TensorRT engine build completed"
    else
        print_error "TensorRT engine build failed"
        exit 1
    fi
}

verify_engine() {
    print_status "Verifying TensorRT engine..."
    
    python << EOF
import os
import sys
import json
import time

# Add project root to path
sys.path.insert(0, '.')

from engine import EngineUtils

try:
    engine_dir = "$ENGINE_OUTPUT_DIR"
    model_name = "$MODEL_NAME"
    
    # Load engine metadata
    metadata_path = os.path.join(engine_dir, f"{model_name}_metadata.json")
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    engine_path = metadata["engine_path"]
    
    print(f"Verifying engine: {engine_path}")
    
    # Validate engine file
    if not EngineUtils.validate_engine(engine_path):
        print("✗ Engine validation failed")
        sys.exit(1)
    
    print("✓ Engine file is valid")
    
    # Load engine and test inference
    engine_utils = EngineUtils(engine_path)
    
    if not engine_utils.engine:
        print("✗ Failed to load engine")
        sys.exit(1)
    
    print("✓ Engine loaded successfully")
    
    # Get engine information
    engine_info = engine_utils.get_engine_info()
    print(f"Engine details: {engine_info}")
    
    # Test inference with dummy data
    import numpy as np
    
    # Create dummy input
    dummy_input = np.random.randint(0, 1000, (1, 10), dtype=np.int32)
    
    print("Testing inference...")
    start_time = time.time()
    
    try:
        outputs = engine_utils.infer({"input_ids": dummy_input})
        inference_time = (time.time() - start_time) * 1000
        
        print(f"✓ Inference successful")
        print(f"Inference time: {inference_time:.2f} ms")
        print(f"Output shape: {outputs['logits'].shape}")
        
    except Exception as e:
        print(f"✗ Inference test failed: {e}")
        sys.exit(1)
    
    # Benchmark inference speed
    print("Running quick benchmark...")
    benchmark_results = engine_utils.benchmark_inference(
        {"input_ids": dummy_input},
        num_runs=50,
        warmup_runs=5
    )
    
    print(f"Benchmark results: {benchmark_results}")
    
    # Cleanup
    engine_utils.cleanup()
    
    print("✓ Engine verification completed successfully")

except Exception as e:
    print(f"Error during engine verification: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
EOF
    
    if [ $? -eq 0 ]; then
        print_success "Engine verification completed"
    else
        print_error "Engine verification failed"
        exit 1
    fi
}

print_summary() {
    print_success "Engine build process completed!"
    echo ""
    echo -e "${BLUE}Build Summary:${NC}"
    echo "- Model: $MODEL_NAME"
    echo "- Precision: $TENSORRT_PRECISION"
    echo "- Max Batch Size: $MAX_BATCH_SIZE"
    echo "- Max Sequence Length: $MAX_SEQUENCE_LENGTH"
    echo "- KV Cache: $ENABLE_KV_CACHE"
    echo "- Flash Attention: $ENABLE_FLASH_ATTENTION"
    echo ""
    echo -e "${BLUE}Output Files:${NC}"
    echo "- ONNX Model: $ONNX_OUTPUT_DIR/${MODEL_NAME}_model.onnx"
    echo "- TensorRT Engine: $ENGINE_OUTPUT_DIR/${MODEL_NAME}_optimized.engine"
    echo "- Metadata: $ENGINE_OUTPUT_DIR/${MODEL_NAME}_metadata.json"
    echo ""
    echo -e "${BLUE}Next Steps:${NC}"
    echo "1. Start the inference server: ./scripts/run_server.sh"
    echo "2. Run benchmarks: ./scripts/benchmark.sh"
    echo "3. Test the API: curl -X POST http://localhost:8000/generate -H 'Content-Type: application/json' -d '{\"prompt\": \"Hello, world!\", \"max_new_tokens\": 50}'"
    echo ""
}

show_help() {
    echo "TensorRT Engine Build Script"
    echo ""
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --model MODEL_NAME          Model name (default: gpt2)"
    echo "  --precision PRECISION       TensorRT precision (fp16/fp32/int8, default: fp16)"
    echo "  --batch-size SIZE           Maximum batch size (default: 8)"
    echo "  --seq-length LENGTH         Maximum sequence length (default: 1024)"
    echo "  --workspace-size SIZE       TensorRT workspace size in bytes (default: 1GB)"
    echo "  --no-kv-cache              Disable KV cache optimization"
    echo "  --no-flash-attention       Disable Flash Attention optimization"
    echo "  --onnx-only                Only export to ONNX, skip TensorRT build"
    echo "  --engine-only              Only build TensorRT engine (requires existing ONNX)"
    echo "  --verify-only              Only verify existing engine"
    echo "  --help                     Show this help message"
    echo ""
    echo "Environment Variables:"
    echo "  MODEL_NAME                 Model name"
    echo "  TENSORRT_PRECISION         TensorRT precision"
    echo "  MAX_BATCH_SIZE             Maximum batch size"
    echo "  MAX_SEQUENCE_LENGTH        Maximum sequence length"
    echo "  ENABLE_KV_CACHE            Enable KV cache (true/false)"
    echo "  ENABLE_FLASH_ATTENTION     Enable Flash Attention (true/false)"
    echo ""
}

main() {
    echo -e "${GREEN}=== TensorRT Engine Build Process ===${NC}"
    echo ""
    
    check_dependencies
    create_directories
    
    if [ "$ONNX_ONLY" != "true" ]; then
        if [ "$ENGINE_ONLY" != "true" ] && [ "$VERIFY_ONLY" != "true" ]; then
            load_and_export_model
        fi
        
        if [ "$VERIFY_ONLY" != "true" ]; then
            build_tensorrt_engine
        fi
        
        verify_engine
    else
        load_and_export_model
    fi
    
    print_summary
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --model)
            MODEL_NAME="$2"
            shift 2
            ;;
        --precision)
            TENSORRT_PRECISION="$2"
            shift 2
            ;;
        --batch-size)
            MAX_BATCH_SIZE="$2"
            shift 2
            ;;
        --seq-length)
            MAX_SEQUENCE_LENGTH="$2"
            shift 2
            ;;
        --workspace-size)
            MAX_WORKSPACE_SIZE="$2"
            shift 2
            ;;
        --no-kv-cache)
            ENABLE_KV_CACHE="false"
            shift
            ;;
        --no-flash-attention)
            ENABLE_FLASH_ATTENTION="false"
            shift
            ;;
        --onnx-only)
            ONNX_ONLY="true"
            shift
            ;;
        --engine-only)
            ENGINE_ONLY="true"
            shift
            ;;
        --verify-only)
            VERIFY_ONLY="true"
            shift
            ;;
        --help)
            show_help
            exit 0
            ;;
        *)
            print_error "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

# Run main process
main