# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

TensorRT-LLM Inference Optimization project focuses on optimizing GPT2-small model inference using NVIDIA TensorRT-LLM with KV Cache and FlashAttention optimizations. The project follows a complete pipeline: PyTorch → ONNX → TensorRT engine compilation, deployed as FastAPI services in Docker containers.

**Target Hardware**: NVIDIA A100, CUDA 11.8, TensorRT 8.6

## Architecture

### Core Pipeline
1. **Model Loading**: GPT2-small from HuggingFace
2. **Export**: PyTorch → ONNX format
3. **Engine Build**: ONNX → TensorRT engine with optimizations
4. **Deployment**: FastAPI service + Docker containerization
5. **Optional**: Triton Inference Server support

### Key Modules
- `models/`: Model loading and ONNX export
- `engine/`: TensorRT engine building and optimization (KV Cache, FlashAttention)
- `server/`: FastAPI inference service
- `docker/`: Containerization and deployment
- `deploy/`: Kubernetes manifests and Triton config
- `benchmark/`: Performance evaluation (throughput, latency)
- `scripts/`: Automation scripts for build and deployment

## Development Commands

### Environment Setup
```bash
./scripts/setup_env.sh          # Configure CUDA/TensorRT environment
```

### Engine Building
```bash
./scripts/build_engine.sh       # Build TensorRT engine from ONNX
```

### Service Management
```bash
./scripts/run_server.sh         # Start FastAPI inference server
```

### Performance Testing
```bash
./scripts/benchmark.sh          # Run throughput and latency tests
```

### Docker Operations
```bash
cd docker/
docker-compose up --build       # Build and run containerized service
```

### Testing
```bash
pytest tests/                   # Run all tests
pytest tests/test_engine.py     # Test TensorRT engine functionality
pytest tests/test_server.py     # Test API endpoints
```

## Configuration Files

- `config/model_config.yaml`: Model parameters and paths
- `config/engine_config.yaml`: TensorRT optimization settings
- `config/server_config.yaml`: FastAPI service configuration
- `deploy/triton/config.pbtxt`: Triton Inference Server model configuration

## Key Dependencies

- TensorRT-LLM: Core inference optimization
- ONNX: Model format conversion
- FastAPI: REST API framework
- Docker: Containerization
- Kubernetes: Orchestration (optional)
- Triton Inference Server: Advanced serving (optional)

## Performance Targets

- Maximize throughput (tokens/second)
- Minimize latency (first token + average response time)
- Efficient memory usage with KV Cache
- FlashAttention integration for attention optimization