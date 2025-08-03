# TensorRT-LLM Inference Optimization

High-performance GPT2 inference using TensorRT-LLM with KV Cache and FlashAttention optimizations, deployed as FastAPI services in Docker containers.

## 🚀 Features

- **Model Support**: GPT2-small with HuggingFace integration
- **Inference Optimization**: TensorRT-LLM with KV Cache & FlashAttention
- **Export Pipeline**: PyTorch → ONNX → TensorRT engine compilation
- **Service Deployment**: FastAPI REST API with Docker containerization
- **Optional Support**: Triton Inference Server integration
- **Target Hardware**: NVIDIA A100, CUDA 11.8, TensorRT 8.6
- **Comprehensive Benchmarking**: Throughput and latency evaluation tools
- **Production Ready**: Kubernetes deployment, monitoring, and scaling

## 📋 Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [Usage](#usage)
- [API Documentation](#api-documentation)
- [Benchmarking](#benchmarking)
- [Deployment](#deployment)
- [Configuration](#configuration)
- [Development](#development)
- [Contributing](#contributing)

## 🛠 Installation

### Prerequisites

- NVIDIA GPU with Compute Capability 7.0+ (Tesla V100, RTX 20/30/40 series, A100)
- CUDA 11.8
- TensorRT 8.6
- Python 3.8+
- Docker (optional)
- Kubernetes (optional)

### Environment Setup

```bash
# Clone the repository
git clone https://github.com/tensorrt-llm/inference-optimization.git
cd tensorrt-llm-inference-optimization

# Run the automated setup script
./scripts/setup_env.sh

# Or manually install dependencies
python -m venv venv
source venv/bin/activate  # On Windows: venv\\Scripts\\activate
pip install -r requirements.txt

# Install the package
pip install -e .
```

## ⚡ Quick Start

### 1. Build TensorRT Engine

```bash
# Build optimized TensorRT engine
./scripts/build_engine.sh

# Or with custom parameters
./scripts/build_engine.sh --model gpt2 --precision fp16 --batch-size 8 --seq-length 1024
```

### 2. Start Inference Server

```bash
# Start FastAPI server
./scripts/run_server.sh

# Or with custom configuration
./scripts/run_server.sh --host 0.0.0.0 --port 8000 --workers 1
```

### 3. Test Inference

```bash
# Test the API
curl -X POST http://localhost:8000/generate \\
  -H "Content-Type: application/json" \\
  -d '{
    "prompt": "The future of artificial intelligence is",
    "max_new_tokens": 100,
    "temperature": 1.0,
    "top_k": 50,
    "top_p": 0.9
  }'
```

### 4. Run Benchmarks

```bash
# Run comprehensive benchmarks
./scripts/benchmark.sh

# Or quick benchmark
./scripts/benchmark.sh --quick
```

## 📁 Project Structure

```
tensorrt-llm-inference-optimization/
├── models/                     # Model loading and ONNX export
│   ├── gpt2_loader.py         # GPT2 model loader from HuggingFace
│   ├── onnx_exporter.py       # PyTorch → ONNX conversion
│   └── model_utils.py         # Model utilities and helpers
├── engine/                     # TensorRT engine management
│   ├── builder.py             # TensorRT engine builder
│   ├── optimizer.py           # KV Cache & FlashAttention optimization
│   └── engine_utils.py        # Engine utilities and inference
├── server/                     # FastAPI inference server
│   ├── api.py                 # REST API endpoints
│   ├── inference.py           # Inference engine wrapper
│   └── middleware.py          # Request handling middleware
├── benchmark/                  # Performance evaluation
│   ├── throughput_test.py     # Throughput benchmarking
│   ├── latency_test.py        # Latency measurement
│   └── metrics.py             # Performance metrics collection
├── scripts/                    # Automation scripts
│   ├── setup_env.sh           # Environment setup
│   ├── build_engine.sh        # Engine building
│   ├── run_server.sh          # Server startup
│   └── benchmark.sh           # Benchmark execution
├── docker/                     # Docker configuration
│   ├── Dockerfile             # Multi-stage Docker build
│   ├── docker-compose.yml     # Service orchestration
│   └── nginx.conf             # Load balancer configuration
├── deploy/                     # Kubernetes deployment
│   ├── deployment.yaml        # Pod deployment
│   ├── service.yaml           # Service configuration
│   ├── configmap.yaml         # Configuration management
│   └── triton/                # Triton Inference Server config
│       └── config.pbtxt       # Model configuration
├── config/                     # Configuration files
│   ├── model_config.yaml      # Model parameters
│   ├── engine_config.yaml     # TensorRT settings
│   └── server_config.yaml     # FastAPI configuration
└── tests/                      # Test suite
    ├── test_models.py         # Model testing
    ├── test_engine.py         # Engine testing
    ├── test_server.py         # API testing
    └── test_benchmark.py      # Benchmark testing
```

## 🔧 Usage

### Model Loading and Export

```python
from models import GPT2Loader, ONNXExporter

# Load GPT2 model
loader = GPT2Loader(model_name="gpt2")
model, tokenizer, config = loader.load_model()

# Export to ONNX
exporter = ONNXExporter(model, tokenizer)
onnx_path = exporter.export_to_onnx(
    batch_size=8,
    seq_length=1024,
    dynamic_axes=True
)
```

### TensorRT Engine Building

```python
from engine import TensorRTBuilder, TensorRTOptimizer

# Initialize builder and optimizer
builder = TensorRTBuilder(precision="fp16", max_batch_size=8)
optimizer = TensorRTOptimizer(
    enable_kv_cache=True,
    enable_flash_attention=True
)

# Build optimized engine
engine_path = builder.build_engine_from_onnx(
    onnx_path="model.onnx",
    engine_path="optimized_engine.trt"
)
```

### Inference Server

```python
from server import TensorRTInferenceEngine

# Initialize inference engine
engine = TensorRTInferenceEngine(
    engine_path="optimized_engine.trt",
    tokenizer_name="gpt2"
)

# Generate text
result = await engine.generate(
    prompt="The future of AI is",
    max_new_tokens=100,
    temperature=1.0
)
```

## 📊 API Documentation

### Endpoints

- **GET** `/` - Service information
- **GET** `/health` - Health check with system status
- **POST** `/generate` - Text generation
- **POST** `/generate/stream` - Streaming text generation
- **GET** `/model/info` - Model and engine information
- **POST** `/benchmark` - Performance benchmarking
- **GET** `/metrics` - Performance metrics
- **POST** `/metrics/reset` - Reset metrics

### Generate Text Request

```json
{
  "prompt": "The future of artificial intelligence is",
  "max_new_tokens": 100,
  "temperature": 1.0,
  "top_k": 50,
  "top_p": 0.9,
  "do_sample": true,
  "repetition_penalty": 1.0,
  "stream": false
}
```

### Generate Text Response

```json
{
  "generated_text": "The future of artificial intelligence is bright and full of possibilities...",
  "input_tokens": 8,
  "output_tokens": 95,
  "total_tokens": 103,
  "inference_time_ms": 245.7,
  "tokens_per_second": 387.2,
  "model_info": {
    "model_name": "gpt2",
    "vocab_size": 50257
  }
}
```

## 📈 Benchmarking

### Throughput Testing

```bash
# Run throughput benchmark
./scripts/benchmark.sh --throughput-only \\
  --batch-sizes 1,2,4,8 \\
  --seq-lengths 128,256,512 \\
  --num-runs 100

# Results saved to benchmark/results/
```

### Latency Testing

```bash
# Run latency benchmark with streaming
./scripts/benchmark.sh --latency-only \\
  --streaming \\
  --token-lengths 50,100,200 \\
  --latency-samples 50
```

### Performance Metrics

- **Throughput**: Requests/second, Tokens/second
- **Latency**: Time to First Token (TTFT), Time Per Token (TPT)
- **System Metrics**: GPU utilization, Memory usage
- **Percentiles**: P50, P95, P99 latency measurements

## 🚀 Deployment

### Docker Deployment

```bash
# Build and run with Docker Compose
cd docker/
docker-compose up --build

# Or build manually
docker build -t tensorrt-llm-inference .
docker run --gpus all -p 8000:8000 tensorrt-llm-inference
```

### Kubernetes Deployment

```bash
# Deploy to Kubernetes
kubectl apply -f deploy/

# Check deployment status
kubectl get pods,services,ingress

# Scale deployment
kubectl scale deployment tensorrt-llm-inference --replicas=3
```

### Triton Inference Server

```bash
# Use Triton configuration
triton_server --model-repository=./deploy/triton/
```

## ⚙️ Configuration

### Model Configuration (`config/model_config.yaml`)

```yaml
model:
  name: "gpt2"
  type: "causal_lm"
  vocab_size: 50257
  n_positions: 1024
  n_embd: 768
  n_layer: 12
  n_head: 12

tokenizer:
  name: "gpt2"
  padding_side: "left"
```

### Engine Configuration (`config/engine_config.yaml`)

```yaml
tensorrt:
  precision: "fp16"
  max_workspace_size: 1073741824  # 1GB
  max_batch_size: 8
  max_sequence_length: 1024

optimization:
  kv_cache:
    enabled: true
    block_size: 16
  flash_attention:
    enabled: true
    causal: true
```

### Server Configuration (`config/server_config.yaml`)

```yaml
server:
  host: "0.0.0.0"
  port: 8000
  workers: 1
  log_level: "info"

inference:
  max_batch_size: 8
  batch_timeout_ms: 100
  enable_streaming: true

monitoring:
  metrics_enabled: true
  prometheus_enabled: true
```

## 🛠 Development

### Setup Development Environment

```bash
# Clone and setup
git clone https://github.com/tensorrt-llm/inference-optimization.git
cd tensorrt-llm-inference-optimization

# Install development dependencies
pip install -e ".[dev]"

# Setup pre-commit hooks
pre-commit install
```

### Running Tests

```bash
# Run all tests
pytest

# Run specific test categories
pytest tests/test_models.py -v
pytest tests/test_engine.py -k "test_builder"
pytest -m "not integration"  # Skip integration tests

# Run with coverage
pytest --cov=models --cov=engine --cov=server --cov=benchmark
```

### Code Quality

```bash
# Format code
black .
isort .

# Lint code
flake8 .
mypy .

# Security scan
bandit -r .
```

### Building Documentation

```bash
# Install documentation dependencies
pip install -e ".[docs]"

# Build documentation
cd docs/
make html

# Serve documentation locally
python -m http.server 8080 -d _build/html/
```

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

### Development Workflow

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/amazing-feature`
3. Make your changes and add tests
4. Run the test suite: `pytest`
5. Format your code: `black . && isort .`
6. Commit your changes: `git commit -m 'Add amazing feature'`
7. Push to the branch: `git push origin feature/amazing-feature`
8. Open a Pull Request

### Reporting Issues

Please report bugs and feature requests through [GitHub Issues](https://github.com/tensorrt-llm/inference-optimization/issues).

## 📄 License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- NVIDIA TensorRT Team for TensorRT-LLM
- HuggingFace for Transformers library
- FastAPI team for the excellent framework
- The open-source AI community

## 📞 Support

- **Documentation**: [https://tensorrt-llm-inference-optimization.readthedocs.io/](https://tensorrt-llm-inference-optimization.readthedocs.io/)
- **Issues**: [GitHub Issues](https://github.com/tensorrt-llm/inference-optimization/issues)
- **Discussions**: [GitHub Discussions](https://github.com/tensorrt-llm/inference-optimization/discussions)
- **Email**: tensorrt-llm@example.com

---

<div align="center">
  <strong>Built with ❤️ by the TensorRT-LLM Team</strong>
</div>