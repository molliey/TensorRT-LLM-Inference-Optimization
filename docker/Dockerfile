# Multi-stage Docker build for TensorRT-LLM Inference Optimization
FROM nvidia/cuda:11.8-devel-ubuntu20.04 as builder

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV CUDA_VERSION=11.8
ENV TENSORRT_VERSION=8.6.1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    build-essential \
    cmake \
    wget \
    curl \
    git \
    vim \
    htop \
    && rm -rf /var/lib/apt/lists/*

# Create symbolic link for python
RUN ln -s /usr/bin/python3 /usr/bin/python

# Upgrade pip
RUN python -m pip install --upgrade pip

# Install TensorRT
RUN pip install nvidia-tensorrt==8.6.1

# Set working directory
WORKDIR /app

# Copy requirements first for better Docker layer caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Install additional TensorRT-LLM dependencies
RUN pip install --no-cache-dir \
    tensorrt-llm \
    nvidia-ml-py3 \
    pycuda \
    onnx \
    onnxruntime-gpu \
    transformers \
    torch \
    fastapi \
    uvicorn \
    psutil \
    GPUtil

# Copy application code
COPY . .

# Create directories for models and engines
RUN mkdir -p /app/models/onnx /app/engines /app/logs

# Runtime stage
FROM nvidia/cuda:11.8-runtime-ubuntu20.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV CUDA_VERSION=11.8
ENV PATH="/usr/local/cuda/bin:${PATH}"
ENV LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH}"

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Create symbolic link for python
RUN ln -s /usr/bin/python3 /usr/bin/python

# Create app user
RUN useradd -m -u 1000 appuser && \
    mkdir -p /app && \
    chown -R appuser:appuser /app

# Set working directory
WORKDIR /app

# Copy from builder stage
COPY --from=builder --chown=appuser:appuser /app /app
COPY --from=builder /usr/local/lib/python3.8/dist-packages /usr/local/lib/python3.8/dist-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Switch to app user
USER appuser

# Create necessary directories
RUN mkdir -p logs models/onnx engines benchmark/results

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default command
CMD ["python", "-m", "uvicorn", "server.api:app", "--host", "0.0.0.0", "--port", "8000"]