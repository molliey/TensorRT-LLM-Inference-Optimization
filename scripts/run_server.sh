#!/bin/bash

# TensorRT-LLM Inference Server Startup Script
# This script starts the FastAPI inference server

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default configuration
SERVER_HOST=${SERVER_HOST:-"0.0.0.0"}
SERVER_PORT=${SERVER_PORT:-8000}
WORKERS=${WORKERS:-1}
LOG_LEVEL=${LOG_LEVEL:-"info"}
ENGINE_PATH=${ENGINE_PATH:-"./engines/gpt2_optimized.engine"}
MODEL_NAME=${MODEL_NAME:-"gpt2"}
MAX_BATCH_SIZE=${MAX_BATCH_SIZE:-8}
MAX_SEQUENCE_LENGTH=${MAX_SEQUENCE_LENGTH:-1024}
RELOAD=${RELOAD:-false}
ACCESS_LOG=${ACCESS_LOG:-true}

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

show_help() {
    echo "TensorRT-LLM Inference Server Startup Script"
    echo ""
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --host HOST                Server host (default: 0.0.0.0)"
    echo "  --port PORT                Server port (default: 8000)"
    echo "  --workers WORKERS          Number of worker processes (default: 1)"
    echo "  --log-level LEVEL          Log level (debug/info/warning/error, default: info)"
    echo "  --engine-path PATH         Path to TensorRT engine file"
    echo "  --model-name NAME          Model name (default: gpt2)"
    echo "  --max-batch-size SIZE      Maximum batch size (default: 8)"
    echo "  --max-seq-length LENGTH    Maximum sequence length (default: 1024)"
    echo "  --reload                   Enable auto-reload for development"
    echo "  --no-access-log            Disable access logging"
    echo "  --daemon                   Run as daemon (background process)"
    echo "  --stop                     Stop running server"
    echo "  --status                   Check server status"
    echo "  --help                     Show this help message"
    echo ""
    echo "Environment Variables:"
    echo "  SERVER_HOST                Server host"
    echo "  SERVER_PORT                Server port"
    echo "  ENGINE_PATH                Path to TensorRT engine"
    echo "  MODEL_NAME                 Model name"
    echo "  LOG_LEVEL                  Log level"
    echo "  MAX_BATCH_SIZE             Maximum batch size"
    echo "  MAX_SEQUENCE_LENGTH        Maximum sequence length"
    echo ""
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
    import fastapi
    import uvicorn
    import tensorrt
    print('✓ All required packages available')
except ImportError as e:
    print(f'✗ Missing package: {e}')
    exit(1)
" || exit 1
    
    print_success "Dependencies check passed"
}

check_engine() {
    print_status "Checking TensorRT engine..."
    
    if [ ! -f "$ENGINE_PATH" ]; then
        print_error "TensorRT engine not found: $ENGINE_PATH"
        print_status "Build the engine first: ./scripts/build_engine.sh"
        exit 1
    fi
    
    # Verify engine
    python -c "
import sys
sys.path.insert(0, '.')
from engine import EngineUtils

if EngineUtils.validate_engine('$ENGINE_PATH'):
    print('✓ Engine file is valid')
else:
    print('✗ Engine file is invalid')
    sys.exit(1)
" || exit 1
    
    print_success "Engine validation passed"
}

setup_logging() {
    print_status "Setting up logging..."
    
    # Create logs directory
    mkdir -p logs
    
    # Setup log rotation
    cat > logs/logging.conf << EOF
[loggers]
keys = root, uvicorn, fastapi

[handlers]
keys = console, file, error_file

[formatters]
keys = default, detailed

[logger_root]
level = $LOG_LEVEL
handlers = console, file

[logger_uvicorn]
level = $LOG_LEVEL
handlers = console, file
qualname = uvicorn
propagate = 0

[logger_fastapi]
level = $LOG_LEVEL
handlers = console, file
qualname = fastapi
propagate = 0

[handler_console]
class = StreamHandler
level = $LOG_LEVEL
formatter = default
args = (sys.stdout,)

[handler_file]
class = handlers.RotatingFileHandler
level = $LOG_LEVEL
formatter = detailed
args = ('logs/server.log', 'a', 10485760, 5)

[handler_error_file]
class = handlers.RotatingFileHandler
level = ERROR
formatter = detailed
args = ('logs/errors.log', 'a', 10485760, 3)

[formatter_default]
format = %(asctime)s - %(name)s - %(levelname)s - %(message)s

[formatter_detailed]
format = %(asctime)s - %(name)s - %(levelname)s - %(module)s - %(funcName)s - %(message)s
EOF
    
    print_success "Logging configured"
}

check_port() {
    print_status "Checking port availability..."
    
    if lsof -Pi :$SERVER_PORT -sTCP:LISTEN -t >/dev/null 2>&1; then
        print_warning "Port $SERVER_PORT is already in use"
        
        # Try to find the process
        process_info=$(lsof -Pi :$SERVER_PORT -sTCP:LISTEN | tail -n +2)
        if [ ! -z "$process_info" ]; then
            print_status "Process using port $SERVER_PORT:"
            echo "$process_info"
        fi
        
        read -p "Do you want to continue anyway? (y/N): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            print_status "Exiting..."
            exit 1
        fi
    else
        print_success "Port $SERVER_PORT is available"
    fi
}

create_server_config() {
    print_status "Creating server configuration..."
    
    cat > server_config.json << EOF
{
    "host": "$SERVER_HOST",
    "port": $SERVER_PORT,
    "workers": $WORKERS,
    "log_level": "$LOG_LEVEL",
    "reload": $RELOAD,
    "access_log": $ACCESS_LOG,
    "engine_config": {
        "engine_path": "$ENGINE_PATH",
        "model_name": "$MODEL_NAME",
        "max_batch_size": $MAX_BATCH_SIZE,
        "max_sequence_length": $MAX_SEQUENCE_LENGTH
    },
    "server_config": {
        "timeout_keep_alive": 30,
        "timeout_graceful_shutdown": 30,
        "limit_concurrency": 100,
        "limit_max_requests": 10000
    }
}
EOF
    
    print_success "Server configuration created"
}

set_environment_variables() {
    print_status "Setting environment variables..."
    
    export ENGINE_PATH="$ENGINE_PATH"
    export MODEL_NAME="$MODEL_NAME"
    export MAX_BATCH_SIZE="$MAX_BATCH_SIZE"
    export MAX_SEQUENCE_LENGTH="$MAX_SEQUENCE_LENGTH"
    export LOG_LEVEL="$LOG_LEVEL"
    export WORKERS="$WORKERS"
    
    # CUDA settings
    export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-"0"}
    export NVIDIA_VISIBLE_DEVICES=${NVIDIA_VISIBLE_DEVICES:-"all"}
    export NVIDIA_DRIVER_CAPABILITIES=${NVIDIA_DRIVER_CAPABILITIES:-"compute,utility"}
    
    print_success "Environment variables set"
}

start_server() {
    print_status "Starting TensorRT-LLM Inference Server..."
    
    # Build uvicorn command
    cmd="uvicorn server.api:app"
    cmd="$cmd --host $SERVER_HOST"
    cmd="$cmd --port $SERVER_PORT"
    cmd="$cmd --workers $WORKERS"
    cmd="$cmd --log-level $LOG_LEVEL"
    
    if [ "$RELOAD" = "true" ]; then
        cmd="$cmd --reload"
    fi
    
    if [ "$ACCESS_LOG" = "false" ]; then
        cmd="$cmd --no-access-log"
    fi
    
    # Additional uvicorn options
    cmd="$cmd --timeout-keep-alive 30"
    cmd="$cmd --timeout-graceful-shutdown 30"
    cmd="$cmd --limit-concurrency 100"
    cmd="$cmd --limit-max-requests 10000"
    
    print_status "Command: $cmd"
    print_status "Server starting on http://$SERVER_HOST:$SERVER_PORT"
    print_status "Documentation available at http://$SERVER_HOST:$SERVER_PORT/docs"
    print_status "Health check: http://$SERVER_HOST:$SERVER_PORT/health"
    echo ""
    print_status "Press Ctrl+C to stop the server"
    echo ""
    
    # Run the server
    exec $cmd
}

start_daemon() {
    print_status "Starting server as daemon..."
    
    # Create PID file directory
    mkdir -p logs
    
    # Start server in background
    nohup ./scripts/run_server.sh > logs/server.out 2>&1 &
    server_pid=$!
    
    # Save PID
    echo $server_pid > logs/server.pid
    
    print_success "Server started as daemon with PID: $server_pid"
    print_status "Log output: logs/server.out"
    print_status "Stop server: ./scripts/run_server.sh --stop"
}

stop_server() {
    print_status "Stopping server..."
    
    if [ -f "logs/server.pid" ]; then
        server_pid=$(cat logs/server.pid)
        
        if kill -0 $server_pid 2>/dev/null; then
            print_status "Stopping server with PID: $server_pid"
            kill -TERM $server_pid
            
            # Wait for graceful shutdown
            for i in {1..30}; do
                if ! kill -0 $server_pid 2>/dev/null; then
                    break
                fi
                sleep 1
            done
            
            # Force kill if still running
            if kill -0 $server_pid 2>/dev/null; then
                print_warning "Server didn't stop gracefully, force killing..."
                kill -KILL $server_pid
            fi
            
            rm -f logs/server.pid
            print_success "Server stopped"
        else
            print_warning "Server PID not found or already stopped"
            rm -f logs/server.pid
        fi
    else
        print_warning "No PID file found"
        
        # Try to find and kill server process
        server_pids=$(pgrep -f "uvicorn.*server.api:app")
        if [ ! -z "$server_pids" ]; then
            print_status "Found server processes: $server_pids"
            echo "$server_pids" | xargs kill -TERM
            print_success "Server processes terminated"
        else
            print_status "No running server processes found"
        fi
    fi
}

check_server_status() {
    print_status "Checking server status..."
    
    if [ -f "logs/server.pid" ]; then
        server_pid=$(cat logs/server.pid)
        
        if kill -0 $server_pid 2>/dev/null; then
            print_success "Server is running with PID: $server_pid"
            
            # Check if server is responding
            if command -v curl &> /dev/null; then
                response=$(curl -s -o /dev/null -w "%{http_code}" http://$SERVER_HOST:$SERVER_PORT/health || echo "000")
                if [ "$response" = "200" ]; then
                    print_success "Server is healthy and responding"
                else
                    print_warning "Server is running but not responding (HTTP $response)"
                fi
            else
                print_status "curl not available, cannot check server response"
            fi
        else
            print_warning "PID file exists but process is not running"
            rm -f logs/server.pid
        fi
    else
        print_status "No PID file found"
        
        # Check for running server processes
        server_pids=$(pgrep -f "uvicorn.*server.api:app" || echo "")
        if [ ! -z "$server_pids" ]; then
            print_warning "Found server processes without PID file: $server_pids"
        else
            print_status "No server processes found"
        fi
    fi
}

pre_flight_check() {
    print_status "Running pre-flight checks..."
    
    # Check GPU status
    if command -v nvidia-smi &> /dev/null; then
        print_status "GPU Status:"
        nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv,noheader
    else
        print_warning "nvidia-smi not found, cannot check GPU status"
    fi
    
    # Check system resources
    print_status "System Resources:"
    echo "CPU Usage: $(top -bn1 | grep "Cpu(s)" | awk '{print $2}' | cut -d'%' -f1)%"
    echo "Memory Usage: $(free | grep Mem | awk '{printf "%.1f%%", $3/$2 * 100.0}')"
    echo "Disk Usage: $(df -h . | tail -1 | awk '{print $5}')"
    
    # Test engine loading
    print_status "Testing engine loading..."
    python -c "
import sys
import time
sys.path.insert(0, '.')

try:
    from server.inference import TensorRTInferenceEngine
    
    start = time.time()
    engine = TensorRTInferenceEngine(
        engine_path='$ENGINE_PATH',
        tokenizer_name='$MODEL_NAME',
        max_batch_size=$MAX_BATCH_SIZE,
        max_sequence_length=$MAX_SEQUENCE_LENGTH
    )
    
    # Don't actually initialize in pre-flight check
    print(f'✓ Engine configuration valid')
    load_time = time.time() - start
    print(f'Configuration check took {load_time:.2f} seconds')
    
except Exception as e:
    print(f'✗ Engine configuration error: {e}')
    sys.exit(1)
" || exit 1
    
    print_success "Pre-flight checks passed"
}

main() {
    echo -e "${GREEN}=== TensorRT-LLM Inference Server ===${NC}"
    echo ""
    
    check_dependencies
    check_engine
    setup_logging
    check_port
    create_server_config
    set_environment_variables
    pre_flight_check
    
    start_server
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --host)
            SERVER_HOST="$2"
            shift 2
            ;;
        --port)
            SERVER_PORT="$2"
            shift 2
            ;;
        --workers)
            WORKERS="$2"
            shift 2
            ;;
        --log-level)
            LOG_LEVEL="$2"
            shift 2
            ;;
        --engine-path)
            ENGINE_PATH="$2"
            shift 2
            ;;
        --model-name)
            MODEL_NAME="$2"
            shift 2
            ;;
        --max-batch-size)
            MAX_BATCH_SIZE="$2"
            shift 2
            ;;
        --max-seq-length)
            MAX_SEQUENCE_LENGTH="$2"
            shift 2
            ;;
        --reload)
            RELOAD="true"
            shift
            ;;
        --no-access-log)
            ACCESS_LOG="false"
            shift
            ;;
        --daemon)
            start_daemon
            exit 0
            ;;
        --stop)
            stop_server
            exit 0
            ;;
        --status)
            check_server_status
            exit 0
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

# Handle signals for graceful shutdown
trap 'print_status "Received shutdown signal, stopping server..."; exit 0' SIGTERM SIGINT

# Run main process
main