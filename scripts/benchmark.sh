#!/bin/bash

# TensorRT-LLM Benchmark Script
# This script runs comprehensive benchmarks for throughput and latency

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default configuration
SERVER_URL=${SERVER_URL:-"http://localhost:8000"}
BENCHMARK_OUTPUT_DIR=${BENCHMARK_OUTPUT_DIR:-"./benchmark/results"}
NUM_WARMUP_RUNS=${NUM_WARMUP_RUNS:-10}
NUM_BENCHMARK_RUNS=${NUM_BENCHMARK_RUNS:-100}
CONCURRENT_REQUESTS=${CONCURRENT_REQUESTS:-10}
BATCH_SIZES=${BATCH_SIZES:-"1,2,4,8"}
SEQUENCE_LENGTHS=${SEQUENCE_LENGTHS:-"128,256,512"}
TOKEN_LENGTHS=${TOKEN_LENGTHS:-"50,100,200"}
NUM_LATENCY_SAMPLES=${NUM_LATENCY_SAMPLES:-50}

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
    echo "TensorRT-LLM Benchmark Script"
    echo ""
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --server-url URL           Server URL (default: http://localhost:8000)"
    echo "  --output-dir DIR           Benchmark output directory"
    echo "  --num-runs RUNS            Number of benchmark runs (default: 100)"
    echo "  --concurrent REQUESTS      Concurrent requests (default: 10)"
    echo "  --batch-sizes SIZES        Comma-separated batch sizes (default: 1,2,4,8)"
    echo "  --seq-lengths LENGTHS      Comma-separated sequence lengths (default: 128,256,512)"
    echo "  --token-lengths LENGTHS    Comma-separated token lengths (default: 50,100,200)"
    echo "  --latency-samples SAMPLES  Number of latency samples (default: 50)"
    echo "  --throughput-only          Run only throughput benchmark"
    echo "  --latency-only             Run only latency benchmark"
    echo "  --streaming                Test streaming inference"
    echo "  --quick                    Run quick benchmark with reduced parameters"
    echo "  --compare BASELINE_FILE    Compare with baseline results"
    echo "  --help                     Show this help message"
    echo ""
    echo "Environment Variables:"
    echo "  SERVER_URL                 Server URL"
    echo "  BENCHMARK_OUTPUT_DIR       Output directory"
    echo "  NUM_BENCHMARK_RUNS         Number of benchmark runs"
    echo "  CONCURRENT_REQUESTS        Concurrent requests"
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
    import requests
    import numpy as np
    import matplotlib.pyplot as plt
    print('✓ All required packages available')
except ImportError as e:
    print(f'✗ Missing package: {e}')
    exit(1)
" || exit 1
    
    print_success "Dependencies check passed"
}

check_server() {
    print_status "Checking server availability..."
    
    # Check if server is running
    if ! curl -s --connect-timeout 5 "$SERVER_URL/health" > /dev/null; then
        print_error "Server is not responding at $SERVER_URL"
        print_status "Start the server first: ./scripts/run_server.sh"
        exit 1
    fi
    
    # Get server info
    server_info=$(curl -s "$SERVER_URL/health" | python -c "
import sys, json
try:
    data = json.load(sys.stdin)
    print(f\"Status: {data.get('status', 'unknown')}\")
    print(f\"Model loaded: {data.get('model_loaded', False)}\")
    if 'engine_info' in data:
        engine_info = data['engine_info']
        print(f\"Engine path: {engine_info.get('engine_path', 'unknown')}\")
        print(f\"Max batch size: {engine_info.get('max_batch_size', 'unknown')}\")
except:
    print('Unable to parse server response')
    sys.exit(1)
")
    
    print_success "Server is healthy"
    echo "$server_info"
}

setup_output_directory() {
    print_status "Setting up output directory..."
    
    # Create output directory with timestamp
    timestamp=$(date +"%Y%m%d_%H%M%S")
    BENCHMARK_OUTPUT_DIR="$BENCHMARK_OUTPUT_DIR/benchmark_$timestamp"
    mkdir -p "$BENCHMARK_OUTPUT_DIR"
    
    print_success "Output directory: $BENCHMARK_OUTPUT_DIR"
}

run_throughput_benchmark() {
    print_status "Running throughput benchmark..."
    
    python << EOF
import sys
import asyncio
import json
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, '.')

from benchmark import ThroughputBenchmark

async def main():
    try:
        # Configuration
        server_url = "$SERVER_URL"
        output_dir = "$BENCHMARK_OUTPUT_DIR"
        batch_sizes = [int(x.strip()) for x in "$BATCH_SIZES".split(',')]
        seq_lengths = [int(x.strip()) for x in "$SEQUENCE_LENGTHS".split(',')]
        num_runs = $NUM_BENCHMARK_RUNS
        concurrent_requests = $CONCURRENT_REQUESTS
        
        print(f"Server URL: {server_url}")
        print(f"Batch sizes: {batch_sizes}")
        print(f"Sequence lengths: {seq_lengths}")
        print(f"Number of runs: {num_runs}")
        print(f"Concurrent requests: {concurrent_requests}")
        
        # Test prompts
        test_prompts = [
            "The quick brown fox jumps over the lazy dog.",
            "In a hole in the ground there lived a hobbit.",
            "It was the best of times, it was the worst of times.",
            "To be or not to be, that is the question.",
            "All happy families are alike; each unhappy family is unhappy in its own way.",
            "In the beginning was the Word, and the Word was with God.",
            "Call me Ishmael. Some years ago—never mind how long precisely.",
            "It is a truth universally acknowledged, that a single man in possession of a good fortune, must be in want of a wife.",
            "In the great libraries of the world, knowledge sleeps like a dragon on its hoard.",
            "The future belongs to those who believe in the beauty of their dreams."
        ]
        
        # Run benchmark
        benchmark = ThroughputBenchmark(server_url=server_url)
        
        results = await benchmark.run_throughput_test(
            prompts=test_prompts,
            batch_sizes=batch_sizes,
            sequence_lengths=seq_lengths,
            num_requests=num_runs,
            max_new_tokens=100,
            concurrent_requests=concurrent_requests
        )
        
        # Save results
        output_path = Path(output_dir) / "throughput_results.json"
        benchmark.save_results(str(output_path))
        
        # Print summary
        summary = benchmark.get_summary()
        print("\\nThroughput Benchmark Summary:")
        print(f"Total tests: {summary['total_tests']}")
        print(f"Best throughput: {summary['best_throughput']['requests_per_sec']:.2f} req/s")
        print(f"Best token throughput: {summary['best_token_throughput']['tokens_per_sec']:.2f} tokens/s")
        print(f"Lowest latency: {summary['lowest_latency']['avg_latency_ms']:.2f} ms")
        
        # Save summary
        summary_path = Path(output_dir) / "throughput_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\\nResults saved to: {output_path}")
        print(f"Summary saved to: {summary_path}")
        
        return True
        
    except Exception as e:
        print(f"Throughput benchmark failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
EOF
    
    if [ $? -eq 0 ]; then
        print_success "Throughput benchmark completed"
    else
        print_error "Throughput benchmark failed"
        return 1
    fi
}

run_latency_benchmark() {
    print_status "Running latency benchmark..."
    
    python << EOF
import sys
import asyncio
import json
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, '.')

from benchmark import LatencyBenchmark

async def main():
    try:
        # Configuration
        server_url = "$SERVER_URL"
        output_dir = "$BENCHMARK_OUTPUT_DIR"
        token_lengths = [int(x.strip()) for x in "$TOKEN_LENGTHS".split(',')]
        num_samples = $NUM_LATENCY_SAMPLES
        streaming = "$STREAMING" == "true"
        
        print(f"Server URL: {server_url}")
        print(f"Token lengths: {token_lengths}")
        print(f"Number of samples: {num_samples}")
        print(f"Streaming mode: {streaming}")
        
        # Test prompts of different lengths
        test_prompts = [
            "Hello, how are you?",  # Short
            "The quick brown fox jumps over the lazy dog. This is a medium length prompt that should generate a reasonable response.",  # Medium
            "In the heart of a bustling metropolis, where towering skyscrapers pierce the clouds and the streets below teem with endless streams of people, there exists a small coffee shop tucked away in a narrow alley. This humble establishment, with its weathered brick walls and vintage wooden furniture, serves as a sanctuary for those seeking respite from the chaos of urban life."  # Long
        ]
        
        # Run benchmark
        benchmark = LatencyBenchmark(server_url=server_url)
        
        results = await benchmark.run_latency_test(
            test_prompts=test_prompts,
            token_lengths=token_lengths,
            num_samples=num_samples,
            temperature=1.0,
            measure_streaming=streaming
        )
        
        # Save results
        output_path = Path(output_dir) / "latency_results.json"
        benchmark.save_results(str(output_path))
        
        # Generate plots
        if not streaming:  # Skip plots for streaming tests as they might fail
            try:
                benchmark.plot_results(output_dir)
                print("✓ Plots generated")
            except Exception as e:
                print(f"Warning: Plot generation failed: {e}")
        
        # Print summary
        summary = benchmark.get_summary()
        print("\\nLatency Benchmark Summary:")
        print(f"Total tests: {summary['total_tests']}")
        print(f"Avg TTFT: {summary['time_to_first_token']['mean_ms']:.2f} ms")
        print(f"P95 TTFT: {summary['time_to_first_token']['p95_ms']:.2f} ms")
        print(f"Avg TPT: {summary['time_per_token']['mean_ms']:.2f} ms")
        print(f"Max throughput: {summary['throughput']['max_tokens_per_sec']:.2f} tokens/s")
        
        # Save summary
        summary_path = Path(output_dir) / "latency_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\\nResults saved to: {output_path}")
        print(f"Summary saved to: {summary_path}")
        
        return True
        
    except Exception as e:
        print(f"Latency benchmark failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
EOF
    
    if [ $? -eq 0 ]; then
        print_success "Latency benchmark completed"
    else
        print_error "Latency benchmark failed"
        return 1
    fi
}

collect_system_metrics() {
    print_status "Collecting system metrics..."
    
    python << EOF
import sys
import json
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, '.')

from benchmark import MetricsCollector

try:
    output_dir = "$BENCHMARK_OUTPUT_DIR"
    
    # Load benchmark results
    throughput_results = None
    latency_results = None
    
    throughput_path = Path(output_dir) / "throughput_results.json"
    if throughput_path.exists():
        with open(throughput_path, 'r') as f:
            throughput_data = json.load(f)
            throughput_results = throughput_data.get('results', [])
    
    latency_path = Path(output_dir) / "latency_results.json"
    if latency_path.exists():
        with open(latency_path, 'r') as f:
            latency_data = json.load(f)
            latency_results = latency_data.get('results', [])
    
    # Collect metrics
    collector = MetricsCollector(output_dir)
    
    # Create mock result objects for metrics collection
    # (In a real implementation, this would use the actual result objects)
    metrics = collector.collect_benchmark_metrics(
        throughput_results=throughput_results,
        latency_results=latency_results
    )
    
    # Save metrics
    metrics_path = collector.save_metrics()
    print(f"Metrics saved to: {metrics_path}")
    
    # Generate report
    report_path = collector.generate_report()
    print(f"Report generated: {report_path}")
    
    print("✓ System metrics collection completed")

except Exception as e:
    print(f"Metrics collection failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
EOF
    
    if [ $? -eq 0 ]; then
        print_success "System metrics collected"
    else
        print_warning "System metrics collection failed (non-critical)"
    fi
}

compare_with_baseline() {
    if [ -z "$BASELINE_FILE" ]; then
        return 0
    fi
    
    print_status "Comparing with baseline: $BASELINE_FILE"
    
    python << EOF
import sys
import json
from pathlib import Path

try:
    baseline_file = "$BASELINE_FILE"
    current_dir = "$BENCHMARK_OUTPUT_DIR"
    
    # Load baseline results
    with open(baseline_file, 'r') as f:
        baseline_data = json.load(f)
    
    # Load current results
    current_summary = {}
    throughput_summary_path = Path(current_dir) / "throughput_summary.json"
    if throughput_summary_path.exists():
        with open(throughput_summary_path, 'r') as f:
            current_summary['throughput'] = json.load(f)
    
    latency_summary_path = Path(current_dir) / "latency_summary.json"
    if latency_summary_path.exists():
        with open(latency_summary_path, 'r') as f:
            current_summary['latency'] = json.load(f)
    
    # Compare results
    comparison = {}
    
    if 'throughput' in baseline_data and 'throughput' in current_summary:
        baseline_tps = baseline_data['throughput']['best_token_throughput']['tokens_per_sec']
        current_tps = current_summary['throughput']['best_token_throughput']['tokens_per_sec']
        
        improvement = ((current_tps - baseline_tps) / baseline_tps) * 100
        comparison['throughput_improvement'] = improvement
        
        print(f"Throughput comparison:")
        print(f"  Baseline: {baseline_tps:.2f} tokens/s")
        print(f"  Current:  {current_tps:.2f} tokens/s")
        print(f"  Change:   {improvement:+.2f}%")
    
    if 'latency' in baseline_data and 'latency' in current_summary:
        baseline_ttft = baseline_data['latency']['time_to_first_token']['mean_ms']
        current_ttft = current_summary['latency']['time_to_first_token']['mean_ms']
        
        improvement = ((baseline_ttft - current_ttft) / baseline_ttft) * 100  # Lower is better
        comparison['latency_improvement'] = improvement
        
        print(f"Latency comparison:")
        print(f"  Baseline TTFT: {baseline_ttft:.2f} ms")
        print(f"  Current TTFT:  {current_ttft:.2f} ms")
        print(f"  Change:        {improvement:+.2f}%")
    
    # Save comparison
    comparison_path = Path(current_dir) / "comparison.json"
    with open(comparison_path, 'w') as f:
        json.dump(comparison, f, indent=2)
    
    print(f"\\nComparison saved to: {comparison_path}")

except Exception as e:
    print(f"Comparison failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
EOF
    
    if [ $? -eq 0 ]; then
        print_success "Baseline comparison completed"
    else
        print_warning "Baseline comparison failed"
    fi
}

create_summary_report() {
    print_status "Creating summary report..."
    
    # Create a simple text summary
    cat > "$BENCHMARK_OUTPUT_DIR/README.txt" << EOF
TensorRT-LLM Benchmark Results
==============================

Benchmark run on: $(date)
Server URL: $SERVER_URL
Output directory: $BENCHMARK_OUTPUT_DIR

Configuration:
- Batch sizes: $BATCH_SIZES
- Sequence lengths: $SEQUENCE_LENGTHS
- Token lengths: $TOKEN_LENGTHS
- Number of runs: $NUM_BENCHMARK_RUNS
- Concurrent requests: $CONCURRENT_REQUESTS
- Latency samples: $NUM_LATENCY_SAMPLES

Files:
- throughput_results.json: Detailed throughput benchmark results
- latency_results.json: Detailed latency benchmark results
- throughput_summary.json: Throughput benchmark summary
- latency_summary.json: Latency benchmark summary
- benchmark_metrics_*.json: System metrics during benchmark
- benchmark_report_*.html: Comprehensive HTML report
- comparison.json: Comparison with baseline (if provided)

Plots (if generated):
- latency_distributions.png: Latency distribution histograms
- latency_vs_length.png: Latency vs prompt/token length
- throughput_comparison.png: Throughput comparison chart
- performance_trends.png: Performance trends over time
- system_metrics.png: System resource usage
- performance_comparison.png: Performance vs resource usage

Instructions:
1. Open the HTML report for a comprehensive view
2. Check JSON files for detailed metrics
3. Use comparison.json to track performance changes over time
EOF
    
    print_success "Summary report created: $BENCHMARK_OUTPUT_DIR/README.txt"
}

print_final_summary() {
    print_success "Benchmark completed successfully!"
    echo ""
    echo -e "${BLUE}Results Summary:${NC}"
    echo "Output directory: $BENCHMARK_OUTPUT_DIR"
    echo ""
    echo -e "${BLUE}Key Files:${NC}"
    ls -la "$BENCHMARK_OUTPUT_DIR"/*.json 2>/dev/null | awk '{print "- " $9 " (" $5 " bytes)"}' || echo "- No JSON files found"
    echo ""
    echo -e "${BLUE}Next Steps:${NC}"
    echo "1. Review the HTML report: open $BENCHMARK_OUTPUT_DIR/benchmark_report_*.html"
    echo "2. Analyze detailed results in JSON files"
    echo "3. Compare with previous runs using --compare option"
    echo "4. Use results to optimize your TensorRT configuration"
    echo ""
}

main() {
    echo -e "${GREEN}=== TensorRT-LLM Benchmark Suite ===${NC}"
    echo ""
    
    check_dependencies
    check_server
    setup_output_directory
    
    # Run benchmarks based on options
    benchmark_success=true
    
    if [ "$THROUGHPUT_ONLY" != "true" ] && [ "$LATENCY_ONLY" != "true" ]; then
        # Run both benchmarks
        run_throughput_benchmark || benchmark_success=false
        run_latency_benchmark || benchmark_success=false
    elif [ "$THROUGHPUT_ONLY" == "true" ]; then
        run_throughput_benchmark || benchmark_success=false
    elif [ "$LATENCY_ONLY" == "true" ]; then
        run_latency_benchmark || benchmark_success=false
    fi
    
    if [ "$benchmark_success" == "true" ]; then
        collect_system_metrics
        compare_with_baseline
        create_summary_report
        print_final_summary
    else
        print_error "Some benchmarks failed, check the logs for details"
        exit 1
    fi
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --server-url)
            SERVER_URL="$2"
            shift 2
            ;;
        --output-dir)
            BENCHMARK_OUTPUT_DIR="$2"
            shift 2
            ;;
        --num-runs)
            NUM_BENCHMARK_RUNS="$2"
            shift 2
            ;;
        --concurrent)
            CONCURRENT_REQUESTS="$2"
            shift 2
            ;;
        --batch-sizes)
            BATCH_SIZES="$2"
            shift 2
            ;;
        --seq-lengths)
            SEQUENCE_LENGTHS="$2"
            shift 2
            ;;
        --token-lengths)
            TOKEN_LENGTHS="$2"
            shift 2
            ;;
        --latency-samples)
            NUM_LATENCY_SAMPLES="$2"
            shift 2
            ;;
        --throughput-only)
            THROUGHPUT_ONLY="true"
            shift
            ;;
        --latency-only)
            LATENCY_ONLY="true"
            shift
            ;;
        --streaming)
            STREAMING="true"
            shift
            ;;
        --quick)
            # Quick benchmark settings
            NUM_BENCHMARK_RUNS=20
            NUM_LATENCY_SAMPLES=10
            CONCURRENT_REQUESTS=5
            BATCH_SIZES="1,4"
            SEQUENCE_LENGTHS="128,512"
            TOKEN_LENGTHS="50,100"
            shift
            ;;
        --compare)
            BASELINE_FILE="$2"
            shift 2
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