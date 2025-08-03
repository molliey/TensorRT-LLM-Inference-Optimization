from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, AsyncIterator
import asyncio
import logging
import time
import json
from contextlib import asynccontextmanager

from .inference import TensorRTInferenceEngine
from .middleware import setup_middleware

logger = logging.getLogger(__name__)

# Global inference engine
inference_engine: Optional[TensorRTInferenceEngine] = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    global inference_engine
    logger.info("Starting TensorRT Inference Server...")
    
    try:
        inference_engine = TensorRTInferenceEngine()
        await inference_engine.initialize()
        logger.info("TensorRT engine initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize inference engine: {e}")
        raise
    
    yield
    
    # Shutdown
    logger.info("Shutting down TensorRT Inference Server...")
    if inference_engine:
        await inference_engine.cleanup()

# Create FastAPI app
app = FastAPI(
    title="TensorRT-LLM Inference API",
    description="High-performance GPT2 inference using TensorRT-LLM with KV Cache and FlashAttention",
    version="1.0.0",
    lifespan=lifespan
)

# Setup middleware
setup_middleware(app)

# Request/Response models
class GenerateRequest(BaseModel):
    prompt: str = Field(..., description="Input text prompt")
    max_new_tokens: int = Field(default=100, ge=1, le=2048, description="Maximum number of tokens to generate")
    temperature: float = Field(default=1.0, ge=0.1, le=2.0, description="Sampling temperature")
    top_k: int = Field(default=50, ge=1, le=100, description="Top-k sampling parameter")
    top_p: float = Field(default=0.9, ge=0.1, le=1.0, description="Top-p (nucleus) sampling parameter")
    do_sample: bool = Field(default=True, description="Whether to use sampling or greedy decoding")
    repetition_penalty: float = Field(default=1.0, ge=0.1, le=2.0, description="Repetition penalty")
    pad_token_id: Optional[int] = Field(default=None, description="Padding token ID")
    eos_token_id: Optional[int] = Field(default=None, description="End-of-sequence token ID")
    stream: bool = Field(default=False, description="Whether to stream the response")

class GenerateResponse(BaseModel):
    generated_text: str = Field(..., description="Generated text")
    input_tokens: int = Field(..., description="Number of input tokens")
    output_tokens: int = Field(..., description="Number of generated tokens")
    total_tokens: int = Field(..., description="Total number of tokens processed")
    inference_time_ms: float = Field(..., description="Inference time in milliseconds")
    tokens_per_second: float = Field(..., description="Generation speed in tokens per second")
    model_info: Dict[str, Any] = Field(..., description="Model information")

class StreamResponse(BaseModel):
    token: str = Field(..., description="Generated token")
    is_final: bool = Field(default=False, description="Whether this is the final token")
    total_tokens: int = Field(..., description="Total tokens generated so far")

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    engine_info: Dict[str, Any]
    system_info: Dict[str, Any]

class BenchmarkRequest(BaseModel):
    prompt: str = Field(default="The quick brown fox", description="Test prompt")
    batch_sizes: List[int] = Field(default=[1, 2, 4, 8], description="Batch sizes to test")
    sequence_lengths: List[int] = Field(default=[128, 256, 512], description="Sequence lengths to test")
    num_runs: int = Field(default=100, ge=10, le=1000, description="Number of benchmark runs")

class BenchmarkResponse(BaseModel):
    results: List[Dict[str, Any]]
    summary: Dict[str, Any]

def get_inference_engine() -> TensorRTInferenceEngine:
    global inference_engine
    if inference_engine is None:
        raise HTTPException(status_code=503, detail="Inference engine not initialized")
    return inference_engine

@app.get("/", response_model=Dict[str, str])
async def root():
    return {
        "message": "TensorRT-LLM Inference API",
        "version": "1.0.0",
        "docs": "/docs"
    }

@app.get("/health", response_model=HealthResponse)
async def health_check(engine: TensorRTInferenceEngine = Depends(get_inference_engine)):
    try:
        engine_info = await engine.get_engine_info()
        system_info = await engine.get_system_info()
        
        return HealthResponse(
            status="healthy",
            model_loaded=engine.is_loaded(),
            engine_info=engine_info,
            system_info=system_info
        )
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=503, detail=f"Service unhealthy: {str(e)}")

@app.post("/generate", response_model=GenerateResponse)
async def generate_text(
    request: GenerateRequest,
    engine: TensorRTInferenceEngine = Depends(get_inference_engine)
):
    try:
        start_time = time.time()
        
        result = await engine.generate(
            prompt=request.prompt,
            max_new_tokens=request.max_new_tokens,
            temperature=request.temperature,
            top_k=request.top_k,
            top_p=request.top_p,
            do_sample=request.do_sample,
            repetition_penalty=request.repetition_penalty,
            pad_token_id=request.pad_token_id,
            eos_token_id=request.eos_token_id
        )
        
        inference_time = (time.time() - start_time) * 1000  # Convert to ms
        tokens_per_second = result["output_tokens"] / (inference_time / 1000) if inference_time > 0 else 0
        
        return GenerateResponse(
            generated_text=result["generated_text"],
            input_tokens=result["input_tokens"],
            output_tokens=result["output_tokens"],
            total_tokens=result["total_tokens"],
            inference_time_ms=inference_time,
            tokens_per_second=tokens_per_second,
            model_info=result["model_info"]
        )
        
    except Exception as e:
        logger.error(f"Generation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")

@app.post("/generate/stream")
async def generate_stream(
    request: GenerateRequest,
    engine: TensorRTInferenceEngine = Depends(get_inference_engine)
):
    if not request.stream:
        request.stream = True
    
    async def stream_generator() -> AsyncIterator[str]:
        try:
            async for token_data in engine.generate_stream(
                prompt=request.prompt,
                max_new_tokens=request.max_new_tokens,
                temperature=request.temperature,
                top_k=request.top_k,
                top_p=request.top_p,
                do_sample=request.do_sample,
                repetition_penalty=request.repetition_penalty,
                pad_token_id=request.pad_token_id,
                eos_token_id=request.eos_token_id
            ):
                response = StreamResponse(
                    token=token_data["token"],
                    is_final=token_data["is_final"],
                    total_tokens=token_data["total_tokens"]
                )
                yield f"data: {response.json()}\n\n"
                
                if token_data["is_final"]:
                    break
                    
        except Exception as e:
            logger.error(f"Stream generation failed: {e}")
            error_response = {"error": str(e)}
            yield f"data: {json.dumps(error_response)}\n\n"
    
    return StreamingResponse(
        stream_generator(),
        media_type="text/plain",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Content-Type": "text/event-stream"
        }
    )

@app.post("/benchmark", response_model=BenchmarkResponse)
async def benchmark_performance(
    request: BenchmarkRequest,
    background_tasks: BackgroundTasks,
    engine: TensorRTInferenceEngine = Depends(get_inference_engine)
):
    try:
        results = []
        
        for batch_size in request.batch_sizes:
            for seq_length in request.sequence_lengths:
                logger.info(f"Benchmarking batch_size={batch_size}, seq_length={seq_length}")
                
                # Create test prompts
                test_prompt = request.prompt + " " * max(0, seq_length - len(request.prompt))
                
                benchmark_result = await engine.benchmark_inference(
                    prompt=test_prompt,
                    batch_size=batch_size,
                    num_runs=request.num_runs
                )
                
                benchmark_result.update({
                    "batch_size": batch_size,
                    "sequence_length": seq_length
                })
                
                results.append(benchmark_result)
        
        # Calculate summary statistics
        avg_latency = sum(r["avg_inference_time_ms"] for r in results) / len(results)
        max_throughput = max(r["throughput_tokens_per_sec"] for r in results)
        
        summary = {
            "total_tests": len(results),
            "avg_latency_ms": avg_latency,
            "max_throughput_tokens_per_sec": max_throughput,
            "test_configuration": {
                "batch_sizes": request.batch_sizes,
                "sequence_lengths": request.sequence_lengths,
                "num_runs": request.num_runs
            }
        }
        
        return BenchmarkResponse(
            results=results,
            summary=summary
        )
        
    except Exception as e:
        logger.error(f"Benchmark failed: {e}")
        raise HTTPException(status_code=500, detail=f"Benchmark failed: {str(e)}")

@app.get("/model/info", response_model=Dict[str, Any])
async def get_model_info(engine: TensorRTInferenceEngine = Depends(get_inference_engine)):
    try:
        model_info = await engine.get_model_info()
        engine_info = await engine.get_engine_info()
        
        return {
            "model_info": model_info,
            "engine_info": engine_info,
            "optimization_info": await engine.get_optimization_info()
        }
    except Exception as e:
        logger.error(f"Failed to get model info: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get model info: {str(e)}")

@app.post("/model/reload")
async def reload_model(
    background_tasks: BackgroundTasks,
    engine: TensorRTInferenceEngine = Depends(get_inference_engine)
):
    try:
        background_tasks.add_task(engine.reload)
        return {"message": "Model reload initiated in background"}
    except Exception as e:
        logger.error(f"Model reload failed: {e}")
        raise HTTPException(status_code=500, detail=f"Model reload failed: {str(e)}")

@app.get("/metrics", response_model=Dict[str, Any])
async def get_metrics(engine: TensorRTInferenceEngine = Depends(get_inference_engine)):
    try:
        metrics = await engine.get_metrics()
        return metrics
    except Exception as e:
        logger.error(f"Failed to get metrics: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get metrics: {str(e)}")

@app.post("/metrics/reset")
async def reset_metrics(engine: TensorRTInferenceEngine = Depends(get_inference_engine)):
    try:
        await engine.reset_metrics()
        return {"message": "Metrics reset successfully"}
    except Exception as e:
        logger.error(f"Failed to reset metrics: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to reset metrics: {str(e)}")

# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return {
        "error": exc.detail,
        "status_code": exc.status_code,
        "timestamp": time.time()
    }

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    logger.error(f"Unhandled exception: {exc}")
    return {
        "error": "Internal server error",
        "status_code": 500,
        "timestamp": time.time()
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "server.api:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info"
    )