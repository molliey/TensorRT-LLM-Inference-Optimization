from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.middleware.gzip import GZipMiddleware
import time
import logging
import json
import uuid
from typing import Callable

logger = logging.getLogger(__name__)

def setup_middleware(app: FastAPI):
    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Configure appropriately for production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Trusted host middleware (configure for production)
    app.add_middleware(
        TrustedHostMiddleware,
        allowed_hosts=["*"]  # Configure appropriately for production
    )
    
    # Gzip compression
    app.add_middleware(GZipMiddleware, minimum_size=1000)
    
    # Custom middleware
    app.add_middleware(LoggingMiddleware)
    app.add_middleware(SecurityMiddleware)
    app.add_middleware(RateLimitMiddleware)

class LoggingMiddleware:
    def __init__(self, app):
        self.app = app
    
    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        
        request_id = str(uuid.uuid4())
        start_time = time.time()
        
        # Extract request info
        method = scope["method"]
        path = scope["path"]
        client_ip = scope.get("client", ["unknown", None])[0]
        
        # Log request
        logger.info(f"[{request_id}] {method} {path} - Client: {client_ip}")
        
        async def send_wrapper(message):
            if message["type"] == "http.response.start":
                status_code = message["status"]
                process_time = time.time() - start_time
                
                # Log response
                logger.info(
                    f"[{request_id}] {method} {path} - "
                    f"Status: {status_code} - "
                    f"Time: {process_time:.3f}s"
                )
                
                # Add custom headers
                headers = dict(message.get("headers", []))
                headers[b"x-request-id"] = request_id.encode()
                headers[b"x-process-time"] = f"{process_time:.3f}".encode()
                message["headers"] = list(headers.items())
            
            await send(message)
        
        await self.app(scope, receive, send_wrapper)

class SecurityMiddleware:
    def __init__(self, app):
        self.app = app
    
    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        
        async def send_wrapper(message):
            if message["type"] == "http.response.start":
                # Add security headers
                headers = dict(message.get("headers", []))
                
                # Security headers
                headers[b"x-content-type-options"] = b"nosniff"
                headers[b"x-frame-options"] = b"DENY"
                headers[b"x-xss-protection"] = b"1; mode=block"
                headers[b"strict-transport-security"] = b"max-age=31536000; includeSubDomains"
                headers[b"referrer-policy"] = b"strict-origin-when-cross-origin"
                
                # API-specific headers
                headers[b"cache-control"] = b"no-cache, no-store, must-revalidate"
                headers[b"pragma"] = b"no-cache"
                headers[b"expires"] = b"0"
                
                message["headers"] = list(headers.items())
            
            await send(message)
        
        await self.app(scope, receive, send_wrapper)

class RateLimitMiddleware:
    def __init__(self, app):
        self.app = app
        self.requests = {}  # Simple in-memory rate limiting
        self.max_requests_per_minute = 60
        self.cleanup_interval = 60  # seconds
        self.last_cleanup = time.time()
    
    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        
        # Get client IP
        client_ip = scope.get("client", ["unknown", None])[0]
        current_time = time.time()
        
        # Cleanup old entries periodically
        if current_time - self.last_cleanup > self.cleanup_interval:
            self._cleanup_old_requests(current_time)
            self.last_cleanup = current_time
        
        # Check rate limit
        if self._is_rate_limited(client_ip, current_time):
            # Send rate limit exceeded response
            response = {
                "error": "Rate limit exceeded",
                "message": f"Maximum {self.max_requests_per_minute} requests per minute allowed",
                "retry_after": 60
            }
            
            await send({
                "type": "http.response.start",
                "status": 429,
                "headers": [
                    [b"content-type", b"application/json"],
                    [b"retry-after", b"60"],
                    [b"x-ratelimit-limit", str(self.max_requests_per_minute).encode()],
                    [b"x-ratelimit-remaining", b"0"],
                    [b"x-ratelimit-reset", str(int(current_time + 60)).encode()]
                ]
            })
            
            await send({
                "type": "http.response.body",
                "body": json.dumps(response).encode()
            })
            return
        
        # Record request
        self._record_request(client_ip, current_time)
        
        # Add rate limit headers
        async def send_wrapper(message):
            if message["type"] == "http.response.start":
                headers = dict(message.get("headers", []))
                
                remaining = max(0, self.max_requests_per_minute - len(self.requests.get(client_ip, [])))
                reset_time = int(current_time + 60)
                
                headers[b"x-ratelimit-limit"] = str(self.max_requests_per_minute).encode()
                headers[b"x-ratelimit-remaining"] = str(remaining).encode()
                headers[b"x-ratelimit-reset"] = str(reset_time).encode()
                
                message["headers"] = list(headers.items())
            
            await send(message)
        
        await self.app(scope, receive, send_wrapper)
    
    def _is_rate_limited(self, client_ip: str, current_time: float) -> bool:
        if client_ip not in self.requests:
            return False
        
        # Count requests in the last minute
        recent_requests = [
            req_time for req_time in self.requests[client_ip]
            if current_time - req_time < 60
        ]
        
        return len(recent_requests) >= self.max_requests_per_minute
    
    def _record_request(self, client_ip: str, current_time: float):
        if client_ip not in self.requests:
            self.requests[client_ip] = []
        
        self.requests[client_ip].append(current_time)
        
        # Keep only requests from the last minute
        self.requests[client_ip] = [
            req_time for req_time in self.requests[client_ip]
            if current_time - req_time < 60
        ]
    
    def _cleanup_old_requests(self, current_time: float):
        clients_to_remove = []
        
        for client_ip in self.requests:
            # Remove old requests
            self.requests[client_ip] = [
                req_time for req_time in self.requests[client_ip]
                if current_time - req_time < 60
            ]
            
            # Remove client if no recent requests
            if not self.requests[client_ip]:
                clients_to_remove.append(client_ip)
        
        for client_ip in clients_to_remove:
            del self.requests[client_ip]

class MetricsMiddleware:
    def __init__(self, app):
        self.app = app
        self.metrics = {
            "total_requests": 0,
            "requests_by_method": {},
            "requests_by_path": {},
            "response_times": [],
            "status_codes": {},
            "error_count": 0
        }
    
    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        
        method = scope["method"]
        path = scope["path"]
        start_time = time.time()
        
        # Update request metrics
        self.metrics["total_requests"] += 1
        self.metrics["requests_by_method"][method] = self.metrics["requests_by_method"].get(method, 0) + 1
        self.metrics["requests_by_path"][path] = self.metrics["requests_by_path"].get(path, 0) + 1
        
        async def send_wrapper(message):
            if message["type"] == "http.response.start":
                status_code = message["status"]
                response_time = time.time() - start_time
                
                # Update response metrics
                self.metrics["response_times"].append(response_time)
                self.metrics["status_codes"][status_code] = self.metrics["status_codes"].get(status_code, 0) + 1
                
                if status_code >= 400:
                    self.metrics["error_count"] += 1
                
                # Keep only last 1000 response times
                if len(self.metrics["response_times"]) > 1000:
                    self.metrics["response_times"] = self.metrics["response_times"][-1000:]
            
            await send(message)
        
        await self.app(scope, receive, send_wrapper)
    
    def get_metrics(self) -> dict:
        metrics = self.metrics.copy()
        
        # Calculate average response time
        if self.metrics["response_times"]:
            metrics["avg_response_time"] = sum(self.metrics["response_times"]) / len(self.metrics["response_times"])
            metrics["min_response_time"] = min(self.metrics["response_times"])
            metrics["max_response_time"] = max(self.metrics["response_times"])
        else:
            metrics["avg_response_time"] = 0
            metrics["min_response_time"] = 0
            metrics["max_response_time"] = 0
        
        # Calculate error rate
        metrics["error_rate"] = self.metrics["error_count"] / max(1, self.metrics["total_requests"])
        
        return metrics

# Request validation middleware
class RequestValidationMiddleware:
    def __init__(self, app):
        self.app = app
        self.max_request_size = 10 * 1024 * 1024  # 10MB
    
    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        
        # Check content length
        headers = dict(scope.get("headers", []))
        content_length = headers.get(b"content-length")
        
        if content_length:
            try:
                length = int(content_length.decode())
                if length > self.max_request_size:
                    # Send request too large response
                    response = {
                        "error": "Request too large",
                        "message": f"Maximum request size is {self.max_request_size} bytes"
                    }
                    
                    await send({
                        "type": "http.response.start",
                        "status": 413,
                        "headers": [[b"content-type", b"application/json"]]
                    })
                    
                    await send({
                        "type": "http.response.body",
                        "body": json.dumps(response).encode()
                    })
                    return
            except ValueError:
                pass
        
        await self.app(scope, receive, send)