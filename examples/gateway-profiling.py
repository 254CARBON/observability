#!/usr/bin/env python3
"""
254Carbon Gateway Service with Pyroscope Profiling Integration

This example demonstrates how to integrate Pyroscope profiling with the gateway service
to enable continuous CPU and memory profiling for performance optimization.
"""

import asyncio
import logging
import time
from typing import Dict, Any
from contextlib import asynccontextmanager

# Pyroscope imports
try:
    import pyroscope
except ImportError:
    print("Pyroscope not installed. Install with: pip install pyroscope-io")
    pyroscope = None

# OpenTelemetry imports
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor
from opentelemetry.instrumentation.redis import RedisInstrumentor
from opentelemetry.instrumentation.psycopg2 import Psycopg2Instrumentor

# FastAPI imports
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
import httpx
import redis
import psycopg2
from psycopg2.extras import RealDictCursor

# Structured logging
import structlog
from structlog.stdlib import LoggerFactory

# Configure structured logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger()

# Pyroscope configuration
PYROSCOPE_CONFIG = {
    "application_name": "254carbon-gateway",
    "server_address": "http://pyroscope:4040",
    "auth_token": None,  # Set if authentication is enabled
    "sample_rate": 100,  # Hz
    "log_level": "info",
    "tags": {
        "service": "gateway",
        "environment": "local",
        "version": "1.0.0"
    }
}

# OpenTelemetry configuration
OTEL_CONFIG = {
    "service_name": "254carbon-gateway",
    "service_version": "1.0.0",
    "deployment_environment": "local",
    "otlp_endpoint": "http://otel-collector:4317"
}

# Database and cache configuration
DB_CONFIG = {
    "host": "postgres",
    "port": 5432,
    "database": "254carbon",
    "user": "gateway",
    "password": "gateway123"
}

REDIS_CONFIG = {
    "host": "redis",
    "port": 6379,
    "db": 0,
    "decode_responses": True
}

# Initialize Pyroscope profiling
def init_pyroscope():
    """Initialize Pyroscope profiling with cost-optimized settings."""
    if pyroscope is None:
        logger.warning("Pyroscope not available, profiling disabled")
        return
        
    try:
        pyroscope.configure(
            application_name=PYROSCOPE_CONFIG["application_name"],
            server_address=PYROSCOPE_CONFIG["server_address"],
            auth_token=PYROSCOPE_CONFIG["auth_token"],
            sample_rate=PYROSCOPE_CONFIG["sample_rate"],
            log_level=PYROSCOPE_CONFIG["log_level"],
            tags=PYROSCOPE_CONFIG["tags"]
        )
        logger.info("Pyroscope profiling initialized", 
                   application=PYROSCOPE_CONFIG["application_name"],
                   server=PYROSCOPE_CONFIG["server_address"])
    except Exception as e:
        logger.error("Failed to initialize Pyroscope", error=str(e))

# Initialize OpenTelemetry tracing
def init_tracing():
    """Initialize OpenTelemetry tracing with span profiling integration."""
    try:
        # Set up tracer provider
        trace.set_tracer_provider(TracerProvider())
        tracer = trace.get_tracer(__name__)
        
        # Set up OTLP exporter
        otlp_exporter = OTLPSpanExporter(
            endpoint=OTEL_CONFIG["otlp_endpoint"],
            insecure=True
        )
        
        # Set up batch span processor
        span_processor = BatchSpanProcessor(otlp_exporter)
        trace.get_tracer_provider().add_span_processor(span_processor)
        
        # Instrument libraries
        FastAPIInstrumentor.instrument_app(app)
        HTTPXClientInstrumentor().instrument()
        RedisInstrumentor().instrument()
        Psycopg2Instrumentor().instrument()
        
        logger.info("OpenTelemetry tracing initialized", 
                   service=OTEL_CONFIG["service_name"],
                   endpoint=OTEL_CONFIG["otlp_endpoint"])
        
        return tracer
    except Exception as e:
        logger.error("Failed to initialize OpenTelemetry", error=str(e))
        return None

# Database connection pool
class DatabasePool:
    """Database connection pool with profiling integration."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.pool = []
        self.max_connections = 10
        
    @asynccontextmanager
    async def get_connection(self):
        """Get database connection with profiling context."""
        if pyroscope:
            with pyroscope.tag_wrapper({"operation": "db_connection"}):
                conn = psycopg2.connect(**self.config)
                try:
                    yield conn
                finally:
                    conn.close()
        else:
            conn = psycopg2.connect(**self.config)
            try:
                yield conn
            finally:
                conn.close()

# Redis cache with profiling
class RedisCache:
    """Redis cache with profiling integration."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.client = redis.Redis(**config)
        
    async def get(self, key: str) -> Any:
        """Get value from cache with profiling."""
        if pyroscope:
            with pyroscope.tag_wrapper({"operation": "cache_get", "key": key}):
                return self.client.get(key)
        else:
            return self.client.get(key)
            
    async def set(self, key: str, value: Any, ttl: int = 300) -> bool:
        """Set value in cache with profiling."""
        if pyroscope:
            with pyroscope.tag_wrapper({"operation": "cache_set", "key": key}):
                return self.client.setex(key, ttl, value)
        else:
            return self.client.setex(key, ttl, value)

# Initialize components
init_pyroscope()
tracer = init_tracing()
db_pool = DatabasePool(DB_CONFIG)
redis_cache = RedisCache(REDIS_CONFIG)

# FastAPI application
app = FastAPI(
    title="254Carbon Gateway Service",
    description="Gateway service with Pyroscope profiling integration",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Profiling decorators
def profile_function(func_name: str):
    """Decorator to profile function execution."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            if pyroscope:
                with pyroscope.tag_wrapper({"function": func_name}):
                    return func(*args, **kwargs)
            else:
                return func(*args, **kwargs)
        return wrapper
    return decorator

def profile_async_function(func_name: str):
    """Decorator to profile async function execution."""
    def decorator(func):
        async def wrapper(*args, **kwargs):
            if pyroscope:
                with pyroscope.tag_wrapper({"function": func_name}):
                    return await func(*args, **kwargs)
            else:
                return await func(*args, **kwargs)
        return wrapper
    return decorator

# Profiled business logic
class GatewayService:
    """Gateway service with profiling integration."""
    
    def __init__(self):
        self.db_pool = db_pool
        self.cache = redis_cache
        
    @profile_async_function("process_request")
    async def process_request(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process incoming request with profiling."""
        start_time = time.time()
        
        try:
            # Simulate business logic
            await asyncio.sleep(0.1)
            
            # Database operation
            result = await self._fetch_data(request_data.get("id"))
            
            # Cache operation
            await self._cache_result(result)
            
            # External API call
            external_data = await self._call_external_api(request_data)
            
            # Combine results
            response = {
                "id": request_data.get("id"),
                "data": result,
                "external": external_data,
                "processed_at": time.time(),
                "duration_ms": (time.time() - start_time) * 1000
            }
            
            logger.info("Request processed successfully", 
                       request_id=request_data.get("id"),
                       duration_ms=response["duration_ms"])
            
            return response
            
        except Exception as e:
            logger.error("Request processing failed", 
                        request_id=request_data.get("id"),
                        error=str(e))
            raise HTTPException(status_code=500, detail=str(e))
    
    @profile_async_function("fetch_data")
    async def _fetch_data(self, data_id: str) -> Dict[str, Any]:
        """Fetch data from database with profiling."""
        if pyroscope:
            with pyroscope.tag_wrapper({"operation": "db_query", "table": "instruments"}):
                async with self.db_pool.get_connection() as conn:
                    cursor = conn.cursor(cursor_factory=RealDictCursor)
                    cursor.execute("SELECT * FROM instruments WHERE id = %s", (data_id,))
                    result = cursor.fetchone()
                    return dict(result) if result else {}
        else:
            async with self.db_pool.get_connection() as conn:
                cursor = conn.cursor(cursor_factory=RealDictCursor)
                cursor.execute("SELECT * FROM instruments WHERE id = %s", (data_id,))
                result = cursor.fetchone()
                return dict(result) if result else {}
    
    @profile_async_function("cache_result")
    async def _cache_result(self, data: Dict[str, Any]) -> bool:
        """Cache result with profiling."""
        cache_key = f"instrument:{data.get('id')}"
        return await self.cache.set(cache_key, str(data), ttl=300)
    
    @profile_async_function("call_external_api")
    async def _call_external_api(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Call external API with profiling."""
        if pyroscope:
            with pyroscope.tag_wrapper({"operation": "external_api", "service": "market_data"}):
                async with httpx.AsyncClient() as client:
                    response = await client.get("http://market-data:8080/api/v1/ticker")
                    return response.json()
        else:
            async with httpx.AsyncClient() as client:
                response = await client.get("http://market-data:8080/api/v1/ticker")
                return response.json()

# Initialize service
gateway_service = GatewayService()

# API endpoints
@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "gateway", "version": "1.0.0"}

@app.post("/api/v1/process")
async def process_request(request: Request):
    """Process request endpoint with profiling."""
    request_data = await request.json()
    
    # Add trace context
    if tracer:
        with tracer.start_as_current_span("gateway.process_request") as span:
            span.set_attribute("request.id", request_data.get("id"))
            span.set_attribute("request.size", len(str(request_data)))
            
            result = await gateway_service.process_request(request_data)
            
            span.set_attribute("response.size", len(str(result)))
            span.set_attribute("response.duration_ms", result["duration_ms"])
            
            return result
    else:
        return await gateway_service.process_request(request_data)

@app.get("/api/v1/instruments/{instrument_id}")
async def get_instrument(instrument_id: str):
    """Get instrument endpoint with profiling."""
    if tracer:
        with tracer.start_as_current_span("gateway.get_instrument") as span:
            span.set_attribute("instrument.id", instrument_id)
            
            # Check cache first
            cache_key = f"instrument:{instrument_id}"
            cached_data = await gateway_service.cache.get(cache_key)
            
            if cached_data:
                span.set_attribute("cache.hit", True)
                return {"id": instrument_id, "data": cached_data, "cached": True}
            
            # Fetch from database
            span.set_attribute("cache.hit", False)
            data = await gateway_service._fetch_data(instrument_id)
            
            if data:
                # Cache the result
                await gateway_service._cache_result(data)
                return {"id": instrument_id, "data": data, "cached": False}
            else:
                raise HTTPException(status_code=404, detail="Instrument not found")
    else:
        # Check cache first
        cache_key = f"instrument:{instrument_id}"
        cached_data = await gateway_service.cache.get(cache_key)
        
        if cached_data:
            return {"id": instrument_id, "data": cached_data, "cached": True}
        
        # Fetch from database
        data = await gateway_service._fetch_data(instrument_id)
        
        if data:
            # Cache the result
            await gateway_service._cache_result(data)
            return {"id": instrument_id, "data": data, "cached": False}
        else:
            raise HTTPException(status_code=404, detail="Instrument not found")

@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint."""
    # This would be implemented with prometheus_client
    return {"message": "Metrics endpoint - implement with prometheus_client"}

# Performance testing endpoints
@app.post("/api/v1/load-test")
async def load_test(request: Request):
    """Load test endpoint for profiling validation."""
    request_data = await request.json()
    num_requests = request_data.get("num_requests", 10)
    
    if pyroscope:
        with pyroscope.tag_wrapper({"operation": "load_test", "num_requests": num_requests}):
            tasks = []
            for i in range(num_requests):
                task = gateway_service.process_request({"id": f"test_{i}"})
                tasks.append(task)
            
            results = await asyncio.gather(*tasks)
            return {"results": results, "count": len(results)}
    else:
        tasks = []
        for i in range(num_requests):
            task = gateway_service.process_request({"id": f"test_{i}"})
            tasks.append(task)
        
        results = await asyncio.gather(*tasks)
        return {"results": results, "count": len(results)}

if __name__ == "__main__":
    import uvicorn
    
    # Start server with profiling
    uvicorn.run(
        "gateway_profiling:app",
        host="0.0.0.0",
        port=8080,
        reload=True,
        log_level="info"
    )
