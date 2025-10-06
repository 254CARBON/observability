#!/usr/bin/env python3
"""
Gateway service instrumentation example for 254Carbon

This example shows a pragmatic, production-lean setup for end-to-end
observability of a FastAPI-based gateway service using OpenTelemetry.

What this file demonstrates:
- Traces: end-to-end spans around HTTP requests and internal operations.
- Metrics: RED metrics (Requests, Errors, Duration) and gauges for WS activity.
- Logs: structured JSON logs correlated with trace/span IDs for triage.

Key design choices and rationale:
- OTLP everywhere: a single, standard protocol to ship all signals to the
  OpenTelemetry Collector. This keeps application code simple and portable.
- Low-cardinality labels: metric attributes avoid user/request identifiers to
  keep series count manageable (e.g., route rather than full URL).
- Trace correlation in logs: logs include `trace_id`/`span_id` so you can pivot
  between logs and traces in Grafana/Tempo/Loki.
- Safe defaults: batch processors for spans, periodic readers for metrics,
  and middleware that always records request duration even on exceptions.

Operational notes:
- OTLP endpoints are set to the in-cluster Collector service by default
  ("http://otel-collector:4317" for gRPC traces, "http://otel-collector:4318"
  for HTTP metrics); override via environment or config when running locally.
- The Prometheus `/metrics` exposition is handled via Collector; the in-app
  route here is informational only.
"""

import time
import asyncio  # Used in simulated I/O below (must be imported at module scope)
import json
import logging
from datetime import datetime
from typing import Dict, Any, Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn

# OpenTelemetry imports
from opentelemetry import trace, metrics
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.requests import RequestsInstrumentor
from opentelemetry.instrumentation.psycopg2 import Psycopg2Instrumentor

# Configure logging
# We emit newline-delimited JSON for ingestion by Loki or any log pipeline.
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Initialize OpenTelemetry
def setup_observability():
    """Initialize OpenTelemetry tracing and metrics.

    Tracing
    - Uses a `TracerProvider` with a BatchSpanProcessor to minimize overhead.
    - Exports spans to the local Collector via OTLP/gRPC.

    Metrics
    - Configures a `MeterProvider` with a `PeriodicExportingMetricReader`.
    - Exports metrics to Collector via OTLP/HTTP (4318).
    """
    
    # Configure tracing
    trace.set_tracer_provider(TracerProvider())
    tracer = trace.get_tracer(__name__)
    
    # Configure OTLP trace exporter
    # OTLP/gRPC exporter to the Collector; use environment/config for overrides
    otlp_trace_exporter = OTLPSpanExporter(endpoint="http://otel-collector:4317")
    span_processor = BatchSpanProcessor(otlp_trace_exporter)
    trace.get_tracer_provider().add_span_processor(span_processor)
    
    # Configure metrics
    # OTLP/HTTP metrics exporter; the Collector converts to Prometheus as needed
    otlp_metric_exporter = OTLPMetricExporter(endpoint="http://otel-collector:4318")
    metric_reader = PeriodicExportingMetricReader(otlp_metric_exporter)
    meter_provider = MeterProvider(metric_readers=[metric_reader])
    metrics.set_meter_provider(meter_provider)
    
    # Auto-instrument frameworks/clients for outbound calls and DB access
    FastAPIInstrumentor.instrument()
    RequestsInstrumentor().instrument()
    Psycopg2Instrumentor().instrument()
    
    return tracer, metrics.get_meter(__name__)

# Initialize observability
tracer, meter = setup_observability()

# Create metrics
# Naming follows <service>_<subsystem>_<metric> with units in the name where
# appropriate (OpenTelemetry SDKs attach units separately as well).
request_counter = meter.create_counter(
    name="gateway_http_requests_total",
    description="Total HTTP requests",
    unit="1"
)

request_duration = meter.create_histogram(
    name="gateway_http_request_duration_seconds",
    description="HTTP request duration",
    unit="s"
)

error_counter = meter.create_counter(
    name="gateway_http_errors_total",
    description="Total HTTP errors",
    unit="1"
)

active_connections = meter.create_up_down_counter(
    name="gateway_active_connections",
    description="Active WebSocket connections",
    unit="1"
)

# Create FastAPI app
# Keep metadata minimal; richer service attributes go into trace/metric resources.
app = FastAPI(
    title="254Carbon Gateway Service",
    description="Example gateway service with full observability",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Structured logging function
def log_with_trace(level: str, message: str, **kwargs):
    """Emit a structured JSON log enriched with active trace context.

    The resulting log line includes `trace_id` and `span_id` when a span is
    recording, enabling cross-navigation from logs to traces in Grafana.
    Avoids high-cardinality fields by default; pass explicit context via kwargs.
    """
    span = trace.get_current_span()
    
    log_data = {
        "ts": datetime.utcnow().isoformat() + "Z",
        "level": level.upper(),
        "service": "254carbon-gateway",
        "message": message,
        **kwargs
    }
    
    # Add trace context if available
    if span and span.is_recording():
        span_context = span.get_span_context()
        log_data.update({
            "trace_id": format(span_context.trace_id, '032x'),
            "span_id": format(span_context.span_id, '016x'),
        })
    
    logger.info(json.dumps(log_data))

# Middleware for request logging and metrics
@app.middleware("http")
async def observability_middleware(request: Request, call_next):
    """Capture RED signals and structured access logs for every HTTP request."""
    start_time = time.time()
    
    # Extract request information
    method = request.method
    path = request.url.path
    user_agent = request.headers.get("user-agent", "")
    client_ip = request.client.host if request.client else "unknown"
    
    # Create span for the request; set semantic attributes used in tracing UIs
    with tracer.start_as_current_span(f"HTTP {method} {path}") as span:
        # Set span attributes
        span.set_attributes({
            "http.method": method,
            "http.url": str(request.url),
            "http.route": path,
            "http.user_agent": user_agent,
            "http.client_ip": client_ip,
            "service.name": "254carbon-gateway",
            "service.version": "1.0.0",
            "deployment.environment": "local",
        })
        
        try:
            # Process request
            response = await call_next(request)
            
            # Calculate duration
            duration = time.time() - start_time
            
            # Set response attributes
            span.set_attributes({
                "http.status_code": response.status_code,
                "http.response_size": response.headers.get("content-length", 0),
            })
            
            # Record metrics with low-cardinality labels only
            request_counter.add(1, {
                "method": method,
                "route": path,
                "status_code": str(response.status_code),
                "service": "gateway"
            })
            
            request_duration.record(duration, {
                "method": method,
                "route": path,
                "service": "gateway"
            })
            
            # Log successful request in a single line for LogQL usability
            log_with_trace("info", f"{method} {path} {response.status_code} {duration*1000:.0f}ms",
                         method=method,
                         path=path,
                         status_code=response.status_code,
                         duration_ms=duration*1000,
                         user_agent=user_agent,
                         client_ip=client_ip,
                         event_type="http_access")
            
            return response
            
        except Exception as e:
            # Calculate duration
            duration = time.time() - start_time
            
            # Set error attributes
            span.set_attributes({
                "error": True,
                "error.type": type(e).__name__,
                "error.message": str(e),
            })
            
            # Record error metrics; do not include exception message as a label
            error_counter.add(1, {
                "method": method,
                "route": path,
                "error_type": type(e).__name__,
                "service": "gateway"
            })
            
            # Log error
            log_with_trace("error", f"Request failed: {str(e)}",
                         method=method,
                         path=path,
                         error_type=type(e).__name__,
                         error_message=str(e),
                         duration_ms=duration*1000,
                         event_type="error")
            
            # Record exception in span
            span.record_exception(e)
            
            # Re-raise the exception
            raise

# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "gateway", "version": "1.0.0"}

# Metrics endpoint (informational)
@app.get("/metrics")
async def metrics_endpoint():
    """Prometheus metrics endpoint"""
    # This would typically be handled by the OpenTelemetry collector
    # or a separate metrics exporter
    return {"message": "Metrics are exported via OTLP to the collector"}

# Example API endpoints
@app.get("/api/v1/instruments")
async def get_instruments():
    """Get list of financial instruments"""
    with tracer.start_as_current_span("get_instruments") as span:
        span.set_attributes({
            "operation": "get_instruments",
            "service": "gateway"
        })
        
        try:
            # Simulate database query
            await asyncio.sleep(0.1)
            
            # Simulate instruments data
            instruments = [
                {"id": "AAPL", "name": "Apple Inc.", "type": "equity"},
                {"id": "MSFT", "name": "Microsoft Corporation", "type": "equity"},
                {"id": "GOOGL", "name": "Alphabet Inc.", "type": "equity"},
            ]
            
            span.set_attributes({
                "instruments.count": len(instruments),
                "operation.success": True
            })
            
            log_with_trace("info", "Retrieved instruments successfully",
                         operation="get_instruments",
                         instrument_count=len(instruments),
                         event_type="business_event")
            
            return {"instruments": instruments, "count": len(instruments)}
            
        except Exception as e:
            span.set_attributes({
                "operation.success": False,
                "error.type": type(e).__name__,
                "error.message": str(e)
            })
            
            log_with_trace("error", "Failed to retrieve instruments",
                         operation="get_instruments",
                         error_type=type(e).__name__,
                         error_message=str(e),
                         event_type="error")
            
            raise HTTPException(status_code=500, detail="Failed to retrieve instruments")

@app.get("/api/v1/instruments/{instrument_id}")
async def get_instrument(instrument_id: str):
    """Get specific instrument details"""
    with tracer.start_as_current_span("get_instrument") as span:
        span.set_attributes({
            "operation": "get_instrument",
            "instrument.id": instrument_id,
            "service": "gateway"
        })
        
        try:
            # Simulate database query
            await asyncio.sleep(0.05)
            
            # Simulate instrument data
            instrument = {
                "id": instrument_id,
                "name": f"{instrument_id} Corporation",
                "type": "equity",
                "price": 150.50,
                "currency": "USD"
            }
            
            span.set_attributes({
                "operation.success": True,
                "instrument.price": instrument["price"]
            })
            
            log_with_trace("info", "Retrieved instrument details",
                         operation="get_instrument",
                         instrument_id=instrument_id,
                         instrument_price=instrument["price"],
                         event_type="business_event")
            
            return instrument
            
        except Exception as e:
            span.set_attributes({
                "operation.success": False,
                "error.type": type(e).__name__,
                "error.message": str(e)
            })
            
            log_with_trace("error", "Failed to retrieve instrument",
                         operation="get_instrument",
                         instrument_id=instrument_id,
                         error_type=type(e).__name__,
                         error_message=str(e),
                         event_type="error")
            
            raise HTTPException(status_code=404, detail="Instrument not found")

@app.post("/api/v1/instruments")
async def create_instrument(instrument_data: Dict[str, Any]):
    """Create a new instrument"""
    with tracer.start_as_current_span("create_instrument") as span:
        span.set_attributes({
            "operation": "create_instrument",
            "service": "gateway"
        })
        
        try:
            # Simulate validation
            if not instrument_data.get("id"):
                raise ValueError("Instrument ID is required")
            
            # Simulate database insert
            await asyncio.sleep(0.2)
            
            # Simulate created instrument
            created_instrument = {
                "id": instrument_data["id"],
                "name": instrument_data.get("name", f"{instrument_data['id']} Corporation"),
                "type": instrument_data.get("type", "equity"),
                "created_at": datetime.utcnow().isoformat() + "Z"
            }
            
            span.set_attributes({
                "operation.success": True,
                "instrument.id": created_instrument["id"]
            })
            
            log_with_trace("info", "Created instrument successfully",
                         operation="create_instrument",
                         instrument_id=created_instrument["id"],
                         event_type="business_event")
            
            return created_instrument
            
        except Exception as e:
            span.set_attributes({
                "operation.success": False,
                "error.type": type(e).__name__,
                "error.message": str(e)
            })
            
            log_with_trace("error", "Failed to create instrument",
                         operation="create_instrument",
                         error_type=type(e).__name__,
                         error_message=str(e),
                         event_type="error")
            
            raise HTTPException(status_code=400, detail=str(e))

# WebSocket endpoint for streaming
@app.websocket("/ws/stream")
async def websocket_stream(websocket):
    """WebSocket endpoint for real-time data streaming"""
    with tracer.start_as_current_span("websocket_stream") as span:
        span.set_attributes({
            "operation": "websocket_stream",
            "service": "gateway"
        })
        
        try:
            # Increment active connections
            active_connections.add(1, {"service": "gateway"})
            
            span.set_attributes({
                "websocket.connection": "established"
            })
            
            log_with_trace("info", "WebSocket connection established",
                         operation="websocket_stream",
                         event_type="business_event")
            
            await websocket.accept()
            
            # Simulate streaming data; emit one message per second
            for i in range(10):
                data = {
                    "timestamp": datetime.utcnow().isoformat() + "Z",
                    "sequence": i,
                    "data": f"stream_data_{i}"
                }
                
                await websocket.send_json(data)
                await asyncio.sleep(1)
            
            # Decrement active connections
            active_connections.add(-1, {"service": "gateway"})
            
            span.set_attributes({
                "websocket.connection": "closed",
                "operation.success": True
            })
            
            log_with_trace("info", "WebSocket connection closed",
                         operation="websocket_stream",
                         event_type="business_event")
            
        except Exception as e:
            # Decrement active connections on error
            active_connections.add(-1, {"service": "gateway"})
            
            span.set_attributes({
                "websocket.connection": "error",
                "error.type": type(e).__name__,
                "error.message": str(e)
            })
            
            log_with_trace("error", "WebSocket connection error",
                         operation="websocket_stream",
                         error_type=type(e).__name__,
                         error_message=str(e),
                         event_type="error")
            
            span.record_exception(e)
            raise

# Error handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler"""
    with tracer.start_as_current_span("global_exception_handler") as span:
        span.set_attributes({
            "error": True,
            "error.type": type(exc).__name__,
            "error.message": str(exc),
            "http.method": request.method,
            "http.url": str(request.url),
        })
        
        log_with_trace("error", f"Global exception: {str(exc)}",
                     method=request.method,
                     path=request.url.path,
                     error_type=type(exc).__name__,
                     error_message=str(exc),
                     event_type="error")
        
        span.record_exception(exc)
        
        return JSONResponse(
            status_code=500,
            content={"error": "Internal server error", "detail": str(exc)}
        )

# Startup event
@app.on_event("startup")
async def startup_event():
    """Application startup event"""
    log_with_trace("info", "Gateway service started",
                 service="254carbon-gateway",
                 version="1.0.0",
                 event_type="business_event")

# Shutdown event
@app.on_event("shutdown")
async def shutdown_event():
    """Application shutdown event"""
    log_with_trace("info", "Gateway service shutting down",
                 service="254carbon-gateway",
                 event_type="business_event")

if __name__ == "__main__":
    import asyncio
    
    # Run the application
    uvicorn.run(
        "gateway_instrumentation:app",
        host="0.0.0.0",
        port=8080,
        reload=True,
        log_level="info"
    )
