# OpenTelemetry Instrumentation Guidelines

## Overview

This document provides guidelines for instrumenting services in the 254Carbon platform with OpenTelemetry for metrics, traces, and logs.

## Quick Start

### 1. Add Dependencies

For Python services:
```bash
pip install opentelemetry-api opentelemetry-sdk opentelemetry-instrumentation-fastapi opentelemetry-exporter-otlp
```

For Node.js services:
```bash
npm install @opentelemetry/api @opentelemetry/sdk-node @opentelemetry/instrumentation-fastify @opentelemetry/exporter-otlp-grpc
```

### 2. Environment Variables

Set these environment variables for all services:

```bash
# OTLP endpoint
export OTEL_EXPORTER_OTLP_ENDPOINT=http://otel-collector:4318

# Service identification
export OTEL_RESOURCE_ATTRIBUTES="service.name=gateway,service.version=1.0.0,deployment.environment=local"

# Optional: Enable auto-instrumentation
export OTEL_PYTHON_LOG_CORRELATION=true
export OTEL_PYTHON_DISABLED_INSTRUMENTATIONS=""
```

### 3. Basic Instrumentation

#### Python (FastAPI)

```python
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.requests import RequestsInstrumentor
from opentelemetry.instrumentation.psycopg2 import Psycopg2Instrumentor

# Initialize tracing
trace.set_tracer_provider(TracerProvider())
tracer = trace.get_tracer(__name__)

# Configure OTLP exporter
otlp_exporter = OTLPSpanExporter(endpoint="http://otel-collector:4317")
span_processor = BatchSpanProcessor(otlp_exporter)
trace.get_tracer_provider().add_span_processor(span_processor)

# Auto-instrument FastAPI
FastAPIInstrumentor.instrument_app(app)
RequestsInstrumentor().instrument()
Psycopg2Instrumentor().instrument()

# Manual instrumentation example
@tracer.start_as_current_span("process_data")
def process_data(data):
    span = trace.get_current_span()
    span.set_attribute("data.size", len(data))
    span.set_attribute("data.type", type(data).__name__)
    
    try:
        # Your business logic here
        result = business_logic(data)
        span.set_attribute("result.success", True)
        return result
    except Exception as e:
        span.set_attribute("error", True)
        span.set_attribute("error.message", str(e))
        span.record_exception(e)
        raise
```

#### Node.js (Fastify)

```javascript
const { NodeSDK } = require('@opentelemetry/sdk-node');
const { getNodeAutoInstrumentations } = require('@opentelemetry/auto-instrumentations-node');
const { OTLPTraceExporter } = require('@opentelemetry/exporter-otlp-grpc');

const sdk = new NodeSDK({
  traceExporter: new OTLPTraceExporter({
    url: 'http://otel-collector:4317',
  }),
  instrumentations: [getNodeAutoInstrumentations()],
});

sdk.start();

// Manual instrumentation
const { trace } = require('@opentelemetry/api');
const tracer = trace.getTracer('gateway-service');

async function processRequest(request) {
  const span = tracer.startSpan('process_request');
  
  try {
    span.setAttributes({
      'request.method': request.method,
      'request.url': request.url,
      'request.user_agent': request.headers['user-agent'],
    });
    
    const result = await businessLogic(request);
    
    span.setAttributes({
      'response.status_code': 200,
      'response.size': JSON.stringify(result).length,
    });
    
    return result;
  } catch (error) {
    span.recordException(error);
    span.setAttributes({
      'error': true,
      'error.message': error.message,
    });
    throw error;
  } finally {
    span.end();
  }
}
```

## Metrics Instrumentation

### RED Metrics (Rate, Errors, Duration)

#### Request Rate
```python
from opentelemetry import metrics
from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader

# Initialize metrics
metric_exporter = OTLPMetricExporter(endpoint="http://otel-collector:4318")
metric_reader = PeriodicExportingMetricReader(metric_exporter)
meter_provider = MeterProvider(metric_readers=[metric_reader])
metrics.set_meter_provider(meter_provider)

meter = metrics.get_meter(__name__)

# Create counters
request_counter = meter.create_counter(
    name="gateway_http_requests_total",
    description="Total HTTP requests",
    unit="1"
)

# Create histograms
request_duration = meter.create_histogram(
    name="gateway_http_request_duration_seconds",
    description="HTTP request duration",
    unit="s"
)

# Use in your code
def handle_request(request):
    start_time = time.time()
    
    try:
        # Process request
        result = process_request(request)
        
        # Record success metrics
        request_counter.add(1, {
            "method": request.method,
            "route": request.path,
            "status_code": "200"
        })
        
        return result
        
    except Exception as e:
        # Record error metrics
        request_counter.add(1, {
            "method": request.method,
            "route": request.path,
            "status_code": "500"
        })
        raise
        
    finally:
        # Record duration
        duration = time.time() - start_time
        request_duration.record(duration, {
            "method": request.method,
            "route": request.path
        })
```

### Custom Business Metrics

```python
# Data processing metrics
records_processed = meter.create_counter(
    name="normalization_records_processed_total",
    description="Total records processed",
    unit="1"
)

processing_duration = meter.create_histogram(
    name="normalization_processing_duration_seconds",
    description="Record processing duration",
    unit="s"
)

# ML inference metrics
inference_latency = meter.create_histogram(
    name="model_serving_inference_latency_seconds",
    description="Model inference latency",
    unit="s"
)

inference_errors = meter.create_counter(
    name="model_serving_inference_errors_total",
    description="Model inference errors",
    unit="1"
)
```

## Tracing Best Practices

### 1. Span Naming Conventions

Use consistent span names:
- `HTTP GET /api/v1/instruments`
- `kafka.consume normalized.market.ticks.v1`
- `clickhouse.query curves_computed`
- `aggregation.compute.bars interval=5m`
- `model.infer curve_forecaster`

### 2. Attribute Standards

Always include these attributes:
```python
span.set_attributes({
    "deployment.environment": "local",
    "service.version": "1.0.0",
    "tenant.id": tenant_id,
    "instrument.group": instrument_group,  # Hashed if high cardinality
})
```

### 3. Error Handling

```python
try:
    result = risky_operation()
    span.set_attribute("operation.success", True)
    return result
except Exception as e:
    span.set_attribute("error", True)
    span.set_attribute("error.type", type(e).__name__)
    span.set_attribute("error.message", str(e))
    span.record_exception(e)
    raise
```

### 4. Span Events

Use span events for important milestones:
```python
span.add_event("cache.miss", {
    "cache.key": cache_key,
    "cache.type": "redis"
})

span.add_event("database.query.start", {
    "query.type": "select",
    "table.name": "instruments"
})
```

## Logging Integration

### Structured Logging with Trace Correlation

```python
import logging
import json
from opentelemetry import trace

# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s',
    handlers=[logging.StreamHandler()]
)

logger = logging.getLogger(__name__)

def log_with_trace(level, message, **kwargs):
    """Log with trace correlation"""
    span = trace.get_current_span()
    
    log_data = {
        "ts": datetime.utcnow().isoformat() + "Z",
        "level": level.upper(),
        "service": "gateway",
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

# Usage
log_with_trace("info", "Processing request", 
               request_id=request_id,
               method=request.method,
               path=request.path)
```

## Sampling Strategy

### Head-based Sampling (Current)

```python
from opentelemetry.sdk.trace.sampling import TraceIdRatioBasedSampler

# 100% sampling in development
sampler = TraceIdRatioBasedSampler(1.0)

# 10% sampling in production
sampler = TraceIdRatioBasedSampler(0.1)

trace.set_tracer_provider(TracerProvider(sampler=sampler))
```

### Tail-based Sampling (Future)

Configure in OpenTelemetry Collector:
```yaml
processors:
  tail_sampling:
    decision_wait: 10s
    num_traces: 50000
    policies:
      - name: errors
        type: status_code
        status_code:
          status_codes: [ERROR]
      - name: slow
        type: latency
        latency:
          threshold_ms: 1000
      - name: baseline
        type: probabilistic
        probabilistic:
          sampling_percentage: 10
```

## Cost Optimization

### 1. Reduce Cardinality

```python
# Bad: High cardinality
span.set_attribute("user_id", user_id)  # Thousands of unique values

# Good: Hash or group
import hashlib
user_hash = hashlib.md5(user_id.encode()).hexdigest()[:8]
span.set_attribute("user_id_hash", user_hash)

# Or group by category
user_tier = "premium" if user.is_premium else "standard"
span.set_attribute("user_tier", user_tier)
```

### 2. Filter Attributes

```python
# Remove sensitive data
def sanitize_attributes(attributes):
    sensitive_keys = ['password', 'token', 'secret', 'key']
    return {k: v for k, v in attributes.items() 
            if not any(sensitive in k.lower() for sensitive in sensitive_keys)}

span.set_attributes(sanitize_attributes(request_attributes))
```

### 3. Limit Span Count

```python
# Only create spans for important operations
if operation_importance > threshold:
    with tracer.start_as_current_span("important_operation"):
        result = important_operation()
else:
    result = important_operation()  # No span
```

## Service-Specific Guidelines

### Gateway Service

```python
# HTTP middleware
@app.middleware("http")
async def add_trace_context(request: Request, call_next):
    with tracer.start_as_current_span(f"HTTP {request.method} {request.url.path}"):
        response = await call_next(request)
        return response

# Rate limiting metrics
rate_limit_counter = meter.create_counter(
    name="gateway_rate_limit_hits_total",
    description="Rate limit hits",
    unit="1"
)
```

### Data Pipeline

```python
# Kafka consumer instrumentation
@tracer.start_as_current_span("kafka.consume")
async def consume_message(message):
    span = trace.get_current_span()
    span.set_attributes({
        "kafka.topic": message.topic,
        "kafka.partition": message.partition,
        "kafka.offset": message.offset,
    })
    
    # Process message
    result = await process_message(message.value)
    
    span.set_attribute("processing.success", True)
    return result
```

### ML Service

```python
# Model inference instrumentation
@tracer.start_as_current_span("model.infer")
def run_inference(model, input_data):
    span = trace.get_current_span()
    span.set_attributes({
        "model.name": model.name,
        "model.version": model.version,
        "input.size": len(input_data),
    })
    
    start_time = time.time()
    try:
        result = model.predict(input_data)
        duration = time.time() - start_time
        
        span.set_attributes({
            "inference.duration": duration,
            "inference.success": True,
            "output.size": len(result),
        })
        
        return result
    except Exception as e:
        span.set_attribute("inference.error", True)
        span.record_exception(e)
        raise
```

## Testing Instrumentation

### Unit Tests

```python
def test_tracing():
    # Mock the tracer
    mock_tracer = Mock()
    with patch('opentelemetry.trace.get_tracer', return_value=mock_tracer):
        result = process_data(test_data)
        
        # Verify span was created
        mock_tracer.start_span.assert_called_once()
        
        # Verify attributes
        span = mock_tracer.start_span.return_value.__enter__.return_value
        span.set_attribute.assert_called()
```

### Integration Tests

```python
def test_metrics_export():
    # Send test requests
    for i in range(10):
        response = client.get("/api/v1/test")
        assert response.status_code == 200
    
    # Wait for metrics to be exported
    time.sleep(5)
    
    # Query Prometheus for metrics
    metrics = prometheus_client.query("gateway_http_requests_total")
    assert len(metrics) > 0
```

## Troubleshooting

### Common Issues

1. **Missing traces**: Check OTLP endpoint configuration
2. **High cardinality**: Review attribute values and hash if needed
3. **Performance impact**: Reduce sampling rate or limit span creation
4. **Memory usage**: Configure batch processor limits

### Debug Mode

```python
# Enable debug logging
import logging
logging.getLogger("opentelemetry").setLevel(logging.DEBUG)

# Use logging exporter for debugging
from opentelemetry.sdk.trace.export import ConsoleSpanExporter
console_exporter = ConsoleSpanExporter()
span_processor = BatchSpanProcessor(console_exporter)
trace.get_tracer_provider().add_span_processor(span_processor)
```

## Migration Guide

### From Custom Metrics

```python
# Old custom metrics
custom_counter = Counter('custom_requests_total', 'Total requests')

# New OpenTelemetry metrics
request_counter = meter.create_counter(
    name="gateway_http_requests_total",
    description="Total HTTP requests",
    unit="1"
)
```

### From Custom Tracing

```python
# Old custom tracing
with custom_tracer.trace('operation'):
    result = operation()

# New OpenTelemetry tracing
with tracer.start_as_current_span('operation'):
    result = operation()
```

## Resources

- [OpenTelemetry Python Documentation](https://opentelemetry.io/docs/languages/python/)
- [OpenTelemetry Node.js Documentation](https://opentelemetry.io/docs/languages/nodejs/)
- [Prometheus Metrics Best Practices](https://prometheus.io/docs/practices/naming/)
- [Grafana Dashboard Best Practices](https://grafana.com/docs/grafana/latest/best-practices/)
