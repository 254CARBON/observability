# OpenTelemetry Semantic Conventions

## Overview

This document defines the semantic conventions used across the 254Carbon platform for consistent observability data.

## Service Naming

### Service Names
- Format: `254carbon-<service>`
- Examples:
  - `254carbon-gateway`
  - `254carbon-streaming`
  - `254carbon-ingestion`
  - `254carbon-normalization`
  - `254carbon-ml-serving`

### Service Versions
- Format: Semantic versioning (e.g., `1.0.0`, `2.1.3`)
- Set via environment variable: `OTEL_RESOURCE_ATTRIBUTES="service.version=1.0.0"`

## Resource Attributes

### Standard Attributes
```yaml
deployment.environment: "local" | "staging" | "production"
service.name: "254carbon-gateway"
service.version: "1.0.0"
service.instance.id: "gateway-pod-12345"
k8s.namespace.name: "default"
k8s.pod.name: "gateway-7d4b8c9f-xyz"
k8s.node.name: "worker-node-1"
```

### Custom Attributes
```yaml
tenant.id: "default" | "premium" | "enterprise"
instrument.group: "equity" | "bond" | "derivative"  # Hashed if high cardinality
data.source: "market_data" | "reference_data" | "user_data"
processing.stage: "ingestion" | "normalization" | "enrichment" | "aggregation"
```

## Span Attributes

### HTTP Attributes
```yaml
http.method: "GET" | "POST" | "PUT" | "DELETE"
http.url: "https://api.254carbon.local/v1/instruments"
http.route: "/api/v1/instruments"
http.status_code: 200 | 400 | 500
http.user_agent: "Mozilla/5.0..."
http.request_content_length: 1024
http.response_content_length: 2048
```

### Database Attributes
```yaml
db.system: "postgresql" | "clickhouse" | "redis"
db.name: "market_data"
db.operation: "SELECT" | "INSERT" | "UPDATE" | "DELETE"
db.sql.table: "instruments"
db.connection_string: "postgresql://user@host:5432/db"  # Sanitized
```

### Messaging Attributes
```yaml
messaging.system: "kafka" | "rabbitmq" | "redis"
messaging.destination: "market.ticks.v1"
messaging.destination_kind: "topic" | "queue"
messaging.operation: "publish" | "receive"
messaging.message_id: "msg-12345"
messaging.consumer_group: "normalization-group"
```

### ML Attributes
```yaml
ml.model.name: "curve_forecaster"
ml.model.version: "v2.1.0"
ml.inference.type: "batch" | "streaming"
ml.inference.batch_size: 100
ml.inference.input_features: 50
ml.inference.output_classes: 10
```

## Metric Names

### Naming Convention
- Format: `<service>_<area>_<metric>_<unit>`
- Use snake_case
- Include unit suffix for clarity

### HTTP Metrics
```yaml
gateway_http_requests_total: Counter
gateway_http_request_duration_seconds: Histogram
gateway_http_request_size_bytes: Histogram
gateway_http_response_size_bytes: Histogram
```

### Business Metrics
```yaml
ingestion_connector_run_duration_seconds: Histogram
ingestion_connector_records_processed_total: Counter
ingestion_connector_errors_total: Counter

normalization_records_processed_total: Counter
normalization_processing_duration_seconds: Histogram
normalization_schema_validation_failures_total: Counter

aggregation_bar_build_duration_seconds: Histogram
aggregation_bars_generated_total: Counter
aggregation_cache_hit_ratio: Gauge

model_serving_inference_latency_seconds: Histogram
model_serving_inference_errors_total: Counter
model_serving_model_load_duration_seconds: Histogram
```

### Infrastructure Metrics
```yaml
clickhouse_query_duration_seconds: Histogram
clickhouse_query_errors_total: Counter
clickhouse_connection_pool_active: Gauge

redis_operation_duration_seconds: Histogram
redis_operation_errors_total: Counter
redis_memory_usage_bytes: Gauge

kafka_consumer_lag_seconds: Gauge
kafka_producer_records_sent_total: Counter
kafka_producer_errors_total: Counter
```

## Label Standards

### Common Labels
```yaml
service: "gateway" | "streaming" | "ingestion" | "normalization" | "ml"
endpoint: "/api/v1/instruments" | "/api/v1/stream" | "/health"
status_code: "200" | "400" | "500"
method: "GET" | "POST" | "PUT" | "DELETE"
tenant_id: "default" | "premium" | "enterprise"
```

### Service-Specific Labels
```yaml
# Gateway
route: "/api/v1/instruments" | "/api/v1/stream"
client_type: "web" | "mobile" | "api"

# Ingestion
connector: "market_data_connector" | "reference_data_connector"
data_source: "bloomberg" | "reuters" | "internal"
market: "NYSE" | "NASDAQ" | "LSE"

# Normalization
schema_version: "v1" | "v2"
data_type: "tick" | "bar" | "reference"
validation_status: "passed" | "failed" | "warning"

# ML
model_name: "curve_forecaster" | "sentiment_analyzer"
model_version: "v1.0.0" | "v2.1.0"
inference_type: "batch" | "streaming"
```

## Histogram Buckets

### Latency Buckets (seconds)
```python
# Gateway HTTP requests
[0.001, 0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 0.75, 1.0, 2.5, 5.0, 7.5, 10.0]

# Database queries
[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 25.0, 50.0, 100.0]

# ML inference
[0.1, 0.5, 1.0, 2.5, 5.0, 10.0, 25.0, 50.0, 100.0, 250.0, 500.0]
```

### Size Buckets (bytes)
```python
# Request/response sizes
[1024, 4096, 16384, 65536, 262144, 1048576, 4194304, 16777216]

# Message sizes
[1024, 4096, 16384, 65536, 262144, 1048576]
```

## Error Conventions

### Error Types
```yaml
error.type: "ValidationError" | "TimeoutError" | "ConnectionError" | "AuthenticationError"
error.message: "Invalid request format"
error.code: "VALIDATION_FAILED" | "TIMEOUT" | "CONNECTION_REFUSED"
```

### Error Labels
```yaml
error_category: "client" | "server" | "network" | "timeout"
error_severity: "low" | "medium" | "high" | "critical"
error_recoverable: "true" | "false"
```

## Log Fields

### Standard Log Schema
```json
{
  "ts": "2025-01-06T12:34:56.789Z",
  "level": "INFO" | "WARN" | "ERROR" | "DEBUG",
  "service": "gateway",
  "trace_id": "abcd1234efgh5678",
  "span_id": "ijkl9012",
  "request_id": "req-xyz-123",
  "tenant_id": "default",
  "event_type": "http_access" | "business_event" | "error",
  "message": "GET /api/v1/instruments 200 42ms",
  "duration_ms": 42,
  "status_code": 200,
  "extra": {
    "cache_hit": true,
    "user_id": "user-123",
    "instrument_count": 150
  }
}
```

### Event Types
```yaml
http_access: HTTP request/response logging
business_event: Important business logic events
error: Error conditions
performance: Performance-related events
security: Security-related events
audit: Audit trail events
```

## Correlation Fields

### Trace Correlation
```yaml
trace_id: 32-character hex string
span_id: 16-character hex string
parent_span_id: 16-character hex string (optional)
```

### Request Correlation
```yaml
request_id: "req-{uuid}" | "req-{timestamp}-{random}"
correlation_id: "corr-{uuid}" | "corr-{timestamp}-{random}"
session_id: "session-{uuid}" | "session-{timestamp}-{random}"
```

### Business Correlation
```yaml
tenant_id: "default" | "premium" | "enterprise"
user_id: "user-{uuid}" | "user-{hash}"
instrument_id: "instrument-{uuid}" | "instrument-{hash}"
order_id: "order-{uuid}" | "order-{timestamp}-{random}"
```

## Sampling Conventions

### Head-based Sampling
```yaml
# Development
sampling_rate: 1.0  # 100%

# Production
sampling_rate: 0.1  # 10%

# High-traffic services
sampling_rate: 0.01  # 1%
```

### Tail-based Sampling Rules
```yaml
# Always sample errors
error_sampling: 1.0

# Always sample slow operations
slow_sampling_threshold: "2x_p95_target"

# Sample baseline operations
baseline_sampling: 0.1
```

## Dashboard Conventions

### Panel Naming
```yaml
# RED metrics
"Request Rate": rate(requests_total[5m])
"Error Rate": rate(errors_total[5m]) / rate(requests_total[5m])
"Request Duration": histogram_quantile(0.95, rate(duration_bucket[5m]))

# USE metrics
"CPU Utilization": rate(cpu_usage[5m])
"Memory Saturation": memory_usage / memory_limit
"Disk Errors": rate(disk_errors[5m])
```

### Dashboard Organization
```yaml
# Folder structure
access/
  gateway_overview.json
  streaming_overview.json
ingestion/
  connectors_health.json
  data_quality.json
data_processing/
  normalization_pipeline.json
  aggregation_performance.json
ml/
  model_serving_latency.json
  inference_throughput.json
infra/
  node_resources.json
  database_health.json
security/
  auth_failures.json
  policy_decisions.json
```

## Alert Conventions

### Alert Naming
```yaml
# Format: ALERT_<SERVICE>_<CONDITION>
ALERT_GATEWAY_HIGH_ERROR_RATE
ALERT_GATEWAY_LATENCY_DEGRADATION
ALERT_STREAMING_CONNECTION_CHURN
ALERT_INGESTION_DATA_LAG
ALERT_ML_INFERENCE_LATENCY
```

### Alert Severity
```yaml
critical: Immediate attention required
warning: Attention required within 15 minutes
info: Informational, no immediate action
```

### Alert Labels
```yaml
severity: "critical" | "warning" | "info"
service: "gateway" | "streaming" | "ingestion"
alert_type: "error_rate" | "latency" | "availability" | "slo_burn"
runbook_url: "https://docs.254carbon.local/runbooks/..."
```

## Best Practices

### 1. Consistency
- Use the same attribute names across all services
- Follow the naming conventions strictly
- Document any deviations

### 2. Performance
- Hash high-cardinality attributes
- Use appropriate sampling rates
- Filter sensitive data

### 3. Maintainability
- Version your semantic conventions
- Provide migration guides
- Use automated validation

### 4. Observability
- Include correlation fields
- Use meaningful span names
- Add context-rich attributes

## Migration Guide

### Version 1.0 to 2.0
```yaml
# Old
service_name: "gateway"

# New
service.name: "254carbon-gateway"
```

### Custom to Standard
```yaml
# Old
custom_request_id: "req-123"

# New
request_id: "req-123"
```

## Validation

### Automated Checks
```python
# Validate attribute names
def validate_attributes(attributes):
    allowed_attributes = load_semantic_conventions()
    for key in attributes:
        if key not in allowed_attributes:
            raise ValueError(f"Unknown attribute: {key}")

# Validate metric names
def validate_metric_name(name):
    pattern = r'^[a-z][a-z0-9_]*_[a-z0-9_]*$'
    if not re.match(pattern, name):
        raise ValueError(f"Invalid metric name: {name}")
```

### Manual Reviews
- Regular convention reviews
- Cross-team alignment
- Documentation updates
