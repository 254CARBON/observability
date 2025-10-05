#````markdown name=README.md
# 254Carbon Observability (`254carbon-observability`)

> Centralized toolkit for **metrics, logs, traces, profiling (future), event telemetry, dashboards, alerts, SLO orchestration, and operational insight** across the 254Carbon platform.

This repository defines deployable manifests, configuration bundles, dashboards, alert rules, correlation conventions, and integration guides for every service (access, ingestion, data-processing, ML, security, infra). It is the single source of truth for how observability is *implemented*, *standardized*, and *evolved*.

---

## Table of Contents
1. Objectives & Non‑Goals  
2. Capabilities Overview  
3. Architectural Diagram  
4. Repository Structure  
5. Core Components  
6. Data Flow (Metrics / Traces / Logs / Events)  
7. Conventions & Naming Standards  
8. Metrics Taxonomy & Guidelines  
9. Tracing Strategy (OpenTelemetry)  
10. Logging Strategy & Schema  
11. Dashboards & Visualization  
12. Alerting & SLO Management  
13. Sampling, Retention & Cost Control  
14. Correlation (Trace ⇄ Log ⇄ Metric ⇄ Event)  
15. Integration Instructions (Per Domain)  
16. Local Development & Quick Start  
17. Configuration & Environment Variables  
18. Security & Multi‑Tenancy Considerations  
19. Performance & Scaling Guidance  
20. Runbooks & Incident Practices  
21. Testing Observability (Instrumentation QA)  
22. Roadmap & Future Enhancements  
23. Contribution Workflow  
24. Troubleshooting Matrix  
25. Glossary  
26. License / Ownership  

---

## 1. Objectives & Non‑Goals

| Objectives | Non‑Goals |
|------------|-----------|
| Unified, low-friction instrumentation | Business analytics reporting (separate BI stack) |
| Deep correlation across signals | Replacing SIEM / full security analytics |
| Standards & automation for all services | Long-term raw log archival (future external) |
| Fast local bootstrap | Heavy multi-tenant billing partitioning (future) |
| SLO-driven alerting over primitive metrics | Manual ad-hoc observability divergence |

---

## 2. Capabilities Overview

| Capability | Tools (Initial) | Notes |
|------------|-----------------|-------|
| Metrics | Prometheus + Alertmanager | Operator or vanilla deployment |
| Dashboards | Grafana | JSON dashboards stored in repo |
| Traces | OpenTelemetry Collector + (Tempo / Jaeger) | Tempo recommended (lightweight) |
| Logs (Phase 2) | Loki stack (optional layer) | Structured JSON focus |
| Profiling (Future) | Pyroscope / Parca | On-demand toggling |
| Event Telemetry | Kafka → Derived metrics or logs | Lightweight counters / derived dashboards |
| SLO Engine | Rules + multi-window burn alerts | Based on RED/USE/Custom |
| Synthetic Probes (Future) | k6 + scheduled jobs | External user journey simulation |

---

## 3. Architectural Diagram

```
                +---------------------------+
                |        Services           |
                | (gateway, streaming, etc) |
                +------------+--------------+
                             | OTLP / HTTP / Prom scrape
                 +-----------+-----------+
                 | OpenTelemetry Collector|
                 +------+--------+-------+
                        |        |
                (metrics export) | (traces export)
                        v        v
                  +-----+--+   +------------------+
                  |Prometheus|  |   Tempo/Jaeger   |
                  +--+-------+  +---------+--------+
                     |                    |
               +-----v------+        +----v--------------+
               | Alertmanager|       |  Grafana (UI)     |
               +------+------+       +---------+---------+
                      |                          |
             (alerts via routing)         (dashboards + logs (Loki later))
                      |
               +------+------+
               | Notification|
               |  Channels   |
               +-------------+
```

(Logs Phase 2: Services → Loki distributor → Querier → Grafana)

---

## 4. Repository Structure

```
/
  k8s/
    base/
      namespaces.yaml
      serviceaccounts.yaml
      rbac.yaml
    prometheus/
      prometheus.yaml
      alertmanager.yaml
      rules/
        general-rules.yaml
        slo-burn-rules.yaml
        recording-rules.yaml
    grafana/
      deployment.yaml
      config/
        datasources/
        dashboards-provisioning/
    otel-collector/
      collector-config.yaml
    tempo/ (or jaeger/)*
      tempo-config.yaml
    loki/ (future)
      loki-config.yaml
    kustomization.yaml
  dashboards/
    access/
      gateway_overview.json
      streaming_ws.json
    ingestion/
      connectors_health.json
    data_processing/
      normalization_pipeline.json
    ml/
      model_serving_latency.json
      embedding_throughput.json
    infra/
      node_resources.json
      clickhouse_health.json
    security/
      policy_decisions.json
  alerts/
    RED/
      gateway_red.yaml
      streaming_red.yaml
    USE/
      infra_use.yaml
    SLO/
      api_latency_slo.yaml
      streaming_delivery_slo.yaml
    ml/
      inference_error_ratio.yaml
  otel/
    semantic_conventions.md
    instrumentation_guidelines.md
  logs/
    schema/
      log_event_schema.json
    processors/
      rewrite_example.yaml
  configs/
    retention-policy.yaml
    sampling-strategy.yaml
  scripts/
    deploy_all.sh
    validate_dashboards.py
    generate_rule_index.py
    test_alerts.sh
    synthetic_probe.sh
  exporters/
    README.md
  .agent/
    context.yaml
  Makefile
  README.md
  CHANGELOG.md
```

---

## 5. Core Components

| Component | Purpose | Mode |
|-----------|---------|------|
| OpenTelemetry Collector | Signal pipeline (receive, process, export) | Stateless |
| Prometheus | Metrics storage & scraping | Stateful (TSDB) |
| Alertmanager | Routing alerts (email/webhook/slack placeholder) | Stateless |
| Grafana | Visualization UI | Stateful (config persist) |
| Tempo / Jaeger | Trace storage | Object store or local volume |
| Loki (planned) | Log storage + indexing | Chunk store + index |
| Synthetic Probe CronJobs (future) | SLO external validation | Scheduled |

---

## 6. Data Flow Details

### Metrics
- Services expose `/metrics` (Prometheus format) OR push custom metrics via OTLP → Collector → Prometheus remote-write (if configured) or sidecar scrape.
- Recording rules aggregate high-cardinality series into stable KPIs.

### Traces
- Services emit spans via OTLP gRPC → Collector
- Collector processors:
  - batch
  - tail_sampling (future)
  - attributes injection (service.version, deployment.environment)
- Export to Tempo (preferred) or Jaeger.

### Logs (Phase 2)
- Structured JSON → Loki (Promtail sidecars or Collector logs pipeline)
- Correlation fields: `trace_id`, `span_id`, `service`, `tenant_id`.

### Events
- Kafka topics feed derived metrics (exporter consumer posts counters via OTLP).

---

## 7. Conventions & Naming Standards

| Signal | Convention |
|--------|------------|
| Metric prefix | `<service>_<area>_<metric>` (snake_case) |
| Histogram buckets | `_bucket` suffix; base unit ms/seconds consistent |
| Labels | `service`, `endpoint`, `tenant_id`, `status_code`, `instrument_id` (hashed optional) |
| Trace service.name | `254carbon-<service>` |
| Log fields | `ts`, `level`, `service`, `trace_id`, `request_id`, `event_type`, `message` |
| Dashboard file names | `<domain>_<focus>.json` |
| Alert names | `ALERT_<SERVICE>_<CONDITION>` |

Time Units:
- Durations → seconds (float) or ms histogram consistent (choose one per metric family; default seconds).
- Currency / numeric amplitude not used (market data handled elsewhere).

---

## 8. Metrics Taxonomy & Guidelines

Frameworks:
- RED (Rate, Errors, Duration) for request/stream flows.
- USE (Utilization, Saturation, Errors) for infrastructure.
- Domain-Specific (ticks processed, curve recomputes, inference latency).

Examples:

| Category | Metric | Type | Labels | Description |
|----------|--------|------|--------|-------------|
| Gateway RED | `gateway_http_requests_total` | Counter | method, route, status | Total requests |
| Gateway RED | `gateway_http_request_duration_seconds` | Histogram | route | Latency distribution |
| Streaming | `streaming_active_ws_connections` | Gauge | tenant_id | Open connections |
| Ingestion | `ingestion_connector_run_duration_seconds` | Histogram | connector | Connector run latency |
| Normalization | `normalization_records_processed_total` | Counter | market | Record processing count |
| Aggregation | `aggregation_bar_build_duration_seconds` | Histogram | interval | Bar build duration |
| ML Inference | `model_serving_inference_latency_seconds` | Histogram | model_version | Inference latency |
| ML Inference | `model_serving_inference_errors_total` | Counter | model_version, error_type | Error classification |
| Security | `security_auth_failures_total` | Counter | reason | Auth failures |
| Infrastructure | `clickhouse_query_duration_seconds` | Histogram | query_type | CH query latency |

Guidelines:
- Avoid high cardinality (e.g., raw instrument_id). Hash if needed.
- Use `*_total` suffix for counters.
- Use monotonic counters; derive rates in PromQL.
- Record histograms with meaningful bucket boundaries (e.g., 10,25,50,75,100,250,500 ms for gateway latency).

---

## 9. Tracing Strategy (OpenTelemetry)

Instrumentation Layers:
1. HTTP server & client requests (FastAPI, aiohttp, requests).
2. Kafka producers/consumers (propagate W3C trace context in headers).
3. DB spans (PostgreSQL, ClickHouse queries).
4. Redis operations (optional, sample only slow commands).
5. Internal tasks (aggregation windows, model loads).

Span Naming:
- `HTTP GET /api/v1/instruments`
- `kafka.consume normalized.market.ticks.v1`
- `clickhouse.query curves_computed`
- `aggregation.compute.bars interval=5m`
- `model.infer curve_forecaster`

Attributes (core):
- `deployment.environment`
- `service.version`
- `tenant.id`
- `instrument.group` (hashed optional)
- `error.type`, `error.message` (on exceptions)

Sampling:
- Head-based 100% in local/dev
- Future: Tail sampling (keep error, slow, or priority traces; sample baseline 10%)

---

## 10. Logging Strategy & Schema

Log Schema (JSON):
```
{
  "ts": "2025-10-06T12:34:56.789Z",
  "level": "INFO",
  "service": "gateway",
  "trace_id": "abcd1234...",
  "span_id": "ef01",
  "request_id": "req-xyz",
  "tenant_id": "default",
  "event_type": "http_access",
  "message": "GET /api/v1/instruments 200 42ms",
  "duration_ms": 42,
  "status_code": 200,
  "extra": { "cache_hit": true }
}
```

Logging Guidelines:
- No secrets / tokens.
- Truncate payload previews.
- Use structured logger; avoid printf logs.

---

## 11. Dashboards & Visualization

Dashboard Domains:
- Access: Gateway RED, Streaming connections & throughput
- Ingestion: Connector success rate, lag, runtime distributions
- Data Processing: Normalization latency, enrichment cache hit %, bar compute backlog
- ML: Inference latency P50/P95/P99, embedding throughput, model load failures
- Infra: Pod resource usage, DB query percentiles, disk space
- Security: Auth failures, entitlement denials, OPA policy decisions
- SLO Views: Error budget burn charts

Provisioning:
- All JSON stored under `dashboards/`
- Grafana sidecar watches config map / volume mount
- Use folder organization by domain

Validation:
```
python scripts/validate_dashboards.py dashboards/access/gateway_overview.json
```

---

## 12. Alerting & SLO Management

Alert Classes:
- Symptom (users feel pain): latency spikes, 5xx rate
- Cause (infrastructure/resource): disk full, memory saturation
- Prevention (exhaustion forecast): error budget fast burn
- Quality: data freshness behind target

Multi-Window Burn Rate Example (Gateway Latency):
- Fast window: 5m P95 > 150ms & burn rate > 4 × SLO threshold
- Slow window: 1h P95 > 150ms & burn rate > 1 × SLO threshold

SLO Definition Template (applies to SLO dashboard + rule):
```
slo:
  name: gateway-availability
  objective: 99.0
  window: 30d
  indicator: (1 - (5xx_requests / total_requests))
```

Common Alerts (examples):
| Alert | Condition | Severity |
|-------|-----------|----------|
| ALERT_GATEWAY_HIGH_ERROR_RATE | 5m error_rate > 2% & 15m > 1% | page |
| ALERT_STREAMING_CONNECTION_CHURN | connections_drop_rate > threshold | warn |
| ALERT_NORMALIZATION_LAG | latest_processed_timestamp lag > 120s | page |
| ALERT_CLICKHOUSE_DISK_HIGH | disk_usage > 80% | warn |
| ALERT_MODEL_INFERENCE_LATENCY | P95 > target for 10m | page |
| ALERT_ENTITLEMENT_CACHE_MISS_SPIKE | miss_ratio > 0.4 15m | info |

---

## 13. Sampling, Retention & Cost Control

| Signal | Retention (Dev) | Target (Future) | Notes |
|--------|-----------------|-----------------|-------|
| Metrics | 15 days | 30–90 days (tiered) | Down-sample older |
| Traces | 2 days full, 7 days sampled | 7–30 days | Tail sample errors/slow |
| Logs | 3 days (dev opt-in) | 14–30 days (prod) | Compress + filter noise |
| Profiles | On demand only | 7 day rolling | Future |

Sampling Strategy (Traces):
- Accept all error spans
- Accept slow spans (duration > 2 × P95 target)
- Baseline 10% of normal requests

---

## 14. Correlation (Trace ⇄ Log ⇄ Metric ⇄ Event)

Correlation Keys:
- `trace_id` in logs & metric exemplars (future)
- `request_id` for HTTP transaction grouping
- `tenant_id` contextual filter
- `instrument_id` hashed or grouped (avoid cardinal explosion)
- Use span events for key milestone markers (e.g., `embedding.batch.complete`)

Dashboards link to:
- Logs: `trace_id`
- Traces: from metrics exemplar or label query
- Metrics: label-based drilldowns

---

## 15. Integration Instructions (Per Domain)

| Domain | Required Actions |
|--------|------------------|
| Access (gateway/streaming) | Add OTel middleware, export RED metrics, propagate trace headers |
| Ingestion | Wrap connector executions in span; record run duration histogram |
| Data Processing | Instrument Kafka consume → process → produce span chain |
| ML | Record model load as startup span; log model version in attributes |
| Security | Emit structured auth failure events; count & histogram decision time |
| Infra | Node exporter / kube-state-metrics (if used) integrated |

---

## 16. Local Development & Quick Start

Deploy baseline (metrics + traces):
```bash
make deploy-core
```

Add optional Tempo:
```bash
make deploy-traces
```

Add (future) Loki:
```bash
make deploy-logs
```

Port forward Grafana:
```bash
kubectl port-forward -n observability svc/grafana 3000:3000
```

Validate scraping:
```bash
curl http://localhost:3000
curl http://PROMETHEUS_SVC:9090/api/v1/targets
```

---

## 17. Configuration & Environment Variables

| Variable | Component | Purpose | Example |
|----------|-----------|---------|---------|
| OTEL_EXPORTER_OTLP_ENDPOINT | Services | Where to send OTLP | http://otel-collector:4318 |
| OTEL_RESOURCE_ATTRIBUTES | Services | `service.name`, env, version | `service.name=gateway,deployment.environment=local` |
| PROM_SCRAPE_INTERVAL | Prometheus | Global scrape interval | 15s |
| PROM_RETENTION | Prometheus | TSDB retention | 15d |
| TEMPO_RETENTION | Tempo | Trace retention | 48h |
| LOKI_RETENTION | Loki | Log retention | 72h |
| ALERT_ROUTING_SLACK_WEBHOOK | Alertmanager | Notification | (placeholder) |

---

## 18. Security & Multi‑Tenancy Considerations

- Separate `observability` namespace with restricted RBAC.
- Limit write access to Prometheus remote-write endpoints.
- Optionally enable auth on Grafana (local: basic auth disabled).
- Traces/logs may contain tenant identifiers → ensure classification policy (no PII).
- Prevent secrets leakage: instrumentation wrappers must redact tokens & keys.

---

## 19. Performance & Scaling Guidance

| Component | Scaling Hint | Tuning |
|-----------|--------------|--------|
| Collector | Add replicas for high throughput (batch size, memory) | Set memory_limiter processor |
| Prometheus | Federation or remote-write later if metrics volume grows | Shard by functional domain (future) |
| Tempo | Scale ingesters & compactors | Enable block storage once capacity high |
| Loki | Deploy only when log volume justifies | Use retention cuts & label cardinality filters |
| Grafana | Single instance fine early | Use persistent volume for dashboards |

---

## 20. Runbooks & Incident Practices

Runbooks (to add under `docs/runbooks/`):
- High API Error Rate
- Streaming Latency Spike
- Kafka Consumer Lag Growth
- ClickHouse Query Saturation
- Model Inference Latency Regression
- Enrichment Cache Miss Spike

General Pattern:
1. Identify alert & correlate with related metrics.
2. Check recent deploy events (meta repo).
3. Examine tracing outliers.
4. Validate data store health (ClickHouse / Postgres).
5. Rollback or activate feature flag mitigations.

---

## 21. Testing Observability (Instrumentation QA)

Checklist (per service PR):
- Exposes /metrics with no scrape errors.
- Key RED metrics present (requests_total, duration histogram).
- Trace created for inbound major requests.
- Log lines include `trace_id`.
- Error path test: intentional failing endpoint emits error span.
- Run `scripts/test_alerts.sh` to simulate threshold crossing.

---

## 22. Roadmap & Future Enhancements

| Milestone | Description | Status |
|-----------|-------------|--------|
| M1 | Core metrics + tracing + dashboards | In progress |
| M2 | Alert rule refinement + SLO burn multi-window | Planned |
| M3 | Loki logging integration | Planned |
| M4 | Trace tail-sampling (error + slow retention) | Planned |
| M5 | Profiling (continuous & ad-hoc) | Future |
| M6 | Synthetic user probe (Web/API) | Future |
| M7 | Anomaly detection on ingestion lag | Future |
| M8 | Automatic instrument cardinality guardrails | Future |
| M9 | Cross-domain dependency graph visualization | Future |

---

## 23. Contribution Workflow

1. Create feature branch.
2. Add / modify:
   - Dashboard JSON (validate with `validate_dashboards.py`)
   - Alert rules (annotate description + runbook link)
   - Collector pipeline config (ensure pipeline names unique)
3. Run:
   ```bash
   make validate
   ```
4. Commit with conventional prefix:
   - `feat(dashboard): add streaming latency panel`
   - `feat(alert): gateway high 5xx burn`
   - `chore(tracing): adjust batch processor size`
5. Open PR → CI:
   - YAML/JSON lint
   - Dashboard schema check
   - Alert rule syntax validation
   - (Future) test ephemeral stack & synthetic query
6. After merge: redeploy via infra or GitOps.

---

## 24. Troubleshooting Matrix

| Symptom | Possible Cause | Diagnostic Command | Resolution |
|---------|----------------|--------------------|------------|
| Prometheus target down | Service not annotated or endpoint mismatch | `kubectl get endpoints -A` | Fix ServiceMonitor / annotations |
| Missing traces | OTLP endpoint misconfigured | Check collector logs | Correct `OTEL_EXPORTER_OTLP_ENDPOINT` |
| High cardinality explosion | Unbounded label value (`instrument_id`) | `topk by(__name__) (count by (label) ...)` | Hash or drop label |
| Alert flood | Too-sensitive thresholds | Alertmanager logs / rule eval | Adjust thresholds / add for clause |
| Grafana dashboards missing | Provisioning failure | Grafana pod logs | Validate provisioning paths |
| Tempo ingestion errors | Storage misconfig | Tempo logs | Fix backend (PVC / object store path) |
| Logs not correlated | Missing trace_id injection | Inspect log format | Update logging middleware |

---

## 25. Glossary

| Term | Definition |
|------|------------|
| RED | Rate, Errors, Duration – KPI model for services |
| USE | Utilization, Saturation, Errors – infra measurement |
| SLO | Service Level Objective (target performance/availability) |
| Error Budget | Allowable failure threshold for SLO window |
| Exemplar | Link in metric pointing to a trace for that sample |
| Tail Sampling | Decide trace retention after inspection |
| Span | A timed unit of work in tracing graph |
| Head Sampling | Decision at trace start (probabilistic) |

---

## 26. License / Ownership

Internal repository; will remain internal until external ecosystem requires a published dashboard/alert library.  
Ownership: Platform Observability (single developer + AI automation).  
License: To be defined (likely Apache 2.0 for generic contrib assets later).

---

## Quick Start Commands

```bash
# Deploy core stack (Prometheus + Grafana + OTel Collector)
make deploy-core

# Deploy tracing backend (Tempo or Jaeger)
make deploy-traces

# Validate dashboards & rules
make validate

# Port-forward Grafana
kubectl port-forward -n observability svc/grafana 3000:3000

# Test an alert rule firing (synthetic load or script)
scripts/test_alerts.sh gateway_high_error
```

---

## Make Targets (Suggested)

| Target | Description |
|--------|-------------|
| make deploy-core | Apply Prometheus + Grafana + Collector |
| make deploy-traces | Deploy Tempo/Jaeger |
| make deploy-logs | Deploy Loki stack (phase 2) |
| make validate | Lint dashboards + rule syntax |
| make reload-dashboards | Force refresh provisioning |
| make test-alerts | Run synthetic rule tests |
| make clean | Remove generated artifacts |

---

> “Observability isn’t a bolt‑on. It’s the feedback nervous system that keeps the platform adaptive, reliable, and confidently evolvable.”

---
````
