# Commentary for JSON Assets (non-commentable)

Some assets in this repository use JSON, which does not support inline comments. To keep runtime behavior correct and imports reliable (Grafana/validators), these files are left unmodified. This document provides the detailed commentary you’d normally see inline.

## dashboards/access/gateway_overview.json
- Purpose: Gateway service overview dashboard focusing on RED metrics and request patterns.
- Key Panels:
  - Request rate (5m and 15m) with anomaly comparison.
  - Error rate and error budget overlays.
  - Latency percentiles (P95/P99) with SLO threshold annotations.
  - Top routes by traffic and errors (label-hashed in rules to control cardinality).
- Usage Notes:
  - Default time range should be broad enough for trend analysis (≥ 1h suggested).
  - Refresh intervals ≥ 10s are generally sufficient; avoid 1s in shared environments.

## dashboards/access/task_manager_overview.json
- Purpose: Monitor Task Manager HTTP health and workflow anomalies from the access layer.
- Key Panels:
  - Request throughput (5m) split by success vs failure.
  - Error ratio trend line with guardrails at 2% (warning) and 5% (critical).
  - P95 latency with thresholds at 1.0s (warning) / 2.0s (critical).
  - Live stat tiles for DLQ rate and total throughput to highlight sudden shifts.
- Usage Notes:
  - Dashboard relies on Prometheus recording rules (`task_manager:*`) introduced in `general-rules.yaml`.
  - Investigate unhandled exception spikes alongside DLQ rate to correlate workflow issues.

## dashboards/data_processing/normalization_pipeline.json
- Purpose: Provide a focused view of normalization throughput, latency, and DLQ pressure.
- Key Panels:
  - Messages processed (5m) split by status for quick success vs failure ratios.
  - Error ratio and DLQ rate tiles aligned with new RED alerts.
  - Processing latency (P95) sourced from recording rule to control query cost.
- Usage Notes:
  - `normalization:*` recording rules aggregate by status and message type; add label filters if a single topic misbehaves.
  - DLQ panel treats both `failed` and `dlq` status labels as failures.

## dashboards/ingestion/connectors_health.json
- Purpose: Monitor ingestion connectors throughput, latency, and failure modes.
- Key Panels:
  - Connector run counts (rate over 5m).
  - Run duration histograms (P50/P95) to spot regressions.
  - Error counters by connector/topic.
- Usage Notes:
  - Break down views by `connector` label; avoid per-record IDs.
  - Consider adding burn-rate style panels for freshness SLOs.

## logs/schema/log_event_schema.json
- Purpose: JSON Schema describing the canonical 254Carbon log event shape.
- Correlation:
  - `trace_id` (32 hex) and `span_id` (16 hex) match OpenTelemetry IDs and enable logs ↔ traces pivots.
- Core Fields:
  - `ts` (ISO 8601 UTC), `level` (DEBUG/INFO/WARN/ERROR/FATAL), `service` (254carbon-<name>), `message` (≤ 1000 chars).
  - Optional `error` object: `type`, `message`, `stack_trace`, `code`.
  - Optional `business` object: domain fields like `instrument_id`, `market`.
- Guidelines:
  - Keep high-cardinality details in `extra` rather than metric labels.
  - Do not include secrets/PII; use the Collector’s filter processors where needed.

---

If you want these comments colocated in your editor for quick reference, consider adding editor-specific JSON “annotations” via separate sidecar files rather than modifying the JSON itself.
