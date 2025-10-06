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

