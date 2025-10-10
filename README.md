# Observability Stack (`254carbon-observability`)

> Platform-wide metrics, traces, logs (in-progress), dashboards, alerts, and runbooks for 254Carbon.

Reference: [Platform Overview](../PLATFORM_OVERVIEW.md)

---

## Scope
- Operate Prometheus, Alertmanager, Grafana, Tempo/Jaeger, OpenTelemetry Collector, Pyroscope, and synthetic probes.
- Provide dashboards and alert rules consumed by all services (access, ingestion, data-processing, ML, infra, security).
- Maintain semantic conventions, sampling strategies, retention policies, and integration guides.

Out of scope: application-specific instrumentation logic (maintained in each service repo) and security incident response (see `../security`).

---

## Components
- `k8s/` – Kubernetes manifests (base, Prometheus, Grafana, OTel collector, Tempo, Pyroscope, Loki (future), synthetic tooling).
- `dashboards/` – Provisioned Grafana JSON for each domain plus synthetic/capacity/profiling views (e.g., `access/gateway_overview.json`, `access/gateway_served_cache.json`).
- `alerts/` – Prometheus rule groups (RED, SLO, ML, infra).
- `scripts/` – Validation (`validate_dashboards.py`), alert smoke tests (`test_alerts.sh`), synthetic probes.
- `Makefile` – Deploy, validate, and status targets.
- `otel/` – Semantic conventions and instrumentation guidelines.

---

## Environments

| Environment | Bootstrap | Entry Points | Notes |
|-------------|-----------|--------------|-------|
| `local` | `make quick-start` (deploy-core + tempo + port-forward) | Grafana `http://localhost:3000` (admin/admin), Prometheus `http://localhost:9090` | Uses single replicas, ephemeral PVCs. |
| `dev` | `make deploy-core deploy-traces` on cluster created via `../infra` | Access via `kubectl port-forward -n observability` | Shared integration; retention trimmed to 7 days. |
| `staging` | GitOps-managed overlay | Ingress `https://grafana.stg.254carbon.local` | Mirror of production retention/alert routing. |
| `production` | GitOps plus sealed secrets for credentials | Ingress `https://grafana.254carbon.com`, Alertmanager integrated with on-call channels | High availability (replicas ≥2) and longer retention (30 days metrics, 90 days traces in object storage). |

Overrides for each environment live under `k8s/overlays/<env>/`.

---

## Runbook

### Daily Checks
- `make status` (or `kubectl get pods -n observability`) – ensure Prometheus, Grafana, Alertmanager, otel-collector, tempo, pyroscope pods Ready.
- Grafana dashboard `dashboards/rca/observability_health.json` (future) or Prometheus `up{job=~"prometheus|grafana|otel-collector"}` – confirm scrape success.
- Verify alert pipeline: `curl http://localhost:9090/api/v1/alerts` (via port-forward) should list firing/resolved alerts.
- Check storage headroom: `kubectl exec prometheus-0 -n observability -- df -h /prometheus`.

### Deploy / Upgrade
1. Modify configuration (dashboards, rules, collector pipelines).
2. `make validate` – lint dashboards and rule syntax via `promtool`.
3. `make deploy-core` (or targeted `deploy-prometheus`, `deploy-grafana`, etc.).
4. `make reload-dashboards` after provisioning updates or `kubectl rollout restart deployment/grafana -n observability`.
5. Run `make test-alerts` to simulate high-error / latency alerts and verify routing.

### Incident Response
- **Prometheus Down / Scrapes Failing**
  - `kubectl logs deployment/prometheus -n observability`.
  - Ensure PVC mounted; if corrupted, `kubectl delete pod` to recreate (data loss limited by retention). Consider snapshot restore from backup.
  - Check `prometheus.yml` config map for syntax errors; rerun `make validate`.
- **Grafana Unreachable**
  - `kubectl describe deployment/grafana -n observability` for image pull or PVC issues.
  - Reset admin password: patch `GF_SECURITY_ADMIN_PASSWORD` secret and restart deployment.
- **Collector Backpressure**
  - Inspect otel collector metrics: `kubectl port-forward svc/otel-collector -n observability 8888:8888` then query `/metrics` for `otelcol_receiver_accepted_spans`.
  - Scale collector: `kubectl scale deployment/otel-collector --replicas=3 -n observability` and adjust exporters batching.
- **Tempo Storage Saturation**
  - Review object storage metrics (S3/MinIO). Trigger retention job via `tempo-compactor` configuration or adjust `retention_period`.
  - Temporarily reduce sampling from services using `OTEL_TRACES_SAMPLER_ARG`.
- **Alert Flood / Noise**
  - Pause noisy rule: `kubectl patch prometheusrule <rule> -n observability --type merge -p '{"spec":{"groups":[...`}}` or disable via GitOps patch.
  - Update SLO burn rate thresholds; rerun `make validate`.

### Backup & Restore
- Prometheus snapshots: `kubectl exec -n observability deploy/prometheus -- promtool snapshot create /prometheus/snapshots`.
- Grafana dashboards stored in Git; datasource credentials stored in secrets (export via `kubectl get secret grafana-admin -n observability -o yaml`).
- Tempo/Alertmanager rely on object store / ConfigMaps; ensure bucket versioning enabled in production.

---

## Configuration

| Config | Description | Location / Default |
|--------|-------------|--------------------|
| `--storage.tsdb.retention.time` | Prometheus retention window | `15d` (`k8s/prometheus/prometheus-deployment.yaml`); override per env. |
| `GF_SECURITY_ADMIN_PASSWORD` | Grafana admin password | Default `admin`; set via secret in overlays. |
| `ALERTMANAGER_CONFIG` | Alert routing destinations | `k8s/prometheus/alertmanager.yaml`; configure Slack/webhook/email. |
| `OTEL_EXPORTER_OTLP_ENDPOINT` | Collector export target (Tempo/Prometheus remote write) | Set in `k8s/otel-collector/collector-config.yaml` pipelines. |
| `TEMPO_STORAGE_BACKEND` | Trace storage backend (`local`, `s3`, etc.) | `tempo/tempo-config.yaml`; prod uses `s3`. |
| `LOKI_BOLTD_PATH` (future) | Loki local index path | `k8s/loki/loki-config.yaml`. |
| `PYROSCOPE_STORAGE_PATH` | Profiling storage | `k8s/pyroscope/pyroscope-config.yaml`. |

Sensitive credentials (Grafana admin, Alertmanager receivers, S3 keys) live in sealed secrets within `k8s/overlays/<env>/secrets/`.

---

## Observability of Observability
- Health KPIs tracked in Grafana dashboards under `dashboards/observability/` (additions pending).
- Prometheus self-scrape metrics (job `prometheus`) highlight scrape duration, rule evaluation, and target counts.
- Alertmanager route status accessible via `/api/v2/status`.
- Synthetic probes (`k6`) validate gateway endpoints and record results in Prometheus under `k6_*` metrics.
- Profiling data accessible via Pyroscope UI (`http://localhost:4040` when port-forwarded).

---

## Troubleshooting

### `make validate` Fails
- Dashboard lint: open referenced JSON and fix schema errors (use `scripts/validate_dashboards.py` locally).
- Prometheus rule errors: run `promtool check rules <file>`; ensure expressions use existing metrics.

### No Metrics from a Service
- Confirm service exposes `/metrics`; check `ServiceMonitor`/scrape config in `k8s/prometheus/prometheus.yaml`.
- Ensure OTEL collector pipelines include correct receiver/exporter; update `k8s/otel-collector/collector-config.yaml`.
- Verify service label `service.name` matches Grafana dashboard queries.

### Alerts Not Leaving Cluster
- Check Alertmanager logs: `kubectl logs statefulset/alertmanager -n observability`.
- Inspect `/api/v2/status` for receiver errors (TLS/SMS).
- Validate networking/egress rules allow outbound communication.

### Tempo Query Empty
- Confirm ingestion: `kubectl logs deployment/tempo -n observability`.
- Ensure exporters send to `tempo:4317`.
- Run sample query via Tempo API: `curl http://localhost:3200/api/traces/<trace-id>` after port-forward.

---

## Reference
- `Makefile` – `make help` for deployment and validation targets.
- `dashboards/` – Grafana JSON definitions (auto-provisioned).
- `alerts/` – Prometheus/Alertmanager rules; align with Platform SLOs.
- `k8s/` – Base manifests and overlays; integrate with `../infra` GitOps flow.
- `scripts/test_alerts.sh` – Smoke-test alert rules before promotion.
- `otel/semantic_conventions.md` – Required labels/attributes for service instrumentation.

For platform-wide context and cross-service relationships, consult the [Platform Overview](../PLATFORM_OVERVIEW.md).
