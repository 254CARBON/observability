# 254Carbon Observability Platform Makefile
#
# Purpose
# - Developer-friendly entry points to deploy, validate, and test alerts.
# - Thin wrapper around `kubectl` and helper scripts; no hidden behavior.

.PHONY: help deploy-core deploy-traces deploy-logs validate clean reload-dashboards test-alerts

WORK_DIR := $(shell pwd)
PROMTOOL_IMAGE ?= prom/prometheus:v2.45.0
PROMTOOL_DOCKER := docker run --rm -v $(WORK_DIR):/work --entrypoint /bin/promtool $(PROMTOOL_IMAGE)
PROMTOOL_BIN := $(shell command -v promtool 2>/dev/null)
ifeq ($(PROMTOOL_BIN),)
PROMTOOL_CMD := $(PROMTOOL_DOCKER)
else
PROMTOOL_CMD := promtool
endif

PROM_RULE_GENERAL_FILE := k8s/prometheus/rules/general-rules.yaml
PROM_RULE_GENERAL_KEY := general-rules.yaml
PROM_RULE_SLO_FILE := k8s/prometheus/rules/slo-burn-rules.yaml
PROM_RULE_SLO_KEY := slo-burn-rules.yaml
ALERTMANAGER_FILE := k8s/prometheus/alertmanager.yaml
ALERTMANAGER_KEY := alertmanager.yml

# Default target
help:
	@echo "254Carbon Observability Platform"
	@echo "Available targets:"
	@echo "  deploy-core     - Deploy Prometheus + Grafana + OTel Collector"
	@echo "  deploy-traces   - Deploy Tempo/Jaeger"
	@echo "  deploy-logs     - Deploy Loki stack (phase 2)"
	@echo "  validate        - Lint dashboards + rule syntax"
	@echo "  reload-dashboards - Force refresh provisioning"
	@echo "  test-alerts     - Run synthetic rule tests"
	@echo "  clean           - Remove generated artifacts"

# Deploy core observability stack
deploy-core: deploy-base deploy-prometheus deploy-grafana deploy-otel-collector deploy-pyroscope deploy-k6 deploy-dependency-graph deploy-capacity-planning
	@echo "Core observability stack deployed successfully"

# Deploy base Kubernetes resources
deploy-base:
	@echo "Deploying base Kubernetes resources..."
	kubectl apply -k k8s/base/

# Deploy Prometheus
deploy-prometheus:
	@echo "Deploying Prometheus..."
	kubectl apply -f k8s/prometheus/prometheus.yaml
	kubectl apply -f k8s/prometheus/prometheus-deployment.yaml
	kubectl apply -f k8s/prometheus/rules/general-rules.yaml
	kubectl apply -f k8s/prometheus/rules/slo-burn-rules.yaml
	kubectl apply -f k8s/prometheus/alertmanager.yaml
	kubectl apply -f k8s/prometheus/alertmanager-deployment.yaml
	kubectl apply -f alerts/RED/gateway_red.yaml
	kubectl apply -f alerts/RED/streaming_red.yaml
	kubectl apply -f alerts/SLO/api_latency_slo.yaml
	# Optional: anomaly detection rules may live outside this repo in some setups
	kubectl apply -f k8s/prometheus/rules/anomaly-detection.yaml

# Deploy Grafana
deploy-grafana:
	@echo "Deploying Grafana..."
	kubectl apply -f k8s/grafana/grafana-config.yaml
	kubectl apply -f k8s/grafana/grafana-deployment.yaml
	kubectl apply -f k8s/grafana/datasources/prometheus.yaml
	kubectl apply -f k8s/grafana/dashboards-provisioning/dashboards.yaml
	@echo "Waiting for Grafana to be ready..."
	kubectl wait --for=condition=available --timeout=300s deployment/grafana -n observability

# Deploy Pyroscope
deploy-pyroscope:
	@echo "Deploying Pyroscope..."
	kubectl apply -f k8s/pyroscope/pyroscope-config.yaml
	kubectl apply -f k8s/pyroscope/pyroscope-deployment.yaml
	kubectl apply -f k8s/pyroscope/pyroscope-agent.yaml
	kubectl apply -f k8s/pyroscope/pyroscope-rules.yaml
	kubectl apply -f k8s/pyroscope/pyroscope-alerts.yaml
	kubectl apply -f k8s/pyroscope/pyroscope-datasource.yaml
	kubectl apply -f k8s/pyroscope/pyroscope-integration.yaml
	kubectl apply -f dashboards/profiling/pyroscope_overview.json
	kubectl apply -f dashboards/profiling/flame_graph_analysis.json
	@echo "Waiting for Pyroscope to be ready..."
	kubectl wait --for=condition=available --timeout=300s deployment/pyroscope -n observability

# Deploy k6 synthetic monitoring
deploy-k6:
	@echo "Deploying k6 synthetic monitoring..."
	kubectl apply -f k8s/k6/k6-crd.yaml
	kubectl apply -f k8s/k6/k6-operator.yaml
	kubectl apply -f k8s/k6/gateway-synthetic-tests.yaml
	kubectl apply -f k8s/k6/k6-test-runs.yaml
	kubectl apply -f k8s/k6/k6-scheduler.yaml
	kubectl apply -f k8s/k6/k6-alerts.yaml
	kubectl apply -f dashboards/synthetic/k6_synthetic_monitoring.json
	@echo "Waiting for k6 operator to be ready..."
	kubectl wait --for=condition=available --timeout=300s deployment/k6-operator -n k6-system

# Deploy dependency graph service
deploy-dependency-graph:
	@echo "Deploying dependency graph service..."
	kubectl apply -f k8s/dependency-graph/dependency-graph-config.yaml
	kubectl apply -f k8s/dependency-graph/dependency-graph-deployment.yaml
	kubectl apply -f k8s/dependency-graph/dependency-graph-rules.yaml
	kubectl apply -f k8s/dependency-graph/dependency-graph-alerts.yaml
	kubectl apply -f dashboards/dependency/service_dependency_graph.json
	@echo "Waiting for dependency graph service to be ready..."
	kubectl wait --for=condition=available --timeout=300s deployment/dependency-graph -n observability

# Deploy capacity planning service
deploy-capacity-planning:
	@echo "Deploying capacity planning service..."
	kubectl apply -f k8s/capacity-planning/capacity-planning-config.yaml
	kubectl apply -f k8s/capacity-planning/capacity-planning-deployment.yaml
	kubectl apply -f k8s/capacity-planning/capacity-planning-rules.yaml
	kubectl apply -f k8s/capacity-planning/capacity-planning-alerts.yaml
	kubectl apply -f dashboards/capacity/capacity_planning_overview.json
	@echo "Waiting for capacity planning service to be ready..."
	kubectl wait --for=condition=available --timeout=300s deployment/capacity-planning -n observability

# Deploy OpenTelemetry Collector
deploy-otel-collector:
	@echo "Deploying OpenTelemetry Collector..."
	kubectl apply -f k8s/otel-collector/collector-config.yaml
	kubectl apply -f k8s/otel-collector/collector-deployment.yaml
	@echo "Waiting for OTel Collector to be ready..."
	kubectl wait --for=condition=available --timeout=300s deployment/otel-collector -n observability

# Deploy tracing backend (Tempo)
deploy-traces: deploy-tempo
	@echo "Tracing backend deployed successfully"

deploy-tempo:
	@echo "Deploying Tempo..."
	kubectl apply -f k8s/tempo/tempo-config.yaml
	kubectl apply -f k8s/tempo/tempo-deployment.yaml
	@echo "Waiting for Tempo to be ready..."
	kubectl wait --for=condition=available --timeout=300s deployment/tempo -n observability

# Deploy logging stack (Loki) - Phase 2
deploy-logs:
	@echo "Deploying Loki stack..."
	kubectl apply -f k8s/loki/loki-config.yaml
	kubectl apply -f k8s/loki/loki-deployment.yaml
	@echo "Loki stack deployed successfully"

# Validate configurations
validate:
	@echo "Validating dashboard JSON files..."
	python3 scripts/validate_dashboards.py dashboards/access/gateway_overview.json dashboards/access/gateway_served_cache.json
	@echo "Validating Prometheus rule files..."
	@echo "Using $(PROMTOOL_CMD) for Prometheus rule validation."
	@tmp=$$(mktemp); \
	python3 scripts/extract_configmap_data.py $(PROM_RULE_GENERAL_FILE) $(PROM_RULE_GENERAL_KEY) > $$tmp; \
	$(PROMTOOL_CMD) check rules $$tmp; \
	rm -f $$tmp
	@tmp=$$(mktemp); \
	python3 scripts/extract_configmap_data.py $(PROM_RULE_SLO_FILE) $(PROM_RULE_SLO_KEY) > $$tmp; \
	$(PROMTOOL_CMD) check rules $$tmp; \
	rm -f $$tmp
	@echo "Validating Alertmanager configuration..."
	@tmp=$$(mktemp); \
	python3 scripts/extract_configmap_data.py $(ALERTMANAGER_FILE) $(ALERTMANAGER_KEY) > $$tmp; \
	$(PROMTOOL_CMD) check config $$tmp; \
	rm -f $$tmp
	@echo "All validations passed"

# Reload dashboards
reload-dashboards:
	@echo "Reloading Grafana dashboards..."
	kubectl rollout restart deployment/grafana -n observability
	@echo "Dashboards reloaded"

# Test alerts
test-alerts:
	@echo "Running synthetic alert tests..."
	./scripts/test_alerts.sh gateway_high_error
	./scripts/test_alerts.sh gateway_latency_degradation
	@echo "Alert tests completed"

# Port forward services for local access
port-forward:
	@echo "Port forwarding services..."
	# Exposes cluster services on localhost for quick exploration. Stop with
	# `make stop-port-forward` or by killing the background kubectl processes.
	@echo "Grafana: http://localhost:3000 (admin/admin)"
	@echo "Prometheus: http://localhost:9090"
	@echo "Alertmanager: http://localhost:9093"
	@echo "Tempo: http://localhost:3200"
	@echo "OTel Collector: http://localhost:8888"
	kubectl port-forward -n observability svc/grafana 3000:3000 &
	kubectl port-forward -n observability svc/prometheus 9090:9090 &
	kubectl port-forward -n observability svc/alertmanager 9093:9093 &
	kubectl port-forward -n observability svc/tempo 3200:3200 &
	kubectl port-forward -n observability svc/otel-collector 8888:8888 &
	@echo "Port forwarding started. Use 'make stop-port-forward' to stop."

# Stop port forwarding
stop-port-forward:
	@echo "Stopping port forwarding..."
	pkill -f "kubectl port-forward" || true
	@echo "Port forwarding stopped"

# Clean up resources
clean:
	@echo "Cleaning up observability resources..."
	kubectl delete -f k8s/capacity-planning/capacity-planning-alerts.yaml --ignore-not-found=true
	kubectl delete -f k8s/capacity-planning/capacity-planning-rules.yaml --ignore-not-found=true
	kubectl delete -f k8s/capacity-planning/capacity-planning-deployment.yaml --ignore-not-found=true
	kubectl delete -f k8s/capacity-planning/capacity-planning-config.yaml --ignore-not-found=true
	kubectl delete -f dashboards/capacity/capacity_planning_overview.json --ignore-not-found=true
	kubectl delete -f k8s/dependency-graph/dependency-graph-alerts.yaml --ignore-not-found=true
	kubectl delete -f k8s/dependency-graph/dependency-graph-rules.yaml --ignore-not-found=true
	kubectl delete -f k8s/dependency-graph/dependency-graph-deployment.yaml --ignore-not-found=true
	kubectl delete -f k8s/dependency-graph/dependency-graph-config.yaml --ignore-not-found=true
	kubectl delete -f dashboards/dependency/service_dependency_graph.json --ignore-not-found=true
	kubectl delete -f k8s/k6/k6-alerts.yaml --ignore-not-found=true
	kubectl delete -f k8s/k6/k6-scheduler.yaml --ignore-not-found=true
	kubectl delete -f k8s/k6/k6-test-runs.yaml --ignore-not-found=true
	kubectl delete -f k8s/k6/gateway-synthetic-tests.yaml --ignore-not-found=true
	kubectl delete -f k8s/k6/k6-operator.yaml --ignore-not-found=true
	kubectl delete -f k8s/k6/k6-crd.yaml --ignore-not-found=true
	kubectl delete -f dashboards/synthetic/k6_synthetic_monitoring.json --ignore-not-found=true
	kubectl delete namespace k6-system --ignore-not-found=true
	kubectl delete -f k8s/pyroscope/pyroscope-alerts.yaml --ignore-not-found=true
	kubectl delete -f k8s/pyroscope/pyroscope-rules.yaml --ignore-not-found=true
	kubectl delete -f k8s/pyroscope/pyroscope-agent.yaml --ignore-not-found=true
	kubectl delete -f k8s/pyroscope/pyroscope-deployment.yaml --ignore-not-found=true
	kubectl delete -f k8s/pyroscope/pyroscope-config.yaml --ignore-not-found=true
	kubectl delete -f k8s/pyroscope/pyroscope-datasource.yaml --ignore-not-found=true
	kubectl delete -f k8s/pyroscope/pyroscope-integration.yaml --ignore-not-found=true
	kubectl delete -f dashboards/profiling/pyroscope_overview.json --ignore-not-found=true
	kubectl delete -f dashboards/profiling/flame_graph_analysis.json --ignore-not-found=true
	kubectl delete namespace observability --ignore-not-found=true
	kubectl delete namespace monitoring --ignore-not-found=true
	@echo "Cleanup completed"

# Check service status
status:
	@echo "Checking service status..."
	kubectl get pods -n observability
	kubectl get services -n observability
	kubectl get configmaps -n observability

# Install dependencies (for validation scripts)
install-deps:
	@echo "Installing validation dependencies..."
	pip3 install --user grafana-dashboard-validator prometheus-client
	@echo "Dependencies installed"

# Quick start - deploy everything
quick-start: deploy-core deploy-traces port-forward
	@echo "Quick start completed!"
	@echo "Access Grafana at: http://localhost:3000 (admin/admin)"
	@echo "Access Prometheus at: http://localhost:9090"
