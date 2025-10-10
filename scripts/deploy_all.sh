#!/bin/bash
# Complete deployment script for 254Carbon Observability Platform
#
# What this does
# - Applies Kubernetes manifests for Prometheus, Alertmanager, OTel Collector,
#   Grafana, Tempo (tracing), and Loki (logging), in a safe order.
# - Waits for Deployments to become available and prints concise status.
# - Provides simple cleanup when something fails mid-flight.
#
# Why the order matters
# - Base RBAC/ServiceAccounts/Namespaces first so subsequent resources bind.
# - Prometheus before rules and alerts so they load successfully.
# - Collector before instrumented apps to accept OTLP data.
# - Grafana after datasources/dashboards ConfigMaps are created.

# Exit on first error; cleanup trap below removes partial state.
set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
# Namespace and environment can be overridden from the shell, e.g.:
#   ENVIRONMENT=staging ./deploy_all.sh
NAMESPACE="observability"
ENVIRONMENT="${ENVIRONMENT:-local}"
TIMEOUT=300
CHECK_INTERVAL=10

# Function to print colored output
print_status() {
    local color=$1
    local message=$2
    echo -e "${color}${message}${NC}"
}

# Function to check if kubectl is available
check_prerequisites() {
    # Fail early if `kubectl` is missing or the current context is invalid
    print_status $BLUE "🔍 Checking prerequisites..."
    
    if ! command -v kubectl &> /dev/null; then
        print_status $RED "❌ kubectl is not installed or not in PATH"
        exit 1
    fi
    
    if ! kubectl cluster-info &> /dev/null; then
        print_status $RED "❌ Cannot connect to Kubernetes cluster"
        exit 1
    fi
    
    print_status $GREEN "✅ Prerequisites check passed"
}

# Function to wait for deployment to be ready
wait_for_deployment() {
    # Blocks until Deployment reports Available condition, or times out
    local deployment_name=$1
    local namespace=$2
    local timeout=${3:-$TIMEOUT}
    
    print_status $YELLOW "⏳ Waiting for deployment $deployment_name to be ready..."
    
    if kubectl wait --for=condition=available --timeout=${timeout}s deployment/$deployment_name -n $namespace; then
        print_status $GREEN "✅ Deployment $deployment_name is ready"
        return 0
    else
        print_status $RED "❌ Deployment $deployment_name failed to become ready within ${timeout}s"
        return 1
    fi
}

# Function to check if namespace exists
check_namespace() {
    # Creates namespace only if it does not already exist
    local namespace=$1
    
    if kubectl get namespace $namespace &> /dev/null; then
        print_status $YELLOW "⚠️  Namespace $namespace already exists"
        return 0
    else
        print_status $BLUE "📦 Creating namespace $namespace..."
        kubectl create namespace $namespace
        print_status $GREEN "✅ Namespace $namespace created"
        return 1
    fi
}

# Function to deploy base resources
deploy_base() {
    # Namespaces, ServiceAccounts, and RBAC
    print_status $BLUE "🏗️  Deploying base Kubernetes resources..."
    
    kubectl apply -k k8s/base/
    
    print_status $GREEN "✅ Base resources deployed"
}

# Function to deploy Prometheus
deploy_prometheus() {
    # Prometheus core + rule ConfigMaps
    print_status $BLUE "📊 Deploying Prometheus..."
    
    kubectl apply -f k8s/prometheus/prometheus.yaml
    kubectl apply -f k8s/prometheus/prometheus-deployment.yaml
    kubectl apply -f k8s/prometheus/rules/general-rules.yaml
    kubectl apply -f k8s/prometheus/rules/slo-burn-rules.yaml
    
    wait_for_deployment "prometheus" $NAMESPACE
    
    print_status $GREEN "✅ Prometheus deployed"
}

# Function to deploy Alertmanager
deploy_alertmanager() {
    # Alertmanager config + deployment (routes critical alerts separately)
    print_status $BLUE "🚨 Deploying Alertmanager..."
    
    kubectl apply -f k8s/prometheus/alertmanager.yaml
    kubectl apply -f k8s/prometheus/alertmanager-deployment.yaml
    
    wait_for_deployment "alertmanager" $NAMESPACE
    
    print_status $GREEN "✅ Alertmanager deployed"
}

# Function to deploy OpenTelemetry Collector
deploy_otel_collector() {
    # OTLP receiver and Prometheus exporter exposed via Service
    print_status $BLUE "🔗 Deploying OpenTelemetry Collector..."
    
    kubectl apply -f k8s/otel-collector/collector-config.yaml
    kubectl apply -f k8s/otel-collector/collector-deployment.yaml
    
    wait_for_deployment "otel-collector" $NAMESPACE
    
    print_status $GREEN "✅ OpenTelemetry Collector deployed"
}

# Function to deploy Grafana
deploy_grafana() {
    # Grafana with datasources and dashboards provisioned via ConfigMaps
    print_status $BLUE "📈 Deploying Grafana..."
    
    kubectl apply -f k8s/grafana/grafana-config.yaml
    kubectl apply -f k8s/grafana/grafana-deployment.yaml
    kubectl apply -f k8s/grafana/datasources/prometheus.yaml
    kubectl apply -f k8s/grafana/dashboards-provisioning/dashboards.yaml
    
    wait_for_deployment "grafana" $NAMESPACE
    
    print_status $GREEN "✅ Grafana deployed"
}

# Function to deploy Tempo
deploy_tempo() {
    # Tempo with MinIO for object storage in local dev
    print_status $BLUE "🔍 Deploying Tempo (tracing backend)..."
    
    kubectl apply -f k8s/tempo/tempo-config.yaml
    kubectl apply -f k8s/tempo/tempo-deployment.yaml
    
    wait_for_deployment "tempo" $NAMESPACE
    wait_for_deployment "minio" $NAMESPACE
    
    print_status $GREEN "✅ Tempo deployed"
}

# Function to deploy Loki
deploy_loki() {
    # Loki + Promtail DaemonSet for node log collection
    print_status $BLUE "📝 Deploying Loki (logging backend)..."
    
    kubectl apply -f k8s/loki/loki-config.yaml
    kubectl apply -f k8s/loki/loki-deployment.yaml
    
    wait_for_deployment "loki" $NAMESPACE
    wait_for_deployment "promtail" $NAMESPACE
    
    print_status $GREEN "✅ Loki deployed"
}

# Function to deploy alert rules
deploy_alerts() {
    # Installs service-specific alerting rules packaged as ConfigMaps
    print_status $BLUE "⚠️  Deploying alert rules..."
    
    kubectl apply -f alerts/RED/gateway_red.yaml
    kubectl apply -f alerts/RED/streaming_red.yaml
    kubectl apply -f alerts/SLO/api_latency_slo.yaml
    
    print_status $GREEN "✅ Alert rules deployed"
}

# Function to deploy dashboards
deploy_dashboards() {
    # ConfigMap wrapper around dashboard JSON to mount into Grafana
    print_status $BLUE "📊 Deploying dashboards..."
    
    # Create ConfigMap for dashboards
    kubectl create configmap grafana-dashboards \
        --from-file=dashboards/access/gateway_overview.json \
        --from-file=dashboards/access/gateway_served_cache.json \
        --dry-run=client -o yaml | kubectl apply -f -
    
    print_status $GREEN "✅ Dashboards deployed"
}

# Function to validate deployment
validate_deployment() {
    # Sanity check Deployments/Services; not a full health audit
    print_status $BLUE "🔍 Validating deployment..."
    
    local failed=0
    
    # Check if all deployments are ready
    local deployments=("prometheus" "grafana" "otel-collector" "tempo" "loki" "alertmanager")
    
    for deployment in "${deployments[@]}"; do
        if kubectl get deployment $deployment -n $NAMESPACE &> /dev/null; then
            local ready=$(kubectl get deployment $deployment -n $NAMESPACE -o jsonpath='{.status.readyReplicas}')
            local desired=$(kubectl get deployment $deployment -n $NAMESPACE -o jsonpath='{.spec.replicas}')
            
            if [ "$ready" = "$desired" ]; then
                print_status $GREEN "✅ $deployment is ready ($ready/$desired)"
            else
                print_status $RED "❌ $deployment is not ready ($ready/$desired)"
                failed=1
            fi
        else
            print_status $RED "❌ $deployment deployment not found"
            failed=1
        fi
    done
    
    # Check if all services are running
    local services=("prometheus" "grafana" "otel-collector" "tempo" "loki" "alertmanager")
    
    for service in "${services[@]}"; do
        if kubectl get service $service -n $NAMESPACE &> /dev/null; then
            print_status $GREEN "✅ $service service is running"
        else
            print_status $RED "❌ $service service not found"
            failed=1
        fi
    done
    
    if [ $failed -eq 0 ]; then
        print_status $GREEN "🎉 All validations passed!"
        return 0
    else
        print_status $RED "❌ Some validations failed"
        return 1
    fi
}

# Function to show access information
show_access_info() {
    # Friendly summary with local URLs; use `make port-forward` if needed
    print_status $BLUE "🌐 Access Information:"
    echo ""
    print_status $YELLOW "Grafana:"
    echo "  URL: http://localhost:3000"
    echo "  Username: admin"
    echo "  Password: admin"
    echo ""
    print_status $YELLOW "Prometheus:"
    echo "  URL: http://localhost:9090"
    echo ""
    print_status $YELLOW "Alertmanager:"
    echo "  URL: http://localhost:9093"
    echo ""
    print_status $YELLOW "Tempo:"
    echo "  URL: http://localhost:3200"
    echo ""
    print_status $YELLOW "Loki:"
    echo "  URL: http://localhost:3100"
    echo ""
    print_status $YELLOW "OpenTelemetry Collector:"
    echo "  Metrics: http://localhost:8888"
    echo "  Traces: http://localhost:4317"
    echo ""
    print_status $BLUE "To port-forward services, run:"
    echo "  make port-forward"
    echo ""
    print_status $BLUE "To validate dashboards, run:"
    echo "  make validate"
    echo ""
    print_status $BLUE "To test alerts, run:"
    echo "  make test-alerts"
}

# Function to cleanup on failure
cleanup_on_failure() {
    # Best-effort teardown in reverse order to free cluster quickly
    print_status $RED "❌ Deployment failed. Cleaning up..."
    
    # Delete resources in reverse order
    kubectl delete -f alerts/RED/gateway_red.yaml --ignore-not-found=true
    kubectl delete -f alerts/RED/streaming_red.yaml --ignore-not-found=true
    kubectl delete -f alerts/SLO/api_latency_slo.yaml --ignore-not-found=true
    
    kubectl delete -f k8s/loki/loki-deployment.yaml --ignore-not-found=true
    kubectl delete -f k8s/tempo/tempo-deployment.yaml --ignore-not-found=true
    kubectl delete -f k8s/grafana/grafana-deployment.yaml --ignore-not-found=true
    kubectl delete -f k8s/otel-collector/collector-deployment.yaml --ignore-not-found=true
    kubectl delete -f k8s/prometheus/alertmanager-deployment.yaml --ignore-not-found=true
    kubectl delete -f k8s/prometheus/prometheus-deployment.yaml --ignore-not-found=true
    
    kubectl delete -k k8s/base/ --ignore-not-found=true
    
    print_status $YELLOW "🧹 Cleanup completed"
}

# Main deployment function
main() {
    local phase=${1:-"all"}
    
    print_status $GREEN "🚀 Starting 254Carbon Observability Platform deployment"
    print_status $BLUE "Environment: $ENVIRONMENT"
    print_status $BLUE "Namespace: $NAMESPACE"
    print_status $BLUE "Phase: $phase"
    echo ""
    
    # Set up error handling
    trap cleanup_on_failure ERR
    
    # Check prerequisites
    check_prerequisites
    
    # Create namespace if it doesn't exist
    check_namespace $NAMESPACE
    
    # Deploy base resources
    deploy_base
    
    # Deploy core components
    deploy_prometheus
    deploy_alertmanager
    deploy_otel_collector
    deploy_grafana
    
    # Deploy tracing (Phase 2)
    if [ "$phase" = "all" ] || [ "$phase" = "traces" ]; then
        deploy_tempo
    fi
    
    # Deploy logging (Phase 3)
    if [ "$phase" = "all" ] || [ "$phase" = "logs" ]; then
        deploy_loki
    fi
    
    # Deploy alert rules
    deploy_alerts
    
    # Deploy dashboards
    deploy_dashboards
    
    # Validate deployment
    if validate_deployment; then
        print_status $GREEN "🎉 Deployment completed successfully!"
        echo ""
        show_access_info
    else
        print_status $RED "❌ Deployment validation failed"
        exit 1
    fi
}

# Parse command line arguments
case "${1:-}" in
    "base")
        main "base"
        ;;
    "traces")
        main "traces"
        ;;
    "logs")
        main "logs"
        ;;
    "all"|"")
        main "all"
        ;;
    "help"|"-h"|"--help")
        echo "Usage: $0 [phase]"
        echo ""
        echo "Phases:"
        echo "  base    - Deploy base components only"
        echo "  traces  - Deploy base + tracing"
        echo "  logs    - Deploy base + logging"
        echo "  all     - Deploy everything (default)"
        echo "  help    - Show this help message"
        echo ""
        echo "Environment variables:"
        echo "  ENVIRONMENT - Set environment (local, staging, production)"
        echo ""
        exit 0
        ;;
    *)
        print_status $RED "❌ Unknown phase: $1"
        echo "Use '$0 help' for usage information"
        exit 1
        ;;
esac
