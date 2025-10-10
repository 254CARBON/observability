#!/bin/bash
# Alert testing script for 254Carbon Observability Platform
#
# Purpose
# - Generate synthetic load/conditions that should trigger specific alerts.
# - Verify alert states via Prometheus and Alertmanager HTTP APIs.
#
# Notes
# - Requires `curl` and `jq` to parse API responses.
# - Assumes local port-forwarding or in-cluster addresses for the endpoints
#   below; override URLs if needed via environment variables.

set -e  # exit on first failure to avoid false positives later in the run

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
PROMETHEUS_URL="http://localhost:9090"
ALERTMANAGER_URL="http://localhost:9093"
NAMESPACE="observability"

# Function to check if service is running
check_service() {
    # Simple reachability check; does not validate auth or version
    local service=$1
    local url=$2
    
    if curl -s "$url" > /dev/null 2>&1; then
        echo -e "${GREEN}✅ $service is running${NC}"
        return 0
    else
        echo -e "${RED}❌ $service is not accessible at $url${NC}"
        return 1
    fi
}

# Function to generate test metrics
generate_test_metrics() {
    # Emit traffic patterns that should satisfy alert rules’ expressions
    local alert_type=$1
    
    echo -e "${YELLOW}🧪 Generating test metrics for: $alert_type${NC}"
    
    case $alert_type in
        "gateway_high_error")
            # Simulate high error rate by sending requests to a non-existent endpoint
            for i in {1..100}; do
                curl -s "http://localhost:8080/nonexistent" > /dev/null 2>&1 || true
            done
            echo "Generated 100 requests to non-existent endpoint"
            ;;
        "gateway_latency_degradation")
            # Simulate slow requests hitting a known slow path in the gateway
            for i in {1..50}; do
                curl -s "http://localhost:8080/api/v1/slow" > /dev/null 2>&1 || true
                sleep 0.1
            done
            echo "Generated 50 slow requests"
            ;;
        "gateway_served_cache_warm")
            echo "Triggering cache warm to populate metrics (errors require projection outage simulation)."
            curl -s -X POST "http://localhost:8080/api/v1/cache/warm" \
                -H "Authorization: Bearer dev-key" > /dev/null 2>&1 || true
            ;;
        "streaming_connection_churn")
            # This would require actual WebSocket connections
            echo "Streaming connection churn test requires WebSocket client"
            ;;
        *)
            echo -e "${RED}❌ Unknown alert type: $alert_type${NC}"
            echo "Available types: gateway_high_error, gateway_latency_degradation, gateway_served_cache_warm, streaming_connection_churn"
            return 1
            ;;
    esac
}

# Function to check alert status
check_alert_status() {
    # Poll Prometheus `/api/v1/alerts` for a single alert’s state
    local alert_name=$1
    local max_wait=60
    local wait_time=0
    
    echo -e "${YELLOW}🔍 Checking alert status: $alert_name${NC}"
    
    while [ $wait_time -lt $max_wait ]; do
        # Check Prometheus alerts
        local alerts=$(curl -s "$PROMETHEUS_URL/api/v1/alerts" | jq -r '.data.alerts[] | select(.labels.alertname == "'$alert_name'") | .state' 2>/dev/null || echo "")
        
        if [ "$alerts" = "firing" ]; then
            echo -e "${GREEN}✅ Alert $alert_name is FIRING${NC}"
            return 0
        elif [ "$alerts" = "pending" ]; then
            echo -e "${YELLOW}⏳ Alert $alert_name is PENDING${NC}"
        fi
        
        sleep 5
        wait_time=$((wait_time + 5))
    done
    
    echo -e "${RED}❌ Alert $alert_name did not fire within $max_wait seconds${NC}"
    return 1
}

# Function to check Alertmanager
check_alertmanager() {
    # Validate the alert reached Alertmanager (routing/notification layer)
    local alert_name=$1
    
    echo -e "${YELLOW}🔍 Checking Alertmanager for: $alert_name${NC}"
    
    # Check Alertmanager alerts
    local alerts=$(curl -s "$ALERTMANAGER_URL/api/v1/alerts" | jq -r '.data[] | select(.labels.alertname == "'$alert_name'") | .status.state' 2>/dev/null || echo "")
    
    if [ "$alerts" = "active" ]; then
        echo -e "${GREEN}✅ Alert $alert_name is active in Alertmanager${NC}"
        return 0
    else
        echo -e "${YELLOW}⏳ Alert $alert_name not yet active in Alertmanager${NC}"
        return 1
    fi
}

# Function to cleanup test data
cleanup_test_data() {
    echo -e "${YELLOW}🧹 Cleaning up test data...${NC}"
    # This would depend on the specific test data generated
    echo "Test data cleanup completed"
}

# Main function
main() {
    local alert_type=$1
    
    if [ -z "$alert_type" ]; then
        echo "Usage: $0 <alert_type>"
        echo "Available alert types:"
        echo "  - gateway_high_error"
        echo "  - gateway_latency_degradation"
        echo "  - gateway_served_cache_warm"
        echo "  - streaming_connection_churn"
        exit 1
    fi
    
    echo -e "${GREEN}🚀 Starting alert test for: $alert_type${NC}"
    
    # Check prerequisites
    echo -e "${YELLOW}📋 Checking prerequisites...${NC}"
    
    if ! check_service "Prometheus" "$PROMETHEUS_URL"; then
        echo "Please ensure Prometheus is running and accessible"
        exit 1
    fi
    
    if ! check_service "Alertmanager" "$ALERTMANAGER_URL"; then
        echo "Please ensure Alertmanager is running and accessible"
        exit 1
    fi
    
    # Generate test metrics
    generate_test_metrics "$alert_type"
    
    # Wait a moment for metrics to be scraped
    echo "Waiting for metrics to be scraped..."
    sleep 30
    
    # Check alert status
    case $alert_type in
        "gateway_high_error")
            check_alert_status "ALERT_GATEWAY_HIGH_ERROR_RATE"
            check_alertmanager "ALERT_GATEWAY_HIGH_ERROR_RATE"
            ;;
        "gateway_latency_degradation")
            check_alert_status "ALERT_GATEWAY_LATENCY_DEGRADATION"
            check_alertmanager "ALERT_GATEWAY_LATENCY_DEGRADATION"
            ;;
        "gateway_served_cache_warm")
            check_alert_status "ALERT_GATEWAY_SERVED_CACHE_WARM_ERRORS"
            check_alertmanager "ALERT_GATEWAY_SERVED_CACHE_WARM_ERRORS"
            check_alert_status "ALERT_GATEWAY_SERVED_PROJECTION_LAG"
            check_alertmanager "ALERT_GATEWAY_SERVED_PROJECTION_LAG"
            ;;
        "streaming_connection_churn")
            check_alert_status "ALERT_STREAMING_CONNECTION_CHURN"
            check_alertmanager "ALERT_STREAMING_CONNECTION_CHURN"
            ;;
    esac
    
    # Cleanup
    cleanup_test_data
    
    echo -e "${GREEN}🎉 Alert test completed for: $alert_type${NC}"
}

# Run main function with all arguments
main "$@"
