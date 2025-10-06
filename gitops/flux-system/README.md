# GitOps Configuration for 254Carbon Observability Platform

This directory contains GitOps configuration files for deploying the observability platform using Flux CD.

## Overview

The GitOps setup uses Flux CD to automatically deploy and manage the observability platform components from a Git repository. This ensures:

- **Automated deployments** from Git commits
- **Configuration drift prevention** through continuous reconciliation
- **Rollback capabilities** through Git history
- **Audit trail** of all changes
- **Multi-environment support** through Git branches

## Components

### Core Flux Components

- **GitRepository**: Defines the source Git repository for configuration
- **Kustomization**: Manages Kubernetes resource deployment using Kustomize
- **HelmRepository**: Defines Helm chart repositories
- **HelmRelease**: Manages Helm chart deployments
- **AlertProvider**: Configures notification channels
- **Alert**: Defines alerting rules for GitOps events

### Observability Stack

- **Prometheus Stack**: Metrics collection, storage, and alerting
- **Grafana**: Visualization and dashboards
- **OpenTelemetry Collector**: Metrics, traces, and logs collection
- **Tempo**: Distributed tracing backend
- **Loki**: Log aggregation and storage
- **Alertmanager**: Alert routing and notification

## Setup Instructions

### 1. Prerequisites

- Kubernetes cluster (1.20+)
- kubectl configured to access the cluster
- Git repository with observability configurations
- Flux CLI installed

### 2. Install Flux CD

```bash
# Install Flux CLI
curl -s https://fluxcd.io/install.sh | sudo bash

# Install Flux components
flux install --components=source-controller,kustomize-controller,helm-controller,notification-controller

# Verify installation
kubectl get pods -n flux-system
```

### 3. Configure Git Repository

```bash
# Create Git repository (if not exists)
git init observability-config
cd observability-config

# Add observability configurations
cp -r ../k8s .
cp -r ../dashboards .
cp -r ../alerts .

# Commit and push
git add .
git commit -m "Initial observability configuration"
git remote add origin https://github.com/254carbon/observability-config.git
git push -u origin main
```

### 4. Deploy GitOps Configuration

```bash
# Apply GitOps configuration
kubectl apply -f gitops/flux-system/

# Verify GitOps setup
flux get sources git
flux get kustomizations
flux get helmreleases
```

### 5. Monitor Deployment

```bash
# Check deployment status
flux get kustomizations observability-platform

# View logs
flux logs --kind=Kustomization --name=observability-platform

# Check health status
kubectl get pods -n observability
```

## Configuration Files

### `gotk-sync.yaml`
Installs and configures Flux CD components with proper RBAC permissions.

### `git-repository.yaml`
Defines the source Git repository and authentication credentials.

### `kustomization.yaml`
Manages the deployment of Kubernetes resources using Kustomize with health checks.

### `alert-provider.yaml`
Configures Slack notifications for GitOps events and deployment status.

### `helm-repository.yaml`
Defines Helm chart repositories for Prometheus, Grafana, and OpenTelemetry.

### `helm-releases.yaml`
Manages Helm chart deployments with custom values and configurations.

## Environment Management

### Local Development
```bash
# Deploy to local environment
kubectl apply -f gitops/flux-system/local/
```

### Staging Environment
```bash
# Deploy to staging environment
kubectl apply -f gitops/flux-system/staging/
```

### Production Environment
```bash
# Deploy to production environment
kubectl apply -f gitops/flux-system/production/
```

## Git Workflow

### 1. Make Changes
```bash
# Edit configuration files
vim k8s/prometheus/prometheus.yaml

# Test locally
make validate
```

### 2. Commit and Push
```bash
# Commit changes
git add .
git commit -m "feat: update Prometheus configuration"

# Push to repository
git push origin main
```

### 3. Monitor Deployment
```bash
# Watch deployment progress
flux get kustomizations observability-platform --watch

# Check for errors
flux logs --kind=Kustomization --name=observability-platform --follow
```

## Troubleshooting

### Common Issues

1. **Git Repository Access**
   ```bash
   # Check Git credentials
   kubectl get secret git-credentials -n flux-system -o yaml
   
   # Update credentials if needed
   kubectl create secret generic git-credentials \
     --from-literal=username=<username> \
     --from-literal=password=<password> \
     --namespace=flux-system \
     --dry-run=client -o yaml | kubectl apply -f -
   ```

2. **Helm Chart Issues**
   ```bash
   # Check Helm repository status
   flux get sources helm
   
   # Suspend and resume Helm release
   flux suspend helmrelease prometheus-stack
   flux resume helmrelease prometheus-stack
   ```

3. **Kustomization Failures**
   ```bash
   # Check Kustomization status
   flux get kustomizations observability-platform
   
   # View detailed logs
   flux logs --kind=Kustomization --name=observability-platform --follow
   
   # Force reconciliation
   flux reconcile kustomization observability-platform
   ```

### Health Checks

```bash
# Check all components
kubectl get pods -n observability
kubectl get services -n observability
kubectl get configmaps -n observability

# Check Flux system
kubectl get pods -n flux-system
flux get all
```

## Security Considerations

### RBAC Permissions
- Flux components have minimal required permissions
- Service accounts are scoped to specific namespaces
- Cluster-wide permissions are limited to necessary resources

### Secret Management
- Git credentials are stored as Kubernetes secrets
- Slack webhook URLs are encrypted
- Consider using external secret management (e.g., Sealed Secrets, External Secrets Operator)

### Network Policies
- Implement network policies to restrict traffic between components
- Use TLS for all inter-component communication
- Enable mTLS for service-to-service communication

## Monitoring and Alerting

### GitOps Events
- Deployment success/failure notifications
- Configuration drift alerts
- Health check failures

### Observability Stack
- Prometheus metrics for all components
- Grafana dashboards for visualization
- Alertmanager for critical alerts

## Best Practices

### Git Repository Structure
```
observability-config/
├── k8s/
│   ├── base/
│   ├── prometheus/
│   ├── grafana/
│   ├── otel-collector/
│   ├── tempo/
│   └── loki/
├── dashboards/
├── alerts/
├── gitops/
│   └── flux-system/
└── README.md
```

### Commit Messages
Use conventional commit format:
- `feat: add new dashboard`
- `fix: update Prometheus configuration`
- `chore: update dependencies`
- `docs: update README`

### Branch Strategy
- `main`: Production environment
- `staging`: Staging environment
- `development`: Development environment
- Feature branches for testing

### Validation
- Use `make validate` before committing
- Run tests in staging environment
- Monitor deployment health after changes

## Support

For issues and questions:
- Check Flux documentation: https://fluxcd.io/docs/
- Review Kubernetes logs: `kubectl logs -n flux-system`
- Use Flux CLI: `flux --help`
- Check observability platform logs: `kubectl logs -n observability`
