#!/usr/bin/env python3
"""
Tenant Management Script for 254Carbon Observability

This script demonstrates basic tenant management operations including:
- Tenant configuration validation
- Quota enforcement simulation
- Access control verification
- Storage isolation checks
"""

import json
import yaml
import requests
import time
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta

@dataclass
class TenantQuota:
    """Represents tenant quota limits"""
    tenant_id: str
    metrics_ingestion_rate: int
    logs_ingestion_rate: int
    traces_ingestion_rate: int
    storage_limit_gb: int
    retention_days: int

@dataclass
class TenantUsage:
    """Represents current tenant usage"""
    tenant_id: str
    metrics_rate: float
    logs_rate: float
    traces_rate: float
    storage_bytes: int
    timestamp: datetime

class TenantManager:
    """Manages tenant operations and quota enforcement"""
    
    def __init__(self, config_path: str = "k8s/multi-tenancy/tenant-config.yaml"):
        self.config_path = config_path
        self.tenants: Dict[str, TenantQuota] = {}
        self.usage_history: List[TenantUsage] = []
        self.load_config()
    
    def load_config(self) -> None:
        """Load tenant configuration from YAML file"""
        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            for tenant_config in config['data']['config.yaml']['tenants']:
                tenant_id = tenant_config['id']
                quota_limits = tenant_config['quota_limits']
                
                self.tenants[tenant_id] = TenantQuota(
                    tenant_id=tenant_id,
                    metrics_ingestion_rate=int(quota_limits['metrics_ingestion_rate']),
                    logs_ingestion_rate=int(quota_limits['logs_ingestion_rate']),
                    traces_ingestion_rate=int(quota_limits['traces_ingestion_rate']),
                    storage_limit_gb=int(quota_limits['storage_limit_gb']),
                    retention_days=tenant_config['data_retention_days']
                )
            
            print(f"Loaded configuration for {len(self.tenants)} tenants")
            
        except FileNotFoundError:
            print(f"Configuration file not found: {self.config_path}")
            self._create_sample_config()
        except Exception as e:
            print(f"Error loading configuration: {e}")
    
    def _create_sample_config(self) -> None:
        """Create sample tenant configuration"""
        sample_tenants = {
            "tenant-a": TenantQuota(
                tenant_id="tenant-a",
                metrics_ingestion_rate=10000,
                logs_ingestion_rate=1000,
                traces_ingestion_rate=1000,
                storage_limit_gb=100,
                retention_days=30
            ),
            "tenant-b": TenantQuota(
                tenant_id="tenant-b",
                metrics_ingestion_rate=1000,
                logs_ingestion_rate=100,
                traces_ingestion_rate=100,
                storage_limit_gb=10,
                retention_days=7
            )
        }
        self.tenants = sample_tenants
        print("Created sample tenant configuration")
    
    def simulate_usage(self, tenant_id: str, duration_minutes: int = 60) -> List[TenantUsage]:
        """Simulate tenant usage over time"""
        if tenant_id not in self.tenants:
            raise ValueError(f"Tenant {tenant_id} not found")
        
        quota = self.tenants[tenant_id]
        usage_data = []
        
        # Simulate varying usage patterns
        base_time = datetime.now()
        for minute in range(duration_minutes):
            # Simulate realistic usage patterns with some randomness
            import random
            
            # Metrics rate with some variation
            metrics_rate = quota.metrics_ingestion_rate * (0.7 + 0.6 * random.random())
            
            # Logs rate with some variation
            logs_rate = quota.logs_ingestion_rate * (0.5 + 0.8 * random.random())
            
            # Traces rate with some variation
            traces_rate = quota.traces_ingestion_rate * (0.6 + 0.7 * random.random())
            
            # Storage usage (cumulative)
            storage_bytes = int(quota.storage_limit_gb * 0.8 * random.random() * 1024**3)
            
            usage = TenantUsage(
                tenant_id=tenant_id,
                metrics_rate=metrics_rate,
                logs_rate=logs_rate,
                traces_rate=traces_rate,
                storage_bytes=storage_bytes,
                timestamp=base_time + timedelta(minutes=minute)
            )
            usage_data.append(usage)
        
        self.usage_history.extend(usage_data)
        return usage_data
    
    def check_quota_violations(self, tenant_id: str) -> Dict[str, Any]:
        """Check for quota violations for a specific tenant"""
        if tenant_id not in self.tenants:
            raise ValueError(f"Tenant {tenant_id} not found")
        
        quota = self.tenants[tenant_id]
        violations = {
            "tenant_id": tenant_id,
            "violations": [],
            "utilization": {}
        }
        
        # Get latest usage data
        latest_usage = None
        for usage in reversed(self.usage_history):
            if usage.tenant_id == tenant_id:
                latest_usage = usage
                break
        
        if not latest_usage:
            return violations
        
        # Check metrics quota
        metrics_utilization = (latest_usage.metrics_rate / quota.metrics_ingestion_rate) * 100
        violations["utilization"]["metrics"] = metrics_utilization
        
        if metrics_utilization > 100:
            violations["violations"].append({
                "type": "metrics_quota_exceeded",
                "current": latest_usage.metrics_rate,
                "limit": quota.metrics_ingestion_rate,
                "utilization": metrics_utilization
            })
        
        # Check logs quota
        logs_utilization = (latest_usage.logs_rate / quota.logs_ingestion_rate) * 100
        violations["utilization"]["logs"] = logs_utilization
        
        if logs_utilization > 100:
            violations["violations"].append({
                "type": "logs_quota_exceeded",
                "current": latest_usage.logs_rate,
                "limit": quota.logs_ingestion_rate,
                "utilization": logs_utilization
            })
        
        # Check traces quota
        traces_utilization = (latest_usage.traces_rate / quota.traces_ingestion_rate) * 100
        violations["utilization"]["traces"] = traces_utilization
        
        if traces_utilization > 100:
            violations["violations"].append({
                "type": "traces_quota_exceeded",
                "current": latest_usage.traces_rate,
                "limit": quota.traces_ingestion_rate,
                "utilization": traces_utilization
            })
        
        # Check storage quota
        storage_limit_bytes = quota.storage_limit_gb * 1024**3
        storage_utilization = (latest_usage.storage_bytes / storage_limit_bytes) * 100
        violations["utilization"]["storage"] = storage_utilization
        
        if storage_utilization > 100:
            violations["violations"].append({
                "type": "storage_quota_exceeded",
                "current": latest_usage.storage_bytes,
                "limit": storage_limit_bytes,
                "utilization": storage_utilization
            })
        
        return violations
    
    def generate_quota_report(self) -> Dict[str, Any]:
        """Generate a comprehensive quota report for all tenants"""
        report = {
            "timestamp": datetime.now().isoformat(),
            "tenants": {}
        }
        
        for tenant_id in self.tenants:
            violations = self.check_quota_violations(tenant_id)
            report["tenants"][tenant_id] = violations
        
        return report
    
    def validate_access_control(self, tenant_id: str, resource: str, action: str) -> bool:
        """Validate if a tenant has access to a specific resource and action"""
        if tenant_id not in self.tenants:
            return False
        
        # Simple access control validation
        # In a real implementation, this would check JWT tokens, RBAC policies, etc.
        
        allowed_resources = ["metrics", "logs", "traces"]
        allowed_actions = ["read", "write"]
        
        if resource not in allowed_resources or action not in allowed_actions:
            return False
        
        # Check if tenant has access to the resource
        # This is a simplified check - in reality, you'd check against actual policies
        return True
    
    def cleanup_old_data(self, tenant_id: str) -> Dict[str, Any]:
        """Simulate cleanup of old data based on retention policy"""
        if tenant_id not in self.tenants:
            raise ValueError(f"Tenant {tenant_id} not found")
        
        quota = self.tenants[tenant_id]
        cutoff_date = datetime.now() - timedelta(days=quota.retention_days)
        
        # Simulate data cleanup
        cleaned_count = 0
        for usage in self.usage_history[:]:
            if usage.tenant_id == tenant_id and usage.timestamp < cutoff_date:
                self.usage_history.remove(usage)
                cleaned_count += 1
        
        return {
            "tenant_id": tenant_id,
            "retention_days": quota.retention_days,
            "cutoff_date": cutoff_date.isoformat(),
            "cleaned_records": cleaned_count
        }

def main():
    """Main function to demonstrate tenant management operations"""
    print("254Carbon Observability - Tenant Management Demo")
    print("=" * 50)
    
    # Initialize tenant manager
    manager = TenantManager()
    
    # Simulate usage for both tenants
    print("\n1. Simulating tenant usage...")
    manager.simulate_usage("tenant-a", duration_minutes=30)
    manager.simulate_usage("tenant-b", duration_minutes=30)
    
    # Generate quota report
    print("\n2. Generating quota report...")
    report = manager.generate_quota_report()
    print(json.dumps(report, indent=2))
    
    # Check access control
    print("\n3. Validating access control...")
    for tenant_id in ["tenant-a", "tenant-b"]:
        for resource in ["metrics", "logs", "traces"]:
            for action in ["read", "write"]:
                has_access = manager.validate_access_control(tenant_id, resource, action)
                print(f"Tenant {tenant_id} -> {resource}:{action} = {'✓' if has_access else '✗'}")
    
    # Simulate data cleanup
    print("\n4. Simulating data cleanup...")
    for tenant_id in ["tenant-a", "tenant-b"]:
        cleanup_result = manager.cleanup_old_data(tenant_id)
        print(f"Tenant {tenant_id}: Cleaned {cleanup_result['cleaned_records']} records")
    
    print("\nTenant management demo completed!")

if __name__ == "__main__":
    main()
