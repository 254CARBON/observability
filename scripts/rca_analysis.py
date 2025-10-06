#!/usr/bin/env python3
"""
Root Cause Analysis Script for 254Carbon Observability

This script demonstrates automated root cause analysis operations including:
- Statistical anomaly detection
- Dependency analysis
- Temporal correlation
- Log pattern analysis
- Trace analysis
- RCA result generation
"""

import json
import yaml
import requests
import time
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import re
from collections import defaultdict
from scipy import stats
from sklearn.ensemble import IsolationForest
from sklearn.cluster import DBSCAN

@dataclass
class Incident:
    """Represents an incident for RCA analysis"""
    id: str
    timestamp: datetime
    severity: str
    description: str
    affected_services: List[str]
    alerts: List[str]
    metrics: Dict[str, float]
    logs: List[str]
    traces: List[str]

@dataclass
class RootCause:
    """Represents a root cause analysis result"""
    incident_id: str
    root_cause: str
    confidence: float
    evidence: List[str]
    affected_services: List[str]
    remediation: str
    algorithm: str
    timestamp: datetime

@dataclass
class ServiceDependency:
    """Represents a service dependency"""
    source: str
    target: str
    weight: float
    latency: float
    error_rate: float

class RootCauseAnalyzer:
    """Performs automated root cause analysis"""
    
    def __init__(self, config_path: str = "k8s/rca/rca-config.yaml"):
        self.config_path = config_path
        self.incidents: List[Incident] = []
        self.root_causes: List[RootCause] = []
        self.service_dependencies: List[ServiceDependency] = []
        self.load_config()
    
    def load_config(self) -> None:
        """Load RCA configuration"""
        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            self.algorithms = config['data']['config.yaml']['algorithms']
            self.rules = config['data']['config.yaml']['rules']
            self.thresholds = config['data']['config.yaml']['performance']
            
            print(f"Loaded RCA configuration with {len(self.algorithms)} algorithms")
            
        except FileNotFoundError:
            print(f"Configuration file not found: {self.config_path}")
            self._create_sample_config()
        except Exception as e:
            print(f"Error loading configuration: {e}")
    
    def _create_sample_config(self) -> None:
        """Create sample RCA configuration"""
        self.algorithms = {
            "statistical_analysis": {"enabled": True, "threshold": 0.8},
            "dependency_analysis": {"enabled": True, "max_depth": 5},
            "temporal_analysis": {"enabled": True, "correlation_window": "15m"},
            "log_analysis": {"enabled": True, "log_levels": ["ERROR", "FATAL"]},
            "trace_analysis": {"enabled": True, "latency_threshold": 1000}
        }
        self.rules = {
            "service_failure": [
                {"name": "database_connection_failure", "pattern": "database.*connection.*failed"},
                {"name": "memory_exhaustion", "pattern": "out of memory|OOM"},
                {"name": "cpu_saturation", "pattern": "cpu.*saturation|high.*cpu"}
            ],
            "network_failure": [
                {"name": "timeout_errors", "pattern": "timeout|connection.*timeout"},
                {"name": "dns_resolution_failure", "pattern": "dns.*resolution.*failed"}
            ]
        }
        self.thresholds = {
            "max_concurrent_analyses": 10,
            "analysis_timeout": 300,
            "confidence_threshold": 0.8
        }
        print("Created sample RCA configuration")
    
    def generate_sample_incidents(self, count: int = 5) -> List[Incident]:
        """Generate sample incidents for analysis"""
        incident_types = [
            {
                "description": "Database connection failures affecting multiple services",
                "severity": "critical",
                "services": ["user-service", "order-service", "payment-service"],
                "alerts": ["DatabaseDown", "HighErrorRate", "ServiceUnavailable"],
                "metrics": {"error_rate": 0.8, "latency": 5000, "cpu_usage": 90},
                "logs": ["database connection failed", "connection timeout", "pool exhausted"],
                "traces": ["db_query_timeout", "connection_pool_error"]
            },
            {
                "description": "Memory exhaustion in analytics service",
                "severity": "critical",
                "services": ["analytics-service"],
                "alerts": ["HighMemoryUsage", "OOMKilled"],
                "metrics": {"memory_usage": 95, "error_rate": 0.6, "latency": 2000},
                "logs": ["out of memory", "OOM killed", "memory allocation failed"],
                "traces": ["memory_allocation_error", "gc_failure"]
            },
            {
                "description": "Network timeout issues between services",
                "severity": "warning",
                "services": ["gateway", "auth-service", "user-service"],
                "alerts": ["HighLatency", "TimeoutErrors"],
                "metrics": {"latency": 3000, "timeout_rate": 0.3, "error_rate": 0.2},
                "logs": ["connection timeout", "request timeout", "network unreachable"],
                "traces": ["network_timeout", "connection_refused"]
            },
            {
                "description": "CPU saturation in payment processing",
                "severity": "warning",
                "services": ["payment-service"],
                "alerts": ["HighCPUUsage", "SlowResponse"],
                "metrics": {"cpu_usage": 85, "latency": 1500, "throughput": 0.5},
                "logs": ["cpu saturation", "slow processing", "high load"],
                "traces": ["cpu_bottleneck", "processing_delay"]
            },
            {
                "description": "Configuration error in notification service",
                "severity": "warning",
                "services": ["notification-service"],
                "alerts": ["ConfigurationError", "ServiceStartupFailed"],
                "metrics": {"error_rate": 0.4, "startup_time": 60, "availability": 0.7},
                "logs": ["configuration error", "invalid config", "startup failed"],
                "traces": ["config_validation_error", "service_init_failure"]
            }
        ]
        
        incidents = []
        base_time = datetime.now() - timedelta(hours=24)
        
        for i in range(min(count, len(incident_types))):
            incident_type = incident_types[i]
            incident = Incident(
                id=f"incident-{i+1}",
                timestamp=base_time + timedelta(hours=i*6),
                severity=incident_type["severity"],
                description=incident_type["description"],
                affected_services=incident_type["services"],
                alerts=incident_type["alerts"],
                metrics=incident_type["metrics"],
                logs=incident_type["logs"],
                traces=incident_type["traces"]
            )
            incidents.append(incident)
        
        self.incidents = incidents
        return incidents
    
    def generate_sample_dependencies(self) -> List[ServiceDependency]:
        """Generate sample service dependencies"""
        dependencies = [
            ("gateway", "auth-service", 0.8, 50, 0.01),
            ("gateway", "user-service", 0.6, 100, 0.02),
            ("gateway", "order-service", 0.4, 150, 0.03),
            ("order-service", "payment-service", 0.7, 200, 0.02),
            ("order-service", "inventory-service", 0.9, 300, 0.05),
            ("user-service", "notification-service", 0.3, 80, 0.01),
            ("gateway", "analytics-service", 0.2, 120, 0.01),
            ("auth-service", "user-service", 0.5, 60, 0.01)
        ]
        
        service_deps = []
        for source, target, weight, latency, error_rate in dependencies:
            dep = ServiceDependency(
                source=source,
                target=target,
                weight=weight,
                latency=latency,
                error_rate=error_rate
            )
            service_deps.append(dep)
        
        self.service_dependencies = service_deps
        return service_deps
    
    def statistical_analysis(self, incident: Incident) -> Optional[RootCause]:
        """Perform statistical analysis for anomaly detection"""
        if not self.algorithms["statistical_analysis"]["enabled"]:
            return None
        
        # Simulate statistical analysis
        metrics = incident.metrics
        anomalies = []
        
        # Check for metric anomalies
        if metrics.get("error_rate", 0) > 0.5:
            anomalies.append("High error rate anomaly detected")
        if metrics.get("latency", 0) > 2000:
            anomalies.append("High latency anomaly detected")
        if metrics.get("cpu_usage", 0) > 80:
            anomalies.append("High CPU usage anomaly detected")
        if metrics.get("memory_usage", 0) > 90:
            anomalies.append("High memory usage anomaly detected")
        
        if anomalies:
            # Calculate confidence based on number of anomalies
            confidence = min(0.9, len(anomalies) * 0.3)
            
            return RootCause(
                incident_id=incident.id,
                root_cause="Statistical anomaly detected in system metrics",
                confidence=confidence,
                evidence=anomalies,
                affected_services=incident.affected_services,
                remediation="Review system metrics and resource utilization",
                algorithm="statistical_analysis",
                timestamp=datetime.now()
            )
        
        return None
    
    def dependency_analysis(self, incident: Incident) -> Optional[RootCause]:
        """Perform dependency analysis for service impact"""
        if not self.algorithms["dependency_analysis"]["enabled"]:
            return None
        
        # Find dependencies of affected services
        affected_deps = []
        for service in incident.affected_services:
            for dep in self.service_dependencies:
                if dep.source == service or dep.target == service:
                    affected_deps.append(dep)
        
        # Check for dependency failures
        high_error_deps = [dep for dep in affected_deps if dep.error_rate > 0.1]
        high_latency_deps = [dep for dep in affected_deps if dep.latency > 1000]
        
        if high_error_deps or high_latency_deps:
            evidence = []
            if high_error_deps:
                evidence.append(f"High error rate in {len(high_error_deps)} dependencies")
            if high_latency_deps:
                evidence.append(f"High latency in {len(high_latency_deps)} dependencies")
            
            confidence = 0.7 if (high_error_deps and high_latency_deps) else 0.5
            
            return RootCause(
                incident_id=incident.id,
                root_cause="Service dependency failure detected",
                confidence=confidence,
                evidence=evidence,
                affected_services=incident.affected_services,
                remediation="Check service dependencies and network connectivity",
                algorithm="dependency_analysis",
                timestamp=datetime.now()
            )
        
        return None
    
    def temporal_analysis(self, incident: Incident) -> Optional[RootCause]:
        """Perform temporal analysis for event correlation"""
        if not self.algorithms["temporal_analysis"]["enabled"]:
            return None
        
        # Simulate temporal correlation analysis
        # Check if multiple alerts occurred within a time window
        if len(incident.alerts) > 2:
            # Check for correlation patterns
            if "DatabaseDown" in incident.alerts and "HighErrorRate" in incident.alerts:
                return RootCause(
                    incident_id=incident.id,
                    root_cause="Database failure causing cascading errors",
                    confidence=0.8,
                    evidence=["DatabaseDown alert", "HighErrorRate alert", "Multiple service failures"],
                    affected_services=incident.affected_services,
                    remediation="Check database health and connectivity",
                    algorithm="temporal_analysis",
                    timestamp=datetime.now()
                )
            
            if "HighMemoryUsage" in incident.alerts and "OOMKilled" in incident.alerts:
                return RootCause(
                    incident_id=incident.id,
                    root_cause="Memory exhaustion causing service failures",
                    confidence=0.9,
                    evidence=["HighMemoryUsage alert", "OOMKilled alert", "Service restart"],
                    affected_services=incident.affected_services,
                    remediation="Increase memory limits or optimize memory usage",
                    algorithm="temporal_analysis",
                    timestamp=datetime.now()
                )
        
        return None
    
    def log_analysis(self, incident: Incident) -> Optional[RootCause]:
        """Perform log analysis for error patterns"""
        if not self.algorithms["log_analysis"]["enabled"]:
            return None
        
        # Analyze log patterns
        log_patterns = defaultdict(int)
        for log in incident.logs:
            for rule_type, rules in self.rules.items():
                for rule in rules:
                    if re.search(rule["pattern"], log, re.IGNORECASE):
                        log_patterns[rule["name"]] += 1
        
        if log_patterns:
            # Find the most common pattern
            most_common = max(log_patterns.items(), key=lambda x: x[1])
            pattern_name = most_common[0]
            count = most_common[1]
            
            # Map pattern to root cause
            root_cause_mapping = {
                "database_connection_failure": "Database connectivity issue",
                "memory_exhaustion": "Memory resource exhaustion",
                "cpu_saturation": "CPU resource saturation",
                "timeout_errors": "Network timeout issues",
                "dns_resolution_failure": "DNS resolution failure"
            }
            
            root_cause = root_cause_mapping.get(pattern_name, "Unknown pattern detected")
            confidence = min(0.9, count * 0.3)
            
            return RootCause(
                incident_id=incident.id,
                root_cause=root_cause,
                confidence=confidence,
                evidence=[f"Pattern '{pattern_name}' detected {count} times in logs"],
                affected_services=incident.affected_services,
                remediation="Review logs and apply appropriate fixes",
                algorithm="log_analysis",
                timestamp=datetime.now()
            )
        
        return None
    
    def trace_analysis(self, incident: Incident) -> Optional[RootCause]:
        """Perform trace analysis for performance issues"""
        if not self.algorithms["trace_analysis"]["enabled"]:
            return None
        
        # Analyze trace patterns
        trace_patterns = defaultdict(int)
        for trace in incident.traces:
            if "timeout" in trace.lower():
                trace_patterns["timeout"] += 1
            elif "error" in trace.lower():
                trace_patterns["error"] += 1
            elif "bottleneck" in trace.lower():
                trace_patterns["bottleneck"] += 1
            elif "failure" in trace.lower():
                trace_patterns["failure"] += 1
        
        if trace_patterns:
            # Find the most common trace pattern
            most_common = max(trace_patterns.items(), key=lambda x: x[1])
            pattern_name = most_common[0]
            count = most_common[1]
            
            # Map pattern to root cause
            root_cause_mapping = {
                "timeout": "Request timeout in service calls",
                "error": "Error propagation through service chain",
                "bottleneck": "Performance bottleneck detected",
                "failure": "Service failure in call chain"
            }
            
            root_cause = root_cause_mapping.get(pattern_name, "Unknown trace pattern")
            confidence = min(0.8, count * 0.4)
            
            return RootCause(
                incident_id=incident.id,
                root_cause=root_cause,
                confidence=confidence,
                evidence=[f"Trace pattern '{pattern_name}' detected {count} times"],
                affected_services=incident.affected_services,
                remediation="Review service call chains and performance",
                algorithm="trace_analysis",
                timestamp=datetime.now()
            )
        
        return None
    
    def analyze_incident(self, incident: Incident) -> List[RootCause]:
        """Perform comprehensive RCA analysis on an incident"""
        results = []
        
        # Run all enabled algorithms
        algorithms = [
            self.statistical_analysis,
            self.dependency_analysis,
            self.temporal_analysis,
            self.log_analysis,
            self.trace_analysis
        ]
        
        for algorithm in algorithms:
            try:
                result = algorithm(incident)
                if result:
                    results.append(result)
            except Exception as e:
                print(f"Error in {algorithm.__name__}: {e}")
        
        # Sort by confidence (highest first)
        results.sort(key=lambda x: x.confidence, reverse=True)
        
        return results
    
    def analyze_all_incidents(self) -> Dict[str, List[RootCause]]:
        """Analyze all incidents and return results"""
        all_results = {}
        
        for incident in self.incidents:
            results = self.analyze_incident(incident)
            all_results[incident.id] = results
            self.root_causes.extend(results)
        
        return all_results
    
    def generate_rca_report(self) -> Dict[str, Any]:
        """Generate comprehensive RCA report"""
        # Analyze all incidents
        analysis_results = self.analyze_all_incidents()
        
        # Calculate summary statistics
        total_incidents = len(self.incidents)
        total_root_causes = len(self.root_causes)
        avg_confidence = np.mean([rc.confidence for rc in self.root_causes]) if self.root_causes else 0
        
        # Group by algorithm
        algorithm_stats = defaultdict(int)
        for rc in self.root_causes:
            algorithm_stats[rc.algorithm] += 1
        
        # Group by root cause type
        root_cause_stats = defaultdict(int)
        for rc in self.root_causes:
            root_cause_stats[rc.root_cause] += 1
        
        report = {
            "timestamp": datetime.now().isoformat(),
            "summary": {
                "total_incidents": total_incidents,
                "total_root_causes": total_root_causes,
                "average_confidence": avg_confidence,
                "analysis_coverage": total_root_causes / total_incidents if total_incidents > 0 else 0
            },
            "algorithm_performance": dict(algorithm_stats),
            "root_cause_distribution": dict(root_cause_stats),
            "incident_analysis": {},
            "recommendations": self._generate_recommendations()
        }
        
        # Add detailed analysis for each incident
        for incident_id, results in analysis_results.items():
            report["incident_analysis"][incident_id] = {
                "incident": next(inc for inc in self.incidents if inc.id == incident_id).__dict__,
                "root_causes": [rc.__dict__ for rc in results],
                "primary_root_cause": results[0].__dict__ if results else None
            }
        
        return report
    
    def _generate_recommendations(self) -> List[str]:
        """Generate recommendations based on analysis results"""
        recommendations = []
        
        # Check algorithm performance
        algorithm_counts = defaultdict(int)
        for rc in self.root_causes:
            algorithm_counts[rc.algorithm] += 1
        
        if algorithm_counts["log_analysis"] > algorithm_counts["statistical_analysis"]:
            recommendations.append("Log analysis is more effective than statistical analysis for this environment")
        
        if algorithm_counts["dependency_analysis"] > 0:
            recommendations.append("Consider implementing service dependency monitoring")
        
        # Check confidence levels
        low_confidence = [rc for rc in self.root_causes if rc.confidence < 0.6]
        if low_confidence:
            recommendations.append(f"Review {len(low_confidence)} low-confidence root causes for accuracy")
        
        # Check common root causes
        root_cause_counts = defaultdict(int)
        for rc in self.root_causes:
            root_cause_counts[rc.root_cause] += 1
        
        common_causes = [cause for cause, count in root_cause_counts.items() if count > 1]
        if common_causes:
            recommendations.append(f"Address recurring root causes: {', '.join(common_causes)}")
        
        return recommendations

def main():
    """Main function to demonstrate RCA analysis operations"""
    print("254Carbon Observability - Root Cause Analysis Demo")
    print("=" * 60)
    
    # Initialize RCA analyzer
    analyzer = RootCauseAnalyzer()
    
    # Generate sample data
    print("\n1. Generating sample incidents...")
    analyzer.generate_sample_incidents(count=5)
    
    print("\n2. Generating sample dependencies...")
    analyzer.generate_sample_dependencies()
    
    # Perform RCA analysis
    print("\n3. Performing RCA analysis...")
    report = analyzer.generate_rca_report()
    
    # Print summary
    print("\n4. RCA Analysis Summary:")
    summary = report['summary']
    print(f"   - Total Incidents: {summary['total_incidents']}")
    print(f"   - Total Root Causes: {summary['total_root_causes']}")
    print(f"   - Average Confidence: {summary['average_confidence']:.2f}")
    print(f"   - Analysis Coverage: {summary['analysis_coverage']:.2f}")
    
    print("\n5. Algorithm Performance:")
    for algorithm, count in report['algorithm_performance'].items():
        print(f"   - {algorithm}: {count} root causes identified")
    
    print("\n6. Root Cause Distribution:")
    for root_cause, count in report['root_cause_distribution'].items():
        print(f"   - {root_cause}: {count} occurrences")
    
    print("\n7. Recommendations:")
    for i, rec in enumerate(report['recommendations'], 1):
        print(f"   {i}. {rec}")
    
    # Save detailed report
    with open('rca_analysis_report.json', 'w') as f:
        json.dump(report, f, indent=2, default=str)
    print("\n8. Detailed report saved to rca_analysis_report.json")
    
    print("\nRCA analysis demo completed!")

if __name__ == "__main__":
    main()
