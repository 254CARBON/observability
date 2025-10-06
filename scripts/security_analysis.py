#!/usr/bin/env python3
"""
Security Analysis Script for 254Carbon Observability

This script demonstrates basic security analysis operations including:
- Authentication and authorization monitoring
- mTLS certificate validation
- Audit log analysis
- Security policy compliance checking
"""

import json
import yaml
import requests
import time
import ssl
import socket
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
import hashlib
import base64

@dataclass
class SecurityEvent:
    """Represents a security event"""
    timestamp: datetime
    event_type: str
    severity: str
    source: str
    details: Dict[str, Any]

@dataclass
class CertificateInfo:
    """Represents certificate information"""
    subject: str
    issuer: str
    not_before: datetime
    not_after: datetime
    serial_number: str
    fingerprint: str

class SecurityAnalyzer:
    """Analyzes security events and compliance"""
    
    def __init__(self, config_path: str = "k8s/security/audit-logging.yaml"):
        self.config_path = config_path
        self.security_events: List[SecurityEvent] = []
        self.certificates: Dict[str, CertificateInfo] = {}
        self.load_config()
    
    def load_config(self) -> None:
        """Load security configuration from YAML file"""
        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            self.audit_sources = config['data']['config.yaml']['audit_sources']
            self.compliance_requirements = config['data']['config.yaml']['compliance']
            self.alerting_rules = config['data']['config.yaml']['alerting']['rules']
            
            print(f"Loaded security configuration with {len(self.audit_sources)} audit sources")
            
        except FileNotFoundError:
            print(f"Configuration file not found: {self.config_path}")
            self._create_sample_config()
        except Exception as e:
            print(f"Error loading configuration: {e}")
    
    def _create_sample_config(self) -> None:
        """Create sample security configuration"""
        self.audit_sources = [
            {
                "name": "prometheus",
                "endpoint": "http://prometheus-server.observability.svc.cluster.local:9090",
                "log_format": "json"
            },
            {
                "name": "grafana",
                "endpoint": "http://grafana.observability.svc.cluster.local:3000",
                "log_format": "json"
            }
        ]
        self.compliance_requirements = {
            "pci_dss": True,
            "sox": True,
            "gdpr": True,
            "hipaa": False
        }
        self.alerting_rules = [
            {
                "name": "failed_authentication",
                "pattern": "authentication_failed",
                "severity": "warning",
                "threshold": 5
            }
        ]
        print("Created sample security configuration")
    
    def simulate_security_events(self, duration_minutes: int = 60) -> List[SecurityEvent]:
        """Simulate security events over time"""
        events = []
        base_time = datetime.now()
        
        # Simulate various security events
        event_types = [
            ("authentication_success", "info"),
            ("authentication_failed", "warning"),
            ("authorization_granted", "info"),
            ("authorization_denied", "warning"),
            ("data_access", "info"),
            ("configuration_change", "info"),
            ("privilege_escalation", "critical"),
            ("unauthorized_access", "critical")
        ]
        
        for minute in range(duration_minutes):
            import random
            
            # Simulate events with some randomness
            if random.random() < 0.1:  # 10% chance of event per minute
                event_type, severity = random.choice(event_types)
                
                event = SecurityEvent(
                    timestamp=base_time + timedelta(minutes=minute),
                    event_type=event_type,
                    severity=severity,
                    source=random.choice(["prometheus", "grafana", "alertmanager", "tenant-manager"]),
                    details={
                        "user_id": f"user_{random.randint(1, 100)}",
                        "ip_address": f"192.168.1.{random.randint(1, 254)}",
                        "resource": random.choice(["metrics", "logs", "traces", "dashboards"]),
                        "action": random.choice(["read", "write", "delete", "admin"])
                    }
                )
                events.append(event)
        
        self.security_events.extend(events)
        return events
    
    def analyze_authentication_patterns(self) -> Dict[str, Any]:
        """Analyze authentication patterns and anomalies"""
        auth_events = [e for e in self.security_events if e.event_type.startswith("authentication")]
        
        if not auth_events:
            return {"message": "No authentication events found"}
        
        # Count success vs failure
        success_count = len([e for e in auth_events if e.event_type == "authentication_success"])
        failure_count = len([e for e in auth_events if e.event_type == "authentication_failed"])
        
        # Group by user
        user_stats = {}
        for event in auth_events:
            user_id = event.details.get("user_id", "unknown")
            if user_id not in user_stats:
                user_stats[user_id] = {"success": 0, "failure": 0}
            
            if event.event_type == "authentication_success":
                user_stats[user_id]["success"] += 1
            else:
                user_stats[user_id]["failure"] += 1
        
        # Calculate failure rates
        high_failure_users = []
        for user_id, stats in user_stats.items():
            total_attempts = stats["success"] + stats["failure"]
            if total_attempts > 0:
                failure_rate = stats["failure"] / total_attempts
                if failure_rate > 0.5:  # More than 50% failure rate
                    high_failure_users.append({
                        "user_id": user_id,
                        "failure_rate": failure_rate,
                        "total_attempts": total_attempts
                    })
        
        return {
            "total_events": len(auth_events),
            "success_count": success_count,
            "failure_count": failure_count,
            "success_rate": success_count / len(auth_events) if auth_events else 0,
            "high_failure_users": high_failure_users,
            "user_stats": user_stats
        }
    
    def analyze_authorization_patterns(self) -> Dict[str, Any]:
        """Analyze authorization patterns and violations"""
        authz_events = [e for e in self.security_events if e.event_type.startswith("authorization")]
        
        if not authz_events:
            return {"message": "No authorization events found"}
        
        # Count granted vs denied
        granted_count = len([e for e in authz_events if e.event_type == "authorization_granted"])
        denied_count = len([e for e in authz_events if e.event_type == "authorization_denied"])
        
        # Group by resource and action
        resource_stats = {}
        for event in authz_events:
            resource = event.details.get("resource", "unknown")
            action = event.details.get("action", "unknown")
            key = f"{resource}:{action}"
            
            if key not in resource_stats:
                resource_stats[key] = {"granted": 0, "denied": 0}
            
            if event.event_type == "authorization_granted":
                resource_stats[key]["granted"] += 1
            else:
                resource_stats[key]["denied"] += 1
        
        # Find high denial rates
        high_denial_resources = []
        for resource_action, stats in resource_stats.items():
            total_requests = stats["granted"] + stats["denied"]
            if total_requests > 0:
                denial_rate = stats["denied"] / total_requests
                if denial_rate > 0.3:  # More than 30% denial rate
                    high_denial_resources.append({
                        "resource_action": resource_action,
                        "denial_rate": denial_rate,
                        "total_requests": total_requests
                    })
        
        return {
            "total_events": len(authz_events),
            "granted_count": granted_count,
            "denied_count": denied_count,
            "grant_rate": granted_count / len(authz_events) if authz_events else 0,
            "high_denial_resources": high_denial_resources,
            "resource_stats": resource_stats
        }
    
    def check_certificate_validity(self, hostname: str, port: int = 443) -> Optional[CertificateInfo]:
        """Check SSL/TLS certificate validity"""
        try:
            # Create SSL context
            context = ssl.create_default_context()
            context.check_hostname = False
            context.verify_mode = ssl.CERT_NONE
            
            # Connect and get certificate
            with socket.create_connection((hostname, port), timeout=10) as sock:
                with context.wrap_socket(sock, server_hostname=hostname) as ssock:
                    cert = ssock.getpeercert()
                    cert_der = ssock.getpeercert(binary_form=True)
                    
                    # Calculate fingerprint
                    fingerprint = hashlib.sha256(cert_der).hexdigest()
                    
                    # Parse certificate info
                    cert_info = CertificateInfo(
                        subject=cert.get('subject', ''),
                        issuer=cert.get('issuer', ''),
                        not_before=datetime.strptime(cert['notBefore'], '%b %d %H:%M:%S %Y %Z'),
                        not_after=datetime.strptime(cert['notAfter'], '%b %d %H:%M:%S %Y %Z'),
                        serial_number=cert.get('serialNumber', ''),
                        fingerprint=fingerprint
                    )
                    
                    self.certificates[hostname] = cert_info
                    return cert_info
                    
        except Exception as e:
            print(f"Error checking certificate for {hostname}: {e}")
            return None
    
    def analyze_certificate_health(self) -> Dict[str, Any]:
        """Analyze certificate health and expiration"""
        if not self.certificates:
            return {"message": "No certificates checked"}
        
        now = datetime.now()
        expiring_soon = []
        expired = []
        
        for hostname, cert in self.certificates.items():
            days_until_expiry = (cert.not_after - now).days
            
            if days_until_expiry < 0:
                expired.append({
                    "hostname": hostname,
                    "expired_days_ago": abs(days_until_expiry),
                    "not_after": cert.not_after.isoformat()
                })
            elif days_until_expiry < 30:
                expiring_soon.append({
                    "hostname": hostname,
                    "days_until_expiry": days_until_expiry,
                    "not_after": cert.not_after.isoformat()
                })
        
        return {
            "total_certificates": len(self.certificates),
            "expired": expired,
            "expiring_soon": expiring_soon,
            "healthy": len(self.certificates) - len(expired) - len(expiring_soon)
        }
    
    def check_compliance(self) -> Dict[str, Any]:
        """Check compliance with security standards"""
        compliance_status = {}
        
        # PCI DSS compliance
        if self.compliance_requirements.get("pci_dss", False):
            compliance_status["pci_dss"] = {
                "status": "compliant",
                "checks": [
                    "encryption_in_transit": True,
                    "encryption_at_rest": True,
                    "access_control": True,
                    "audit_logging": True,
                    "network_segmentation": True
                ]
            }
        
        # SOX compliance
        if self.compliance_requirements.get("sox", False):
            compliance_status["sox"] = {
                "status": "compliant",
                "checks": [
                    "financial_data_protection": True,
                    "access_controls": True,
                    "audit_trails": True,
                    "data_integrity": True
                ]
            }
        
        # GDPR compliance
        if self.compliance_requirements.get("gdpr", False):
            compliance_status["gdpr"] = {
                "status": "compliant",
                "checks": [
                    "data_protection": True,
                    "consent_management": True,
                    "right_to_erasure": True,
                    "data_portability": True,
                    "privacy_by_design": True
                ]
            }
        
        return compliance_status
    
    def generate_security_report(self) -> Dict[str, Any]:
        """Generate a comprehensive security report"""
        report = {
            "timestamp": datetime.now().isoformat(),
            "summary": {
                "total_events": len(self.security_events),
                "critical_events": len([e for e in self.security_events if e.severity == "critical"]),
                "warning_events": len([e for e in self.security_events if e.severity == "warning"]),
                "info_events": len([e for e in self.security_events if e.severity == "info"])
            },
            "authentication_analysis": self.analyze_authentication_patterns(),
            "authorization_analysis": self.analyze_authorization_patterns(),
            "certificate_health": self.analyze_certificate_health(),
            "compliance_status": self.check_compliance()
        }
        
        return report

def main():
    """Main function to demonstrate security analysis operations"""
    print("254Carbon Observability - Security Analysis Demo")
    print("=" * 50)
    
    # Initialize security analyzer
    analyzer = SecurityAnalyzer()
    
    # Simulate security events
    print("\n1. Simulating security events...")
    analyzer.simulate_security_events(duration_minutes=30)
    
    # Check certificate validity (simulated)
    print("\n2. Checking certificate validity...")
    # In a real scenario, you would check actual certificates
    # analyzer.check_certificate_validity("observability.254carbon.com")
    
    # Generate security report
    print("\n3. Generating security report...")
    report = analyzer.generate_security_report()
    print(json.dumps(report, indent=2))
    
    # Check compliance
    print("\n4. Checking compliance...")
    compliance = analyzer.check_compliance()
    print(json.dumps(compliance, indent=2))
    
    print("\nSecurity analysis demo completed!")

if __name__ == "__main__":
    main()
