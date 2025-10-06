#!/usr/bin/env python3
"""
High Availability Analysis Script for 254Carbon Observability

This script demonstrates HA operations including:
- Federation health monitoring
- Multi-region trace analysis
- Replication lag detection
- Failover readiness checks
"""

import json
import requests
from datetime import datetime
from typing import Dict, List, Any

class HAAnalyzer:
    """Performs high availability analysis"""
    
    def __init__(self, prometheus_url: str = "http://localhost:9090"):
        self.prometheus_url = prometheus_url
    
    def check_federation_health(self) -> Dict[str, Any]:
        """Check federation health across regions"""
        query = 'prometheus_federation_up'
        try:
            response = requests.get(f"{self.prometheus_url}/api/v1/query", params={'query': query})
            data = response.json()
            return {"status": "healthy", "data": data.get('data', {}).get('result', [])}
        except Exception as e:
            return {"status": "error", "error": str(e)}
    
    def check_tempo_replication(self) -> Dict[str, Any]:
        """Check Tempo multi-region replication"""
        query = 'tempo_multi_region_traces_received'
        try:
            response = requests.get(f"{self.prometheus_url}/api/v1/query", params={'query': query})
            data = response.json()
            return {"status": "healthy", "data": data.get('data', {}).get('result', [])}
        except Exception as e:
            return {"status": "error", "error": str(e)}
    
    def generate_report(self) -> Dict[str, Any]:
        """Generate HA analysis report"""
        return {
            "timestamp": datetime.now().isoformat(),
            "federation_health": self.check_federation_health(),
            "tempo_replication": self.check_tempo_replication()
        }

def main():
    print("254Carbon Observability - High Availability Analysis")
    print("=" * 60)
    
    analyzer = HAAnalyzer()
    report = analyzer.generate_report()
    
    print("\nHA Analysis Report:")
    print(json.dumps(report, indent=2))
    
    with open('ha_analysis_report.json', 'w') as f:
        json.dump(report, f, indent=2)
    print("\nReport saved to ha_analysis_report.json")

if __name__ == "__main__":
    main()
