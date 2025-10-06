#!/usr/bin/env python3
"""
Alert Intelligence Analysis Script for 254Carbon Observability

This script provides alert intelligence analysis capabilities including:
- Alert correlation analysis
- Alert deduplication analysis
- Alert pattern recognition
- Alert trend analysis
- Alert intelligence scoring
"""

import os
import sys
import json
import yaml
import argparse
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
import logging
from collections import defaultdict, deque
import hashlib

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class Alert:
    """Alert data structure"""
    id: str
    name: str
    severity: str
    status: str
    service: str
    instance: str
    labels: Dict[str, str]
    annotations: Dict[str, str]
    starts_at: datetime
    ends_at: Optional[datetime]
    fingerprint: str
    value: float
    source: str

@dataclass
class AlertIntelligenceConfig:
    """Configuration for alert intelligence analysis"""
    alert_correlation_endpoint: str
    prometheus_endpoint: str
    alertmanager_endpoint: str
    analysis_time_window: str
    correlation_thresholds: Dict[str, float]
    deduplication_thresholds: Dict[str, float]
    intelligence_thresholds: Dict[str, float]

class AlertIntelligenceAnalyzer:
    """Main alert intelligence analyzer"""
    
    def __init__(self, config_path: str):
        self.config = self._load_config(config_path)
        self.alert_correlation_endpoint = self.config['alert_correlation_endpoint']
        self.prometheus_endpoint = self.config['prometheus_endpoint']
        self.alertmanager_endpoint = self.config['alertmanager_endpoint']
        
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.error(f"Failed to load config: {e}")
            raise
            
    def _query_prometheus(self, query: str, start_time: datetime, end_time: datetime) -> List[Dict]:
        """Query Prometheus for metrics"""
        try:
            params = {
                'query': query,
                'start': start_time.timestamp(),
                'end': end_time.timestamp(),
                'step': '1m'
            }
            
            response = requests.get(
                f"{self.prometheus_endpoint}/api/v1/query_range",
                params=params,
                timeout=30
            )
            response.raise_for_status()
            
            data = response.json()
            if data['status'] == 'success':
                return data['data']['result']
            else:
                raise Exception(f"Prometheus query failed: {data.get('error', 'Unknown error')}")
                
        except Exception as e:
            logger.error(f"Prometheus query error: {e}")
            raise
            
    def _query_alert_correlation(self, endpoint: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Query alert correlation service"""
        try:
            response = requests.post(
                f"{self.alert_correlation_endpoint}{endpoint}",
                json=data,
                timeout=30
            )
            response.raise_for_status()
            return response.json()
            
        except Exception as e:
            logger.error(f"Alert correlation query error: {e}")
            raise
            
    def _query_alertmanager(self, endpoint: str) -> Dict[str, Any]:
        """Query Alertmanager for alerts"""
        try:
            response = requests.get(
                f"{self.alertmanager_endpoint}/api/v1/{endpoint}",
                timeout=10
            )
            response.raise_for_status()
            return response.json()
            
        except Exception as e:
            logger.error(f"Alertmanager query error: {e}")
            raise
            
    def analyze_alert_correlation(self, hours: int = 24) -> Dict[str, Any]:
        """Analyze alert correlation"""
        logger.info(f"Analyzing alert correlation for last {hours} hours")
        
        try:
            # Get alerts from Alertmanager
            alerts_data = self._query_alertmanager('alerts')
            alerts = [Alert(**alert) for alert in alerts_data.get('data', [])]
            
            if len(alerts) < 2:
                return {
                    'correlation_analysis': {
                        'total_alerts': len(alerts),
                        'correlation_groups': [],
                        'average_correlation_score': 0.0,
                        'high_correlation_alerts': 0
                    },
                    'analysis_timestamp': datetime.now().isoformat()
                }
                
            # Correlate alerts
            correlation_result = self._query_alert_correlation('/api/v1/alerts/correlate', {
                'alerts': [asdict(alert) for alert in alerts]
            })
            
            # Analyze correlation results
            correlation_score = correlation_result.get('correlation_score', 0.0)
            confidence = correlation_result.get('confidence', 0.0)
            reasoning = correlation_result.get('reasoning', '')
            
            # Group alerts by correlation
            correlation_groups = self._group_alerts_by_correlation(alerts, correlation_score)
            
            # Calculate high correlation alerts
            high_correlation_alerts = sum(
                1 for group in correlation_groups 
                if group['correlation_score'] > 0.7
            )
            
            return {
                'correlation_analysis': {
                    'total_alerts': len(alerts),
                    'correlation_groups': correlation_groups,
                    'average_correlation_score': correlation_score,
                    'confidence': confidence,
                    'reasoning': reasoning,
                    'high_correlation_alerts': high_correlation_alerts
                },
                'analysis_timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error analyzing alert correlation: {e}")
            raise
            
    def analyze_alert_deduplication(self, hours: int = 24) -> Dict[str, Any]:
        """Analyze alert deduplication"""
        logger.info(f"Analyzing alert deduplication for last {hours} hours")
        
        try:
            # Get alerts from Alertmanager
            alerts_data = self._query_alertmanager('alerts')
            alerts = [Alert(**alert) for alert in alerts_data.get('data', [])]
            
            if len(alerts) < 2:
                return {
                    'deduplication_analysis': {
                        'total_alerts': len(alerts),
                        'deduplicated_alerts': len(alerts),
                        'deduplication_rate': 0.0,
                        'strategy': 'none',
                        'confidence': 1.0,
                        'reasoning': 'Insufficient alerts for deduplication'
                    },
                    'analysis_timestamp': datetime.now().isoformat()
                }
                
            # Deduplicate alerts
            deduplication_result = self._query_alert_correlation('/api/v1/alerts/deduplicate', {
                'alerts': [asdict(alert) for alert in alerts]
            })
            
            # Analyze deduplication results
            original_count = len(alerts)
            deduplicated_count = len(deduplication_result.get('deduplicated_alerts', []))
            deduplication_rate = 1.0 - (deduplicated_count / original_count) if original_count > 0 else 0.0
            strategy = deduplication_result.get('strategy', 'unknown')
            confidence = deduplication_result.get('confidence', 0.0)
            reasoning = deduplication_result.get('reasoning', '')
            
            return {
                'deduplication_analysis': {
                    'total_alerts': original_count,
                    'deduplicated_alerts': deduplicated_count,
                    'deduplication_rate': deduplication_rate,
                    'strategy': strategy,
                    'confidence': confidence,
                    'reasoning': reasoning
                },
                'analysis_timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error analyzing alert deduplication: {e}")
            raise
            
    def analyze_alert_patterns(self, hours: int = 168) -> Dict[str, Any]:
        """Analyze alert patterns"""
        logger.info(f"Analyzing alert patterns for last {hours} hours")
        
        try:
            # Get alert patterns from correlation service
            patterns_result = self._query_alert_correlation('/api/v1/alerts/patterns', {})
            
            # Analyze patterns
            patterns = patterns_result.get('patterns', {})
            
            # Calculate pattern metrics
            pattern_metrics = {}
            
            # Seasonal patterns
            if 'seasonal' in patterns:
                seasonal = patterns['seasonal']
                if 'hourly' in seasonal:
                    hourly_counts = seasonal['hourly']
                    peak_hour = max(hourly_counts.items(), key=lambda x: x[1])[0]
                    pattern_metrics['peak_hour'] = peak_hour
                    pattern_metrics['hourly_variance'] = np.var(list(hourly_counts.values()))
                    
            # Correlated patterns
            if 'correlated' in patterns:
                correlated = patterns['correlated']
                if 'services' in correlated:
                    service_counts = correlated['services']
                    top_service = max(service_counts.items(), key=lambda x: x[1])[0]
                    pattern_metrics['top_service'] = top_service
                    pattern_metrics['service_diversity'] = len(service_counts)
                    
            return {
                'pattern_analysis': {
                    'patterns': patterns,
                    'pattern_metrics': pattern_metrics,
                    'analysis_period_hours': hours
                },
                'analysis_timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error analyzing alert patterns: {e}")
            raise
            
    def analyze_alert_trends(self, hours: int = 168) -> Dict[str, Any]:
        """Analyze alert trends"""
        logger.info(f"Analyzing alert trends for last {hours} hours")
        
        try:
            # Get alert trends from correlation service
            trends_result = self._query_alert_correlation('/api/v1/alerts/trends', {})
            
            # Analyze trends
            trends = trends_result.get('trends', {})
            
            # Calculate trend metrics
            trend_metrics = {}
            
            # Volume trends
            if 'volume' in trends and 'daily' in trends['volume']:
                daily_counts = trends['volume']['daily']
                if daily_counts:
                    counts = list(daily_counts.values())
                    trend_metrics['average_daily_alerts'] = np.mean(counts)
                    trend_metrics['alert_volume_trend'] = self._calculate_trend(counts)
                    trend_metrics['alert_volume_volatility'] = np.std(counts)
                    
            # Severity trends
            if 'severity' in trends and 'distribution' in trends['severity']:
                severity_dist = trends['severity']['distribution']
                total_alerts = sum(severity_dist.values())
                if total_alerts > 0:
                    trend_metrics['critical_percentage'] = (severity_dist.get('critical', 0) / total_alerts) * 100
                    trend_metrics['warning_percentage'] = (severity_dist.get('warning', 0) / total_alerts) * 100
                    trend_metrics['info_percentage'] = (severity_dist.get('info', 0) / total_alerts) * 100
                    
            return {
                'trend_analysis': {
                    'trends': trends,
                    'trend_metrics': trend_metrics,
                    'analysis_period_hours': hours
                },
                'analysis_timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error analyzing alert trends: {e}")
            raise
            
    def analyze_alert_intelligence(self, hours: int = 24) -> Dict[str, Any]:
        """Analyze alert intelligence"""
        logger.info(f"Analyzing alert intelligence for last {hours} hours")
        
        try:
            # Get alerts from Alertmanager
            alerts_data = self._query_alertmanager('alerts')
            alerts = [Alert(**alert) for alert in alerts_data.get('data', [])]
            
            if not alerts:
                return {
                    'intelligence_analysis': {
                        'total_alerts': 0,
                        'intelligence_scores': {},
                        'average_intelligence_score': 0.0,
                        'high_intelligence_alerts': 0
                    },
                    'analysis_timestamp': datetime.now().isoformat()
                }
                
            # Analyze intelligence for each alert
            intelligence_scores = {}
            total_score = 0.0
            high_intelligence_count = 0
            
            for alert in alerts:
                intelligence_result = self._query_alert_correlation('/api/v1/alerts/intelligence', {
                    'alert': asdict(alert)
                })
                
                scores = intelligence_result.get('intelligence_scores', {})
                intelligence_scores[alert.id] = scores
                
                # Calculate average score for this alert
                if scores:
                    avg_score = np.mean(list(scores.values()))
                    total_score += avg_score
                    
                    if avg_score > 0.7:
                        high_intelligence_count += 1
                        
            average_intelligence_score = total_score / len(alerts) if alerts else 0.0
            
            return {
                'intelligence_analysis': {
                    'total_alerts': len(alerts),
                    'intelligence_scores': intelligence_scores,
                    'average_intelligence_score': average_intelligence_score,
                    'high_intelligence_alerts': high_intelligence_count
                },
                'analysis_timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error analyzing alert intelligence: {e}")
            raise
            
    def generate_alert_intelligence_report(self, hours: int = 24) -> Dict[str, Any]:
        """Generate comprehensive alert intelligence report"""
        logger.info(f"Generating comprehensive alert intelligence report for last {hours} hours")
        
        try:
            # Collect all analyses
            correlation = self.analyze_alert_correlation(hours)
            deduplication = self.analyze_alert_deduplication(hours)
            patterns = self.analyze_alert_patterns(hours * 7)  # Use 7x hours for patterns
            trends = self.analyze_alert_trends(hours * 7)  # Use 7x hours for trends
            intelligence = self.analyze_alert_intelligence(hours)
            
            # Calculate overall intelligence score
            intelligence_score = self._calculate_overall_intelligence_score(
                correlation, deduplication, patterns, trends, intelligence
            )
            
            # Generate executive summary
            executive_summary = self._generate_intelligence_executive_summary(
                correlation, deduplication, patterns, trends, intelligence, intelligence_score
            )
            
            return {
                'report_period': {
                    'hours': hours,
                    'start_date': (datetime.now() - timedelta(hours=hours)).isoformat(),
                    'end_date': datetime.now().isoformat()
                },
                'executive_summary': executive_summary,
                'intelligence_score': intelligence_score,
                'correlation_analysis': correlation,
                'deduplication_analysis': deduplication,
                'pattern_analysis': patterns,
                'trend_analysis': trends,
                'intelligence_analysis': intelligence,
                'generated_at': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error generating alert intelligence report: {e}")
            raise
            
    def _group_alerts_by_correlation(self, alerts: List[Alert], correlation_score: float) -> List[Dict[str, Any]]:
        """Group alerts by correlation"""
        groups = []
        
        # Simple grouping based on service
        service_groups = defaultdict(list)
        for alert in alerts:
            service_groups[alert.service].append(alert)
            
        for service, service_alerts in service_groups.items():
            if len(service_alerts) > 1:
                groups.append({
                    'service': service,
                    'alert_count': len(service_alerts),
                    'correlation_score': correlation_score,
                    'alerts': [asdict(alert) for alert in service_alerts]
                })
                
        return groups
        
    def _calculate_trend(self, values: List[float]) -> str:
        """Calculate trend direction"""
        if len(values) < 2:
            return 'stable'
            
        # Simple linear trend
        x = np.arange(len(values))
        slope = np.polyfit(x, values, 1)[0]
        
        if slope > 0.1:
            return 'increasing'
        elif slope < -0.1:
            return 'decreasing'
        else:
            return 'stable'
            
    def _calculate_overall_intelligence_score(self, correlation: Dict, deduplication: Dict, 
                                            patterns: Dict, trends: Dict, intelligence: Dict) -> float:
        """Calculate overall intelligence score"""
        try:
            score = 0.0
            
            # Correlation score (30%)
            corr_score = correlation['correlation_analysis']['average_correlation_score']
            score += corr_score * 0.3
            
            # Deduplication score (25%)
            dedup_score = deduplication['deduplication_analysis']['deduplication_rate']
            score += dedup_score * 0.25
            
            # Pattern score (20%)
            pattern_score = 0.5  # Placeholder
            score += pattern_score * 0.2
            
            # Trend score (15%)
            trend_score = 0.5  # Placeholder
            score += trend_score * 0.15
            
            # Intelligence score (10%)
            intel_score = intelligence['intelligence_analysis']['average_intelligence_score']
            score += intel_score * 0.1
            
            return min(1.0, score)
            
        except Exception as e:
            logger.error(f"Error calculating overall intelligence score: {e}")
            return 0.5  # Default score
            
    def _generate_intelligence_executive_summary(self, correlation: Dict, deduplication: Dict,
                                               patterns: Dict, trends: Dict, intelligence: Dict,
                                               intelligence_score: float) -> Dict[str, Any]:
        """Generate executive summary for alert intelligence"""
        try:
            # Extract key metrics
            total_alerts = correlation['correlation_analysis']['total_alerts']
            correlation_score = correlation['correlation_analysis']['average_correlation_score']
            deduplication_rate = deduplication['deduplication_analysis']['deduplication_rate']
            intelligence_score_avg = intelligence['intelligence_analysis']['average_intelligence_score']
            
            # Determine overall status
            if intelligence_score >= 0.8:
                status = "excellent"
            elif intelligence_score >= 0.6:
                status = "good"
            elif intelligence_score >= 0.4:
                status = "fair"
            else:
                status = "poor"
                
            # Generate key insights
            insights = []
            
            if correlation_score > 0.7:
                insights.append(f"High alert correlation detected (score: {correlation_score:.2f})")
            elif correlation_score < 0.3:
                insights.append(f"Low alert correlation (score: {correlation_score:.2f})")
                
            if deduplication_rate > 0.5:
                insights.append(f"Effective alert deduplication (rate: {deduplication_rate:.1%})")
            elif deduplication_rate < 0.2:
                insights.append(f"Low alert deduplication (rate: {deduplication_rate:.1%})")
                
            if intelligence_score_avg > 0.7:
                insights.append(f"High alert intelligence (score: {intelligence_score_avg:.2f})")
            elif intelligence_score_avg < 0.4:
                insights.append(f"Low alert intelligence (score: {intelligence_score_avg:.2f})")
                
            # Generate recommendations
            recommendations = []
            
            if correlation_score < 0.5:
                recommendations.append("Improve alert correlation algorithms")
                
            if deduplication_rate < 0.3:
                recommendations.append("Enhance alert deduplication strategies")
                
            if intelligence_score_avg < 0.5:
                recommendations.append("Upgrade alert intelligence models")
                
            return {
                'status': status,
                'intelligence_score': intelligence_score,
                'total_alerts': total_alerts,
                'correlation_score': correlation_score,
                'deduplication_rate': deduplication_rate,
                'intelligence_score_avg': intelligence_score_avg,
                'key_insights': insights,
                'recommendations': recommendations
            }
            
        except Exception as e:
            logger.error(f"Error generating intelligence executive summary: {e}")
            return {
                'status': 'unknown',
                'intelligence_score': intelligence_score,
                'error': str(e)
            }

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Alert Intelligence Analysis Script for 254Carbon Observability')
    parser.add_argument('--config', '-c', required=True, help='Path to configuration file')
    parser.add_argument('--hours', '-h', type=int, default=24, help='Number of hours to analyze')
    parser.add_argument('--output', '-o', help='Output file path (JSON format)')
    parser.add_argument('--format', '-f', choices=['json', 'yaml'], default='json', help='Output format')
    parser.add_argument('--analysis', '-a', choices=['correlation', 'deduplication', 'patterns', 'trends', 'intelligence', 'report'], 
                       default='report', help='Type of analysis to perform')
    
    args = parser.parse_args()
    
    try:
        # Initialize analyzer
        analyzer = AlertIntelligenceAnalyzer(args.config)
        
        # Perform analysis
        if args.analysis == 'correlation':
            result = analyzer.analyze_alert_correlation(args.hours)
        elif args.analysis == 'deduplication':
            result = analyzer.analyze_alert_deduplication(args.hours)
        elif args.analysis == 'patterns':
            result = analyzer.analyze_alert_patterns(args.hours)
        elif args.analysis == 'trends':
            result = analyzer.analyze_alert_trends(args.hours)
        elif args.analysis == 'intelligence':
            result = analyzer.analyze_alert_intelligence(args.hours)
        else:  # report
            result = analyzer.generate_alert_intelligence_report(args.hours)
            
        # Output results
        if args.output:
            with open(args.output, 'w') as f:
                if args.format == 'json':
                    json.dump(result, f, indent=2)
                else:
                    yaml.dump(result, f, default_flow_style=False)
            print(f"Analysis results written to {args.output}")
        else:
            if args.format == 'json':
                print(json.dumps(result, indent=2))
            else:
                print(yaml.dump(result, default_flow_style=False))
                
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()
