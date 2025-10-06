#!/usr/bin/env python3
"""
Cost Analysis Script for 254Carbon Observability

This script provides cost analysis capabilities including:
- Cost breakdown by service and tenant
- Cost trend analysis
- Optimization recommendations
- Cost efficiency metrics
- Budget utilization tracking
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
from dataclasses import dataclass
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class CostAnalysisConfig:
    """Configuration for cost analysis"""
    prometheus_endpoint: str
    cost_analytics_endpoint: str
    budget_daily: float
    budget_weekly: float
    budget_monthly: float
    cost_thresholds: Dict[str, float]
    optimization_thresholds: Dict[str, float]

class CostAnalyzer:
    """Main cost analysis class"""
    
    def __init__(self, config_path: str):
        self.config = self._load_config(config_path)
        self.prometheus_endpoint = self.config['prometheus_endpoint']
        self.cost_analytics_endpoint = self.config['cost_analytics_endpoint']
        
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
                'step': '1h'
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
            
    def _query_cost_analytics(self, endpoint: str) -> Dict[str, Any]:
        """Query cost analytics service"""
        try:
            response = requests.get(
                f"{self.cost_analytics_endpoint}{endpoint}",
                timeout=30
            )
            response.raise_for_status()
            return response.json()
            
        except Exception as e:
            logger.error(f"Cost analytics query error: {e}")
            raise
            
    def analyze_cost_breakdown(self, days: int = 7) -> Dict[str, Any]:
        """Analyze cost breakdown by service and tenant"""
        logger.info(f"Analyzing cost breakdown for last {days} days")
        
        try:
            breakdown = self._query_cost_analytics('/api/v1/cost/breakdown')
            
            # Calculate additional metrics
            total_cost = breakdown['total_cost']
            service_breakdown = breakdown['service_breakdown']
            
            # Calculate cost allocation percentages
            for service, data in service_breakdown.items():
                data['cost_percentage'] = (data['total_cost'] / total_cost) * 100
                
            # Identify top cost drivers
            top_services = sorted(
                service_breakdown.items(),
                key=lambda x: x[1]['total_cost'],
                reverse=True
            )[:5]
            
            return {
                'period_days': days,
                'total_cost': total_cost,
                'service_breakdown': service_breakdown,
                'top_cost_drivers': top_services,
                'analysis_timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error analyzing cost breakdown: {e}")
            raise
            
    def analyze_cost_trends(self, days: int = 30) -> Dict[str, Any]:
        """Analyze cost trends over time"""
        logger.info(f"Analyzing cost trends for last {days} days")
        
        try:
            trends = self._query_cost_analytics('/api/v1/cost/trends')
            
            # Calculate additional trend metrics
            daily_costs = trends['daily_costs']
            trend_data = trends['trend']
            
            # Calculate volatility
            costs = list(daily_costs.values())
            if len(costs) > 1:
                volatility = np.std(costs) / np.mean(costs)
            else:
                volatility = 0.0
                
            # Calculate cost acceleration
            if len(costs) >= 3:
                recent_slope = (costs[-1] - costs[-3]) / 2
                earlier_slope = (costs[-3] - costs[-5]) / 2 if len(costs) >= 5 else recent_slope
                acceleration = recent_slope - earlier_slope
            else:
                acceleration = 0.0
                
            return {
                'period_days': days,
                'daily_costs': daily_costs,
                'trend': trend_data,
                'volatility': volatility,
                'acceleration': acceleration,
                'total_cost': trends['total_cost'],
                'average_daily_cost': trends['average_daily_cost'],
                'analysis_timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error analyzing cost trends: {e}")
            raise
            
    def analyze_optimization_recommendations(self) -> Dict[str, Any]:
        """Analyze optimization recommendations"""
        logger.info("Analyzing optimization recommendations")
        
        try:
            recommendations = self._query_cost_analytics('/api/v1/cost/optimization')
            
            # Categorize recommendations by priority
            high_priority = [r for r in recommendations['recommendations'] if r['priority'] == 'high']
            medium_priority = [r for r in recommendations['recommendations'] if r['priority'] == 'medium']
            low_priority = [r for r in recommendations['recommendations'] if r['priority'] == 'low']
            
            # Calculate ROI for each recommendation
            for rec in recommendations['recommendations']:
                if rec['potential_savings'] > 0:
                    rec['roi'] = rec['potential_savings'] / 100.0  # Simplified ROI calculation
                else:
                    rec['roi'] = 0.0
                    
            # Sort by potential savings
            recommendations['recommendations'].sort(
                key=lambda x: x['potential_savings'],
                reverse=True
            )
            
            return {
                'recommendations': recommendations['recommendations'],
                'total_potential_savings': recommendations['total_potential_savings'],
                'recommendation_count': recommendations['recommendation_count'],
                'priority_breakdown': {
                    'high': len(high_priority),
                    'medium': len(medium_priority),
                    'low': len(low_priority)
                },
                'analysis_timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error analyzing optimization recommendations: {e}")
            raise
            
    def analyze_cost_efficiency(self, days: int = 7) -> Dict[str, Any]:
        """Analyze cost efficiency metrics"""
        logger.info(f"Analyzing cost efficiency for last {days} days")
        
        try:
            efficiency = self._query_cost_analytics('/api/v1/cost/efficiency')
            
            # Calculate efficiency score
            efficiency_metrics = efficiency['efficiency_metrics']
            total_cost = efficiency['total_cost']
            
            # Calculate efficiency score (lower is better)
            efficiency_score = sum(efficiency_metrics.values()) / len(efficiency_metrics)
            
            # Calculate efficiency trends
            end_time = datetime.now()
            start_time = end_time - timedelta(days=days)
            
            # Query historical efficiency data
            efficiency_query = 'cost_analytics_efficiency_score'
            efficiency_data = self._query_prometheus(efficiency_query, start_time, end_time)
            
            efficiency_trend = "stable"
            if efficiency_data and len(efficiency_data[0]['values']) > 1:
                values = [float(v[1]) for v in efficiency_data[0]['values']]
                if values[-1] > values[0] * 1.1:
                    efficiency_trend = "deteriorating"
                elif values[-1] < values[0] * 0.9:
                    efficiency_trend = "improving"
                    
            return {
                'period_days': days,
                'efficiency_metrics': efficiency_metrics,
                'efficiency_score': efficiency_score,
                'efficiency_trend': efficiency_trend,
                'total_cost': total_cost,
                'analysis_timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error analyzing cost efficiency: {e}")
            raise
            
    def analyze_budget_utilization(self, days: int = 7) -> Dict[str, Any]:
        """Analyze budget utilization"""
        logger.info(f"Analyzing budget utilization for last {days} days")
        
        try:
            # Get current cost breakdown
            breakdown = self.analyze_cost_breakdown(days)
            total_cost = breakdown['total_cost']
            
            # Calculate budget utilization
            daily_budget = self.config['budget_daily']
            weekly_budget = self.config['budget_weekly']
            monthly_budget = self.config['budget_monthly']
            
            daily_utilization = (total_cost / days) / daily_budget
            weekly_utilization = total_cost / weekly_budget
            monthly_utilization = (total_cost / days) * 30 / monthly_budget
            
            # Calculate budget burn rate
            burn_rate = total_cost / days
            
            # Calculate days until budget exhaustion
            days_until_daily_exhaustion = daily_budget / burn_rate if burn_rate > 0 else float('inf')
            days_until_weekly_exhaustion = weekly_budget / burn_rate if burn_rate > 0 else float('inf')
            days_until_monthly_exhaustion = monthly_budget / burn_rate if burn_rate > 0 else float('inf')
            
            return {
                'period_days': days,
                'total_cost': total_cost,
                'burn_rate': burn_rate,
                'budget_utilization': {
                    'daily': daily_utilization,
                    'weekly': weekly_utilization,
                    'monthly': monthly_utilization
                },
                'days_until_exhaustion': {
                    'daily': days_until_daily_exhaustion,
                    'weekly': days_until_weekly_exhaustion,
                    'monthly': days_until_monthly_exhaustion
                },
                'analysis_timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error analyzing budget utilization: {e}")
            raise
            
    def generate_cost_report(self, days: int = 7) -> Dict[str, Any]:
        """Generate comprehensive cost report"""
        logger.info(f"Generating comprehensive cost report for last {days} days")
        
        try:
            # Collect all analyses
            breakdown = self.analyze_cost_breakdown(days)
            trends = self.analyze_cost_trends(days)
            recommendations = self.analyze_optimization_recommendations()
            efficiency = self.analyze_cost_efficiency(days)
            budget = self.analyze_budget_utilization(days)
            
            # Calculate overall health score
            health_score = self._calculate_health_score(breakdown, trends, efficiency, budget)
            
            # Generate executive summary
            executive_summary = self._generate_executive_summary(
                breakdown, trends, recommendations, efficiency, budget, health_score
            )
            
            return {
                'report_period': {
                    'days': days,
                    'start_date': (datetime.now() - timedelta(days=days)).isoformat(),
                    'end_date': datetime.now().isoformat()
                },
                'executive_summary': executive_summary,
                'health_score': health_score,
                'cost_breakdown': breakdown,
                'cost_trends': trends,
                'optimization_recommendations': recommendations,
                'cost_efficiency': efficiency,
                'budget_utilization': budget,
                'generated_at': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error generating cost report: {e}")
            raise
            
    def _calculate_health_score(self, breakdown: Dict, trends: Dict, efficiency: Dict, budget: Dict) -> float:
        """Calculate overall cost health score (0-100)"""
        try:
            score = 100.0
            
            # Penalize high costs
            total_cost = breakdown['total_cost']
            if total_cost > 100:
                score -= min(20, (total_cost - 100) / 10)
                
            # Penalize high growth rate
            growth_rate = abs(trends['trend']['growth_rate'])
            if growth_rate > 0.5:
                score -= min(20, growth_rate * 40)
                
            # Penalize low efficiency
            efficiency_score = efficiency['efficiency_score']
            if efficiency_score > 0.1:
                score -= min(20, efficiency_score * 200)
                
            # Penalize high budget utilization
            budget_utilization = max(
                budget['budget_utilization']['daily'],
                budget['budget_utilization']['weekly'],
                budget['budget_utilization']['monthly']
            )
            if budget_utilization > 0.8:
                score -= min(20, (budget_utilization - 0.8) * 100)
                
            return max(0, score)
            
        except Exception as e:
            logger.error(f"Error calculating health score: {e}")
            return 50.0  # Default score
            
    def _generate_executive_summary(self, breakdown: Dict, trends: Dict, recommendations: Dict, 
                                  efficiency: Dict, budget: Dict, health_score: float) -> Dict[str, Any]:
        """Generate executive summary"""
        try:
            total_cost = breakdown['total_cost']
            growth_rate = trends['trend']['growth_rate']
            potential_savings = recommendations['total_potential_savings']
            efficiency_score = efficiency['efficiency_score']
            
            # Determine overall status
            if health_score >= 80:
                status = "excellent"
            elif health_score >= 60:
                status = "good"
            elif health_score >= 40:
                status = "fair"
            else:
                status = "poor"
                
            # Generate key insights
            insights = []
            
            if growth_rate > 0.2:
                insights.append(f"Cost growth rate is {growth_rate:.1%}, indicating increasing spend")
            elif growth_rate < -0.1:
                insights.append(f"Cost growth rate is {growth_rate:.1%}, indicating cost reduction")
                
            if potential_savings > total_cost * 0.1:
                insights.append(f"Optimization opportunities could save ${potential_savings:.2f}")
                
            if efficiency_score > 0.05:
                insights.append(f"Cost efficiency is below optimal (score: ${efficiency_score:.4f})")
                
            # Generate recommendations
            top_recommendations = recommendations['recommendations'][:3]
            
            return {
                'status': status,
                'health_score': health_score,
                'total_cost': total_cost,
                'growth_rate': growth_rate,
                'potential_savings': potential_savings,
                'efficiency_score': efficiency_score,
                'key_insights': insights,
                'top_recommendations': top_recommendations
            }
            
        except Exception as e:
            logger.error(f"Error generating executive summary: {e}")
            return {
                'status': 'unknown',
                'health_score': health_score,
                'error': str(e)
            }

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Cost Analysis Script for 254Carbon Observability')
    parser.add_argument('--config', '-c', required=True, help='Path to configuration file')
    parser.add_argument('--days', '-d', type=int, default=7, help='Number of days to analyze')
    parser.add_argument('--output', '-o', help='Output file path (JSON format)')
    parser.add_argument('--format', '-f', choices=['json', 'yaml'], default='json', help='Output format')
    parser.add_argument('--analysis', '-a', choices=['breakdown', 'trends', 'optimization', 'efficiency', 'budget', 'report'], 
                       default='report', help='Type of analysis to perform')
    
    args = parser.parse_args()
    
    try:
        # Initialize analyzer
        analyzer = CostAnalyzer(args.config)
        
        # Perform analysis
        if args.analysis == 'breakdown':
            result = analyzer.analyze_cost_breakdown(args.days)
        elif args.analysis == 'trends':
            result = analyzer.analyze_cost_trends(args.days)
        elif args.analysis == 'optimization':
            result = analyzer.analyze_optimization_recommendations()
        elif args.analysis == 'efficiency':
            result = analyzer.analyze_cost_efficiency(args.days)
        elif args.analysis == 'budget':
            result = analyzer.analyze_budget_utilization(args.days)
        else:  # report
            result = analyzer.generate_cost_report(args.days)
            
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
