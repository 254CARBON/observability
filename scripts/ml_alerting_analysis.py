#!/usr/bin/env python3
"""
ML Alerting Analysis Script for 254Carbon Observability

This script provides ML alerting analysis capabilities including:
- Model performance analysis
- Prediction accuracy analysis
- Model drift detection
- Feature importance analysis
- Training performance analysis
- Model health monitoring
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
import pickle
import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import seaborn as sns

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class MLModelPerformance:
    """ML model performance data structure"""
    model_name: str
    model_type: str
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    auc_roc: float
    mae: float
    rmse: float
    mape: float
    training_duration: float
    prediction_latency: float
    model_drift: float
    feature_importance: Dict[str, float]
    last_trained: datetime
    version: str

@dataclass
class MLAlertingConfig:
    """Configuration for ML alerting analysis"""
    ml_alerting_endpoint: str
    prometheus_endpoint: str
    model_storage_path: str
    analysis_time_window: str
    performance_thresholds: Dict[str, float]
    drift_thresholds: Dict[str, float]
    accuracy_thresholds: Dict[str, float]

class MLAlertingAnalyzer:
    """Main ML alerting analyzer"""
    
    def __init__(self, config_path: str):
        self.config = self._load_config(config_path)
        self.ml_alerting_endpoint = self.config['ml_alerting_endpoint']
        self.prometheus_endpoint = self.config['prometheus_endpoint']
        self.model_storage_path = self.config['model_storage_path']
        
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
            
    def _query_ml_alerting(self, endpoint: str, data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Query ML alerting service"""
        try:
            if data:
                response = requests.post(
                    f"{self.ml_alerting_endpoint}{endpoint}",
                    json=data,
                    timeout=30
                )
            else:
                response = requests.get(
                    f"{self.ml_alerting_endpoint}{endpoint}",
                    timeout=30
                )
            response.raise_for_status()
            return response.json()
            
        except Exception as e:
            logger.error(f"ML alerting query error: {e}")
            raise
            
    def analyze_model_performance(self, hours: int = 24) -> Dict[str, Any]:
        """Analyze ML model performance"""
        logger.info(f"Analyzing ML model performance for last {hours} hours")
        
        try:
            # Get model status from ML alerting service
            model_status = self._query_ml_alerting('/api/v1/models/status')
            
            # Get model performance from ML alerting service
            model_performance = self._query_ml_alerting('/api/v1/models/performance')
            
            # Query Prometheus for additional metrics
            end_time = datetime.now()
            start_time = end_time - timedelta(hours=hours)
            
            # Get prediction accuracy
            accuracy_query = 'ml_alerting_prediction_accuracy:avg'
            accuracy_data = self._query_prometheus(accuracy_query, start_time, end_time)
            
            # Get prediction latency
            latency_query = 'ml_alerting_prediction_latency:avg'
            latency_data = self._query_prometheus(latency_query, start_time, end_time)
            
            # Get model drift
            drift_query = 'ml_alerting_model_drift:avg'
            drift_data = self._query_prometheus(drift_query, start_time, end_time)
            
            # Analyze performance metrics
            performance_analysis = {}
            
            for model_name, status in model_status['models'].items():
                performance_analysis[model_name] = {
                    'model_type': status['model_type'],
                    'version': status['version'],
                    'last_trained': status['last_trained'],
                    'performance_metrics': status['performance_metrics'],
                    'features_count': status['features_count'],
                    'status': status['status']
                }
                
                # Add Prometheus metrics
                if accuracy_data:
                    for result in accuracy_data:
                        if result['metric'].get('model_type') == model_name:
                            performance_analysis[model_name]['prometheus_accuracy'] = [
                                float(v[1]) for v in result['values']
                            ]
                            
                if latency_data:
                    for result in latency_data:
                        if result['metric'].get('model_type') == model_name:
                            performance_analysis[model_name]['prometheus_latency'] = [
                                float(v[1]) for v in result['values']
                            ]
                            
                if drift_data:
                    for result in drift_data:
                        if result['metric'].get('model_type') == model_name:
                            performance_analysis[model_name]['prometheus_drift'] = [
                                float(v[1]) for v in result['values']
                            ]
                            
            return {
                'performance_analysis': performance_analysis,
                'total_models': len(performance_analysis),
                'analysis_period_hours': hours,
                'analysis_timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error analyzing model performance: {e}")
            raise
            
    def analyze_prediction_accuracy(self, hours: int = 24) -> Dict[str, Any]:
        """Analyze prediction accuracy"""
        logger.info(f"Analyzing prediction accuracy for last {hours} hours")
        
        try:
            end_time = datetime.now()
            start_time = end_time - timedelta(hours=hours)
            
            # Query Prometheus for accuracy metrics
            accuracy_query = 'ml_alerting_prediction_accuracy:avg'
            accuracy_data = self._query_prometheus(accuracy_query, start_time, end_time)
            
            # Query Prometheus for confidence metrics
            confidence_query = 'ml_alerting_prediction_confidence:avg'
            confidence_data = self._query_prometheus(confidence_query, start_time, end_time)
            
            # Analyze accuracy metrics
            accuracy_analysis = {}
            
            for result in accuracy_data:
                model_type = result['metric'].get('model_type', 'unknown')
                prediction_type = result['metric'].get('prediction_type', 'unknown')
                
                if model_type not in accuracy_analysis:
                    accuracy_analysis[model_type] = {}
                    
                values = [float(v[1]) for v in result['values']]
                accuracy_analysis[model_type][prediction_type] = {
                    'values': values,
                    'average': np.mean(values),
                    'min': np.min(values),
                    'max': np.max(values),
                    'std': np.std(values),
                    'trend': self._calculate_trend(values)
                }
                
            # Analyze confidence metrics
            confidence_analysis = {}
            
            for result in confidence_data:
                model_type = result['metric'].get('model_type', 'unknown')
                prediction_type = result['metric'].get('prediction_type', 'unknown')
                
                if model_type not in confidence_analysis:
                    confidence_analysis[model_type] = {}
                    
                values = [float(v[1]) for v in result['values']]
                confidence_analysis[model_type][prediction_type] = {
                    'values': values,
                    'average': np.mean(values),
                    'min': np.min(values),
                    'max': np.max(values),
                    'std': np.std(values),
                    'trend': self._calculate_trend(values)
                }
                
            return {
                'accuracy_analysis': accuracy_analysis,
                'confidence_analysis': confidence_analysis,
                'analysis_period_hours': hours,
                'analysis_timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error analyzing prediction accuracy: {e}")
            raise
            
    def analyze_model_drift(self, hours: int = 168) -> Dict[str, Any]:
        """Analyze model drift"""
        logger.info(f"Analyzing model drift for last {hours} hours")
        
        try:
            end_time = datetime.now()
            start_time = end_time - timedelta(hours=hours)
            
            # Query Prometheus for drift metrics
            drift_query = 'ml_alerting_model_drift:avg'
            drift_data = self._query_prometheus(drift_query, start_time, end_time)
            
            # Query Prometheus for feature importance
            feature_importance_query = 'ml_alerting_feature_importance:avg'
            feature_importance_data = self._query_prometheus(feature_importance_query, start_time, end_time)
            
            # Analyze drift metrics
            drift_analysis = {}
            
            for result in drift_data:
                model_type = result['metric'].get('model_type', 'unknown')
                values = [float(v[1]) for v in result['values']]
                
                drift_analysis[model_type] = {
                    'values': values,
                    'average': np.mean(values),
                    'min': np.min(values),
                    'max': np.max(values),
                    'std': np.std(values),
                    'trend': self._calculate_trend(values),
                    'drift_score': np.mean(values),
                    'drift_threshold': 0.1,  # Default threshold
                    'drift_detected': np.mean(values) > 0.1
                }
                
            # Analyze feature importance
            feature_importance_analysis = {}
            
            for result in feature_importance_data:
                model_type = result['metric'].get('model_type', 'unknown')
                feature = result['metric'].get('feature', 'unknown')
                
                if model_type not in feature_importance_analysis:
                    feature_importance_analysis[model_type] = {}
                    
                values = [float(v[1]) for v in result['values']]
                feature_importance_analysis[model_type][feature] = {
                    'values': values,
                    'average': np.mean(values),
                    'min': np.min(values),
                    'max': np.max(values),
                    'std': np.std(values),
                    'trend': self._calculate_trend(values)
                }
                
            return {
                'drift_analysis': drift_analysis,
                'feature_importance_analysis': feature_importance_analysis,
                'analysis_period_hours': hours,
                'analysis_timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error analyzing model drift: {e}")
            raise
            
    def analyze_training_performance(self, hours: int = 168) -> Dict[str, Any]:
        """Analyze training performance"""
        logger.info(f"Analyzing training performance for last {hours} hours")
        
        try:
            end_time = datetime.now()
            start_time = end_time - timedelta(hours=hours)
            
            # Query Prometheus for training metrics
            training_duration_query = 'ml_alerting_training_duration:avg'
            training_duration_data = self._query_prometheus(training_duration_query, start_time, end_time)
            
            training_errors_query = 'ml_alerting_training_errors:rate5m'
            training_errors_data = self._query_prometheus(training_errors_query, start_time, end_time)
            
            # Analyze training performance
            training_analysis = {}
            
            for result in training_duration_data:
                model_type = result['metric'].get('model_type', 'unknown')
                values = [float(v[1]) for v in result['values']]
                
                training_analysis[model_type] = {
                    'training_duration': {
                        'values': values,
                        'average': np.mean(values),
                        'min': np.min(values),
                        'max': np.max(values),
                        'std': np.std(values),
                        'trend': self._calculate_trend(values)
                    }
                }
                
            # Analyze training errors
            for result in training_errors_data:
                model_type = result['metric'].get('model_type', 'unknown')
                error_type = result['metric'].get('error_type', 'unknown')
                
                if model_type not in training_analysis:
                    training_analysis[model_type] = {}
                    
                if 'training_errors' not in training_analysis[model_type]:
                    training_analysis[model_type]['training_errors'] = {}
                    
                values = [float(v[1]) for v in result['values']]
                training_analysis[model_type]['training_errors'][error_type] = {
                    'values': values,
                    'average': np.mean(values),
                    'min': np.min(values),
                    'max': np.max(values),
                    'std': np.std(values),
                    'trend': self._calculate_trend(values)
                }
                
            return {
                'training_analysis': training_analysis,
                'analysis_period_hours': hours,
                'analysis_timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error analyzing training performance: {e}")
            raise
            
    def analyze_model_health(self, hours: int = 24) -> Dict[str, Any]:
        """Analyze model health"""
        logger.info(f"Analyzing model health for last {hours} hours")
        
        try:
            end_time = datetime.now()
            start_time = end_time - timedelta(hours=hours)
            
            # Query Prometheus for health metrics
            health_query = 'ml_alerting_model_health_score'
            health_data = self._query_prometheus(health_query, start_time, end_time)
            
            performance_query = 'ml_alerting_model_performance_score'
            performance_data = self._query_prometheus(performance_query, start_time, end_time)
            
            efficiency_query = 'ml_alerting_model_efficiency_score'
            efficiency_data = self._query_prometheus(efficiency_query, start_time, end_time)
            
            quality_query = 'ml_alerting_model_quality_score'
            quality_data = self._query_prometheus(quality_query, start_time, end_time)
            
            # Analyze health metrics
            health_analysis = {}
            
            # Health score
            if health_data:
                for result in health_data:
                    values = [float(v[1]) for v in result['values']]
                    health_analysis['health_score'] = {
                        'values': values,
                        'average': np.mean(values),
                        'min': np.min(values),
                        'max': np.max(values),
                        'std': np.std(values),
                        'trend': self._calculate_trend(values),
                        'status': 'healthy' if np.mean(values) > 0.7 else 'unhealthy'
                    }
                    
            # Performance score
            if performance_data:
                for result in performance_data:
                    values = [float(v[1]) for v in result['values']]
                    health_analysis['performance_score'] = {
                        'values': values,
                        'average': np.mean(values),
                        'min': np.min(values),
                        'max': np.max(values),
                        'std': np.std(values),
                        'trend': self._calculate_trend(values),
                        'status': 'good' if np.mean(values) > 0.6 else 'poor'
                    }
                    
            # Efficiency score
            if efficiency_data:
                for result in efficiency_data:
                    values = [float(v[1]) for v in result['values']]
                    health_analysis['efficiency_score'] = {
                        'values': values,
                        'average': np.mean(values),
                        'min': np.min(values),
                        'max': np.max(values),
                        'std': np.std(values),
                        'trend': self._calculate_trend(values),
                        'status': 'efficient' if np.mean(values) > 10.0 else 'inefficient'
                    }
                    
            # Quality score
            if quality_data:
                for result in quality_data:
                    values = [float(v[1]) for v in result['values']]
                    health_analysis['quality_score'] = {
                        'values': values,
                        'average': np.mean(values),
                        'min': np.min(values),
                        'max': np.max(values),
                        'std': np.std(values),
                        'trend': self._calculate_trend(values),
                        'status': 'high' if np.mean(values) > 0.7 else 'low'
                    }
                    
            # Calculate overall health score
            overall_health = np.mean([
                health_analysis.get('health_score', {}).get('average', 0.5),
                health_analysis.get('performance_score', {}).get('average', 0.5),
                health_analysis.get('efficiency_score', {}).get('average', 0.5) / 10.0,  # Normalize
                health_analysis.get('quality_score', {}).get('average', 0.5)
            ])
            
            health_analysis['overall_health'] = {
                'score': overall_health,
                'status': 'healthy' if overall_health > 0.7 else 'unhealthy',
                'recommendations': self._generate_health_recommendations(health_analysis)
            }
            
            return {
                'health_analysis': health_analysis,
                'analysis_period_hours': hours,
                'analysis_timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error analyzing model health: {e}")
            raise
            
    def generate_ml_alerting_report(self, hours: int = 24) -> Dict[str, Any]:
        """Generate comprehensive ML alerting report"""
        logger.info(f"Generating comprehensive ML alerting report for last {hours} hours")
        
        try:
            # Collect all analyses
            performance = self.analyze_model_performance(hours)
            accuracy = self.analyze_prediction_accuracy(hours)
            drift = self.analyze_model_drift(hours * 7)  # Use 7x hours for drift
            training = self.analyze_training_performance(hours * 7)  # Use 7x hours for training
            health = self.analyze_model_health(hours)
            
            # Calculate overall ML score
            ml_score = self._calculate_overall_ml_score(performance, accuracy, drift, training, health)
            
            # Generate executive summary
            executive_summary = self._generate_ml_executive_summary(
                performance, accuracy, drift, training, health, ml_score
            )
            
            return {
                'report_period': {
                    'hours': hours,
                    'start_date': (datetime.now() - timedelta(hours=hours)).isoformat(),
                    'end_date': datetime.now().isoformat()
                },
                'executive_summary': executive_summary,
                'ml_score': ml_score,
                'performance_analysis': performance,
                'accuracy_analysis': accuracy,
                'drift_analysis': drift,
                'training_analysis': training,
                'health_analysis': health,
                'generated_at': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error generating ML alerting report: {e}")
            raise
            
    def _calculate_trend(self, values: List[float]) -> str:
        """Calculate trend direction"""
        if len(values) < 2:
            return 'stable'
            
        # Simple linear trend
        x = np.arange(len(values))
        slope = np.polyfit(x, values, 1)[0]
        
        if slope > 0.01:
            return 'increasing'
        elif slope < -0.01:
            return 'decreasing'
        else:
            return 'stable'
            
    def _generate_health_recommendations(self, health_analysis: Dict[str, Any]) -> List[str]:
        """Generate health recommendations"""
        recommendations = []
        
        # Health score recommendations
        if health_analysis.get('health_score', {}).get('average', 0) < 0.7:
            recommendations.append("Improve model health score by retraining models")
            
        # Performance score recommendations
        if health_analysis.get('performance_score', {}).get('average', 0) < 0.6:
            recommendations.append("Optimize model performance by tuning hyperparameters")
            
        # Efficiency score recommendations
        if health_analysis.get('efficiency_score', {}).get('average', 0) < 10.0:
            recommendations.append("Improve model efficiency by optimizing features")
            
        # Quality score recommendations
        if health_analysis.get('quality_score', {}).get('average', 0) < 0.7:
            recommendations.append("Enhance model quality by improving data quality")
            
        return recommendations
        
    def _calculate_overall_ml_score(self, performance: Dict, accuracy: Dict, drift: Dict, 
                                  training: Dict, health: Dict) -> float:
        """Calculate overall ML score"""
        try:
            score = 0.0
            
            # Performance score (30%)
            if performance['performance_analysis']:
                perf_scores = []
                for model_name, model_data in performance['performance_analysis'].items():
                    if 'prometheus_accuracy' in model_data:
                        perf_scores.append(np.mean(model_data['prometheus_accuracy']))
                if perf_scores:
                    score += np.mean(perf_scores) * 0.3
                    
            # Accuracy score (25%)
            if accuracy['accuracy_analysis']:
                acc_scores = []
                for model_type, model_data in accuracy['accuracy_analysis'].items():
                    for prediction_type, pred_data in model_data.items():
                        acc_scores.append(pred_data['average'])
                if acc_scores:
                    score += np.mean(acc_scores) * 0.25
                    
            # Drift score (20%)
            if drift['drift_analysis']:
                drift_scores = []
                for model_type, model_data in drift['drift_analysis'].items():
                    drift_scores.append(1.0 - model_data['drift_score'])  # Invert drift score
                if drift_scores:
                    score += np.mean(drift_scores) * 0.2
                    
            # Training score (15%)
            if training['training_analysis']:
                training_scores = []
                for model_type, model_data in training['training_analysis'].items():
                    if 'training_errors' in model_data:
                        error_rates = []
                        for error_type, error_data in model_data['training_errors'].items():
                            error_rates.append(error_data['average'])
                        if error_rates:
                            training_scores.append(1.0 - np.mean(error_rates))  # Invert error rate
                if training_scores:
                    score += np.mean(training_scores) * 0.15
                    
            # Health score (10%)
            if health['health_analysis'].get('overall_health', {}).get('score'):
                score += health['health_analysis']['overall_health']['score'] * 0.1
                
            return min(1.0, score)
            
        except Exception as e:
            logger.error(f"Error calculating overall ML score: {e}")
            return 0.5  # Default score
            
    def _generate_ml_executive_summary(self, performance: Dict, accuracy: Dict, drift: Dict,
                                      training: Dict, health: Dict, ml_score: float) -> Dict[str, Any]:
        """Generate executive summary for ML alerting"""
        try:
            # Extract key metrics
            total_models = performance['total_models']
            health_status = health['health_analysis'].get('overall_health', {}).get('status', 'unknown')
            
            # Calculate average accuracy
            avg_accuracy = 0.0
            if accuracy['accuracy_analysis']:
                acc_scores = []
                for model_type, model_data in accuracy['accuracy_analysis'].items():
                    for prediction_type, pred_data in model_data.items():
                        acc_scores.append(pred_data['average'])
                if acc_scores:
                    avg_accuracy = np.mean(acc_scores)
                    
            # Calculate average drift
            avg_drift = 0.0
            if drift['drift_analysis']:
                drift_scores = []
                for model_type, model_data in drift['drift_analysis'].items():
                    drift_scores.append(model_data['drift_score'])
                if drift_scores:
                    avg_drift = np.mean(drift_scores)
                    
            # Determine overall status
            if ml_score >= 0.8:
                status = "excellent"
            elif ml_score >= 0.6:
                status = "good"
            elif ml_score >= 0.4:
                status = "fair"
            else:
                status = "poor"
                
            # Generate key insights
            insights = []
            
            if avg_accuracy > 0.8:
                insights.append(f"High prediction accuracy achieved (average: {avg_accuracy:.1%})")
            elif avg_accuracy < 0.6:
                insights.append(f"Low prediction accuracy detected (average: {avg_accuracy:.1%})")
                
            if avg_drift > 0.1:
                insights.append(f"Model drift detected (average: {avg_drift:.2f})")
            elif avg_drift < 0.05:
                insights.append(f"Models are stable (drift: {avg_drift:.2f})")
                
            if health_status == 'healthy':
                insights.append("All models are in healthy state")
            elif health_status == 'unhealthy':
                insights.append("Some models require attention")
                
            # Generate recommendations
            recommendations = []
            
            if avg_accuracy < 0.7:
                recommendations.append("Retrain models to improve accuracy")
                
            if avg_drift > 0.1:
                recommendations.append("Address model drift by updating training data")
                
            if health_status == 'unhealthy':
                recommendations.append("Investigate and fix model health issues")
                
            return {
                'status': status,
                'ml_score': ml_score,
                'total_models': total_models,
                'average_accuracy': avg_accuracy,
                'average_drift': avg_drift,
                'health_status': health_status,
                'key_insights': insights,
                'recommendations': recommendations
            }
            
        except Exception as e:
            logger.error(f"Error generating ML executive summary: {e}")
            return {
                'status': 'unknown',
                'ml_score': ml_score,
                'error': str(e)
            }

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='ML Alerting Analysis Script for 254Carbon Observability')
    parser.add_argument('--config', '-c', required=True, help='Path to configuration file')
    parser.add_argument('--hours', '-h', type=int, default=24, help='Number of hours to analyze')
    parser.add_argument('--output', '-o', help='Output file path (JSON format)')
    parser.add_argument('--format', '-f', choices=['json', 'yaml'], default='json', help='Output format')
    parser.add_argument('--analysis', '-a', choices=['performance', 'accuracy', 'drift', 'training', 'health', 'report'], 
                       default='report', help='Type of analysis to perform')
    
    args = parser.parse_args()
    
    try:
        # Initialize analyzer
        analyzer = MLAlertingAnalyzer(args.config)
        
        # Perform analysis
        if args.analysis == 'performance':
            result = analyzer.analyze_model_performance(args.hours)
        elif args.analysis == 'accuracy':
            result = analyzer.analyze_prediction_accuracy(args.hours)
        elif args.analysis == 'drift':
            result = analyzer.analyze_model_drift(args.hours)
        elif args.analysis == 'training':
            result = analyzer.analyze_training_performance(args.hours)
        elif args.analysis == 'health':
            result = analyzer.analyze_model_health(args.hours)
        else:  # report
            result = analyzer.generate_ml_alerting_report(args.hours)
            
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
