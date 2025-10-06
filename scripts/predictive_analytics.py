#!/usr/bin/env python3
"""
Predictive Analytics Script for 254Carbon Observability

This script demonstrates predictive analytics operations including:
- Failure prediction models
- Capacity exhaustion forecasting
- Anomaly detection
- Performance optimization
- Model training and validation
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
import warnings
warnings.filterwarnings('ignore')

# Machine learning imports
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from scipy import stats
from scipy.stats import zscore

@dataclass
class PredictionResult:
    """Represents a prediction result"""
    timestamp: datetime
    prediction_type: str
    value: float
    confidence: float
    model: str
    features: Dict[str, float]
    metadata: Dict[str, Any]

@dataclass
class CapacityForecast:
    """Represents a capacity forecast"""
    resource_type: str
    current_usage: float
    forecasted_usage: List[float]
    forecast_horizon: int
    confidence_interval: Tuple[float, float]
    exhaustion_probability: float
    timestamp: datetime

@dataclass
class AnomalyDetection:
    """Represents an anomaly detection result"""
    timestamp: datetime
    anomaly_score: float
    anomaly_type: str
    features: Dict[str, float]
    algorithm: str
    confidence: float

class PredictiveAnalytics:
    """Performs predictive analytics operations"""
    
    def __init__(self, config_path: str = "k8s/predictive-analytics/predictive-analytics-config.yaml"):
        self.config_path = config_path
        self.models = {}
        self.predictions: List[PredictionResult] = []
        self.forecasts: List[CapacityForecast] = []
        self.anomalies: List[AnomalyDetection] = []
        self.load_config()
    
    def load_config(self) -> None:
        """Load predictive analytics configuration"""
        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            self.failure_prediction = config['data']['config.yaml']['failure_prediction']
            self.capacity_forecasting = config['data']['config.yaml']['capacity_forecasting']
            self.anomaly_detection = config['data']['config.yaml']['anomaly_detection']
            self.performance_optimization = config['data']['config.yaml']['performance_optimization']
            
            print(f"Loaded predictive analytics configuration")
            
        except FileNotFoundError:
            print(f"Configuration file not found: {self.config_path}")
            self._create_sample_config()
        except Exception as e:
            print(f"Error loading configuration: {e}")
    
    def _create_sample_config(self) -> None:
        """Create sample predictive analytics configuration"""
        self.failure_prediction = {
            "enabled": True,
            "models": {
                "service_failure": {
                    "algorithm": "random_forest",
                    "features": ["cpu_usage", "memory_usage", "error_rate", "latency"],
                    "confidence_threshold": 0.8
                }
            }
        }
        self.capacity_forecasting = {
            "enabled": True,
            "models": {
                "cpu_capacity": {
                    "algorithm": "linear_regression",
                    "features": ["cpu_usage", "request_rate", "pod_count"],
                    "forecast_horizon": 7
                }
            }
        }
        self.anomaly_detection = {
            "enabled": True,
            "models": {
                "statistical": {
                    "algorithm": "z_score",
                    "threshold": 3.0
                }
            }
        }
        self.performance_optimization = {
            "enabled": True,
            "models": {
                "auto_scaling": {
                    "algorithm": "reinforcement_learning",
                    "features": ["cpu_usage", "memory_usage", "request_rate"]
                }
            }
        }
        print("Created sample predictive analytics configuration")
    
    def generate_sample_data(self, duration_hours: int = 168) -> pd.DataFrame:
        """Generate sample time series data for training"""
        # Generate timestamps
        timestamps = pd.date_range(
            start=datetime.now() - timedelta(hours=duration_hours),
            end=datetime.now(),
            freq='5T'  # 5-minute intervals
        )
        
        # Generate synthetic metrics
        data = []
        for i, ts in enumerate(timestamps):
            # Base values with trends and seasonality
            base_cpu = 50 + 20 * np.sin(i * 2 * np.pi / 288)  # Daily seasonality
            base_memory = 60 + 15 * np.cos(i * 2 * np.pi / 144)  # Half-day seasonality
            base_error_rate = 0.02 + 0.01 * np.sin(i * 2 * np.pi / 576)  # Weekly seasonality
            
            # Add some noise
            cpu_usage = max(0, min(100, base_cpu + np.random.normal(0, 5)))
            memory_usage = max(0, min(100, base_memory + np.random.normal(0, 3)))
            error_rate = max(0, min(1, base_error_rate + np.random.normal(0, 0.005)))
            latency = max(10, 100 + 50 * np.sin(i * 2 * np.pi / 288) + np.random.normal(0, 20))
            request_rate = max(0, 1000 + 200 * np.sin(i * 2 * np.pi / 144) + np.random.normal(0, 50))
            pod_count = max(1, 5 + int(2 * np.sin(i * 2 * np.pi / 576) + np.random.normal(0, 0.5)))
            
            # Simulate occasional failures
            failure_probability = 0.05
            if np.random.random() < failure_probability:
                cpu_usage = min(100, cpu_usage + 30)
                memory_usage = min(100, memory_usage + 25)
                error_rate = min(1, error_rate + 0.3)
                latency = latency * 2
            
            data.append({
                'timestamp': ts,
                'cpu_usage': cpu_usage,
                'memory_usage': memory_usage,
                'error_rate': error_rate,
                'latency': latency,
                'request_rate': request_rate,
                'pod_count': pod_count,
                'disk_usage': 70 + 10 * np.sin(i * 2 * np.pi / 720) + np.random.normal(0, 2),
                'network_io': 100 + 50 * np.sin(i * 2 * np.pi / 288) + np.random.normal(0, 10)
            })
        
        return pd.DataFrame(data)
    
    def train_failure_prediction_model(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Train failure prediction model"""
        if not self.failure_prediction["enabled"]:
            return {"status": "disabled"}
        
        # Prepare features
        feature_columns = ['cpu_usage', 'memory_usage', 'error_rate', 'latency', 'request_rate', 'pod_count']
        X = data[feature_columns].values
        
        # Create failure labels (simplified: high error rate + high resource usage)
        y = ((data['error_rate'] > 0.1) | 
             (data['cpu_usage'] > 80) | 
             (data['memory_usage'] > 85)).astype(int)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train model
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train_scaled, y_train)
        
        # Evaluate model
        y_pred = model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        
        # Store model
        self.models['failure_prediction'] = {
            'model': model,
            'scaler': scaler,
            'features': feature_columns,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall
        }
        
        return {
            'status': 'trained',
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'feature_importance': dict(zip(feature_columns, model.feature_importances_))
        }
    
    def predict_failures(self, data: pd.DataFrame) -> List[PredictionResult]:
        """Predict service failures"""
        if 'failure_prediction' not in self.models:
            return []
        
        model_info = self.models['failure_prediction']
        model = model_info['model']
        scaler = model_info['scaler']
        features = model_info['features']
        
        predictions = []
        for _, row in data.iterrows():
            # Prepare features
            X = np.array([row[features].values])
            X_scaled = scaler.transform(X)
            
            # Make prediction
            failure_prob = model.predict_proba(X_scaled)[0][1]
            confidence = model.predict_proba(X_scaled)[0].max()
            
            prediction = PredictionResult(
                timestamp=row['timestamp'],
                prediction_type='service_failure',
                value=failure_prob,
                confidence=confidence,
                model='random_forest',
                features=row[features].to_dict(),
                metadata={'threshold': 0.5}
            )
            predictions.append(prediction)
        
        self.predictions.extend(predictions)
        return predictions
    
    def train_capacity_forecasting_model(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Train capacity forecasting model"""
        if not self.capacity_forecasting["enabled"]:
            return {"status": "disabled"}
        
        # Prepare features for CPU capacity forecasting
        feature_columns = ['cpu_usage', 'request_rate', 'pod_count']
        X = data[feature_columns].values
        y = data['cpu_usage'].values
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train model
        model = LinearRegression()
        model.fit(X_train, y_train)
        
        # Evaluate model
        y_pred = model.predict(X_test)
        mse = np.mean((y_test - y_pred) ** 2)
        r2_score = model.score(X_test, y_test)
        
        # Store model
        self.models['capacity_forecasting'] = {
            'model': model,
            'features': feature_columns,
            'mse': mse,
            'r2_score': r2_score
        }
        
        return {
            'status': 'trained',
            'mse': mse,
            'r2_score': r2_score,
            'coefficients': dict(zip(feature_columns, model.coef_))
        }
    
    def forecast_capacity(self, data: pd.DataFrame, horizon_hours: int = 24) -> List[CapacityForecast]:
        """Forecast capacity exhaustion"""
        if 'capacity_forecasting' not in self.models:
            return []
        
        model_info = self.models['capacity_forecasting']
        model = model_info['model']
        features = model_info['features']
        
        forecasts = []
        
        # Forecast for each resource type
        resource_types = ['cpu', 'memory', 'disk', 'network']
        
        for resource in resource_types:
            # Get current usage
            current_usage = data[resource + '_usage'].iloc[-1]
            
            # Generate forecast (simplified linear extrapolation)
            forecasted_usage = []
            for i in range(horizon_hours):
                # Simple trend-based forecast
                trend = 0.1 * i  # 0.1% increase per hour
                forecast_value = min(100, current_usage + trend + np.random.normal(0, 2))
                forecasted_usage.append(forecast_value)
            
            # Calculate exhaustion probability
            exhaustion_probability = sum(1 for val in forecasted_usage if val > 90) / len(forecasted_usage)
            
            # Confidence interval (simplified)
            confidence_interval = (
                np.mean(forecasted_usage) - 2 * np.std(forecasted_usage),
                np.mean(forecasted_usage) + 2 * np.std(forecasted_usage)
            )
            
            forecast = CapacityForecast(
                resource_type=resource,
                current_usage=current_usage,
                forecasted_usage=forecasted_usage,
                forecast_horizon=horizon_hours,
                confidence_interval=confidence_interval,
                exhaustion_probability=exhaustion_probability,
                timestamp=datetime.now()
            )
            forecasts.append(forecast)
        
        self.forecasts.extend(forecasts)
        return forecasts
    
    def detect_anomalies(self, data: pd.DataFrame) -> List[AnomalyDetection]:
        """Detect anomalies in the data"""
        if not self.anomaly_detection["enabled"]:
            return []
        
        anomalies = []
        
        # Statistical anomaly detection (Z-score)
        numeric_columns = ['cpu_usage', 'memory_usage', 'error_rate', 'latency', 'request_rate']
        
        for _, row in data.iterrows():
            anomaly_scores = {}
            anomaly_features = {}
            
            for col in numeric_columns:
                # Calculate Z-score for this column
                col_data = data[col].values
                z_scores = np.abs(zscore(col_data))
                current_z_score = z_scores[data.index.get_loc(row.name)]
                
                anomaly_scores[col] = current_z_score
                anomaly_features[col] = row[col]
            
            # Overall anomaly score
            overall_score = max(anomaly_scores.values())
            
            if overall_score > 3.0:  # Threshold for anomaly
                anomaly = AnomalyDetection(
                    timestamp=row['timestamp'],
                    anomaly_score=overall_score,
                    anomaly_type='statistical',
                    features=anomaly_features,
                    algorithm='z_score',
                    confidence=min(1.0, overall_score / 5.0)
                )
                anomalies.append(anomaly)
        
        # Machine learning anomaly detection (Isolation Forest)
        if len(data) > 100:  # Need sufficient data
            feature_columns = ['cpu_usage', 'memory_usage', 'error_rate', 'latency', 'request_rate']
            X = data[feature_columns].values
            
            # Train isolation forest
            iso_forest = IsolationForest(contamination=0.1, random_state=42)
            anomaly_labels = iso_forest.fit_predict(X)
            anomaly_scores = iso_forest.decision_function(X)
            
            for i, (_, row) in enumerate(data.iterrows()):
                if anomaly_labels[i] == -1:  # Anomaly detected
                    anomaly = AnomalyDetection(
                        timestamp=row['timestamp'],
                        anomaly_score=abs(anomaly_scores[i]),
                        anomaly_type='ml_anomaly',
                        features=row[feature_columns].to_dict(),
                        algorithm='isolation_forest',
                        confidence=min(1.0, abs(anomaly_scores[i]) / 2.0)
                    )
                    anomalies.append(anomaly)
        
        self.anomalies.extend(anomalies)
        return anomalies
    
    def optimize_performance(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Generate performance optimization recommendations"""
        if not self.performance_optimization["enabled"]:
            return {"status": "disabled"}
        
        recommendations = []
        
        # Analyze resource utilization
        avg_cpu = data['cpu_usage'].mean()
        avg_memory = data['memory_usage'].mean()
        avg_error_rate = data['error_rate'].mean()
        avg_latency = data['latency'].mean()
        
        # CPU optimization
        if avg_cpu > 80:
            recommendations.append({
                'type': 'cpu_optimization',
                'priority': 'high',
                'description': 'High CPU usage detected',
                'recommendation': 'Scale horizontally or optimize CPU-intensive operations',
                'potential_savings': '20-30%'
            })
        elif avg_cpu < 30:
            recommendations.append({
                'type': 'cpu_optimization',
                'priority': 'low',
                'description': 'Low CPU usage detected',
                'recommendation': 'Consider reducing CPU requests to save costs',
                'potential_savings': '10-15%'
            })
        
        # Memory optimization
        if avg_memory > 85:
            recommendations.append({
                'type': 'memory_optimization',
                'priority': 'high',
                'description': 'High memory usage detected',
                'recommendation': 'Increase memory limits or optimize memory usage',
                'potential_savings': '15-25%'
            })
        elif avg_memory < 40:
            recommendations.append({
                'type': 'memory_optimization',
                'priority': 'low',
                'description': 'Low memory usage detected',
                'recommendation': 'Consider reducing memory requests to save costs',
                'potential_savings': '8-12%'
            })
        
        # Error rate optimization
        if avg_error_rate > 0.05:
            recommendations.append({
                'type': 'error_optimization',
                'priority': 'high',
                'description': 'High error rate detected',
                'recommendation': 'Investigate and fix error sources',
                'potential_savings': '30-40%'
            })
        
        # Latency optimization
        if avg_latency > 500:
            recommendations.append({
                'type': 'latency_optimization',
                'priority': 'medium',
                'description': 'High latency detected',
                'recommendation': 'Optimize database queries and network calls',
                'potential_savings': '20-30%'
            })
        
        return {
            'status': 'completed',
            'recommendations': recommendations,
            'summary': {
                'total_recommendations': len(recommendations),
                'high_priority': len([r for r in recommendations if r['priority'] == 'high']),
                'medium_priority': len([r for r in recommendations if r['priority'] == 'medium']),
                'low_priority': len([r for r in recommendations if r['priority'] == 'low'])
            }
        }
    
    def generate_predictive_report(self) -> Dict[str, Any]:
        """Generate comprehensive predictive analytics report"""
        report = {
            "timestamp": datetime.now().isoformat(),
            "summary": {
                "total_predictions": len(self.predictions),
                "total_forecasts": len(self.forecasts),
                "total_anomalies": len(self.anomalies),
                "models_trained": len(self.models)
            },
            "failure_predictions": {
                "high_confidence_predictions": len([p for p in self.predictions if p.confidence > 0.8]),
                "average_confidence": np.mean([p.confidence for p in self.predictions]) if self.predictions else 0,
                "predictions_by_type": {}
            },
            "capacity_forecasts": {
                "exhaustion_probabilities": {},
                "forecast_horizons": {},
                "confidence_intervals": {}
            },
            "anomaly_detection": {
                "total_anomalies": len(self.anomalies),
                "anomalies_by_algorithm": {},
                "average_anomaly_score": np.mean([a.anomaly_score for a in self.anomalies]) if self.anomalies else 0
            },
            "model_performance": {},
            "recommendations": []
        }
        
        # Failure predictions summary
        if self.predictions:
            prediction_types = {}
            for pred in self.predictions:
                pred_type = pred.prediction_type
                if pred_type not in prediction_types:
                    prediction_types[pred_type] = 0
                prediction_types[pred_type] += 1
            report["failure_predictions"]["predictions_by_type"] = prediction_types
        
        # Capacity forecasts summary
        if self.forecasts:
            for forecast in self.forecasts:
                report["capacity_forecasts"]["exhaustion_probabilities"][forecast.resource_type] = forecast.exhaustion_probability
                report["capacity_forecasts"]["forecast_horizons"][forecast.resource_type] = forecast.forecast_horizon
                report["capacity_forecasts"]["confidence_intervals"][forecast.resource_type] = forecast.confidence_interval
        
        # Anomaly detection summary
        if self.anomalies:
            anomaly_algorithms = {}
            for anomaly in self.anomalies:
                algo = anomaly.algorithm
                if algo not in anomaly_algorithms:
                    anomaly_algorithms[algo] = 0
                anomaly_algorithms[algo] += 1
            report["anomaly_detection"]["anomalies_by_algorithm"] = anomaly_algorithms
        
        # Model performance
        for model_name, model_info in self.models.items():
            if 'accuracy' in model_info:
                report["model_performance"][model_name] = {
                    "accuracy": model_info['accuracy'],
                    "precision": model_info.get('precision', 0),
                    "recall": model_info.get('recall', 0)
                }
            elif 'r2_score' in model_info:
                report["model_performance"][model_name] = {
                    "r2_score": model_info['r2_score'],
                    "mse": model_info.get('mse', 0)
                }
        
        # Generate recommendations
        if self.forecasts:
            high_exhaustion = [f for f in self.forecasts if f.exhaustion_probability > 0.7]
            if high_exhaustion:
                report["recommendations"].append(f"High capacity exhaustion probability detected for {len(high_exhaustion)} resources")
        
        if self.anomalies:
            high_anomalies = [a for a in self.anomalies if a.anomaly_score > 3.0]
            if high_anomalies:
                report["recommendations"].append(f"{len(high_anomalies)} high-severity anomalies detected")
        
        if self.predictions:
            high_confidence = [p for p in self.predictions if p.confidence > 0.8]
            if high_confidence:
                report["recommendations"].append(f"{len(high_confidence)} high-confidence failure predictions")
        
        return report

def main():
    """Main function to demonstrate predictive analytics operations"""
    print("254Carbon Observability - Predictive Analytics Demo")
    print("=" * 60)
    
    # Initialize predictive analytics
    analyzer = PredictiveAnalytics()
    
    # Generate sample data
    print("\n1. Generating sample data...")
    data = analyzer.generate_sample_data(duration_hours=168)  # 1 week
    print(f"   - Generated {len(data)} data points")
    
    # Train models
    print("\n2. Training models...")
    failure_model_result = analyzer.train_failure_prediction_model(data)
    print(f"   - Failure prediction model: {failure_model_result['status']}")
    if failure_model_result['status'] == 'trained':
        print(f"     Accuracy: {failure_model_result['accuracy']:.3f}")
        print(f"     Precision: {failure_model_result['precision']:.3f}")
        print(f"     Recall: {failure_model_result['recall']:.3f}")
    
    capacity_model_result = analyzer.train_capacity_forecasting_model(data)
    print(f"   - Capacity forecasting model: {capacity_model_result['status']}")
    if capacity_model_result['status'] == 'trained':
        print(f"     R² Score: {capacity_model_result['r2_score']:.3f}")
        print(f"     MSE: {capacity_model_result['mse']:.3f}")
    
    # Make predictions
    print("\n3. Making predictions...")
    failure_predictions = analyzer.predict_failures(data)
    print(f"   - Generated {len(failure_predictions)} failure predictions")
    
    capacity_forecasts = analyzer.forecast_capacity(data, horizon_hours=24)
    print(f"   - Generated {len(capacity_forecasts)} capacity forecasts")
    
    # Detect anomalies
    print("\n4. Detecting anomalies...")
    anomalies = analyzer.detect_anomalies(data)
    print(f"   - Detected {len(anomalies)} anomalies")
    
    # Optimize performance
    print("\n5. Optimizing performance...")
    optimization_result = analyzer.optimize_performance(data)
    print(f"   - Performance optimization: {optimization_result['status']}")
    if optimization_result['status'] == 'completed':
        print(f"     Generated {len(optimization_result['recommendations'])} recommendations")
    
    # Generate report
    print("\n6. Generating predictive analytics report...")
    report = analyzer.generate_predictive_report()
    
    # Print summary
    print("\n7. Predictive Analytics Summary:")
    summary = report['summary']
    print(f"   - Total Predictions: {summary['total_predictions']}")
    print(f"   - Total Forecasts: {summary['total_forecasts']}")
    print(f"   - Total Anomalies: {summary['total_anomalies']}")
    print(f"   - Models Trained: {summary['models_trained']}")
    
    print("\n8. Failure Predictions:")
    failure_summary = report['failure_predictions']
    print(f"   - High Confidence Predictions: {failure_summary['high_confidence_predictions']}")
    print(f"   - Average Confidence: {failure_summary['average_confidence']:.3f}")
    
    print("\n9. Capacity Forecasts:")
    capacity_summary = report['capacity_forecasts']
    for resource, prob in capacity_summary['exhaustion_probabilities'].items():
        print(f"   - {resource.title()} Exhaustion Probability: {prob:.3f}")
    
    print("\n10. Anomaly Detection:")
    anomaly_summary = report['anomaly_detection']
    print(f"   - Total Anomalies: {anomaly_summary['total_anomalies']}")
    print(f"   - Average Anomaly Score: {anomaly_summary['average_anomaly_score']:.3f}")
    
    print("\n11. Recommendations:")
    for i, rec in enumerate(report['recommendations'], 1):
        print(f"   {i}. {rec}")
    
    # Save detailed report
    with open('predictive_analytics_report.json', 'w') as f:
        json.dump(report, f, indent=2, default=str)
    print("\n12. Detailed report saved to predictive_analytics_report.json")
    
    print("\nPredictive analytics demo completed!")

if __name__ == "__main__":
    main()
