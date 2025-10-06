#!/usr/bin/env python3
"""
Visualization Analysis Script for 254Carbon Observability

This script demonstrates advanced visualization analysis operations including:
- 3D dependency graph generation
- Heatmap data processing
- Interactive visualization data preparation
- Performance metrics visualization
"""

import json
import yaml
import requests
import time
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import networkx as nx
from scipy import stats

@dataclass
class ServiceNode:
    """Represents a service node in the dependency graph"""
    id: str
    name: str
    status: str
    position: Tuple[float, float, float]
    metrics: Dict[str, float]
    dependencies: List[str]

@dataclass
class ServiceEdge:
    """Represents a dependency edge between services"""
    source: str
    target: str
    weight: float
    latency: float
    error_rate: float

@dataclass
class HeatmapData:
    """Represents heatmap data point"""
    x: str
    y: str
    value: float
    timestamp: datetime
    metadata: Dict[str, Any]

class VisualizationAnalyzer:
    """Analyzes and generates visualization data"""
    
    def __init__(self, config_path: str = "visualizations/config.yaml"):
        self.config_path = config_path
        self.services: List[ServiceNode] = []
        self.edges: List[ServiceEdge] = []
        self.heatmap_data: List[HeatmapData] = []
        self.load_config()
    
    def load_config(self) -> None:
        """Load visualization configuration"""
        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            self.visualization_settings = config.get('visualization', {})
            self.color_schemes = config.get('color_schemes', {})
            self.layout_algorithms = config.get('layout_algorithms', {})
            
            print(f"Loaded visualization configuration")
            
        except FileNotFoundError:
            print(f"Configuration file not found: {self.config_path}")
            self._create_sample_config()
        except Exception as e:
            print(f"Error loading configuration: {e}")
    
    def _create_sample_config(self) -> None:
        """Create sample visualization configuration"""
        self.visualization_settings = {
            "node_size_range": [0.5, 3.0],
            "edge_thickness_range": [0.1, 5.0],
            "animation_speed_range": [0.1, 3.0],
            "color_schemes": ["viridis", "plasma", "inferno", "magma"]
        }
        self.color_schemes = {
            "healthy": "#00ff00",
            "warning": "#ffaa00",
            "critical": "#ff0000"
        }
        self.layout_algorithms = {
            "circular": {"radius": 15, "height_variation": 10},
            "force_directed": {"iterations": 100, "k": 1.0},
            "hierarchical": {"levels": 3, "spacing": 8}
        }
        print("Created sample visualization configuration")
    
    def generate_sample_services(self, count: int = 8) -> List[ServiceNode]:
        """Generate sample service data for visualization"""
        service_names = [
            "API Gateway", "Auth Service", "User Service", "Order Service",
            "Payment Service", "Inventory Service", "Notification Service", "Analytics Service"
        ]
        
        services = []
        for i in range(min(count, len(service_names))):
            # Generate random metrics
            requests = np.random.uniform(50, 1000)
            errors = np.random.uniform(0, 50)
            latency = np.random.uniform(10, 200)
            
            # Determine status based on metrics
            if errors > 30 or latency > 150:
                status = "critical"
            elif errors > 10 or latency > 100:
                status = "warning"
            else:
                status = "healthy"
            
            service = ServiceNode(
                id=f"service-{i}",
                name=service_names[i],
                status=status,
                position=(0, 0, 0),  # Will be calculated by layout
                metrics={
                    "requests": requests,
                    "errors": errors,
                    "latency": latency,
                    "cpu_usage": np.random.uniform(10, 90),
                    "memory_usage": np.random.uniform(20, 80)
                },
                dependencies=[]
            )
            services.append(service)
        
        self.services = services
        return services
    
    def generate_sample_dependencies(self) -> List[ServiceEdge]:
        """Generate sample dependency edges"""
        dependencies = [
            ("service-0", "service-1", 0.8),  # Gateway -> Auth
            ("service-0", "service-2", 0.6),  # Gateway -> User
            ("service-0", "service-3", 0.4),  # Gateway -> Order
            ("service-3", "service-4", 0.7),  # Order -> Payment
            ("service-3", "service-5", 0.9),  # Order -> Inventory
            ("service-2", "service-6", 0.3),  # User -> Notification
            ("service-0", "service-7", 0.2),  # Gateway -> Analytics
        ]
        
        edges = []
        for source, target, weight in dependencies:
            latency = np.random.uniform(10, 100)
            error_rate = np.random.uniform(0, 5)
            
            edge = ServiceEdge(
                source=source,
                target=target,
                weight=weight,
                latency=latency,
                error_rate=error_rate
            )
            edges.append(edge)
        
        self.edges = edges
        return edges
    
    def calculate_circular_layout(self, radius: float = 15) -> None:
        """Calculate circular layout for services"""
        angle_step = (2 * np.pi) / len(self.services)
        
        for i, service in enumerate(self.services):
            angle = i * angle_step
            x = np.cos(angle) * radius
            z = np.sin(angle) * radius
            y = np.random.uniform(-5, 5)  # Random height
            
            service.position = (x, y, z)
    
    def calculate_force_directed_layout(self, iterations: int = 100) -> None:
        """Calculate force-directed layout for services"""
        # Initialize random positions
        for service in self.services:
            x = np.random.uniform(-10, 10)
            y = np.random.uniform(-5, 5)
            z = np.random.uniform(-10, 10)
            service.position = (x, y, z)
        
        # Force-directed algorithm
        k = np.sqrt((4 * np.pi * 15 * 15) / len(self.services))
        
        for iteration in range(iterations):
            for service in self.services:
                fx, fy, fz = 0, 0, 0
                
                # Repulsive forces from other nodes
                for other in self.services:
                    if service != other:
                        dx = service.position[0] - other.position[0]
                        dy = service.position[1] - other.position[1]
                        dz = service.position[2] - other.position[2]
                        distance = np.sqrt(dx*dx + dy*dy + dz*dz)
                        
                        if distance > 0:
                            force = (k * k) / distance
                            fx += (dx / distance) * force
                            fy += (dy / distance) * force
                            fz += (dz / distance) * force
                
                # Attractive forces from edges
                for edge in self.edges:
                    if edge.source == service.id:
                        target_service = next((s for s in self.services if s.id == edge.target), None)
                        if target_service:
                            dx = target_service.position[0] - service.position[0]
                            dy = target_service.position[1] - service.position[1]
                            dz = target_service.position[2] - service.position[2]
                            distance = np.sqrt(dx*dx + dy*dy + dz*dz)
                            
                            if distance > 0:
                                force = (distance * distance) / k
                                fx += (dx / distance) * force
                                fy += (dy / distance) * force
                                fz += (dz / distance) * force
                
                # Apply forces
                service.position = (
                    service.position[0] + fx * 0.01,
                    service.position[1] + fy * 0.01,
                    service.position[2] + fz * 0.01
                )
    
    def calculate_hierarchical_layout(self, levels: int = 3) -> None:
        """Calculate hierarchical layout for services"""
        nodes_per_level = len(self.services) // levels
        
        for i, service in enumerate(self.services):
            level = i // nodes_per_level
            position_in_level = i % nodes_per_level
            angle = (position_in_level / nodes_per_level) * 2 * np.pi
            radius = 5 + level * 5
            
            x = np.cos(angle) * radius
            z = np.sin(angle) * radius
            y = level * 8 - 8
            
            service.position = (x, y, z)
    
    def generate_heatmap_data(self, duration_hours: int = 24) -> List[HeatmapData]:
        """Generate sample heatmap data"""
        heatmap_data = []
        base_time = datetime.now() - timedelta(hours=duration_hours)
        
        # Generate data for service interactions
        for hour in range(duration_hours):
            for service in self.services:
                for other_service in self.services:
                    if service != other_service:
                        # Generate interaction value based on time and service characteristics
                        time_factor = np.sin(hour * np.pi / 12)  # Daily pattern
                        service_factor = service.metrics["requests"] / 1000
                        interaction_value = time_factor * service_factor * np.random.uniform(0.5, 1.5)
                        
                        data_point = HeatmapData(
                            x=service.name,
                            y=other_service.name,
                            value=max(0, interaction_value),
                            timestamp=base_time + timedelta(hours=hour),
                            metadata={
                                "source_service": service.id,
                                "target_service": other_service.id,
                                "interaction_type": "api_call"
                            }
                        )
                        heatmap_data.append(data_point)
        
        self.heatmap_data = heatmap_data
        return heatmap_data
    
    def export_3d_visualization_data(self) -> Dict[str, Any]:
        """Export data for 3D visualization"""
        nodes_data = []
        for service in self.services:
            nodes_data.append({
                "id": service.id,
                "name": service.name,
                "status": service.status,
                "position": {
                    "x": service.position[0],
                    "y": service.position[1],
                    "z": service.position[2]
                },
                "metrics": service.metrics,
                "size": self._calculate_node_size(service),
                "color": self._get_status_color(service.status)
            })
        
        edges_data = []
        for edge in self.edges:
            edges_data.append({
                "source": edge.source,
                "target": edge.target,
                "weight": edge.weight,
                "latency": edge.latency,
                "error_rate": edge.error_rate,
                "thickness": self._calculate_edge_thickness(edge),
                "color": self._get_edge_color(edge)
            })
        
        return {
            "nodes": nodes_data,
            "edges": edges_data,
            "metadata": {
                "generated_at": datetime.now().isoformat(),
                "total_services": len(self.services),
                "total_dependencies": len(self.edges),
                "layout_algorithm": "force_directed"
            }
        }
    
    def export_heatmap_data(self) -> Dict[str, Any]:
        """Export data for heatmap visualization"""
        # Group data by time intervals
        time_intervals = {}
        for data_point in self.heatmap_data:
            hour_key = data_point.timestamp.strftime("%Y-%m-%d %H:00")
            if hour_key not in time_intervals:
                time_intervals[hour_key] = []
            time_intervals[hour_key].append(data_point)
        
        # Convert to heatmap format
        heatmap_data = []
        for hour_key, data_points in time_intervals.items():
            for point in data_points:
                heatmap_data.append({
                    "x": point.x,
                    "y": point.y,
                    "value": point.value,
                    "timestamp": point.timestamp.isoformat(),
                    "metadata": point.metadata
                })
        
        return {
            "heatmap_data": heatmap_data,
            "metadata": {
                "generated_at": datetime.now().isoformat(),
                "total_data_points": len(heatmap_data),
                "time_range": {
                    "start": min(point.timestamp for point in self.heatmap_data).isoformat(),
                    "end": max(point.timestamp for point in self.heatmap_data).isoformat()
                }
            }
        }
    
    def _calculate_node_size(self, service: ServiceNode) -> float:
        """Calculate node size based on metrics"""
        base_size = 1.0
        request_factor = service.metrics["requests"] / 1000
        return max(0.5, min(3.0, base_size + request_factor))
    
    def _calculate_edge_thickness(self, edge: ServiceEdge) -> float:
        """Calculate edge thickness based on weight"""
        return max(0.1, min(5.0, edge.weight * 3))
    
    def _get_status_color(self, status: str) -> str:
        """Get color for service status"""
        return self.color_schemes.get(status, "#00ff00")
    
    def _get_edge_color(self, edge: ServiceEdge) -> str:
        """Get color for edge based on error rate"""
        if edge.error_rate > 3:
            return "#ff0000"  # Red for high error rate
        elif edge.error_rate > 1:
            return "#ffaa00"  # Orange for medium error rate
        else:
            return "#00ff00"  # Green for low error rate
    
    def generate_visualization_report(self) -> Dict[str, Any]:
        """Generate comprehensive visualization report"""
        report = {
            "timestamp": datetime.now().isoformat(),
            "summary": {
                "total_services": len(self.services),
                "total_dependencies": len(self.edges),
                "total_heatmap_points": len(self.heatmap_data),
                "healthy_services": len([s for s in self.services if s.status == "healthy"]),
                "warning_services": len([s for s in self.services if s.status == "warning"]),
                "critical_services": len([s for s in self.services if s.status == "critical"])
            },
            "3d_visualization": self.export_3d_visualization_data(),
            "heatmap_visualization": self.export_heatmap_data(),
            "recommendations": self._generate_recommendations()
        }
        
        return report
    
    def _generate_recommendations(self) -> List[str]:
        """Generate visualization recommendations"""
        recommendations = []
        
        # Check for critical services
        critical_services = [s for s in self.services if s.status == "critical"]
        if critical_services:
            recommendations.append(f"Focus on {len(critical_services)} critical services in visualization")
        
        # Check for high error rates
        high_error_edges = [e for e in self.edges if e.error_rate > 3]
        if high_error_edges:
            recommendations.append(f"Highlight {len(high_error_edges)} high-error dependencies")
        
        # Check for service isolation
        isolated_services = [s for s in self.services if not any(e.source == s.id or e.target == s.id for e in self.edges)]
        if isolated_services:
            recommendations.append(f"Consider connecting {len(isolated_services)} isolated services")
        
        return recommendations

def main():
    """Main function to demonstrate visualization analysis operations"""
    print("254Carbon Observability - Visualization Analysis Demo")
    print("=" * 60)
    
    # Initialize visualization analyzer
    analyzer = VisualizationAnalyzer()
    
    # Generate sample data
    print("\n1. Generating sample services...")
    analyzer.generate_sample_services(count=8)
    
    print("\n2. Generating sample dependencies...")
    analyzer.generate_sample_dependencies()
    
    # Calculate layouts
    print("\n3. Calculating layouts...")
    analyzer.calculate_circular_layout()
    print("   - Circular layout calculated")
    
    analyzer.calculate_force_directed_layout()
    print("   - Force-directed layout calculated")
    
    analyzer.calculate_hierarchical_layout()
    print("   - Hierarchical layout calculated")
    
    # Generate heatmap data
    print("\n4. Generating heatmap data...")
    analyzer.generate_heatmap_data(duration_hours=24)
    print(f"   - Generated {len(analyzer.heatmap_data)} heatmap data points")
    
    # Export visualization data
    print("\n5. Exporting visualization data...")
    report = analyzer.generate_visualization_report()
    
    # Save to files
    with open('visualization_report.json', 'w') as f:
        json.dump(report, f, indent=2)
    print("   - Saved visualization_report.json")
    
    with open('3d_visualization_data.json', 'w') as f:
        json.dump(report['3d_visualization'], f, indent=2)
    print("   - Saved 3d_visualization_data.json")
    
    with open('heatmap_data.json', 'w') as f:
        json.dump(report['heatmap_visualization'], f, indent=2)
    print("   - Saved heatmap_data.json")
    
    # Print summary
    print("\n6. Visualization Summary:")
    summary = report['summary']
    print(f"   - Total Services: {summary['total_services']}")
    print(f"   - Total Dependencies: {summary['total_dependencies']}")
    print(f"   - Healthy Services: {summary['healthy_services']}")
    print(f"   - Warning Services: {summary['warning_services']}")
    print(f"   - Critical Services: {summary['critical_services']}")
    print(f"   - Heatmap Data Points: {summary['total_heatmap_points']}")
    
    print("\n7. Recommendations:")
    for i, rec in enumerate(report['recommendations'], 1):
        print(f"   {i}. {rec}")
    
    print("\nVisualization analysis demo completed!")

if __name__ == "__main__":
    main()
