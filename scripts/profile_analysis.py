#!/usr/bin/env python3
"""
Pyroscope Profile Analysis Script

This script analyzes Pyroscope profiles to identify performance bottlenecks,
memory leaks, and optimization opportunities.
"""

import json
import time
import requests
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
import argparse
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ProfileAnalysis:
    """Profile analysis result."""
    service: str
    profile_type: str
    timestamp: datetime
    duration: float
    cpu_usage: float
    memory_usage: float
    top_functions: List[Dict[str, Any]]
    bottlenecks: List[str]
    recommendations: List[str]
    score: float  # Performance score 0-100

class PyroscopeAnalyzer:
    """Pyroscope profile analyzer."""
    
    def __init__(self, pyroscope_url: str = "http://localhost:4040"):
        self.pyroscope_url = pyroscope_url
        self.session = requests.Session()
        
    def get_profiles(self, service: str, profile_type: str = "cpu", 
                    start_time: Optional[datetime] = None, 
                    end_time: Optional[datetime] = None) -> List[Dict[str, Any]]:
        """Get profiles from Pyroscope."""
        try:
            params = {
                "query": f"service={service}",
                "profileType": profile_type
            }
            
            if start_time:
                params["start"] = start_time.isoformat()
            if end_time:
                params["end"] = end_time.isoformat()
                
            response = self.session.get(f"{self.pyroscope_url}/api/v1/query", params=params)
            response.raise_for_status()
            
            return response.json().get("profiles", [])
            
        except requests.RequestException as e:
            logger.error(f"Failed to get profiles: {e}")
            return []
    
    def analyze_profile(self, profile: Dict[str, Any]) -> ProfileAnalysis:
        """Analyze a single profile."""
        service = profile.get("service", "unknown")
        profile_type = profile.get("profileType", "cpu")
        timestamp = datetime.fromisoformat(profile.get("timestamp", datetime.now().isoformat()))
        
        # Extract profile data
        profile_data = profile.get("data", {})
        duration = profile_data.get("duration", 0)
        cpu_usage = profile_data.get("cpuUsage", 0)
        memory_usage = profile_data.get("memoryUsage", 0)
        
        # Analyze top functions
        top_functions = self._analyze_top_functions(profile_data)
        
        # Identify bottlenecks
        bottlenecks = self._identify_bottlenecks(profile_data, top_functions)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(bottlenecks, top_functions)
        
        # Calculate performance score
        score = self._calculate_performance_score(cpu_usage, memory_usage, bottlenecks)
        
        return ProfileAnalysis(
            service=service,
            profile_type=profile_type,
            timestamp=timestamp,
            duration=duration,
            cpu_usage=cpu_usage,
            memory_usage=memory_usage,
            top_functions=top_functions,
            bottlenecks=bottlenecks,
            recommendations=recommendations,
            score=score
        )
    
    def _analyze_top_functions(self, profile_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Analyze top functions by CPU/memory usage."""
        functions = profile_data.get("functions", [])
        
        # Sort by usage (CPU or memory)
        sorted_functions = sorted(functions, key=lambda x: x.get("usage", 0), reverse=True)
        
        # Return top 10 functions
        return sorted_functions[:10]
    
    def _identify_bottlenecks(self, profile_data: Dict[str, Any], 
                             top_functions: List[Dict[str, Any]]) -> List[str]:
        """Identify performance bottlenecks."""
        bottlenecks = []
        
        # Check for high CPU usage functions
        for func in top_functions[:5]:
            if func.get("usage", 0) > 10:  # More than 10% usage
                bottlenecks.append(f"High CPU usage in {func.get('name', 'unknown')}: {func.get('usage', 0):.2f}%")
        
        # Check for memory leaks
        memory_functions = [f for f in top_functions if f.get("type") == "memory"]
        if memory_functions:
            total_memory = sum(f.get("usage", 0) for f in memory_functions)
            if total_memory > 50:  # More than 50% memory usage
                bottlenecks.append(f"High memory usage: {total_memory:.2f}%")
        
        # Check for blocking operations
        blocking_functions = [f for f in top_functions if "block" in f.get("name", "").lower()]
        if blocking_functions:
            bottlenecks.append(f"Blocking operations detected: {len(blocking_functions)} functions")
        
        # Check for inefficient algorithms
        inefficient_patterns = ["nested_loop", "recursive", "sort", "search"]
        for func in top_functions:
            func_name = func.get("name", "").lower()
            for pattern in inefficient_patterns:
                if pattern in func_name and func.get("usage", 0) > 5:
                    bottlenecks.append(f"Inefficient algorithm in {func.get('name', 'unknown')}")
        
        return bottlenecks
    
    def _generate_recommendations(self, bottlenecks: List[str], 
                                 top_functions: List[Dict[str, Any]]) -> List[str]:
        """Generate optimization recommendations."""
        recommendations = []
        
        # CPU optimization recommendations
        cpu_functions = [f for f in top_functions if f.get("type") == "cpu"]
        if cpu_functions:
            top_cpu = cpu_functions[0]
            if top_cpu.get("usage", 0) > 20:
                recommendations.append(f"Optimize {top_cpu.get('name', 'unknown')} - highest CPU consumer")
        
        # Memory optimization recommendations
        memory_functions = [f for f in top_functions if f.get("type") == "memory"]
        if memory_functions:
            total_memory = sum(f.get("usage", 0) for f in memory_functions)
            if total_memory > 30:
                recommendations.append("Consider memory pooling or caching strategies")
        
        # Algorithm optimization recommendations
        for func in top_functions:
            func_name = func.get("name", "").lower()
            if "sort" in func_name and func.get("usage", 0) > 10:
                recommendations.append("Consider using more efficient sorting algorithms")
            elif "search" in func_name and func.get("usage", 0) > 10:
                recommendations.append("Consider using indexed search or caching")
            elif "loop" in func_name and func.get("usage", 0) > 15:
                recommendations.append("Optimize loop performance or consider parallel processing")
        
        # General recommendations
        if len(bottlenecks) > 5:
            recommendations.append("Consider code review and refactoring")
        
        if not recommendations:
            recommendations.append("Profile looks healthy - continue monitoring")
        
        return recommendations
    
    def _calculate_performance_score(self, cpu_usage: float, memory_usage: float, 
                                   bottlenecks: List[str]) -> float:
        """Calculate performance score (0-100)."""
        score = 100.0
        
        # Deduct points for high CPU usage
        if cpu_usage > 50:
            score -= 30
        elif cpu_usage > 20:
            score -= 15
        elif cpu_usage > 10:
            score -= 5
        
        # Deduct points for high memory usage
        if memory_usage > 50:
            score -= 25
        elif memory_usage > 20:
            score -= 10
        elif memory_usage > 10:
            score -= 5
        
        # Deduct points for bottlenecks
        score -= len(bottlenecks) * 5
        
        # Ensure score is between 0 and 100
        return max(0, min(100, score))
    
    def generate_report(self, analyses: List[ProfileAnalysis]) -> Dict[str, Any]:
        """Generate comprehensive analysis report."""
        if not analyses:
            return {"error": "No profiles to analyze"}
        
        # Calculate aggregate metrics
        total_analyses = len(analyses)
        avg_score = sum(a.score for a in analyses) / total_analyses
        avg_cpu = sum(a.cpu_usage for a in analyses) / total_analyses
        avg_memory = sum(a.memory_usage for a in analyses) / total_analyses
        
        # Find worst performing profiles
        worst_profiles = sorted(analyses, key=lambda x: x.score)[:3]
        
        # Aggregate bottlenecks
        all_bottlenecks = []
        for analysis in analyses:
            all_bottlenecks.extend(analysis.bottlenecks)
        
        # Count bottleneck frequency
        bottleneck_counts = {}
        for bottleneck in all_bottlenecks:
            bottleneck_counts[bottleneck] = bottleneck_counts.get(bottleneck, 0) + 1
        
        # Aggregate recommendations
        all_recommendations = []
        for analysis in analyses:
            all_recommendations.extend(analysis.recommendations)
        
        # Count recommendation frequency
        recommendation_counts = {}
        for rec in all_recommendations:
            recommendation_counts[rec] = recommendation_counts.get(rec, 0) + 1
        
        return {
            "summary": {
                "total_profiles": total_analyses,
                "average_score": round(avg_score, 2),
                "average_cpu_usage": round(avg_cpu, 2),
                "average_memory_usage": round(avg_memory, 2),
                "analysis_timestamp": datetime.now().isoformat()
            },
            "worst_performing_profiles": [
                {
                    "service": p.service,
                    "score": p.score,
                    "cpu_usage": p.cpu_usage,
                    "memory_usage": p.memory_usage,
                    "bottlenecks": p.bottlenecks,
                    "recommendations": p.recommendations
                }
                for p in worst_profiles
            ],
            "common_bottlenecks": [
                {"bottleneck": k, "frequency": v}
                for k, v in sorted(bottleneck_counts.items(), key=lambda x: x[1], reverse=True)
            ],
            "top_recommendations": [
                {"recommendation": k, "frequency": v}
                for k, v in sorted(recommendation_counts.items(), key=lambda x: x[1], reverse=True)
            ],
            "detailed_analyses": [
                {
                    "service": a.service,
                    "profile_type": a.profile_type,
                    "timestamp": a.timestamp.isoformat(),
                    "duration": a.duration,
                    "cpu_usage": a.cpu_usage,
                    "memory_usage": a.memory_usage,
                    "top_functions": a.top_functions,
                    "bottlenecks": a.bottlenecks,
                    "recommendations": a.recommendations,
                    "score": a.score
                }
                for a in analyses
            ]
        }

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Analyze Pyroscope profiles")
    parser.add_argument("--service", required=True, help="Service name to analyze")
    parser.add_argument("--profile-type", default="cpu", choices=["cpu", "memory", "goroutines"],
                       help="Profile type to analyze")
    parser.add_argument("--pyroscope-url", default="http://localhost:4040",
                       help="Pyroscope server URL")
    parser.add_argument("--hours", type=int, default=24,
                       help="Number of hours to look back")
    parser.add_argument("--output", help="Output file for report")
    
    args = parser.parse_args()
    
    # Initialize analyzer
    analyzer = PyroscopeAnalyzer(args.pyroscope_url)
    
    # Calculate time range
    end_time = datetime.now()
    start_time = end_time - timedelta(hours=args.hours)
    
    # Get profiles
    logger.info(f"Getting profiles for service: {args.service}")
    profiles = analyzer.get_profiles(args.service, args.profile_type, start_time, end_time)
    
    if not profiles:
        logger.warning("No profiles found")
        return
    
    # Analyze profiles
    logger.info(f"Analyzing {len(profiles)} profiles")
    analyses = []
    for profile in profiles:
        try:
            analysis = analyzer.analyze_profile(profile)
            analyses.append(analysis)
        except Exception as e:
            logger.error(f"Failed to analyze profile: {e}")
    
    # Generate report
    report = analyzer.generate_report(analyses)
    
    # Output report
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(report, f, indent=2)
        logger.info(f"Report saved to {args.output}")
    else:
        print(json.dumps(report, indent=2))

if __name__ == "__main__":
    main()
