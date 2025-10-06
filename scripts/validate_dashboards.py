#!/usr/bin/env python3
"""
Dashboard validation script for 254Carbon Observability Platform

What it checks
- JSON syntax correctness (fast fail on malformed files)
- Presence of baseline fields (`title`, `panels`, `time`, `refresh`)
- Basic panel/target structure (`targets[*].expr`)
- Gentle best-practice warnings (refresh/time range, RED coverage hints)

Limitations
- Does not validate PromQL semantics or data source connectivity
- Treats absence of RED metrics as a warning unless the dashboard is tagged
  `red`, to avoid false negatives for specialized dashboards
"""

import json
import sys
import os
from pathlib import Path

def validate_dashboard_json(file_path):
    """Validate a single dashboard JSON file.

    Returns True on success, False otherwise; prints human-friendly diagnostics
    to stdout for fast iteration during local development.
    """
    try:
        with open(file_path, 'r') as f:
            dashboard = json.load(f)
        
        # Basic structure validation
        required_fields = ['title', 'panels', 'time', 'refresh']
        missing_fields = [field for field in required_fields if field not in dashboard]
        
        if missing_fields:
            print(f"❌ {file_path}: Missing required fields: {missing_fields}")
            return False
        
        # Panel validation: ensure targets exist and contain an expression
        for i, panel in enumerate(dashboard.get('panels', [])):
            if 'targets' not in panel:
                print(f"❌ {file_path}: Panel {i} missing targets")
                return False
            
            for j, target in enumerate(panel['targets']):
                if 'expr' not in target:
                    print(f"❌ {file_path}: Panel {i}, Target {j} missing expr")
                    return False
        
        # Check for best practices (non-fatal guidance)
        warnings = []
        
        # Check refresh interval is within a typical set; flag oddball values
        refresh = dashboard.get('refresh', '')
        if refresh and refresh not in ['5s', '10s', '30s', '1m', '5m']:
            warnings.append(f"Refresh interval '{refresh}' may be too frequent")
        
        # Check time range default encourages trend analysis for oncall
        time_from = dashboard.get('time', {}).get('from', '')
        if time_from and 'now-' in time_from:
            duration = time_from.replace('now-', '')
            if duration in ['5m', '15m']:
                warnings.append(f"Time range '{time_from}' may be too short for trend analysis")
        
        # Check for RED metrics coverage; helpful for service dashboards
        red_metrics = ['rate(', 'histogram_quantile(', 'error_rate']
        has_red_metrics = any(
            any(red_metric in target.get('expr', '') for red_metric in red_metrics)
            for panel in dashboard.get('panels', [])
            for target in panel.get('targets', [])
        )
        
        if not has_red_metrics and 'red' in dashboard.get('tags', []):
            warnings.append("Dashboard tagged as RED but no RED metrics found")
        
        if warnings:
            print(f"⚠️  {file_path}: Warnings:")
            for warning in warnings:
                print(f"   - {warning}")
        else:
            print(f"✅ {file_path}: Valid")
        
        return True
        
    except json.JSONDecodeError as e:
        print(f"❌ {file_path}: JSON syntax error: {e}")
        return False
    except Exception as e:
        print(f"❌ {file_path}: Validation error: {e}")
        return False

def main():
    """Main validation function"""
    if len(sys.argv) < 2:
        print("Usage: python3 validate_dashboards.py <dashboard_file_or_directory>")
        sys.exit(1)
    
    input_path = Path(sys.argv[1])
    
    if not input_path.exists():
        print(f"❌ Path does not exist: {input_path}")
        sys.exit(1)
    
    valid_count = 0
    total_count = 0
    
    if input_path.is_file():
        # Single file
        if input_path.suffix == '.json':
            total_count = 1
            if validate_dashboard_json(input_path):
                valid_count = 1
        else:
            print(f"❌ Not a JSON file: {input_path}")
            sys.exit(1)
    
    elif input_path.is_dir():
        # Directory - find all JSON files
        json_files = list(input_path.rglob('*.json'))
        
        if not json_files:
            print(f"❌ No JSON files found in: {input_path}")
            sys.exit(1)
        
        for json_file in json_files:
            total_count += 1
            if validate_dashboard_json(json_file):
                valid_count += 1
    
    # Summary
    print(f"\n📊 Validation Summary:")
    print(f"   Valid: {valid_count}/{total_count}")
    
    if valid_count == total_count:
        print("🎉 All dashboards are valid!")
        sys.exit(0)
    else:
        print("❌ Some dashboards have issues")
        sys.exit(1)

if __name__ == '__main__':
    main()
