import React from 'react';
import { PanelProps } from '@grafana/data';
import { Options } from './types';
import DependencyGraph3D from './components/DependencyGraph3D';

interface Props extends PanelProps<Options> {}

export const SimplePanel: React.FC<Props> = ({ options, data, width, height }) => {
  // Transform Grafana data into nodes and edges
  const nodes = React.useMemo(() => {
    if (!data.series || data.series.length === 0) {
      return [];
    }

    // Extract nodes from the first data series
    const series = data.series[0];
    const nodes: any[] = [];

    if (series.fields) {
      const nameField = series.fields.find(f => f.name === 'service' || f.name === 'name');
      const statusField = series.fields.find(f => f.name === 'status' || f.name === 'health');
      const requestsField = series.fields.find(f => f.name === 'requests' || f.name === 'rate');
      const errorsField = series.fields.find(f => f.name === 'errors' || f.name === 'error_rate');
      const latencyField = series.fields.find(f => f.name === 'latency' || f.name === 'duration');

      if (nameField && nameField.values) {
        for (let i = 0; i < nameField.values.length; i++) {
          const name = nameField.values.get(i);
          const status = statusField?.values?.get(i) || 'healthy';
          const requests = requestsField?.values?.get(i) || 0;
          const errors = errorsField?.values?.get(i) || 0;
          const latency = latencyField?.values?.get(i) || 0;

          // Determine color based on status
          let color = '#00ff00'; // green for healthy
          if (status === 'warning') color = '#ffaa00';
          if (status === 'critical') color = '#ff0000';

          nodes.push({
            id: name,
            name: name,
            x: 0, // Will be positioned by the 3D layout
            y: 0,
            z: 0,
            size: Math.max(0.5, Math.min(2.0, requests / 100)), // Size based on request rate
            color: color,
            status: status,
            metrics: {
              requests: requests,
              errors: errors,
              latency: latency
            }
          });
        }
      }
    }

    return nodes;
  }, [data.series]);

  const edges = React.useMemo(() => {
    if (!data.series || data.series.length < 2) {
      return [];
    }

    // Extract edges from the second data series (if available)
    const series = data.series[1];
    const edges: any[] = [];

    if (series.fields) {
      const sourceField = series.fields.find(f => f.name === 'source' || f.name === 'from');
      const targetField = series.fields.find(f => f.name === 'target' || f.name === 'to');
      const weightField = series.fields.find(f => f.name === 'weight' || f.name === 'calls');

      if (sourceField && targetField && sourceField.values && targetField.values) {
        for (let i = 0; i < sourceField.values.length; i++) {
          const source = sourceField.values.get(i);
          const target = targetField.values.get(i);
          const weight = weightField?.values?.get(i) || 1;

          edges.push({
            source: source,
            target: target,
            weight: Math.max(0.1, Math.min(3.0, weight / 10)), // Normalize weight
            color: '#ffffff',
            status: 'healthy'
          });
        }
      }
    }

    return edges;
  }, [data.series]);

  const handleNodeClick = (node: any) => {
    console.log('Node clicked:', node);
    // You can implement navigation or detailed view here
  };

  const handleEdgeClick = (edge: any) => {
    console.log('Edge clicked:', edge);
    // You can implement edge details view here
  };

  return (
    <div style={{ width, height }}>
      <DependencyGraph3D
        nodes={nodes}
        edges={edges}
        onNodeClick={handleNodeClick}
        onEdgeClick={handleEdgeClick}
      />
    </div>
  );
};
