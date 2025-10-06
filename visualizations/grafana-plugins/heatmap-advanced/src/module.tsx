import React, { useMemo } from 'react';
import { PanelProps } from '@grafana/data';
import { Options } from './types';
import AdvancedHeatmap from './components/AdvancedHeatmap';

interface Props extends PanelProps<Options> {}

export const SimplePanel: React.FC<Props> = ({ options, data, width, height }) => {
  // Transform Grafana data into heatmap format
  const heatmapData = useMemo(() => {
    if (!data.series || data.series.length === 0) {
      return [];
    }

    const series = data.series[0];
    const heatmapData: any[] = [];

    if (series.fields) {
      const xField = series.fields.find(f => f.name === 'x' || f.name === 'source' || f.name === 'service');
      const yField = series.fields.find(f => f.name === 'y' || f.name === 'target' || f.name === 'endpoint');
      const valueField = series.fields.find(f => f.name === 'value' || f.name === 'count' || f.name === 'rate');
      const timeField = series.fields.find(f => f.type === 'time');

      if (xField && yField && valueField && xField.values && yField.values && valueField.values) {
        for (let i = 0; i < xField.values.length; i++) {
          const x = xField.values.get(i);
          const y = yField.values.get(i);
          const value = valueField.values.get(i);
          const timestamp = timeField?.values?.get(i) || Date.now();

          heatmapData.push({
            x: String(x),
            y: String(y),
            value: Number(value) || 0,
            timestamp: timestamp
          });
        }
      }
    }

    return heatmapData;
  }, [data.series]);

  const handleCellClick = (cellData: any) => {
    console.log('Cell clicked:', cellData);
    // You can implement navigation or detailed view here
  };

  return (
    <div style={{ width, height }}>
      <AdvancedHeatmap
        data={heatmapData}
        width={width}
        height={height}
        colorScheme={options.colorScheme}
        showTooltip={options.showTooltip}
        showLegend={options.showLegend}
        cellSize={options.cellSize}
        onCellClick={handleCellClick}
      />
    </div>
  );
};
