import React, { useRef, useEffect, useMemo } from 'react';
import * as d3 from 'd3';
import { scaleSequential, interpolateViridis } from 'd3-scale-chromatic';

interface HeatmapData {
  x: string;
  y: string;
  value: number;
  timestamp: number;
}

interface AdvancedHeatmapProps {
  data: HeatmapData[];
  width: number;
  height: number;
  colorScheme: string;
  showTooltip: boolean;
  showLegend: boolean;
  cellSize: number;
  onCellClick?: (data: HeatmapData) => void;
}

const AdvancedHeatmap: React.FC<AdvancedHeatmapProps> = ({
  data,
  width,
  height,
  colorScheme,
  showTooltip,
  showLegend,
  cellSize,
  onCellClick
}) => {
  const svgRef = useRef<SVGSVGElement>(null);
  const tooltipRef = useRef<HTMLDivElement>(null);

  // Process data for heatmap
  const processedData = useMemo(() => {
    if (!data || data.length === 0) return { cells: [], xLabels: [], yLabels: [] };

    // Get unique x and y labels
    const xLabels = Array.from(new Set(data.map(d => d.x))).sort();
    const yLabels = Array.from(new Set(data.map(d => d.y))).sort();

    // Create cells for heatmap
    const cells = data.map(d => ({
      ...d,
      xIndex: xLabels.indexOf(d.x),
      yIndex: yLabels.indexOf(d.y)
    }));

    return { cells, xLabels, yLabels };
  }, [data]);

  // Create color scale
  const colorScale = useMemo(() => {
    const values = data.map(d => d.value);
    const min = Math.min(...values);
    const max = Math.max(...values);
    
    return scaleSequential(interpolateViridis)
      .domain([min, max]);
  }, [data]);

  useEffect(() => {
    if (!svgRef.current || !processedData.cells.length) return;

    const svg = d3.select(svgRef.current);
    svg.selectAll('*').remove();

    const margin = { top: 20, right: 20, bottom: 60, left: 60 };
    const chartWidth = width - margin.left - margin.right;
    const chartHeight = height - margin.top - margin.bottom;

    const g = svg.append('g')
      .attr('transform', `translate(${margin.left},${margin.top})`);

    // Create scales
    const xScale = d3.scaleBand()
      .domain(processedData.xLabels)
      .range([0, chartWidth])
      .padding(0.05);

    const yScale = d3.scaleBand()
      .domain(processedData.yLabels)
      .range([0, chartHeight])
      .padding(0.05);

    // Create cells
    const cells = g.selectAll('.cell')
      .data(processedData.cells)
      .enter()
      .append('rect')
      .attr('class', 'cell')
      .attr('x', d => xScale(d.x) || 0)
      .attr('y', d => yScale(d.y) || 0)
      .attr('width', xScale.bandwidth())
      .attr('height', yScale.bandwidth())
      .attr('fill', d => colorScale(d.value))
      .attr('stroke', '#fff')
      .attr('stroke-width', 1)
      .style('cursor', 'pointer')
      .on('click', (event, d) => {
        if (onCellClick) {
          onCellClick(d);
        }
      })
      .on('mouseover', (event, d) => {
        if (showTooltip && tooltipRef.current) {
          const tooltip = d3.select(tooltipRef.current);
          tooltip
            .style('opacity', 1)
            .html(`
              <div>
                <strong>${d.x} → ${d.y}</strong><br/>
                Value: ${d.value.toFixed(2)}<br/>
                Time: ${new Date(d.timestamp).toLocaleString()}
              </div>
            `)
            .style('left', (event.pageX + 10) + 'px')
            .style('top', (event.pageY - 10) + 'px');
        }
      })
      .on('mouseout', () => {
        if (showTooltip && tooltipRef.current) {
          d3.select(tooltipRef.current).style('opacity', 0);
        }
      });

    // Add x-axis
    g.append('g')
      .attr('transform', `translate(0,${chartHeight})`)
      .call(d3.axisBottom(xScale))
      .selectAll('text')
      .style('text-anchor', 'end')
      .attr('dx', '-.8em')
      .attr('dy', '.15em')
      .attr('transform', 'rotate(-45)');

    // Add y-axis
    g.append('g')
      .call(d3.axisLeft(yScale));

    // Add legend if enabled
    if (showLegend) {
      const legendWidth = 200;
      const legendHeight = 20;
      
      const legend = svg.append('g')
        .attr('class', 'legend')
        .attr('transform', `translate(${width - legendWidth - 20}, 20)`);

      const legendScale = d3.scaleLinear()
        .domain(colorScale.domain())
        .range([0, legendWidth]);

      const legendAxis = d3.axisBottom(legendScale)
        .ticks(5)
        .tickFormat(d3.format('.2f'));

      legend.append('g')
        .attr('transform', `translate(0,${legendHeight})`)
        .call(legendAxis);

      // Create gradient for legend
      const defs = svg.append('defs');
      const gradient = defs.append('linearGradient')
        .attr('id', 'legend-gradient')
        .attr('x1', '0%')
        .attr('x2', '100%')
        .attr('y1', '0%')
        .attr('y2', '0%');

      const stops = colorScale.domain();
      gradient.selectAll('stop')
        .data(d3.range(0, 1, 0.01))
        .enter()
        .append('stop')
        .attr('offset', d => `${d * 100}%`)
        .attr('stop-color', d => colorScale(d3.interpolate(stops[0], stops[1])(d)));

      legend.append('rect')
        .attr('width', legendWidth)
        .attr('height', legendHeight)
        .style('fill', 'url(#legend-gradient)')
        .style('stroke', '#000')
        .style('stroke-width', 1);
    }

  }, [processedData, colorScale, width, height, showTooltip, showLegend, onCellClick]);

  return (
    <div style={{ position: 'relative', width, height }}>
      <svg
        ref={svgRef}
        width={width}
        height={height}
        style={{ background: '#f8f9fa' }}
      />
      {showTooltip && (
        <div
          ref={tooltipRef}
          style={{
            position: 'absolute',
            padding: '8px',
            background: 'rgba(0, 0, 0, 0.8)',
            color: 'white',
            borderRadius: '4px',
            fontSize: '12px',
            pointerEvents: 'none',
            opacity: 0,
            zIndex: 1000
          }}
        />
      )}
    </div>
  );
};

export default AdvancedHeatmap;
