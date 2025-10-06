import { PanelOptionsEditorBuilder } from '@grafana/data';

export interface Options {
  // Visualization options
  colorScheme: 'viridis' | 'plasma' | 'inferno' | 'magma' | 'cividis';
  showTooltip: boolean;
  showLegend: boolean;
  cellSize: number;
  
  // Interaction options
  enableZoom: boolean;
  enablePan: boolean;
  enableSelection: boolean;
  
  // Display options
  showGrid: boolean;
  showLabels: boolean;
  labelRotation: number;
  
  // Animation options
  enableAnimation: boolean;
  animationDuration: number;
}

export const defaults: Options = {
  colorScheme: 'viridis',
  showTooltip: true,
  showLegend: true,
  cellSize: 20,
  enableZoom: true,
  enablePan: true,
  enableSelection: true,
  showGrid: true,
  showLabels: true,
  labelRotation: -45,
  enableAnimation: true,
  animationDuration: 1000,
};

export const optionsBuilder = (builder: PanelOptionsEditorBuilder<Options>) => {
  builder
    .addSelect({
      path: 'colorScheme',
      name: 'Color Scheme',
      description: 'Color scheme for the heatmap',
      defaultValue: defaults.colorScheme,
      settings: {
        options: [
          { value: 'viridis', label: 'Viridis' },
          { value: 'plasma', label: 'Plasma' },
          { value: 'inferno', label: 'Inferno' },
          { value: 'magma', label: 'Magma' },
          { value: 'cividis', label: 'Cividis' },
        ],
      },
    })
    .addBooleanSwitch({
      path: 'showTooltip',
      name: 'Show Tooltip',
      description: 'Display tooltip on hover',
      defaultValue: defaults.showTooltip,
    })
    .addBooleanSwitch({
      path: 'showLegend',
      name: 'Show Legend',
      description: 'Display color legend',
      defaultValue: defaults.showLegend,
    })
    .addSliderInput({
      path: 'cellSize',
      name: 'Cell Size',
      description: 'Size of heatmap cells',
      defaultValue: defaults.cellSize,
      settings: {
        min: 10,
        max: 50,
        step: 5,
      },
    })
    .addBooleanSwitch({
      path: 'enableZoom',
      name: 'Enable Zoom',
      description: 'Allow zooming in/out',
      defaultValue: defaults.enableZoom,
    })
    .addBooleanSwitch({
      path: 'enablePan',
      name: 'Enable Pan',
      description: 'Allow panning the view',
      defaultValue: defaults.enablePan,
    })
    .addBooleanSwitch({
      path: 'enableSelection',
      name: 'Enable Selection',
      description: 'Allow selecting cells',
      defaultValue: defaults.enableSelection,
    })
    .addBooleanSwitch({
      path: 'showGrid',
      name: 'Show Grid',
      description: 'Display grid lines',
      defaultValue: defaults.showGrid,
    })
    .addBooleanSwitch({
      path: 'showLabels',
      name: 'Show Labels',
      description: 'Display axis labels',
      defaultValue: defaults.showLabels,
    })
    .addSliderInput({
      path: 'labelRotation',
      name: 'Label Rotation',
      description: 'Rotation angle for labels',
      defaultValue: defaults.labelRotation,
      settings: {
        min: -90,
        max: 90,
        step: 15,
      },
    })
    .addBooleanSwitch({
      path: 'enableAnimation',
      name: 'Enable Animation',
      description: 'Enable cell animations',
      defaultValue: defaults.enableAnimation,
    })
    .addSliderInput({
      path: 'animationDuration',
      name: 'Animation Duration',
      description: 'Duration of animations in milliseconds',
      defaultValue: defaults.animationDuration,
      settings: {
        min: 100,
        max: 3000,
        step: 100,
      },
    });
};
