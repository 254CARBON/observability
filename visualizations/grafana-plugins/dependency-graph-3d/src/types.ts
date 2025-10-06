import { PanelOptionsEditorBuilder } from '@grafana/data';

export interface Options {
  // Visualization options
  showLabels: boolean;
  showMetrics: boolean;
  nodeSize: 'small' | 'medium' | 'large';
  edgeThickness: 'thin' | 'medium' | 'thick';
  
  // Animation options
  enableAnimation: boolean;
  animationSpeed: number;
  
  // Color options
  healthyColor: string;
  warningColor: string;
  criticalColor: string;
  
  // Layout options
  layout: 'circular' | 'force-directed' | 'hierarchical';
  spacing: number;
  
  // Interaction options
  enableZoom: boolean;
  enableRotation: boolean;
  enablePan: boolean;
}

export const defaults: Options = {
  showLabels: true,
  showMetrics: true,
  nodeSize: 'medium',
  edgeThickness: 'medium',
  enableAnimation: true,
  animationSpeed: 1.0,
  healthyColor: '#00ff00',
  warningColor: '#ffaa00',
  criticalColor: '#ff0000',
  layout: 'circular',
  spacing: 5,
  enableZoom: true,
  enableRotation: true,
  enablePan: true,
};

export const optionsBuilder = (builder: PanelOptionsEditorBuilder<Options>) => {
  builder
    .addBooleanSwitch({
      path: 'showLabels',
      name: 'Show Labels',
      description: 'Display service names on nodes',
      defaultValue: defaults.showLabels,
    })
    .addBooleanSwitch({
      path: 'showMetrics',
      name: 'Show Metrics',
      description: 'Display metrics on nodes',
      defaultValue: defaults.showMetrics,
    })
    .addSelect({
      path: 'nodeSize',
      name: 'Node Size',
      description: 'Size of the nodes',
      defaultValue: defaults.nodeSize,
      settings: {
        options: [
          { value: 'small', label: 'Small' },
          { value: 'medium', label: 'Medium' },
          { value: 'large', label: 'Large' },
        ],
      },
    })
    .addSelect({
      path: 'edgeThickness',
      name: 'Edge Thickness',
      description: 'Thickness of the edges',
      defaultValue: defaults.edgeThickness,
      settings: {
        options: [
          { value: 'thin', label: 'Thin' },
          { value: 'medium', label: 'Medium' },
          { value: 'thick', label: 'Thick' },
        ],
      },
    })
    .addBooleanSwitch({
      path: 'enableAnimation',
      name: 'Enable Animation',
      description: 'Enable node animations',
      defaultValue: defaults.enableAnimation,
    })
    .addSliderInput({
      path: 'animationSpeed',
      name: 'Animation Speed',
      description: 'Speed of animations',
      defaultValue: defaults.animationSpeed,
      settings: {
        min: 0.1,
        max: 3.0,
        step: 0.1,
      },
    })
    .addColorPicker({
      path: 'healthyColor',
      name: 'Healthy Color',
      description: 'Color for healthy nodes',
      defaultValue: defaults.healthyColor,
    })
    .addColorPicker({
      path: 'warningColor',
      name: 'Warning Color',
      description: 'Color for warning nodes',
      defaultValue: defaults.warningColor,
    })
    .addColorPicker({
      path: 'criticalColor',
      name: 'Critical Color',
      description: 'Color for critical nodes',
      defaultValue: defaults.criticalColor,
    })
    .addSelect({
      path: 'layout',
      name: 'Layout',
      description: 'Node layout algorithm',
      defaultValue: defaults.layout,
      settings: {
        options: [
          { value: 'circular', label: 'Circular' },
          { value: 'force-directed', label: 'Force Directed' },
          { value: 'hierarchical', label: 'Hierarchical' },
        ],
      },
    })
    .addSliderInput({
      path: 'spacing',
      name: 'Spacing',
      description: 'Spacing between nodes',
      defaultValue: defaults.spacing,
      settings: {
        min: 1,
        max: 20,
        step: 1,
      },
    })
    .addBooleanSwitch({
      path: 'enableZoom',
      name: 'Enable Zoom',
      description: 'Allow zooming in/out',
      defaultValue: defaults.enableZoom,
    })
    .addBooleanSwitch({
      path: 'enableRotation',
      name: 'Enable Rotation',
      description: 'Allow rotating the view',
      defaultValue: defaults.enableRotation,
    })
    .addBooleanSwitch({
      path: 'enablePan',
      name: 'Enable Pan',
      description: 'Allow panning the view',
      defaultValue: defaults.enablePan,
    });
};
