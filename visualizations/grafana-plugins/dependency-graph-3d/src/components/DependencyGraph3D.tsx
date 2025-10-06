import React, { useRef, useMemo } from 'react';
import { Canvas, useFrame } from '@react-three/fiber';
import { OrbitControls, Text, Sphere, Line } from '@react-three/drei';
import * as THREE from 'three';

interface Node {
  id: string;
  name: string;
  x: number;
  y: number;
  z: number;
  size: number;
  color: string;
  status: 'healthy' | 'warning' | 'critical';
  metrics: {
    requests: number;
    errors: number;
    latency: number;
  };
}

interface Edge {
  source: string;
  target: string;
  weight: number;
  color: string;
  status: 'healthy' | 'warning' | 'critical';
}

interface DependencyGraph3DProps {
  nodes: Node[];
  edges: Edge[];
  onNodeClick?: (node: Node) => void;
  onEdgeClick?: (edge: Edge) => void;
}

const NodeComponent: React.FC<{ node: Node; onClick?: (node: Node) => void }> = ({ node, onClick }) => {
  const meshRef = useRef<THREE.Mesh>(null);
  
  useFrame((state) => {
    if (meshRef.current) {
      // Animate node based on status
      const scale = node.status === 'critical' ? 1.2 : node.status === 'warning' ? 1.1 : 1.0;
      meshRef.current.scale.setScalar(scale);
      
      // Pulse animation for critical nodes
      if (node.status === 'critical') {
        meshRef.current.position.y = node.y + Math.sin(state.clock.elapsedTime * 2) * 0.1;
      }
    }
  });

  const handleClick = () => {
    if (onClick) {
      onClick(node);
    }
  };

  return (
    <group position={[node.x, node.y, node.z]}>
      <Sphere
        ref={meshRef}
        args={[node.size, 32, 32]}
        onClick={handleClick}
        onPointerOver={(e) => {
          e.stopPropagation();
          document.body.style.cursor = 'pointer';
        }}
        onPointerOut={() => {
          document.body.style.cursor = 'default';
        }}
      >
        <meshStandardMaterial
          color={node.color}
          emissive={node.status === 'critical' ? '#ff0000' : node.status === 'warning' ? '#ffaa00' : '#00ff00'}
          emissiveIntensity={0.2}
        />
      </Sphere>
      <Text
        position={[0, node.size + 0.5, 0]}
        fontSize={0.3}
        color="white"
        anchorX="center"
        anchorY="middle"
      >
        {node.name}
      </Text>
      <Text
        position={[0, node.size + 0.8, 0]}
        fontSize={0.2}
        color="gray"
        anchorX="center"
        anchorY="middle"
      >
        {node.metrics.requests} req/s
      </Text>
    </group>
  );
};

const EdgeComponent: React.FC<{ edge: Edge; nodes: Node[]; onClick?: (edge: Edge) => void }> = ({ 
  edge, 
  nodes, 
  onClick 
}) => {
  const sourceNode = nodes.find(n => n.id === edge.source);
  const targetNode = nodes.find(n => n.id === edge.target);
  
  if (!sourceNode || !targetNode) return null;

  const points = [
    new THREE.Vector3(sourceNode.x, sourceNode.y, sourceNode.z),
    new THREE.Vector3(targetNode.x, targetNode.y, targetNode.z)
  ];

  const handleClick = () => {
    if (onClick) {
      onClick(edge);
    }
  };

  return (
    <Line
      points={points}
      color={edge.color}
      lineWidth={edge.weight * 2}
      onClick={handleClick}
      onPointerOver={(e) => {
        e.stopPropagation();
        document.body.style.cursor = 'pointer';
      }}
      onPointerOut={() => {
        document.body.style.cursor = 'default';
      }}
    />
  );
};

const DependencyGraph3D: React.FC<DependencyGraph3DProps> = ({ 
  nodes, 
  edges, 
  onNodeClick, 
  onEdgeClick 
}) => {
  const processedNodes = useMemo(() => {
    // Position nodes in 3D space using force-directed layout
    const positions = new Map<string, { x: number; y: number; z: number }>();
    
    // Simple circular layout for demonstration
    const radius = 5;
    const angleStep = (2 * Math.PI) / nodes.length;
    
    nodes.forEach((node, index) => {
      const angle = index * angleStep;
      const x = Math.cos(angle) * radius;
      const z = Math.sin(angle) * radius;
      const y = (Math.random() - 0.5) * 2; // Random height
      
      positions.set(node.id, { x, y, z });
    });
    
    return nodes.map(node => {
      const pos = positions.get(node.id);
      return {
        ...node,
        x: pos?.x || 0,
        y: pos?.y || 0,
        z: pos?.z || 0
      };
    });
  }, [nodes]);

  return (
    <Canvas
      camera={{ position: [10, 10, 10], fov: 60 }}
      style={{ width: '100%', height: '100%' }}
    >
      <ambientLight intensity={0.5} />
      <pointLight position={[10, 10, 10]} />
      <pointLight position={[-10, -10, -10]} />
      
      {/* Render edges first so they appear behind nodes */}
      {edges.map((edge, index) => (
        <EdgeComponent
          key={`edge-${index}`}
          edge={edge}
          nodes={processedNodes}
          onClick={onEdgeClick}
        />
      ))}
      
      {/* Render nodes */}
      {processedNodes.map((node) => (
        <NodeComponent
          key={node.id}
          node={node}
          onClick={onNodeClick}
        />
      ))}
      
      <OrbitControls
        enablePan={true}
        enableZoom={true}
        enableRotate={true}
        minDistance={5}
        maxDistance={50}
      />
    </Canvas>
  );
};

export default DependencyGraph3D;
