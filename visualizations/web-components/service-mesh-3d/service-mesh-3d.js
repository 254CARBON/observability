// 3D Service Mesh Visualization
// 254Carbon Observability Platform

class ServiceMesh3D {
    constructor() {
        this.scene = null;
        this.camera = null;
        this.renderer = null;
        this.controls = null;
        this.nodes = [];
        this.edges = [];
        this.animating = true;
        this.selectedNode = null;
        
        this.init();
        this.loadSampleData();
        this.setupEventListeners();
        this.animate();
    }
    
    init() {
        // Create scene
        this.scene = new THREE.Scene();
        this.scene.background = new THREE.Color(0x1a1a1a);
        
        // Create camera
        this.camera = new THREE.PerspectiveCamera(
            75,
            window.innerWidth / window.innerHeight,
            0.1,
            1000
        );
        this.camera.position.set(20, 20, 20);
        
        // Create renderer
        this.renderer = new THREE.WebGLRenderer({ antialias: true });
        this.renderer.setSize(window.innerWidth, window.innerHeight);
        this.renderer.shadowMap.enabled = true;
        this.renderer.shadowMap.type = THREE.PCFSoftShadowMap;
        
        // Add renderer to DOM
        const container = document.getElementById('container');
        container.appendChild(this.renderer.domElement);
        
        // Create controls
        this.controls = new THREE.OrbitControls(this.camera, this.renderer.domElement);
        this.controls.enableDamping = true;
        this.controls.dampingFactor = 0.05;
        this.controls.maxDistance = 100;
        this.controls.minDistance = 5;
        
        // Add lighting
        const ambientLight = new THREE.AmbientLight(0x404040, 0.6);
        this.scene.add(ambientLight);
        
        const directionalLight = new THREE.DirectionalLight(0xffffff, 0.8);
        directionalLight.position.set(50, 50, 50);
        directionalLight.castShadow = true;
        directionalLight.shadow.mapSize.width = 2048;
        directionalLight.shadow.mapSize.height = 2048;
        this.scene.add(directionalLight);
        
        // Add grid
        const gridHelper = new THREE.GridHelper(100, 100, 0x444444, 0x444444);
        this.scene.add(gridHelper);
        
        // Hide loading screen
        document.getElementById('loading').style.display = 'none';
    }
    
    loadSampleData() {
        // Sample service mesh data
        const services = [
            { id: 'gateway', name: 'API Gateway', status: 'healthy', requests: 1000, errors: 5, latency: 50 },
            { id: 'auth', name: 'Auth Service', status: 'healthy', requests: 800, errors: 2, latency: 30 },
            { id: 'user', name: 'User Service', status: 'healthy', requests: 600, errors: 1, latency: 40 },
            { id: 'order', name: 'Order Service', status: 'warning', requests: 400, errors: 20, latency: 80 },
            { id: 'payment', name: 'Payment Service', status: 'healthy', requests: 300, errors: 3, latency: 60 },
            { id: 'inventory', name: 'Inventory Service', status: 'critical', requests: 200, errors: 50, latency: 200 },
            { id: 'notification', name: 'Notification Service', status: 'healthy', requests: 150, errors: 1, latency: 25 },
            { id: 'analytics', name: 'Analytics Service', status: 'healthy', requests: 100, errors: 0, latency: 35 }
        ];
        
        const dependencies = [
            { source: 'gateway', target: 'auth', weight: 0.8 },
            { source: 'gateway', target: 'user', weight: 0.6 },
            { source: 'gateway', target: 'order', weight: 0.4 },
            { source: 'order', target: 'payment', weight: 0.7 },
            { source: 'order', target: 'inventory', weight: 0.9 },
            { source: 'user', target: 'notification', weight: 0.3 },
            { source: 'gateway', target: 'analytics', weight: 0.2 }
        ];
        
        this.createNodes(services);
        this.createEdges(dependencies);
        this.positionNodes();
    }
    
    createNodes(services) {
        services.forEach(service => {
            const geometry = new THREE.SphereGeometry(1, 32, 32);
            
            let color = 0x00ff00; // green for healthy
            if (service.status === 'warning') color = 0xffaa00; // orange for warning
            if (service.status === 'critical') color = 0xff0000; // red for critical
            
            const material = new THREE.MeshLambertMaterial({ 
                color: color,
                emissive: color,
                emissiveIntensity: 0.2
            });
            
            const node = new THREE.Mesh(geometry, material);
            node.userData = service;
            node.castShadow = true;
            node.receiveShadow = true;
            
            // Add click event
            node.onClick = () => this.selectNode(node);
            
            this.nodes.push(node);
            this.scene.add(node);
        });
    }
    
    createEdges(dependencies) {
        dependencies.forEach(dep => {
            const sourceNode = this.nodes.find(n => n.userData.id === dep.source);
            const targetNode = this.nodes.find(n => n.userData.id === dep.target);
            
            if (sourceNode && targetNode) {
                const geometry = new THREE.BufferGeometry().setFromPoints([
                    sourceNode.position,
                    targetNode.position
                ]);
                
                const material = new THREE.LineBasicMaterial({ 
                    color: 0xffffff,
                    opacity: 0.6,
                    transparent: true
                });
                
                const edge = new THREE.Line(geometry, material);
                edge.userData = dep;
                
                this.edges.push(edge);
                this.scene.add(edge);
            }
        });
    }
    
    positionNodes() {
        const radius = 15;
        const angleStep = (2 * Math.PI) / this.nodes.length;
        
        this.nodes.forEach((node, index) => {
            const angle = index * angleStep;
            const x = Math.cos(angle) * radius;
            const z = Math.sin(angle) * radius;
            const y = (Math.random() - 0.5) * 10; // Random height
            
            node.position.set(x, y, z);
        });
        
        // Update edge positions
        this.edges.forEach(edge => {
            const sourceNode = this.nodes.find(n => n.userData.id === edge.userData.source);
            const targetNode = this.nodes.find(n => n.userData.id === edge.userData.target);
            
            if (sourceNode && targetNode) {
                edge.geometry.setFromPoints([sourceNode.position, targetNode.position]);
            }
        });
    }
    
    selectNode(node) {
        // Deselect previous node
        if (this.selectedNode) {
            this.selectedNode.material.emissiveIntensity = 0.2;
        }
        
        // Select new node
        this.selectedNode = node;
        node.material.emissiveIntensity = 0.5;
        
        // Update info panel
        this.updateInfoPanel(node.userData);
    }
    
    updateInfoPanel(service) {
        document.getElementById('selected-service').textContent = service.name;
        document.getElementById('request-rate').textContent = `${service.requests} req/s`;
        document.getElementById('error-rate').textContent = `${((service.errors / service.requests) * 100).toFixed(1)}%`;
        document.getElementById('latency').textContent = `${service.latency}ms`;
        
        // Count dependencies
        const dependencies = this.edges.filter(e => 
            e.userData.source === service.id || e.userData.target === service.id
        ).length;
        document.getElementById('dependencies').textContent = dependencies;
    }
    
    setupEventListeners() {
        // Raycaster for mouse interactions
        this.raycaster = new THREE.Raycaster();
        this.mouse = new THREE.Vector2();
        
        this.renderer.domElement.addEventListener('click', (event) => {
            this.mouse.x = (event.clientX / window.innerWidth) * 2 - 1;
            this.mouse.y = -(event.clientY / window.innerHeight) * 2 + 1;
            
            this.raycaster.setFromCamera(this.mouse, this.camera);
            const intersects = this.raycaster.intersectObjects(this.nodes);
            
            if (intersects.length > 0) {
                intersects[0].object.onClick();
            }
        });
        
        // Control event listeners
        document.getElementById('layout').addEventListener('change', (e) => {
            this.changeLayout(e.target.value);
        });
        
        document.getElementById('nodeSize').addEventListener('input', (e) => {
            this.updateNodeSize(parseFloat(e.target.value));
        });
        
        document.getElementById('edgeThickness').addEventListener('input', (e) => {
            this.updateEdgeThickness(parseFloat(e.target.value));
        });
        
        document.getElementById('animationSpeed').addEventListener('input', (e) => {
            this.animationSpeed = parseFloat(e.target.value);
        });
        
        document.getElementById('resetView').addEventListener('click', () => {
            this.resetView();
        });
        
        document.getElementById('toggleAnimation').addEventListener('click', () => {
            this.toggleAnimation();
        });
        
        document.getElementById('exportImage').addEventListener('click', () => {
            this.exportImage();
        });
        
        document.getElementById('exportData').addEventListener('click', () => {
            this.exportData();
        });
        
        // Window resize
        window.addEventListener('resize', () => {
            this.camera.aspect = window.innerWidth / window.innerHeight;
            this.camera.updateProjectionMatrix();
            this.renderer.setSize(window.innerWidth, window.innerHeight);
        });
    }
    
    changeLayout(layout) {
        switch (layout) {
            case 'circular':
                this.positionNodesCircular();
                break;
            case 'force-directed':
                this.positionNodesForceDirected();
                break;
            case 'hierarchical':
                this.positionNodesHierarchical();
                break;
        }
    }
    
    positionNodesCircular() {
        const radius = 15;
        const angleStep = (2 * Math.PI) / this.nodes.length;
        
        this.nodes.forEach((node, index) => {
            const angle = index * angleStep;
            const x = Math.cos(angle) * radius;
            const z = Math.sin(angle) * radius;
            const y = (Math.random() - 0.5) * 10;
            
            node.position.set(x, y, z);
        });
        
        this.updateEdges();
    }
    
    positionNodesForceDirected() {
        // Simple force-directed layout simulation
        const iterations = 100;
        const k = Math.sqrt((4 * Math.PI * 15 * 15) / this.nodes.length);
        
        for (let i = 0; i < iterations; i++) {
            this.nodes.forEach(node => {
                let fx = 0, fy = 0, fz = 0;
                
                // Repulsive forces
                this.nodes.forEach(other => {
                    if (node !== other) {
                        const dx = node.position.x - other.position.x;
                        const dy = node.position.y - other.position.y;
                        const dz = node.position.z - other.position.z;
                        const distance = Math.sqrt(dx * dx + dy * dy + dz * dz);
                        
                        if (distance > 0) {
                            const force = (k * k) / distance;
                            fx += (dx / distance) * force;
                            fy += (dy / distance) * force;
                            fz += (dz / distance) * force;
                        }
                    }
                });
                
                // Attractive forces from edges
                this.edges.forEach(edge => {
                    const sourceNode = this.nodes.find(n => n.userData.id === edge.userData.source);
                    const targetNode = this.nodes.find(n => n.userData.id === edge.userData.target);
                    
                    if (sourceNode === node && targetNode) {
                        const dx = targetNode.position.x - node.position.x;
                        const dy = targetNode.position.y - node.position.y;
                        const dz = targetNode.position.z - node.position.z;
                        const distance = Math.sqrt(dx * dx + dy * dy + dz * dz);
                        
                        if (distance > 0) {
                            const force = (distance * distance) / k;
                            fx += (dx / distance) * force;
                            fy += (dy / distance) * force;
                            fz += (dz / distance) * force;
                        }
                    }
                });
                
                // Apply forces
                node.position.x += fx * 0.01;
                node.position.y += fy * 0.01;
                node.position.z += fz * 0.01;
            });
        }
        
        this.updateEdges();
    }
    
    positionNodesHierarchical() {
        // Simple hierarchical layout
        const levels = 3;
        const nodesPerLevel = Math.ceil(this.nodes.length / levels);
        
        this.nodes.forEach((node, index) => {
            const level = Math.floor(index / nodesPerLevel);
            const positionInLevel = index % nodesPerLevel;
            const angle = (positionInLevel / nodesPerLevel) * 2 * Math.PI;
            const radius = 5 + level * 5;
            
            const x = Math.cos(angle) * radius;
            const z = Math.sin(angle) * radius;
            const y = level * 8 - 8;
            
            node.position.set(x, y, z);
        });
        
        this.updateEdges();
    }
    
    updateEdges() {
        this.edges.forEach(edge => {
            const sourceNode = this.nodes.find(n => n.userData.id === edge.userData.source);
            const targetNode = this.nodes.find(n => n.userData.id === edge.userData.target);
            
            if (sourceNode && targetNode) {
                edge.geometry.setFromPoints([sourceNode.position, targetNode.position]);
            }
        });
    }
    
    updateNodeSize(size) {
        this.nodes.forEach(node => {
            node.scale.setScalar(size);
        });
    }
    
    updateEdgeThickness(thickness) {
        this.edges.forEach(edge => {
            edge.material.linewidth = thickness;
        });
    }
    
    resetView() {
        this.camera.position.set(20, 20, 20);
        this.controls.reset();
    }
    
    toggleAnimation() {
        this.animating = !this.animating;
        document.getElementById('toggleAnimation').textContent = 
            this.animating ? 'Pause Animation' : 'Start Animation';
    }
    
    exportImage() {
        const link = document.createElement('a');
        link.download = 'service-mesh-3d.png';
        link.href = this.renderer.domElement.toDataURL();
        link.click();
    }
    
    exportData() {
        const data = {
            nodes: this.nodes.map(n => ({
                id: n.userData.id,
                name: n.userData.name,
                status: n.userData.status,
                position: {
                    x: n.position.x,
                    y: n.position.y,
                    z: n.position.z
                }
            })),
            edges: this.edges.map(e => e.userData)
        };
        
        const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
        const url = URL.createObjectURL(blob);
        const link = document.createElement('a');
        link.download = 'service-mesh-data.json';
        link.href = url;
        link.click();
        URL.revokeObjectURL(url);
    }
    
    animate() {
        requestAnimationFrame(() => this.animate());
        
        if (this.animating) {
            const time = Date.now() * 0.001 * this.animationSpeed;
            
            // Animate nodes based on status
            this.nodes.forEach((node, index) => {
                if (node.userData.status === 'critical') {
                    node.position.y += Math.sin(time + index) * 0.02;
                    node.rotation.y += 0.01;
                } else if (node.userData.status === 'warning') {
                    node.position.y += Math.sin(time * 0.5 + index) * 0.01;
                }
            });
            
            // Animate edges
            this.edges.forEach((edge, index) => {
                edge.material.opacity = 0.6 + Math.sin(time + index) * 0.2;
            });
        }
        
        this.controls.update();
        this.renderer.render(this.scene, this.camera);
    }
}

// Initialize the visualization
try {
    new ServiceMesh3D();
} catch (error) {
    console.error('Error initializing 3D visualization:', error);
    document.getElementById('loading').style.display = 'none';
    document.getElementById('error').style.display = 'block';
}
