/* ============================================
   Network Renderer - Canvas Visualization
   ============================================ */

class NetworkRenderer {
  constructor(canvasId) {
    this.canvas = document.getElementById(canvasId);
    if (!this.canvas) return;
    this.ctx = this.canvas.getContext('2d');
    this.nodes = [];      // [{x, y, layer, index, value}]
    this.connections = []; // [{from, to, weight}]
    this.activeNode = null;
    this.activeLayer = -1;
    this.animationProgress = 0;
    this.isAnimating = false;
    this.highlightedPath = [];
    this.layerColors = [
      '#00E676', // input - green
      '#26C6DA', // cyan
      '#42A5F5', // blue
      '#5C6BC0', // indigo
      '#7E57C2', // deep purple
      '#AB47BC', // purple
      '#EC407A', // pink
      '#EF5350', // red
      '#FF7043', // deep orange
      '#FFA726', // orange
      '#FFCA28', // amber
      '#D4E157', // lime
      '#66BB6A', // green
      '#26A69A', // teal
      '#FF9800', // output - amber
    ];
    this.dropoutMasks = null;  // Array of masks per hidden layer
    this.dropoutLayers = null; // Network layers array for offset calculation
  }

  /**
   * Render the network structure
   */
  render(network, animate = true) {
    if (!this.canvas || !network) return;

    this.network = network;
    this.resize();
    this.calculateLayout(network);

    if (animate) {
      this.animationProgress = 0;
      this.isAnimating = true;
      this.animateBuild();
    } else {
      this.animationProgress = 1;
      this.draw();
    }
  }

  /**
   * Set dropout masks for visual rendering
   */
  setDropoutMasks(masks, layers) {
    this.dropoutMasks = masks;
    this.dropoutLayers = layers;
  }

  /**
   * Check if a node is dropped out
   */
  isNodeDropped(node) {
    if (!this.dropoutMasks || !this.dropoutLayers) return false;
    // Dropout only applies to hidden layers (layer index 1 to L-2)
    const numLayers = this.dropoutLayers.length;
    if (node.layer === 0 || node.layer === numLayers - 1) return false;
    const hiddenLayerIdx = node.layer - 1; // Map to dropout mask index
    if (hiddenLayerIdx >= 0 && hiddenLayerIdx < this.dropoutMasks.length) {
      return this.dropoutMasks[hiddenLayerIdx][node.index] === 0;
    }
    return false;
  }

  resize() {
    const parent = this.canvas.parentElement;
    const dpr = window.devicePixelRatio || 1;
    const w = parent.clientWidth;
    const h = parent.clientHeight;
    this.canvas.width = w * dpr;
    this.canvas.height = h * dpr;
    this.ctx.scale(dpr, dpr);
    this.canvas.style.width = w + 'px';
    this.canvas.style.height = h + 'px';
    this.width = w;
    this.height = h;
  }

  calculateLayout(network) {
    const { layers } = network;
    const numLayers = layers.length;
    const padding = numLayers > 10 ? 30 : 60;
    const availableWidth = this.width - padding * 2;
    const availableHeight = this.height - padding * 2;
    const layerSpacing = availableWidth / (numLayers - 1 || 1);

    this.nodes = [];
    this.connections = [];

    // Calculate node positions - RTL so input is on the right
    for (let l = 0; l < numLayers; l++) {
      const numNodes = layers[l];
      const maxRadius = numLayers > 10 ? 10 : (numNodes > 10 ? 12 : 18);
      const nodeSpacing = Math.min(availableHeight / (numNodes + 1), numNodes > 10 ? 25 : 60);
      const startY = (this.height - nodeSpacing * (numNodes - 1)) / 2;

      for (let n = 0; n < numNodes; n++) {
        this.nodes.push({
          x: this.width - padding - l * layerSpacing,
          y: startY + n * nodeSpacing,
          layer: l,
          index: n,
          value: null,
          radius: Math.max(5, Math.min(maxRadius, 30 - numNodes)),
          color: this.getLayerColor(l, numLayers)
        });
      }
    }

    // Calculate connections
    let nodeOffset = 0;
    for (let l = 0; l < numLayers - 1; l++) {
      const currentLayerNodes = layers[l];
      const nextLayerNodes = layers[l + 1];
      const nextOffset = nodeOffset + currentLayerNodes;

      for (let i = 0; i < currentLayerNodes; i++) {
        for (let j = 0; j < nextLayerNodes; j++) {
          this.connections.push({
            from: nodeOffset + i,
            to: nextOffset + j,
            weight: network.weights[l][i][j],
            layer: l
          });
        }
      }
      nodeOffset += currentLayerNodes;
    }
  }

  getLayerColor(layerIndex, totalLayers) {
    if (layerIndex === 0) return this.layerColors[0];
    if (layerIndex === totalLayers - 1) return this.layerColors[this.layerColors.length - 1];
    const t = layerIndex / (totalLayers - 1);
    const idx = Math.floor(t * (this.layerColors.length - 2)) + 1;
    return this.layerColors[Math.min(idx, this.layerColors.length - 2)];
  }

  animateBuild() {
    this.animationProgress += 0.025;
    if (this.animationProgress >= 1) {
      this.animationProgress = 1;
      this.isAnimating = false;
    }
    this.draw();
    if (this.isAnimating) {
      requestAnimationFrame(() => this.animateBuild());
    }
  }

  draw() {
    const ctx = this.ctx;
    ctx.clearRect(0, 0, this.width, this.height);

    // Draw background grid
    this.drawGrid();

    // Draw connections
    const visibleConnections = Math.floor(this.connections.length * this.animationProgress);
    for (let i = 0; i < visibleConnections; i++) {
      this.drawConnection(this.connections[i]);
    }

    // Draw nodes
    const visibleNodes = Math.floor(this.nodes.length * this.animationProgress);
    for (let i = 0; i < visibleNodes; i++) {
      this.drawNode(this.nodes[i]);
    }

    // Draw layer labels
    if (this.animationProgress >= 1) {
      this.drawLayerLabels();
    }
  }

  drawGrid() {
    const ctx = this.ctx;
    ctx.strokeStyle = 'rgba(108, 99, 255, 0.03)';
    ctx.lineWidth = 1;

    for (let x = 0; x < this.width; x += 30) {
      ctx.beginPath();
      ctx.moveTo(x, 0);
      ctx.lineTo(x, this.height);
      ctx.stroke();
    }
    for (let y = 0; y < this.height; y += 30) {
      ctx.beginPath();
      ctx.moveTo(0, y);
      ctx.lineTo(this.width, y);
      ctx.stroke();
    }
  }

  drawConnection(conn) {
    const ctx = this.ctx;
    const from = this.nodes[conn.from];
    const to = this.nodes[conn.to];

    // Check if either endpoint is dropped out
    const fromDropped = this.isNodeDropped(from);
    const toDropped = this.isNodeDropped(to);
    const isDroppedConn = fromDropped || toDropped;

    const isHighlighted = this.highlightedPath.some(
      p => p.from === conn.from && p.to === conn.to
    );

    // Weight determines color intensity
    const absWeight = Math.abs(conn.weight);
    let alpha = 0.15 + absWeight * 0.3;
    let color;

    if (isDroppedConn) {
      // Faded connection for dropped neurons
      color = 'rgba(107, 111, 141, 0.06)';
      ctx.lineWidth = 0.5;
      ctx.setLineDash([3, 4]);
    } else if (isHighlighted) {
      color = `rgba(0, 229, 255, 0.8)`;
      ctx.lineWidth = 2.5;
    } else {
      color = conn.weight >= 0
        ? `rgba(0, 230, 118, ${alpha})`
        : `rgba(255, 82, 82, ${alpha})`;
      ctx.lineWidth = 1;
    }

    ctx.beginPath();
    ctx.moveTo(from.x, from.y);
    ctx.lineTo(to.x, to.y);
    ctx.strokeStyle = color;
    ctx.stroke();
    ctx.setLineDash([]);

    // Draw weight label on hover/highlight
    if (isHighlighted) {
      const midX = (from.x + to.x) / 2;
      const midY = (from.y + to.y) / 2;
      ctx.save();
      ctx.font = '10px JetBrains Mono';
      ctx.fillStyle = '#00E5FF';
      ctx.textAlign = 'center';
      ctx.fillText(conn.weight.toFixed(4), midX, midY - 5);
      ctx.restore();
    }
  }

  drawNode(node) {
    const ctx = this.ctx;
    const r = node.radius;
    const isActive = this.activeNode &&
      this.activeNode.layer === node.layer &&
      this.activeNode.index === node.index;

    // Glow effect
    if (isActive) {
      const gradient = ctx.createRadialGradient(node.x, node.y, 0, node.x, node.y, r * 3);
      gradient.addColorStop(0, node.color + '40');
      gradient.addColorStop(1, 'transparent');
      ctx.beginPath();
      ctx.arc(node.x, node.y, r * 3, 0, Math.PI * 2);
      ctx.fillStyle = gradient;
      ctx.fill();
    }

    // Outer ring
    ctx.beginPath();
    ctx.arc(node.x, node.y, r + 2, 0, Math.PI * 2);
    ctx.fillStyle = isActive ? node.color + '60' : node.color + '20';
    ctx.fill();

    // Main circle
    ctx.beginPath();
    ctx.arc(node.x, node.y, r, 0, Math.PI * 2);
    const grad = ctx.createRadialGradient(node.x - r * 0.3, node.y - r * 0.3, 0, node.x, node.y, r);
    grad.addColorStop(0, node.color);
    grad.addColorStop(1, this.darkenColor(node.color, 0.5));
    ctx.fillStyle = grad;
    ctx.fill();

    ctx.strokeStyle = isActive ? '#fff' : node.color + '80';
    ctx.lineWidth = isActive ? 2 : 1;
    ctx.stroke();

    // Value label
    if (node.value !== null && node.value !== undefined) {
      ctx.save();
      ctx.font = `bold ${Math.max(8, r * 0.6)}px JetBrains Mono`;
      ctx.fillStyle = '#fff';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillText(
        typeof node.value === 'number' ? node.value.toFixed(2) : node.value,
        node.x,
        node.y
      );
      ctx.restore();
    }
  }

  drawLayerLabels() {
    const ctx = this.ctx;
    const { layers } = this.network;
    const numLayers = layers.length;
    const labels = this.network.layerNames;

    const drawn = new Set();
    this.nodes.forEach(node => {
      if (!drawn.has(node.layer)) {
        drawn.add(node.layer);
        ctx.save();
        ctx.font = 'bold 11px Cairo';
        ctx.fillStyle = node.color + 'CC';
        ctx.textAlign = 'center';
        ctx.fillText(labels[node.layer], node.x, this.height - 15);
        ctx.font = '10px JetBrains Mono';
        ctx.fillStyle = '#6B6F8D';
        ctx.fillText(`(${layers[node.layer]})`, node.x, this.height - 2);
        ctx.restore();
      }
    });
  }

  darkenColor(hex, factor) {
    // Simple darkening for hex colors
    if (hex.startsWith('#')) {
      const r = parseInt(hex.slice(1, 3), 16);
      const g = parseInt(hex.slice(3, 5), 16);
      const b = parseInt(hex.slice(5, 7), 16);
      return `rgb(${Math.floor(r * factor)}, ${Math.floor(g * factor)}, ${Math.floor(b * factor)})`;
    }
    return hex;
  }

  /**
   * Set node values for visualization
   */
  setLayerValues(layerIndex, values) {
    let nodeIdx = 0;
    for (let l = 0; l < layerIndex; l++) {
      nodeIdx += this.network.layers[l];
    }
    for (let i = 0; i < values.length; i++) {
      if (this.nodes[nodeIdx + i]) {
        this.nodes[nodeIdx + i].value = values[i];
      }
    }
    this.draw();
  }

  /**
   * Highlight a specific node
   */
  highlightNode(layerIndex, nodeIndex) {
    this.activeNode = { layer: layerIndex, index: nodeIndex };
    this.draw();
  }

  /**
   * Highlight connections from a node
   */
  highlightNodeConnections(layerIndex, nodeIndex) {
    this.highlightedPath = [];
    // Find the global index of this node
    let globalIdx = 0;
    for (let l = 0; l < layerIndex; l++) {
      globalIdx += this.network.layers[l];
    }
    globalIdx += nodeIndex;

    this.connections.forEach(conn => {
      if (conn.from === globalIdx || conn.to === globalIdx) {
        this.highlightedPath.push(conn);
      }
    });
    this.highlightNode(layerIndex, nodeIndex);
  }

  /**
   * Clear all highlights
   */
  clearHighlights() {
    this.activeNode = null;
    this.activeLayer = -1;
    this.highlightedPath = [];
    this.nodes.forEach(n => n.value = null);
    this.draw();
  }

  /**
   * Update all node values after feedforward
   */
  setAllValues(layerOutputs) {
    let nodeIdx = 0;
    for (let l = 0; l < layerOutputs.length; l++) {
      for (let i = 0; i < layerOutputs[l].length; i++) {
        if (this.nodes[nodeIdx]) {
          this.nodes[nodeIdx].value = layerOutputs[l][i];
        }
        nodeIdx++;
      }
    }
    this.draw();
  }

  /**
   * Animate forward pass — reveal values layer by layer (input → output)
   * @param {Array<Array<number>>} layerOutputs - values for each layer
   * @param {number} delayMs - delay between layers in ms
   * @returns {Promise} resolves when animation is complete
   */
  animateForwardPass(layerOutputs, delayMs = 150) {
    return new Promise(resolve => {
      // Clear all values first
      this.nodes.forEach(n => n.value = null);
      this.activeLayer = -1;
      this.passDirection = 'forward';
      this.draw();

      let layerIdx = 0;
      const totalLayers = layerOutputs.length;

      const step = () => {
        if (layerIdx >= totalLayers) {
          this.activeLayer = -1;
          this.passDirection = null;
          this.draw();
          resolve();
          return;
        }

        // Set values for this layer
        let nodeOffset = 0;
        for (let l = 0; l < layerIdx; l++) {
          nodeOffset += this.network.layers[l];
        }
        for (let i = 0; i < layerOutputs[layerIdx].length; i++) {
          if (this.nodes[nodeOffset + i]) {
            this.nodes[nodeOffset + i].value = layerOutputs[layerIdx][i];
          }
        }

        this.activeLayer = layerIdx;
        this.draw();

        layerIdx++;
        setTimeout(step, delayMs);
      };

      step();
    });
  }

  /**
   * Animate backward pass — reveal gradient/error values layer by layer (output → input)
   * @param {Array<Array<number>>} layerGradients - gradient values for each layer
   * @param {number} delayMs - delay between layers in ms
   * @returns {Promise} resolves when animation is complete
   */
  animateBackwardPass(layerGradients, delayMs = 150) {
    return new Promise(resolve => {
      // Clear all values first
      this.nodes.forEach(n => n.value = null);
      this.activeLayer = -1;
      this.passDirection = 'backward';
      this.draw();

      const totalLayers = layerGradients.length;
      let step_i = 0;

      const step = () => {
        if (step_i >= totalLayers) {
          this.activeLayer = -1;
          this.passDirection = null;
          this.draw();
          resolve();
          return;
        }

        // Reverse: start from output, go to input
        const layerIdx = totalLayers - 1 - step_i;
        
        // layerGradients (deltas) has length L, but network.layers has length L + 1.
        // So layerIdx in layerGradients corresponds to layerIdx + 1 in network.layers.
        const targetNetworkLayer = layerIdx + 1;

        let nodeOffset = 0;
        for (let l = 0; l < targetNetworkLayer; l++) {
          nodeOffset += this.network.layers[l];
        }
        for (let i = 0; i < layerGradients[layerIdx].length; i++) {
          if (this.nodes[nodeOffset + i]) {
            this.nodes[nodeOffset + i].value = layerGradients[layerIdx][i];
          }
        }

        this.activeLayer = targetNetworkLayer;
        this.draw();

        step_i++;
        setTimeout(step, delayMs);
      };

      step();
    });
  }

  /**
   * Override draw to show active layer glow during pass animation
   */
  drawNode(node) {
    const ctx = this.ctx;
    const r = node.radius;
    const isActive = this.activeNode &&
      this.activeNode.layer === node.layer &&
      this.activeNode.index === node.index;
    const isActiveLayer = this.activeLayer === node.layer;
    const isDropped = this.isNodeDropped(node);

    // Active layer glow during pass animation
    if (isActiveLayer && !isDropped) {
      const glowColor = this.passDirection === 'backward' ? 'rgba(255, 82, 82,' : 'rgba(0, 230, 118,';
      const gradient = ctx.createRadialGradient(node.x, node.y, 0, node.x, node.y, r * 3.5);
      gradient.addColorStop(0, glowColor + '0.5)');
      gradient.addColorStop(1, 'transparent');
      ctx.beginPath();
      ctx.arc(node.x, node.y, r * 3.5, 0, Math.PI * 2);
      ctx.fillStyle = gradient;
      ctx.fill();
    }

    // Glow effect for individually active node
    if (isActive && !isDropped) {
      const gradient = ctx.createRadialGradient(node.x, node.y, 0, node.x, node.y, r * 3);
      gradient.addColorStop(0, node.color + '40');
      gradient.addColorStop(1, 'transparent');
      ctx.beginPath();
      ctx.arc(node.x, node.y, r * 3, 0, Math.PI * 2);
      ctx.fillStyle = gradient;
      ctx.fill();
    }

    if (isDropped) {
      // Dropped neuron: dimmed with dashed outline
      ctx.beginPath();
      ctx.arc(node.x, node.y, r + 2, 0, Math.PI * 2);
      ctx.fillStyle = 'rgba(107, 111, 141, 0.08)';
      ctx.fill();

      ctx.beginPath();
      ctx.arc(node.x, node.y, r, 0, Math.PI * 2);
      ctx.fillStyle = 'rgba(20, 24, 50, 0.7)';
      ctx.fill();

      ctx.setLineDash([3, 3]);
      ctx.strokeStyle = 'rgba(224, 64, 251, 0.4)';
      ctx.lineWidth = 1.5;
      ctx.stroke();
      ctx.setLineDash([]);

      // Draw X marker
      const xSize = r * 0.5;
      ctx.beginPath();
      ctx.moveTo(node.x - xSize, node.y - xSize);
      ctx.lineTo(node.x + xSize, node.y + xSize);
      ctx.moveTo(node.x + xSize, node.y - xSize);
      ctx.lineTo(node.x - xSize, node.y + xSize);
      ctx.strokeStyle = 'rgba(224, 64, 251, 0.6)';
      ctx.lineWidth = 2;
      ctx.stroke();
      return;
    }

    // Outer ring
    ctx.beginPath();
    ctx.arc(node.x, node.y, r + 2, 0, Math.PI * 2);
    ctx.fillStyle = (isActive || isActiveLayer) ? node.color + '60' : node.color + '20';
    ctx.fill();

    // Main circle
    ctx.beginPath();
    ctx.arc(node.x, node.y, r, 0, Math.PI * 2);
    const grad = ctx.createRadialGradient(node.x - r * 0.3, node.y - r * 0.3, 0, node.x, node.y, r);
    grad.addColorStop(0, node.color);
    grad.addColorStop(1, this.darkenColor(node.color, 0.5));
    ctx.fillStyle = grad;
    ctx.fill();

    ctx.strokeStyle = isActiveLayer
      ? (this.passDirection === 'backward' ? '#FF5252' : '#00E676')
      : (isActive ? '#fff' : node.color + '80');
    ctx.lineWidth = (isActive || isActiveLayer) ? 2 : 1;
    ctx.stroke();

    // Value label
    if (node.value !== null && node.value !== undefined) {
      ctx.save();
      ctx.font = `bold ${Math.max(8, r * 0.6)}px JetBrains Mono`;
      ctx.fillStyle = '#fff';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillText(
        typeof node.value === 'number' ? node.value.toFixed(2) : node.value,
        node.x,
        node.y
      );
      ctx.restore();
    }
  }
}
