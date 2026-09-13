/* ============================================
   RNN Trainer - Training Loop, BPTT & Visualization
   ============================================ */

class RNNTrainer {
  constructor(builder) {
    this.builder = builder;
    this.isTraining = false;
    this.shouldStop = false;
    this.history = { loss: [], epoch: [] };
    this.currentEpoch = 0;

    // Animation callbacks (same pattern as DNN)
    this.onPassAnimation = null;  // async callback(states, sequence) - forward/BPTT animation
    this.onEpochEnd = null;       // callback(epoch, loss)
    this.onTrainingEnd = null;    // callback(history)

    // Adam optimizer state
    this._adamState = null;
  }

  // ── Training Loop ──

  async train(onEpoch, onComplete) {
    if (!this.builder.isBuilt) return;

    this.isTraining = true;
    this.shouldStop = false;
    this.history = { loss: [], epoch: [] };
    this.currentEpoch = 0;

    const { epochs, learningRate, batchSize, optimizer } = this.builder.config;
    const { sequences, targets } = this.builder.data;
    const n = sequences.length;

    // Initialize Adam state
    if (optimizer === 'adam') {
      this._initAdamState();
    }

    for (let epoch = 0; epoch < epochs; epoch++) {
      if (this.shouldStop) break;

      let epochLoss = 0;

      // Shuffle data indices
      const indices = Array.from({ length: n }, (_, i) => i);
      for (let i = indices.length - 1; i > 0; i--) {
        const j = Math.floor(Math.random() * (i + 1));
        [indices[i], indices[j]] = [indices[j], indices[i]];
      }

      // Process in batches
      for (let b = 0; b < n; b += batchSize) {
        if (this.shouldStop) break;

        const batchEnd = Math.min(b + batchSize, n);
        const batchGrads = {};

        for (let i = b; i < batchEnd; i++) {
          const idx = indices[i];
          const seq = sequences[idx];
          const target = targets[idx];

          // Forward
          const states = this.builder.forward(seq);
          const loss = this.builder.computeLoss(states.outputs, target);
          epochLoss += loss;

          // Animation: show forward/BPTT on first sample of first batch each epoch
          if (b === 0 && i === b && this.onPassAnimation) {
            await this.onPassAnimation(states, seq);
          }

          // Backward (BPTT)
          const grads = this._bptt(states, seq, target);

          // Accumulate gradients
          this._accumulateGrads(batchGrads, grads);
        }

        // Average and apply gradients
        const count = batchEnd - b;
        this._averageGrads(batchGrads, count);

        if (optimizer === 'adam') {
          this._adamUpdate(batchGrads, epoch * Math.ceil(n / batchSize) + Math.floor(b / batchSize) + 1);
        } else {
          this._sgdUpdate(batchGrads, learningRate);
        }
      }

      epochLoss /= n;
      this.currentEpoch = epoch + 1;
      this.history.loss.push(epochLoss);
      this.history.epoch.push(epoch + 1);

      // Call epoch-end callbacks
      if (onEpoch) {
        onEpoch(epoch + 1, epochLoss);
      }
      if (this.onEpochEnd) {
        this.onEpochEnd(epoch + 1, epochLoss);
      }

      // Yield to UI every epoch
      await new Promise(r => setTimeout(r, 16));
    }

    this.isTraining = false;
    if (onComplete) onComplete(this.history);
    if (this.onTrainingEnd) this.onTrainingEnd(this.history);
  }

  stop() {
    this.shouldStop = true;
    this.isTraining = false;
  }

  // ── Backpropagation Through Time (Simple RNN) ──

  _bptt(states, sequence, target) {
    const w = this.builder.network.weights;
    const T = sequence.length;
    const H = this.builder.config.hiddenUnits;
    const { outputMode, clipGradient, numLayers } = this.builder.config;

    // Initialize gradient accumulators
    const grads = {
      W_xh: RNNBuilder._createMatrix(H, this.builder.config.inputFeatures).map(r => r.map(() => 0)),
      W_hh: RNNBuilder._createMatrix(H, H).map(r => r.map(() => 0)),
      b_h: RNNBuilder._createVector(H, 0),
      W_hy: RNNBuilder._createMatrix(this.builder.config.outputSize, H).map(r => r.map(() => 0)),
      b_y: RNNBuilder._createVector(this.builder.config.outputSize, 0),
    };

    if (numLayers === 2) {
      grads.W_xh2 = RNNBuilder._createMatrix(H, H).map(r => r.map(() => 0));
      grads.W_hh2 = RNNBuilder._createMatrix(H, H).map(r => r.map(() => 0));
      grads.b_h2 = RNNBuilder._createVector(H, 0);
    }

    // Output error at each time step
    const dOutputs = [];
    for (let t = 0; t < T; t++) {
      if (outputMode === 'many-to-one') {
        if (t === T - 1) {
          // Only compute error for last time step
          dOutputs.push(RNNBuilder._vecSub(states.outputs[t], target));
        } else {
          dOutputs.push(RNNBuilder._createVector(this.builder.config.outputSize, 0));
        }
      } else {
        // many-to-many: error at every step
        dOutputs.push(RNNBuilder._vecSub(states.outputs[t], target[t]));
      }
    }

    // BPTT: go backwards through time
    let dh_next = RNNBuilder._createVector(H, 0);
    let dh_next2 = RNNBuilder._createVector(H, 0);

    for (let t = T - 1; t >= 0; t--) {
      const dy = dOutputs[t];
      const x_t = states.inputs[t];

      if (numLayers === 2) {
        const h_t2 = states.hiddens2[t + 1];
        const h_prev2 = states.hiddens2[t];
        const h_t1 = states.hiddens[t + 1]; // This is x_t2 for layer 2
        const h_prev1 = states.hiddens[t];

        // --- Layer 2 ---
        // Gradients for output layer
        grads.W_hy = RNNBuilder._matAdd(grads.W_hy, RNNBuilder._outerProduct(dy, h_t2));
        grads.b_y = RNNBuilder._vecAdd(grads.b_y, dy);

        const dh_from_output = RNNBuilder._matVecMul(RNNBuilder._transpose(w.W_hy), dy);
        const dh2 = RNNBuilder._vecAdd(dh_from_output, dh_next2);

        // dz2 = dh2 * activation'(z2)
        const dz2 = dh2.map((v, i) => v * this.builder.activateDeriv(h_t2[i]));

        // Gradients for layer 2
        grads.W_xh2 = RNNBuilder._matAdd(grads.W_xh2, RNNBuilder._outerProduct(dz2, h_t1));
        grads.W_hh2 = RNNBuilder._matAdd(grads.W_hh2, RNNBuilder._outerProduct(dz2, h_prev2));
        grads.b_h2 = RNNBuilder._vecAdd(grads.b_h2, dz2);

        dh_next2 = RNNBuilder._matVecMul(RNNBuilder._transpose(w.W_hh2), dz2);

        // --- Layer 1 ---
        // Gradient from layer 2 + next time step
        const dh_from_layer2 = RNNBuilder._matVecMul(RNNBuilder._transpose(w.W_xh2), dz2);
        const dh1 = RNNBuilder._vecAdd(dh_from_layer2, dh_next);

        // dz1 = dh1 * activation'(z1)
        const dz1 = dh1.map((v, i) => v * this.builder.activateDeriv(h_t1[i]));

        // Gradients for layer 1
        grads.W_xh = RNNBuilder._matAdd(grads.W_xh, RNNBuilder._outerProduct(dz1, x_t));
        grads.W_hh = RNNBuilder._matAdd(grads.W_hh, RNNBuilder._outerProduct(dz1, h_prev1));
        grads.b_h = RNNBuilder._vecAdd(grads.b_h, dz1);

        dh_next = RNNBuilder._matVecMul(RNNBuilder._transpose(w.W_hh), dz1);

      } else {
        // --- Single Layer ---
        const h_t = states.hiddens[t + 1];
        const h_prev = states.hiddens[t];

        // Gradients for output layer
        grads.W_hy = RNNBuilder._matAdd(grads.W_hy, RNNBuilder._outerProduct(dy, h_t));
        grads.b_y = RNNBuilder._vecAdd(grads.b_y, dy);

        // dh from output + from next time step
        const dh_from_output = RNNBuilder._matVecMul(RNNBuilder._transpose(w.W_hy), dy);
        const dh = RNNBuilder._vecAdd(dh_from_output, dh_next);

        // dz = dh * activation'(z)
        const dz = dh.map((v, i) => v * this.builder.activateDeriv(h_t[i]));

        // Gradients for hidden layer
        grads.W_xh = RNNBuilder._matAdd(grads.W_xh, RNNBuilder._outerProduct(dz, x_t));
        grads.W_hh = RNNBuilder._matAdd(grads.W_hh, RNNBuilder._outerProduct(dz, h_prev));
        grads.b_h = RNNBuilder._vecAdd(grads.b_h, dz);

        // Propagate gradient to previous time step
        dh_next = RNNBuilder._matVecMul(RNNBuilder._transpose(w.W_hh), dz);
      }
    }

    // Gradient clipping
    this._clipGrads(grads, clipGradient);

    return grads;
  }

  // ── Gradient Clipping ──

  _clipGrads(grads, threshold) {
    for (const key in grads) {
      if (Array.isArray(grads[key][0])) {
        // Matrix
        for (let i = 0; i < grads[key].length; i++) {
          for (let j = 0; j < grads[key][i].length; j++) {
            grads[key][i][j] = Math.max(-threshold, Math.min(threshold, grads[key][i][j]));
          }
        }
      } else {
        // Vector
        for (let i = 0; i < grads[key].length; i++) {
          grads[key][i] = Math.max(-threshold, Math.min(threshold, grads[key][i]));
        }
      }
    }
  }

  // ── Accumulate Gradients ──

  _accumulateGrads(accum, grads) {
    for (const key in grads) {
      if (!accum[key]) {
        accum[key] = RNNBuilder._deepCopy(grads[key]);
      } else {
        if (Array.isArray(grads[key][0])) {
          accum[key] = RNNBuilder._matAdd(accum[key], grads[key]);
        } else {
          accum[key] = RNNBuilder._vecAdd(accum[key], grads[key]);
        }
      }
    }
  }

  _averageGrads(grads, count) {
    for (const key in grads) {
      if (Array.isArray(grads[key][0])) {
        grads[key] = RNNBuilder._matScale(grads[key], 1 / count);
      } else {
        grads[key] = RNNBuilder._vecScale(grads[key], 1 / count);
      }
    }
  }

  // ── SGD Update ──

  _sgdUpdate(grads, lr) {
    const w = this.builder.network.weights;
    const updateKeys = ['W_xh', 'W_hh', 'b_h', 'W_hy', 'b_y'];
    
    if (this.builder.config.numLayers === 2) {
      updateKeys.push('W_xh2', 'W_hh2', 'b_h2');
    }

    for (const key of updateKeys) {
      if (!grads[key]) continue;
      if (Array.isArray(w[key][0])) {
        for (let i = 0; i < w[key].length; i++) {
          for (let j = 0; j < w[key][i].length; j++) {
            w[key][i][j] -= lr * grads[key][i][j];
          }
        }
      } else {
        for (let i = 0; i < w[key].length; i++) {
          w[key][i] -= lr * grads[key][i];
        }
      }
    }
  }

  // ── Adam Optimizer ──

  _initAdamState() {
    const w = this.builder.network.weights;
    this._adamState = {};
    for (const key of ['W_xh', 'W_hh', 'b_h', 'W_hy', 'b_y']) {
      if (Array.isArray(w[key][0])) {
        this._adamState[key] = {
          m: w[key].map(r => r.map(() => 0)),
          v: w[key].map(r => r.map(() => 0)),
        };
      } else {
        this._adamState[key] = {
          m: w[key].map(() => 0),
          v: w[key].map(() => 0),
        };
      }
    }
  }

  _adamUpdate(grads, t) {
    const w = this.builder.network.weights;
    const lr = this.builder.config.learningRate;
    const beta1 = 0.9, beta2 = 0.999, eps = 1e-8;

    for (const key of ['W_xh', 'W_hh', 'b_h', 'W_hy', 'b_y']) {
      if (!grads[key] || !this._adamState[key]) continue;
      const s = this._adamState[key];

      if (Array.isArray(w[key][0])) {
        for (let i = 0; i < w[key].length; i++) {
          for (let j = 0; j < w[key][i].length; j++) {
            s.m[i][j] = beta1 * s.m[i][j] + (1 - beta1) * grads[key][i][j];
            s.v[i][j] = beta2 * s.v[i][j] + (1 - beta2) * grads[key][i][j] * grads[key][i][j];
            const mHat = s.m[i][j] / (1 - Math.pow(beta1, t));
            const vHat = s.v[i][j] / (1 - Math.pow(beta2, t));
            w[key][i][j] -= lr * mHat / (Math.sqrt(vHat) + eps);
          }
        }
      } else {
        for (let i = 0; i < w[key].length; i++) {
          s.m[i] = beta1 * s.m[i] + (1 - beta1) * grads[key][i];
          s.v[i] = beta2 * s.v[i] + (1 - beta2) * grads[key][i] * grads[key][i];
          const mHat = s.m[i] / (1 - Math.pow(beta1, t));
          const vHat = s.v[i] / (1 - Math.pow(beta2, t));
          w[key][i] -= lr * mHat / (Math.sqrt(vHat) + eps);
        }
      }
    }
  }

  // ── Draw Loss Chart ──

  static drawLossChart(canvasId, history) {
    const canvas = document.getElementById(canvasId);
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    const parent = canvas.parentElement;
    canvas.width = parent.clientWidth;
    canvas.height = parent.clientHeight || 300;

    const W = canvas.width;
    const H = canvas.height;
    const pad = { top: 40, right: 30, bottom: 50, left: 65 };

    ctx.clearRect(0, 0, W, H);

    // Background
    ctx.fillStyle = '#0a0e1a';
    ctx.fillRect(0, 0, W, H);

    const losses = history.loss;
    if (losses.length === 0) return;

    const maxLoss = Math.max(...losses) * 1.1 || 1;
    const minLoss = Math.min(0, Math.min(...losses));
    const plotW = W - pad.left - pad.right;
    const plotH = H - pad.top - pad.bottom;

    // Grid lines
    ctx.strokeStyle = 'rgba(255,255,255,0.06)';
    ctx.lineWidth = 1;
    const numGridLines = 5;
    for (let i = 0; i <= numGridLines; i++) {
      const y = pad.top + (plotH / numGridLines) * i;
      ctx.beginPath();
      ctx.moveTo(pad.left, y);
      ctx.lineTo(W - pad.right, y);
      ctx.stroke();

      // Y-axis labels
      const val = maxLoss - ((maxLoss - minLoss) / numGridLines) * i;
      ctx.fillStyle = 'rgba(255,255,255,0.4)';
      ctx.font = '11px JetBrains Mono, monospace';
      ctx.textAlign = 'right';
      ctx.fillText(val.toFixed(4), pad.left - 8, y + 4);
    }

    // X-axis labels
    const xLabelCount = Math.min(losses.length, 10);
    const xStep = Math.floor(losses.length / xLabelCount) || 1;
    ctx.fillStyle = 'rgba(255,255,255,0.4)';
    ctx.textAlign = 'center';
    for (let i = 0; i < losses.length; i += xStep) {
      const x = pad.left + (i / (losses.length - 1 || 1)) * plotW;
      ctx.fillText(history.epoch[i], x, H - pad.bottom + 20);
    }

    // Loss curve
    ctx.beginPath();
    ctx.strokeStyle = '#00E676';
    ctx.lineWidth = 2.5;
    ctx.shadowColor = '#00E676';
    ctx.shadowBlur = 8;

    for (let i = 0; i < losses.length; i++) {
      const x = pad.left + (i / (losses.length - 1 || 1)) * plotW;
      const y = pad.top + plotH - ((losses[i] - minLoss) / (maxLoss - minLoss)) * plotH;
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.stroke();
    ctx.shadowBlur = 0;

    // Fill under curve
    const lastX = pad.left + plotW;
    const lastY = pad.top + plotH - ((losses[losses.length - 1] - minLoss) / (maxLoss - minLoss)) * plotH;
    ctx.lineTo(lastX, pad.top + plotH);
    ctx.lineTo(pad.left, pad.top + plotH);
    ctx.closePath();
    const grad = ctx.createLinearGradient(0, pad.top, 0, pad.top + plotH);
    grad.addColorStop(0, 'rgba(0, 230, 118, 0.15)');
    grad.addColorStop(1, 'rgba(0, 230, 118, 0)');
    ctx.fillStyle = grad;
    ctx.fill();

    // Current loss indicator
    ctx.fillStyle = '#00E676';
    ctx.beginPath();
    ctx.arc(lastX, lastY, 5, 0, Math.PI * 2);
    ctx.fill();

    // Title
    ctx.fillStyle = 'rgba(255,255,255,0.7)';
    ctx.font = 'bold 13px Cairo, sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText('منحنى الخطأ (Loss Curve)', W / 2, 20);

    // Current loss value
    ctx.fillStyle = '#00E676';
    ctx.font = 'bold 12px JetBrains Mono, monospace';
    ctx.textAlign = 'left';
    ctx.fillText(`Loss: ${losses[losses.length - 1].toFixed(6)}`, pad.left + 10, 20);

    // Epoch label
    ctx.fillStyle = 'rgba(255,255,255,0.4)';
    ctx.font = '11px Cairo, sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText('Epoch', W / 2, H - 5);
  }

  // ── Draw Unfolded RNN Architecture ──

  static drawUnfoldedRNN(canvasId, config, states, highlightStep) {
    const canvas = document.getElementById(canvasId);
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    const parent = canvas.parentElement;
    canvas.width = parent.clientWidth;
    canvas.height = parent.clientHeight || 500;

    const W = canvas.width;
    const H = canvas.height;

    ctx.clearRect(0, 0, W, H);

    // Background
    ctx.fillStyle = 'transparent';
    ctx.fillRect(0, 0, W, H);

    const { timeSteps, hiddenUnits, inputFeatures, cellType, numLayers = 1 } = config;
    const T = timeSteps;

    // Layout calculations
    const pad = { top: 60, bottom: 60, left: 50, right: 50 };
    const usableW = W - pad.left - pad.right;
    const usableH = H - pad.top - pad.bottom;

    const stepWidth = usableW / T;
    const inputY = H - pad.bottom;
    const outputY = pad.top;

    const layerYs = [];
    if (numLayers === 1) {
      layerYs.push(pad.top + usableH * 0.5);
    } else {
      layerYs.push(pad.top + usableH * 0.66); // Layer 1
      layerYs.push(pad.top + usableH * 0.33); // Layer 2
    }

    const nodeRadius = Math.min(22, stepWidth * 0.15);
    const innerNodeR = Math.min(8, nodeRadius * 0.5);

    // ── Draw each time step ──
    for (let t = 0; t < T; t++) {
      const centerX = pad.left + stepWidth * t + stepWidth / 2;
      const isHighlighted = highlightStep === t;
      const alpha = isHighlighted ? 1 : 0.6;

      // ── Time step label ──
      ctx.fillStyle = `rgba(255,255,255,${alpha * 0.5})`;
      ctx.font = 'bold 11px JetBrains Mono, monospace';
      ctx.textAlign = 'center';
      ctx.fillText(`t = ${t}`, centerX, H - 10);

      // ── Input node ──
      ctx.beginPath();
      ctx.arc(centerX, inputY - 20, nodeRadius, 0, Math.PI * 2);
      ctx.fillStyle = `rgba(0, 229, 255, ${alpha * 0.15})`;
      ctx.fill();
      ctx.strokeStyle = `rgba(0, 229, 255, ${alpha})`;
      ctx.lineWidth = isHighlighted ? 2.5 : 1.5;
      ctx.stroke();

      // Input label
      ctx.fillStyle = `rgba(0, 229, 255, ${alpha})`;
      ctx.font = `bold ${Math.min(11, nodeRadius * 0.7)}px JetBrains Mono`;
      ctx.fillText(`x${t}`, centerX, inputY - 16);

      // Input value
      if (states && states.inputs[t]) {
        ctx.fillStyle = `rgba(255,255,255,${alpha * 0.5})`;
        ctx.font = '9px JetBrains Mono';
        const val = states.inputs[t].map(v => v.toFixed(2)).join(',');
        ctx.fillText(val.length > 8 ? val.substring(0, 8) + '..' : val, centerX, inputY - 3);
      }

      // ── Connection: Input → Layer 1 ──
      ctx.beginPath();
      ctx.moveTo(centerX, inputY - 20 - nodeRadius);
      ctx.lineTo(centerX, layerYs[0] + nodeRadius * 2 + 5);
      ctx.strokeStyle = `rgba(0, 229, 255, ${alpha * 0.4})`;
      ctx.lineWidth = isHighlighted ? 2 : 1;
      ctx.stroke();

      // Arrow
      RNNTrainer._drawArrow(ctx, centerX, layerYs[0] + nodeRadius * 2 + 5, 'up', `rgba(0, 229, 255, ${alpha * 0.4})`);

      // ── Draw Layers ──
      const cellW = Math.min(nodeRadius * 3, stepWidth * 0.5);
      const cellH = nodeRadius * 2.5;
      const cornerR = 8;
      
      const layerStates = [];
      if (states && states.hiddens) layerStates.push(states.hiddens);
      if (states && states.hiddens2) layerStates.push(states.hiddens2);

      for (let l = 0; l < numLayers; l++) {
        const lY = layerYs[l];
        
        // Connection L1 -> L2
        if (l > 0) {
          ctx.beginPath();
          ctx.moveTo(centerX, layerYs[l-1] - cellH / 2 - 5);
          ctx.lineTo(centerX, lY + cellH / 2 + 5);
          ctx.strokeStyle = `rgba(0, 229, 255, ${alpha * 0.4})`;
          ctx.lineWidth = isHighlighted ? 2 : 1;
          ctx.stroke();
          RNNTrainer._drawArrow(ctx, centerX, lY + cellH / 2 + 5, 'up', `rgba(0, 229, 255, ${alpha * 0.4})`);
        }

        const rx = centerX - cellW / 2;
        const ry = lY - cellH / 2;
        
        const displayNodes = Math.min(hiddenUnits, 3);
        const offset = 6;
        const cellColor = isHighlighted ? 'rgba(108, 99, 255, 0.25)' : 'rgba(108, 99, 255, 0.1)';
        const borderColor = `rgba(108, 99, 255, ${alpha})`;
        const lineWidth = isHighlighted ? 2 : 1.5;

        // Draw from back to front
        for (let n = displayNodes - 1; n >= 0; n--) {
          const bx = rx + n * offset;
          const by = ry - n * offset;
          
          // Draw dashed lines connecting back box to front box for depth
          if (n === displayNodes - 1 && n > 0) {
            ctx.beginPath();
            // Top-left
            ctx.moveTo(bx + cornerR, by);
            ctx.lineTo(rx + cornerR, ry);
            // Top-right
            ctx.moveTo(bx + cellW - cornerR, by);
            ctx.lineTo(rx + cellW - cornerR, ry);
            // Bottom-right
            ctx.moveTo(bx + cellW, by + cellH - cornerR);
            ctx.lineTo(rx + cellW, ry + cellH - cornerR);
            // Bottom-left
            ctx.moveTo(bx, by + cellH - cornerR);
            ctx.lineTo(rx, ry + cellH - cornerR);
            
            ctx.strokeStyle = `rgba(255, 255, 255, ${alpha * 0.3})`;
            ctx.setLineDash([3, 3]);
            ctx.lineWidth = 1;
            ctx.stroke();
            ctx.setLineDash([]);
          }

          // Cell background
          ctx.beginPath();
          ctx.moveTo(bx + cornerR, by);
          ctx.lineTo(bx + cellW - cornerR, by);
          ctx.quadraticCurveTo(bx + cellW, by, bx + cellW, by + cornerR);
          ctx.lineTo(bx + cellW, by + cellH - cornerR);
          ctx.quadraticCurveTo(bx + cellW, by + cellH, bx + cellW - cornerR, by + cellH);
          ctx.lineTo(bx + cornerR, by + cellH);
          ctx.quadraticCurveTo(bx, by + cellH, bx, by + cellH - cornerR);
          ctx.lineTo(bx, by + cornerR);
          ctx.quadraticCurveTo(bx, by, bx + cornerR, by);
          ctx.closePath();

          ctx.fillStyle = cellColor;
          ctx.fill();
          ctx.strokeStyle = borderColor;
          ctx.lineWidth = lineWidth;
          ctx.stroke();
        }

        if (hiddenUnits > 3) {
           ctx.fillStyle = `rgba(255,255,255,${alpha})`;
           ctx.font = '8px Arial';
           ctx.fillText('...', centerX + displayNodes * offset, lY + cellH/2 - displayNodes * offset);
        }

        // Cell type label
        ctx.fillStyle = `rgba(108, 99, 255, ${alpha})`;
        ctx.font = `bold ${Math.min(10, cellW * 0.2)}px JetBrains Mono`;
        const cellLabel = cellType === 'lstm' ? 'LSTM' : cellType === 'gru' ? 'GRU' : 'tanh';
        ctx.fillText(`L${l+1}`, centerX, lY - cellH / 2 + 10);
        
        // Hidden state label
        ctx.fillStyle = `rgba(255,255,255,${alpha * 0.5})`;
        ctx.font = '8px JetBrains Mono';
        
        // Hidden value
        if (layerStates[l] && layerStates[l][t + 1]) {
          ctx.fillStyle = `rgba(255,255,255,${alpha * 0.4})`;
          const hVal = layerStates[l][t + 1].slice(0, 2).map(v => v.toFixed(1)).join(',');
          ctx.fillText(`[${hVal}${hiddenUnits > 2 ? '..' : ''}]`, centerX, lY + cellH / 2 + 12);
        }
        
        // ── Recurrent connection: h_{t-1} → h_t ──
        if (t > 0) {
          const prevCenterX = pad.left + stepWidth * (t - 1) + stepWidth / 2;
          const startX = prevCenterX + cellW / 2 + 2;
          const endX = centerX - cellW / 2 - 2;
          const arrowY = lY;
  
          ctx.beginPath();
          ctx.moveTo(startX, arrowY);
          ctx.lineTo(endX - 8, arrowY);
          ctx.strokeStyle = `rgba(224, 64, 251, ${alpha * 0.7})`;
          ctx.lineWidth = isHighlighted ? 2.5 : 1.5;
          ctx.setLineDash([]);
          ctx.stroke();
  
          RNNTrainer._drawArrow(ctx, endX - 4, arrowY, 'left', `rgba(224, 64, 251, ${alpha * 0.7})`);
  
          if (l === 0) {
            ctx.fillStyle = `rgba(224, 64, 251, ${alpha * 0.6})`;
            ctx.font = '9px JetBrains Mono';
            ctx.fillText(`h${t - 1}→`, (startX + endX) / 2, arrowY - 8);
          }
        }
  
        // ── Self-loop for first cell ──
        if (t === 0) {
          const loopX = centerX - cellW / 2 - 15;
          ctx.beginPath();
          ctx.arc(loopX, lY, 12, -Math.PI * 0.3, Math.PI * 0.3);
          ctx.strokeStyle = `rgba(224, 64, 251, ${alpha * 0.4})`;
          ctx.lineWidth = 1.5;
          ctx.setLineDash([3, 3]);
          ctx.stroke();
          ctx.setLineDash([]);
  
          ctx.fillStyle = `rgba(224, 64, 251, ${alpha * 0.4})`;
          ctx.font = '8px JetBrains Mono';
          ctx.fillText('h₀=0', loopX - 10, lY - 16);
        }
      }

      // ── Connection: Top Layer → Output ──
      const topLayerY = layerYs[numLayers - 1];
      const drawOutput = config.outputMode !== 'many-to-one' || t === T - 1;
      
      if (drawOutput) {
        ctx.beginPath();
        ctx.moveTo(centerX, topLayerY - cellH / 2 - 5);
        ctx.lineTo(centerX, outputY + nodeRadius + 5);
        ctx.strokeStyle = `rgba(0, 230, 118, ${alpha * 0.4})`;
        ctx.lineWidth = isHighlighted ? 2 : 1;
        ctx.stroke();

        RNNTrainer._drawArrow(ctx, centerX, outputY + nodeRadius + 5, 'up', `rgba(0, 230, 118, ${alpha * 0.4})`);

        // ── Output node ──
        ctx.beginPath();
        ctx.arc(centerX, outputY, nodeRadius, 0, Math.PI * 2);
        ctx.fillStyle = `rgba(0, 230, 118, ${alpha * 0.15})`;
        ctx.fill();
        ctx.strokeStyle = `rgba(0, 230, 118, ${alpha})`;
        ctx.lineWidth = isHighlighted ? 2.5 : 1.5;
        ctx.stroke();

        ctx.fillStyle = `rgba(0, 230, 118, ${alpha})`;
        ctx.font = `bold ${Math.min(11, nodeRadius * 0.7)}px JetBrains Mono`;
        ctx.fillText(`y${t}`, centerX, outputY + 4);

        // Output value
        if (states && states.outputs[t]) {
          ctx.fillStyle = `rgba(255,255,255,${alpha * 0.5})`;
          ctx.font = '9px JetBrains Mono';
          const oVal = states.outputs[t].map(v => v.toFixed(3)).join(',');
          ctx.fillText(oVal.length > 8 ? oVal.substring(0, 8) + '..' : oVal, centerX, outputY - nodeRadius - 6);
        }
      }
    }

    // ── Shared weights label ──
    ctx.fillStyle = 'rgba(255, 214, 0, 0.5)';
    ctx.font = 'bold 11px Cairo, sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText('⚠ الأوزان مشتركة عبر جميع الخطوات الزمنية (Weight Sharing)', W / 2, H - pad.bottom + 40);

    // ── Layer labels ──
    ctx.fillStyle = 'rgba(255,255,255,0.3)';
    ctx.font = 'bold 12px Cairo, sans-serif';
    ctx.textAlign = 'right';
    ctx.fillText('المخرجات', pad.left - 10, outputY + 4);
    for (let l = 0; l < numLayers; l++) {
      ctx.fillText(`الطبقة ${l+1}`, pad.left - 10, layerYs[l] + 4);
    }
    ctx.fillText('المدخلات', pad.left - 10, inputY - 16);
  }

  // ── Helper: Draw Arrow ──

  static _drawArrow(ctx, x, y, direction, color) {
    const size = 5;
    ctx.fillStyle = color;
    ctx.beginPath();
    if (direction === 'up') {
      ctx.moveTo(x, y);
      ctx.lineTo(x - size, y + size * 1.5);
      ctx.lineTo(x + size, y + size * 1.5);
    } else if (direction === 'left') {
      ctx.moveTo(x, y);
      ctx.lineTo(x + size * 1.5, y - size);
      ctx.lineTo(x + size * 1.5, y + size);
    } else if (direction === 'right') {
      ctx.moveTo(x, y);
      ctx.lineTo(x - size * 1.5, y - size);
      ctx.lineTo(x - size * 1.5, y + size);
    }
    ctx.closePath();
    ctx.fill();
  }

  // ── Draw Weights Comparison Table ──

  static renderWeightsTable(weights, tableId) {
    const table = document.getElementById(tableId);
    if (!table || !weights) return;

    let html = '<thead><tr><th>الوزن</th><th>الأبعاد</th><th>القيم (أول 5)</th></tr></thead><tbody>';

    const weightNames = {
      W_xh: 'W_xh (إدخال→مخفي)',
      W_hh: 'W_hh (مخفي→مخفي)',
      b_h: 'b_h (انحياز مخفي)',
      W_hy: 'W_hy (مخفي→مخرج)',
      b_y: 'b_y (انحياز مخرج)',
    };

    for (const key of ['W_xh', 'W_hh', 'b_h', 'W_hy', 'b_y']) {
      if (!weights[key]) continue;
      const w = weights[key];
      const isMatrix = Array.isArray(w[0]);
      const dims = isMatrix ? `${w.length}×${w[0].length}` : `${w.length}`;

      let values;
      if (isMatrix) {
        values = w[0].slice(0, 5).map(v => v.toFixed(4)).join(', ');
      } else {
        values = w.slice(0, 5).map(v => v.toFixed(4)).join(', ');
      }

      html += `<tr>
        <td style="color: var(--primary-light); font-family: var(--font-mono)">${weightNames[key] || key}</td>
        <td style="font-family: var(--font-mono)">${dims}</td>
        <td style="font-family: var(--font-mono); font-size: 0.8rem">${values}${(isMatrix ? w[0].length : w.length) > 5 ? ' ...' : ''}</td>
      </tr>`;
    }

    html += '</tbody>';
    table.innerHTML = html;
  }
}
