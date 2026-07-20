/* ============================================
   DNN Trainer - Training Loop with Optimizers
   ============================================ */

class DNNTrainer {
  constructor(builder) {
    this.builder = builder;
    this.isTraining = false;
    this.shouldStop = false;
    this.history = { loss: [], accuracy: [] };
    this.currentEpoch = 0;
    this.onEpochEnd = null;    // callback(epoch, loss, accuracy)
    this.onTrainingEnd = null; // callback(history)
    this.onBatchEnd = null;    // callback(epoch, batch, loss)
    this.onEarlyStop = null;   // callback(epoch, bestLoss)

    // Adam / RMSProp state
    this.mWeights = null; // first moment (Adam)
    this.vWeights = null; // second moment (Adam/RMSProp)
    this.mBiases = null;
    this.vBiases = null;
    this.t = 0; // timestep for Adam
  }

  // ── Initialize Optimizer State ──

  initOptimizerState() {
    const { weights, biases } = this.builder.network;
    const L = weights.length;

    this.mWeights = [];
    this.vWeights = [];
    this.mBiases = [];
    this.vBiases = [];

    for (let l = 0; l < L; l++) {
      this.mWeights[l] = weights[l].map(row => row.map(() => 0));
      this.vWeights[l] = weights[l].map(row => row.map(() => 0));
      this.mBiases[l] = biases[l].map(() => 0);
      this.vBiases[l] = biases[l].map(() => 0);
    }

    this.t = 0;
  }

  // ── Apply Gradients with Optimizer ──

  applyGradients(weightGrads, biasGrads) {
    const { weights, biases } = this.builder.network;
    const lr = this.builder.config.learningRate;
    const optimizer = this.builder.config.optimizer;
    const L = weights.length;

    switch (optimizer) {
      case 'sgd':
        this.applySGD(weights, biases, weightGrads, biasGrads, lr, L);
        break;
      case 'adam':
        this.applyAdam(weights, biases, weightGrads, biasGrads, lr, L);
        break;
      case 'rmsprop':
        this.applyRMSProp(weights, biases, weightGrads, biasGrads, lr, L);
        break;
      default:
        this.applySGD(weights, biases, weightGrads, biasGrads, lr, L);
    }
  }

  applySGD(weights, biases, wGrads, bGrads, lr, L) {
    for (let l = 0; l < L; l++) {
      for (let i = 0; i < weights[l].length; i++) {
        for (let j = 0; j < weights[l][i].length; j++) {
          weights[l][i][j] -= lr * wGrads[l][i][j];
        }
      }
      for (let j = 0; j < biases[l].length; j++) {
        biases[l][j] -= lr * bGrads[l][j];
      }
    }
  }

  applyAdam(weights, biases, wGrads, bGrads, lr, L) {
    const beta1 = 0.9;
    const beta2 = 0.999;
    const eps = 1e-8;
    this.t++;

    for (let l = 0; l < L; l++) {
      for (let i = 0; i < weights[l].length; i++) {
        for (let j = 0; j < weights[l][i].length; j++) {
          this.mWeights[l][i][j] = beta1 * this.mWeights[l][i][j] + (1 - beta1) * wGrads[l][i][j];
          this.vWeights[l][i][j] = beta2 * this.vWeights[l][i][j] + (1 - beta2) * wGrads[l][i][j] * wGrads[l][i][j];

          const mHat = this.mWeights[l][i][j] / (1 - Math.pow(beta1, this.t));
          const vHat = this.vWeights[l][i][j] / (1 - Math.pow(beta2, this.t));

          weights[l][i][j] -= lr * mHat / (Math.sqrt(vHat) + eps);
        }
      }
      for (let j = 0; j < biases[l].length; j++) {
        this.mBiases[l][j] = beta1 * this.mBiases[l][j] + (1 - beta1) * bGrads[l][j];
        this.vBiases[l][j] = beta2 * this.vBiases[l][j] + (1 - beta2) * bGrads[l][j] * bGrads[l][j];

        const mHat = this.mBiases[l][j] / (1 - Math.pow(beta1, this.t));
        const vHat = this.vBiases[l][j] / (1 - Math.pow(beta2, this.t));

        biases[l][j] -= lr * mHat / (Math.sqrt(vHat) + eps);
      }
    }
  }

  applyRMSProp(weights, biases, wGrads, bGrads, lr, L) {
    const decay = 0.9;
    const eps = 1e-8;

    for (let l = 0; l < L; l++) {
      for (let i = 0; i < weights[l].length; i++) {
        for (let j = 0; j < weights[l][i].length; j++) {
          this.vWeights[l][i][j] = decay * this.vWeights[l][i][j] + (1 - decay) * wGrads[l][i][j] * wGrads[l][i][j];
          weights[l][i][j] -= lr * wGrads[l][i][j] / (Math.sqrt(this.vWeights[l][i][j]) + eps);
        }
      }
      for (let j = 0; j < biases[l].length; j++) {
        this.vBiases[l][j] = decay * this.vBiases[l][j] + (1 - decay) * bGrads[l][j] * bGrads[l][j];
        biases[l][j] -= lr * bGrads[l][j] / (Math.sqrt(this.vBiases[l][j]) + eps);
      }
    }
  }

  // ── Accumulate Gradients ──

  accumulateGradients(accumulated, newGrads, count) {
    if (!accumulated) return newGrads;
    const L = accumulated.length;
    for (let l = 0; l < L; l++) {
      for (let i = 0; i < accumulated[l].length; i++) {
        if (Array.isArray(accumulated[l][i])) {
          for (let j = 0; j < accumulated[l][i].length; j++) {
            accumulated[l][i][j] += newGrads[l][i][j];
          }
        } else {
          accumulated[l][i] += newGrads[l][i];
        }
      }
    }
    return accumulated;
  }

  averageGradients(grads, batchSize) {
    return grads.map(layerGrad => {
      if (Array.isArray(layerGrad[0])) {
        return layerGrad.map(row => row.map(v => v / batchSize));
      }
      return layerGrad.map(v => v / batchSize);
    });
  }

  // ── Training Loop ──

  async train() {
    if (this.isTraining) return;

    this.isTraining = true;
    this.shouldStop = false;
    this.history = { loss: [], accuracy: [] };
    this.currentEpoch = 0;
    this.initOptimizerState();

    const { features, labels } = this.builder.data;
    const { epochs, batchSize } = this.builder.config;
    const numSamples = features.length;
    const useDropout = this.builder.config.dropoutRate > 0;

    // Early stopping state
    const earlyStop = this.builder.config.earlyStopEnabled;
    const patience = this.builder.config.earlyStopPatience;
    let bestLoss = Infinity;
    let patienceCounter = 0;
    let earlyStopTriggered = false;

    for (let epoch = 0; epoch < epochs; epoch++) {
      if (this.shouldStop) break;
      this.currentEpoch = epoch;

      let epochLoss = 0;
      let correct = 0;

      // Shuffle data indices
      const indices = Array.from({ length: numSamples }, (_, i) => i);
      for (let i = indices.length - 1; i > 0; i--) {
        const j = Math.floor(Math.random() * (i + 1));
        [indices[i], indices[j]] = [indices[j], indices[i]];
      }

      // Mini-batch processing
      const numBatches = Math.ceil(numSamples / batchSize);

      for (let b = 0; b < numBatches; b++) {
        if (this.shouldStop) break;

        const batchStart = b * batchSize;
        const batchEnd = Math.min(batchStart + batchSize, numSamples);
        const actualBatchSize = batchEnd - batchStart;

        let accWGrads = null;
        let accBGrads = null;

        for (let s = batchStart; s < batchEnd; s++) {
          const idx = indices[s];

          // Generate new dropout masks for each sample during training
          if (useDropout) {
            this.builder.generateDropoutMasks();
          }

          const result = this.builder.backpropagate(features[idx], labels[idx], useDropout);

          if (b === 0 && s === batchStart && this.onPassAnimation) {
            await this.onPassAnimation(result.layerOutputs, result.deltas);
          }

          epochLoss += result.loss;

          // Check accuracy
          if (this.builder.config.outputType === 'binary') {
            const pred = result.prediction >= 0.5 ? 1 : 0;
            if (pred === labels[idx]) correct++;
          } else {
            const predClass = result.prediction.indexOf(Math.max(...result.prediction));
            if (predClass === labels[idx]) correct++;
          }

          // Accumulate gradients
          accWGrads = this.accumulateGradients(accWGrads, result.weightGradients, actualBatchSize);
          accBGrads = this.accumulateGradients(accBGrads, result.biasGradients, actualBatchSize);
        }

        // Average and apply gradients
        accWGrads = this.averageGradients(accWGrads, actualBatchSize);
        accBGrads = this.averageGradients(accBGrads, actualBatchSize);
        this.applyGradients(accWGrads, accBGrads);
      }

      const avgLoss = epochLoss / numSamples;
      const accuracy = correct / numSamples;

      this.history.loss.push(parseFloat(avgLoss.toFixed(6)));
      this.history.accuracy.push(parseFloat(accuracy.toFixed(4)));

      // Early stopping check
      if (earlyStop) {
        if (avgLoss < bestLoss - 1e-6) {
          bestLoss = avgLoss;
          patienceCounter = 0;
        } else {
          patienceCounter++;
          if (patienceCounter >= patience) {
            earlyStopTriggered = true;
          }
        }
      }

      if (this.onEpochEnd) {
        this.onEpochEnd(epoch, avgLoss, accuracy);
      }

      // Check early stopping after callback
      if (earlyStopTriggered) {
        if (this.onEarlyStop) {
          this.onEarlyStop(epoch, bestLoss);
        }
        break;
      }

      // Yield to UI every epoch
      await new Promise(resolve => setTimeout(resolve, 0));
    }

    this.isTraining = false;
    if (this.onTrainingEnd) {
      this.onTrainingEnd(this.history);
    }
  }

  stop() {
    this.shouldStop = true;
  }

  // ── Loss Chart Drawing ──

  static drawLossChart(canvasId, history) {
    const canvas = document.getElementById(canvasId);
    if (!canvas || canvas.clientWidth === 0) return;
    
    const ctx = canvas.getContext('2d');
    const dpr = window.devicePixelRatio || 1;

    canvas.width = canvas.clientWidth * dpr;
    canvas.height = canvas.clientHeight * dpr;
    ctx.scale(dpr, dpr);

    const w = canvas.clientWidth;
    const h = canvas.clientHeight;
    const pad = { top: 30, right: 20, bottom: 40, left: 55 };
    const chartW = w - pad.left - pad.right;
    const chartH = h - pad.top - pad.bottom;

    ctx.clearRect(0, 0, w, h);

    // Background
    ctx.fillStyle = '#0A0E27';
    ctx.fillRect(0, 0, w, h);

    const lossData = history.loss;
    const accData = history.accuracy;

    if (lossData.length === 0) return;

    // Calculate scales
    const maxLoss = Math.max(...lossData) * 1.1 || 1;
    const minLoss = 0;
    const numPoints = lossData.length;

    // Grid lines
    ctx.strokeStyle = 'rgba(108, 99, 255, 0.1)';
    ctx.lineWidth = 1;
    const gridLines = 5;
    for (let i = 0; i <= gridLines; i++) {
      const y = pad.top + (i / gridLines) * chartH;
      ctx.beginPath();
      ctx.moveTo(pad.left, y);
      ctx.lineTo(pad.left + chartW, y);
      ctx.stroke();

      // Y-axis labels
      const val = maxLoss - (i / gridLines) * (maxLoss - minLoss);
      ctx.fillStyle = '#6B6F8D';
      ctx.font = '11px JetBrains Mono';
      ctx.textAlign = 'right';
      ctx.fillText(val.toFixed(2), pad.left - 8, y + 4);
    }

    // X-axis labels
    const xStep = Math.max(1, Math.floor(numPoints / 8));
    ctx.fillStyle = '#6B6F8D';
    ctx.font = '11px JetBrains Mono';
    ctx.textAlign = 'center';
    for (let i = 0; i < numPoints; i += xStep) {
      const x = pad.left + (i / (numPoints - 1 || 1)) * chartW;
      ctx.fillText(i + 1, x, h - pad.bottom + 18);
    }

    // Axis labels
    ctx.fillStyle = '#A0A3BD';
    ctx.font = '12px Tajawal';
    ctx.textAlign = 'center';
    ctx.fillText('Epoch', w / 2, h - 5);

    ctx.save();
    ctx.translate(14, h / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.fillText('Loss', 0, 0);
    ctx.restore();

    // Draw Loss line
    ctx.strokeStyle = '#6C63FF';
    ctx.lineWidth = 2.5;
    ctx.lineJoin = 'round';
    ctx.lineCap = 'round';
    ctx.beginPath();

    for (let i = 0; i < numPoints; i++) {
      const x = pad.left + (i / (numPoints - 1 || 1)) * chartW;
      const y = pad.top + (1 - (lossData[i] - minLoss) / (maxLoss - minLoss)) * chartH;
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.stroke();

    // Glow effect
    ctx.shadowColor = '#6C63FF';
    ctx.shadowBlur = 10;
    ctx.stroke();
    ctx.shadowBlur = 0;

    // Draw Accuracy line (scaled 0-1 mapped to chart)
    if (accData.length > 0) {
      ctx.strokeStyle = '#00E676';
      ctx.lineWidth = 2;
      ctx.setLineDash([6, 4]);
      ctx.beginPath();

      for (let i = 0; i < accData.length; i++) {
        const x = pad.left + (i / (numPoints - 1 || 1)) * chartW;
        // Map accuracy (0-1) to chart area using maxLoss as scale reference
        const y = pad.top + (1 - accData[i]) * chartH;
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      }
      ctx.stroke();
      ctx.setLineDash([]);
    }

    // Legend
    const legendX = pad.left + 10;
    const legendY = pad.top + 10;

    ctx.fillStyle = 'rgba(10, 14, 39, 0.8)';
    ctx.fillRect(legendX, legendY, 150, 50);
    ctx.strokeStyle = 'rgba(108, 99, 255, 0.3)';
    ctx.strokeRect(legendX, legendY, 150, 50);

    // Loss legend
    ctx.strokeStyle = '#6C63FF';
    ctx.lineWidth = 3;
    ctx.setLineDash([]);
    ctx.beginPath();
    ctx.moveTo(legendX + 10, legendY + 16);
    ctx.lineTo(legendX + 35, legendY + 16);
    ctx.stroke();
    ctx.fillStyle = '#A0A3BD';
    ctx.font = '12px Tajawal';
    ctx.textAlign = 'left';
    ctx.fillText('Loss', legendX + 42, legendY + 20);

    // Accuracy legend
    ctx.strokeStyle = '#00E676';
    ctx.lineWidth = 2;
    ctx.setLineDash([6, 4]);
    ctx.beginPath();
    ctx.moveTo(legendX + 10, legendY + 38);
    ctx.lineTo(legendX + 35, legendY + 38);
    ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillText('Accuracy', legendX + 42, legendY + 42);

    // Title
    ctx.fillStyle = '#FFFFFF';
    ctx.font = 'bold 13px Cairo';
    ctx.textAlign = 'center';
    ctx.fillText('منحنى التدريب (Training Curve)', w / 2, 18);
  }
}
