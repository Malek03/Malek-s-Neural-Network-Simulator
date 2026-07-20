/* ============================================
   DNN Builder - Deep Neural Network Configuration & State
   ============================================ */

class DNNBuilder {
  constructor() {
    this.config = {
      inputNodes: 2,
      hiddenLayers: [4, 4],       // Array of neuron counts
      activations: ['relu', 'relu'], // Activation per hidden layer
      outputType: 'binary',       // 'binary' or 'multiclass'
      outputClasses: 3,           // Number of classes for multiclass
      optimizer: 'adam',          // 'sgd', 'adam', 'rmsprop'
      learningRate: 0.01,
      epochs: 50,
      batchSize: 8,
    };
    this.network = null;
    this.data = null;
    this.isBuilt = false;
  }

  // ── Configuration Setters ──

  setInputNodes(n) {
    this.config.inputNodes = Math.max(1, Math.min(255, n));
  }

  setHiddenLayerCount(n) {
    const count = Math.max(1, Math.min(20, n));
    const current = this.config.hiddenLayers;
    const currentAct = this.config.activations;
    if (count > current.length) {
      while (this.config.hiddenLayers.length < count) {
        this.config.hiddenLayers.push(4);
        this.config.activations.push('relu');
      }
    } else {
      this.config.hiddenLayers = current.slice(0, count);
      this.config.activations = currentAct.slice(0, count);
    }
  }

  setHiddenLayerNodes(layerIndex, nodes) {
    if (layerIndex >= 0 && layerIndex < this.config.hiddenLayers.length) {
      this.config.hiddenLayers[layerIndex] = Math.max(1, Math.min(255, nodes));
    }
  }

  setLayerActivation(layerIndex, activation) {
    if (layerIndex < this.config.activations.length) {
      this.config.activations[layerIndex] = activation;
    }
  }

  setOutputType(type) {
    this.config.outputType = type; // 'binary' or 'multiclass'
  }

  setOutputClasses(n) {
    this.config.outputClasses = Math.max(2, Math.min(10, n));
  }

  setOptimizer(opt) {
    this.config.optimizer = opt;
  }

  setLearningRate(lr) {
    this.config.learningRate = Math.max(0.0001, Math.min(1, lr));
  }

  setEpochs(n) {
    this.config.epochs = Math.max(1, Math.min(1000, n));
  }

  setBatchSize(n) {
    this.config.batchSize = Math.max(1, Math.min(128, n));
  }

  get outputNodes() {
    return this.config.outputType === 'binary' ? 1 : this.config.outputClasses;
  }

  get outputActivation() {
    return this.config.outputType === 'binary' ? 'sigmoid' : 'softmax';
  }

  // ── Build Network ──

  build() {
    const layers = [
      this.config.inputNodes,
      ...this.config.hiddenLayers,
      this.outputNodes
    ];

    const weights = [];
    const biases = [];

    for (let i = 0; i < layers.length - 1; i++) {
      weights.push(MathUtils.initWeightMatrix(layers[i], layers[i + 1]));
      biases.push(MathUtils.initBiasVector(layers[i + 1]));
    }

    // Build activation list: one per connection (hidden layers + output)
    const activations = [
      ...this.config.activations,
      this.outputActivation
    ];

    this.network = {
      layers,
      weights,
      biases,
      activations,
      layerNames: this.getLayerNames(layers),
      initialWeights: JSON.parse(JSON.stringify(weights))
    };

    // Generate data
    const numSamples = Math.max(20, this.config.batchSize * 4);
    if (this.config.outputType === 'binary') {
      this.data = DNNBuilder.generateBinaryData(this.config.inputNodes, numSamples);
    } else {
      this.data = DNNBuilder.generateMulticlassData(this.config.inputNodes, numSamples, this.config.outputClasses);
    }

    this.isBuilt = true;
    return this.network;
  }

  getLayerNames(layers) {
    const names = ['طبقة الإدخال'];
    for (let i = 1; i < layers.length - 1; i++) {
      names.push(`طبقة مخفية ${i}`);
    }
    names.push('طبقة الإخراج');
    return names;
  }

  // ── Feedforward ──

  feedforward(inputRow) {
    const { weights, biases, activations, layers } = this.network;
    const layerOutputs = [inputRow];
    const layerLinear = [];

    let currentInput = inputRow;

    for (let l = 0; l < weights.length; l++) {
      const W = weights[l];
      const b = biases[l];
      const actName = activations[l];
      const z = [];
      const a = [];

      // Compute z for each neuron
      for (let j = 0; j < W[0].length; j++) {
        let sum = 0;
        for (let i = 0; i < currentInput.length; i++) {
          sum += currentInput[i] * W[i][j];
        }
        sum += b[j];
        z.push(parseFloat(sum.toFixed(6)));
      }

      // Apply activation
      if (actName === 'softmax') {
        const sm = MathUtils.softmax(z);
        for (let j = 0; j < sm.length; j++) {
          a.push(parseFloat(sm[j].toFixed(6)));
        }
      } else {
        for (let j = 0; j < z.length; j++) {
          a.push(parseFloat(MathUtils.activate(z[j], actName).toFixed(6)));
        }
      }

      layerLinear.push(z);
      layerOutputs.push(a);
      currentInput = a;
    }

    return {
      layerOutputs,
      layerLinear,
      prediction: this.config.outputType === 'binary' ? currentInput[0] : currentInput
    };
  }

  // ── Backpropagation (single sample) ──

  backpropagate(inputRow, yTrue) {
    const { weights, biases, activations } = this.network;
    const ff = this.feedforward(inputRow);
    const { layerOutputs, layerLinear } = ff;
    const L = weights.length;

    const deltas = new Array(L);
    const weightGradients = new Array(L);
    const biasGradients = new Array(L);

    // Compute loss
    let loss;
    if (this.config.outputType === 'binary') {
      loss = MathUtils.binaryCrossEntropy(yTrue, ff.prediction);
    } else {
      // yTrue is an index, convert to one-hot
      const oneHot = new Array(this.outputNodes).fill(0);
      oneHot[yTrue] = 1;
      loss = MathUtils.categoricalCrossEntropy(oneHot, ff.prediction);
    }

    // Output layer delta
    const outputA = layerOutputs[L];
    if (this.config.outputType === 'binary') {
      deltas[L - 1] = outputA.map((a) => parseFloat((a - yTrue).toFixed(6)));
    } else {
      // For softmax + CCE: delta = a - y_one_hot
      const oneHot = new Array(this.outputNodes).fill(0);
      oneHot[yTrue] = 1;
      deltas[L - 1] = outputA.map((a, i) => parseFloat((a - oneHot[i]).toFixed(6)));
    }

    // Hidden layer deltas
    for (let l = L - 2; l >= 0; l--) {
      const z = layerLinear[l];
      const nextDelta = deltas[l + 1];
      const W = weights[l + 1];
      const actName = activations[l];

      const delta = [];
      for (let j = 0; j < z.length; j++) {
        let sum = 0;
        for (let k = 0; k < nextDelta.length; k++) {
          sum += nextDelta[k] * W[j][k];
        }
        const dAct = MathUtils.activateDerivative(z[j], actName);
        delta.push(parseFloat((sum * dAct).toFixed(6)));
      }
      deltas[l] = delta;
    }

    // Compute gradients
    for (let l = 0; l < L; l++) {
      const input = layerOutputs[l];
      const delta = deltas[l];

      const wGrad = [];
      for (let i = 0; i < input.length; i++) {
        wGrad[i] = [];
        for (let j = 0; j < delta.length; j++) {
          wGrad[i][j] = parseFloat((input[i] * delta[j]).toFixed(6));
        }
      }

      const bGrad = delta.map(d => parseFloat(d.toFixed(6)));

      weightGradients[l] = wGrad;
      biasGradients[l] = bGrad;
    }

    return {
      loss: parseFloat(loss.toFixed(6)),
      prediction: ff.prediction,
      weightGradients,
      biasGradients,
      layerOutputs,
      deltas
    };
  }

  // ── Data Generation ──

  static generateBinaryData(numFeatures, numSamples) {
    const features = [];
    const labels = [];
    for (let i = 0; i < numSamples; i++) {
      const row = [];
      for (let j = 0; j < numFeatures; j++) {
        row.push(parseFloat((Math.random() * 2 - 1).toFixed(4)));
      }
      features.push(row);
      // Simple classification rule: sum > 0 => 1, else 0
      const sum = row.reduce((a, b) => a + b, 0);
      labels.push(sum > 0 ? 1 : 0);
    }
    return { features, labels, type: 'binary' };
  }

  static generateMulticlassData(numFeatures, numSamples, numClasses) {
    const features = [];
    const labels = [];
    for (let i = 0; i < numSamples; i++) {
      const row = [];
      for (let j = 0; j < numFeatures; j++) {
        row.push(parseFloat((Math.random() * 2 - 1).toFixed(4)));
      }
      features.push(row);
      // Assign class based on angle / region
      const angle = Math.atan2(row[1] || row[0], row[0]) + Math.PI;
      const classIdx = Math.floor((angle / (2 * Math.PI)) * numClasses) % numClasses;
      labels.push(classIdx);
    }
    return { features, labels, type: 'multiclass', numClasses };
  }

  // ── Render Data Table ──

  renderDataTable() {
    if (!this.data) return '<p>لا توجد بيانات</p>';
    const { features, labels } = this.data;
    const numFeatures = features[0].length;

    let html = '<table class="data-table" id="dnnDataTable"><thead><tr>';
    html += '<th>#</th>';
    for (let i = 0; i < numFeatures; i++) {
      html += `<th>X${i + 1}</th>`;
    }
    html += '<th>Y (Label)</th>';
    html += '</tr></thead><tbody>';

    for (let i = 0; i < features.length; i++) {
      html += `<tr data-row="${i}">`;
      html += `<td>${i + 1}</td>`;
      for (let j = 0; j < numFeatures; j++) {
        html += `<td>${features[i][j]}</td>`;
      }
      if (this.config.outputType === 'binary') {
        html += `<td class="label-${labels[i]}">${labels[i]}</td>`;
      } else {
        html += `<td style="color: var(--accent); font-weight: 700;">Class ${labels[i]}</td>`;
      }
      html += '</tr>';
    }

    html += '</tbody></table>';
    return html;
  }

  // ── Network Summary ──

  getSummary() {
    if (!this.network) return null;
    const { layers, weights } = this.network;
    let totalParams = 0;
    for (let i = 0; i < weights.length; i++) {
      totalParams += layers[i] * layers[i + 1];
      totalParams += layers[i + 1];
    }
    return {
      layers: layers.join(' → '),
      totalLayers: layers.length,
      totalParams,
      architecture: layers
    };
  }
}
