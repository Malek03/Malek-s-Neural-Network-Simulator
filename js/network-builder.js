/* ============================================
   Network Builder - Configuration & State
   ============================================ */

class NetworkBuilder {
  constructor() {
    this.config = {
      inputNodes: 2,
      hiddenLayers: [3],
      outputNodes: 1,
    };
    this.network = null; // Built network with weights
    this.data = null;    // Generated dataset
    this.isBuilt = false;
  }

  /**
   * Update configuration
   */
  setInputNodes(n) {
    this.config.inputNodes = Math.max(1, Math.min(10, n));
  }

  setHiddenLayerCount(n) {
    const count = Math.max(1, Math.min(5, n));
    const current = this.config.hiddenLayers;
    if (count > current.length) {
      while (this.config.hiddenLayers.length < count) {
        this.config.hiddenLayers.push(3);
      }
    } else {
      this.config.hiddenLayers = current.slice(0, count);
    }
  }

  setHiddenLayerNodes(layerIndex, nodes) {
    if (layerIndex < this.config.hiddenLayers.length) {
      this.config.hiddenLayers[layerIndex] = Math.max(1, Math.min(10, nodes));
    }
  }

  /**
   * Build the network with random weights and biases
   */
  build() {
    const layers = [
      this.config.inputNodes,
      ...this.config.hiddenLayers,
      this.config.outputNodes
    ];

    const weights = [];
    const biases = [];

    for (let i = 0; i < layers.length - 1; i++) {
      weights.push(MathUtils.initWeightMatrix(layers[i], layers[i + 1]));
      biases.push(MathUtils.initBiasVector(layers[i + 1]));
    }

    this.network = {
      layers,
      weights,
      biases,
      layerNames: this.getLayerNames(layers)
    };

    // Generate data
    this.data = DataGenerator.generate(this.config.inputNodes, 10);
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

  /**
   * Perform feedforward for a single input sample
   * Returns all intermediate values for visualization
   */
  feedforward(inputRow) {
    const { weights, biases, layers } = this.network;
    const layerOutputs = [inputRow]; // a[0] = input
    const layerLinear = [];          // z values

    let currentInput = inputRow;

    for (let l = 0; l < weights.length; l++) {
      const W = weights[l];
      const b = biases[l];
      const z = [];
      const a = [];

      for (let j = 0; j < W[0].length; j++) {
        let sum = 0;
        for (let i = 0; i < currentInput.length; i++) {
          sum += currentInput[i] * W[i][j];
        }
        sum += b[j];
        const zVal = parseFloat(sum.toFixed(6));
        z.push(zVal);
        a.push(parseFloat(MathUtils.sigmoid(zVal).toFixed(6)));
      }

      layerLinear.push(z);
      layerOutputs.push(a);
      currentInput = a;
    }

    return {
      layerOutputs,  // a[0]=input, a[1]=hidden1, ..., a[n]=output
      layerLinear,   // z[0]=first hidden z, ..., z[n-1]=output z
      prediction: currentInput[0]
    };
  }

  /**
   * Perform feedforward using matrix operations
   * Returns step-by-step matrix computations
   */
  feedforwardMatrix(inputRow) {
    const { weights, biases } = this.network;
    const steps = [];
    let X = [inputRow]; // 1×n matrix

    for (let l = 0; l < weights.length; l++) {
      const W = weights[l];
      const b = biases[l];

      // Z = X · W
      const Z_noBias = MathUtils.matMul(X, W);

      // Z = X · W + B
      const Z = MathUtils.addBias(Z_noBias, b);

      // A = σ(Z)
      const A = MathUtils.applySigmoid(Z);

      steps.push({
        layerIndex: l,
        X: X,
        W: W,
        B: [b],
        Z_noBias: Z_noBias,
        Z: Z,
        A: A,
      });

      X = A;
    }

    return {
      steps,
      prediction: X[0][0]
    };
  }

  /**
   * Perform backpropagation for a single sample
   * Returns step-by-step computations
   */
  backpropagate(inputRow, yTrue, learningRate = 0.1) {
    const { weights, biases } = this.network;
    const ff = this.feedforward(inputRow);
    const { layerOutputs, layerLinear } = ff;
    const L = weights.length;

    const steps = [];
    const deltas = new Array(L);
    const weightGradients = new Array(L);
    const biasGradients = new Array(L);

    // Step 1: Compute loss
    const yPred = ff.prediction;
    const loss = MathUtils.binaryCrossEntropy(yTrue, yPred);

    steps.push({
      type: 'loss',
      yTrue,
      yPred: MathUtils.formatNum(yPred),
      loss: MathUtils.formatNum(loss)
    });

    // Step 2: Output layer delta
    // dL/da = -(y/a) + (1-y)/(1-a)
    // da/dz = a*(1-a)
    // delta = a - y (for BCE + sigmoid)
    const outputA = layerOutputs[L];
    const outputDelta = outputA.map((a, i) => parseFloat((a - yTrue).toFixed(6)));
    deltas[L - 1] = outputDelta;

    steps.push({
      type: 'output_delta',
      layerIndex: L - 1,
      output: outputA.map(v => MathUtils.formatNum(v)),
      yTrue,
      delta: outputDelta.map(v => MathUtils.formatNum(v))
    });

    // Step 3: Hidden layer deltas (back-propagate)
    for (let l = L - 2; l >= 0; l--) {
      const a = layerOutputs[l + 1];
      const nextDelta = deltas[l + 1];
      const W = weights[l + 1];

      const delta = [];
      for (let j = 0; j < a.length; j++) {
        let sum = 0;
        for (let k = 0; k < nextDelta.length; k++) {
          sum += nextDelta[k] * W[j][k];
        }
        delta.push(parseFloat((sum * a[j] * (1 - a[j])).toFixed(6)));
      }
      deltas[l] = delta;

      steps.push({
        type: 'hidden_delta',
        layerIndex: l,
        activation: a.map(v => MathUtils.formatNum(v)),
        delta: delta.map(v => MathUtils.formatNum(v))
      });
    }

    // Step 4: Compute gradients and update weights
    const oldWeights = weights.map(w => w.map(row => [...row]));
    const oldBiases = biases.map(b => [...b]);

    for (let l = 0; l < L; l++) {
      const input = layerOutputs[l];
      const delta = deltas[l];

      const wGrad = [];
      for (let i = 0; i < input.length; i++) {
        wGrad[i] = [];
        for (let j = 0; j < delta.length; j++) {
          wGrad[i][j] = parseFloat((input[i] * delta[j]).toFixed(6));
          // Update weight
          weights[l][i][j] = parseFloat((weights[l][i][j] - learningRate * wGrad[i][j]).toFixed(6));
        }
      }

      const bGrad = delta.map(d => parseFloat(d.toFixed(6)));
      for (let j = 0; j < biases[l].length; j++) {
        biases[l][j] = parseFloat((biases[l][j] - learningRate * bGrad[j]).toFixed(6));
      }

      weightGradients[l] = wGrad;
      biasGradients[l] = bGrad;

      steps.push({
        type: 'weight_update',
        layerIndex: l,
        oldWeights: oldWeights[l],
        newWeights: weights[l].map(row => [...row]),
        oldBiases: oldBiases[l],
        newBiases: [...biases[l]],
        weightGradients: wGrad,
        biasGradients: bGrad,
        learningRate
      });
    }

    return {
      steps,
      loss: parseFloat(loss.toFixed(6)),
      prediction: yPred
    };
  }

  /**
   * Get network summary
   */
  getSummary() {
    if (!this.network) return null;
    const { layers, weights } = this.network;
    let totalParams = 0;
    for (let i = 0; i < weights.length; i++) {
      totalParams += layers[i] * layers[i + 1]; // weights
      totalParams += layers[i + 1];              // biases
    }
    return {
      layers: layers.join(' → '),
      totalLayers: layers.length,
      totalParams,
      architecture: layers
    };
  }
}
