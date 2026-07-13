/* ============================================
   Math Utilities for Neural Network
   ============================================ */

const MathUtils = {
  // Sigmoid activation function
  sigmoid(x) {
    return 1 / (1 + Math.exp(-x));
  },

  // Sigmoid derivative
  sigmoidDerivative(x) {
    const s = MathUtils.sigmoid(x);
    return s * (1 - s);
  },

  // ReLU activation function
  relu(x) {
    return Math.max(0, x);
  },

  // ReLU derivative
  reluDerivative(x) {
    return x > 0 ? 1 : 0;
  },

  // Tanh activation function
  tanh(x) {
    return Math.tanh(x);
  },

  // Tanh derivative
  tanhDerivative(x) {
    const t = Math.tanh(x);
    return 1 - t * t;
  },

  // Leaky ReLU activation function
  leakyRelu(x, alpha = 0.01) {
    return x > 0 ? x : alpha * x;
  },

  // Leaky ReLU derivative
  leakyReluDerivative(x, alpha = 0.01) {
    return x > 0 ? 1 : alpha;
  },

  // Softmax activation (for array/vector)
  softmax(arr) {
    const maxVal = Math.max(...arr);
    const exps = arr.map(x => Math.exp(x - maxVal));
    const sum = exps.reduce((a, b) => a + b, 0);
    return exps.map(e => e / sum);
  },

  // Categorical Cross Entropy Loss
  categoricalCrossEntropy(yTrueArr, yPredArr) {
    const eps = 1e-7;
    let loss = 0;
    for (let i = 0; i < yTrueArr.length; i++) {
      loss -= yTrueArr[i] * Math.log(Math.max(eps, yPredArr[i]));
    }
    return loss;
  },

  // Apply activation by name (scalar)
  activate(x, name) {
    switch (name) {
      case 'sigmoid': return MathUtils.sigmoid(x);
      case 'relu': return MathUtils.relu(x);
      case 'tanh': return MathUtils.tanh(x);
      case 'leaky_relu': return MathUtils.leakyRelu(x);
      default: return MathUtils.relu(x);
    }
  },

  // Apply activation derivative by name (scalar)
  activateDerivative(x, name) {
    switch (name) {
      case 'sigmoid': return MathUtils.sigmoidDerivative(x);
      case 'relu': return MathUtils.reluDerivative(x);
      case 'tanh': return MathUtils.tanhDerivative(x);
      case 'leaky_relu': return MathUtils.leakyReluDerivative(x);
      default: return MathUtils.reluDerivative(x);
    }
  },

  // Apply activation to matrix by name
  applyActivation(matrix, name) {
    if (name === 'softmax') {
      return matrix.map(row => {
        const sm = MathUtils.softmax(row);
        return sm.map(v => parseFloat(v.toFixed(6)));
      });
    }
    return matrix.map(row =>
      row.map(val => parseFloat(MathUtils.activate(val, name).toFixed(6)))
    );
  },

  // Random weight initialization (Xavier/Glorot)
  randomWeight(fanIn, fanOut) {
    const limit = Math.sqrt(6 / (fanIn + fanOut));
    return (Math.random() * 2 - 1) * limit;
  },

  // Random bias initialization
  randomBias() {
    return parseFloat((Math.random() * 0.4 - 0.2).toFixed(4));
  },

  // Initialize weight matrix
  initWeightMatrix(rows, cols) {
    const matrix = [];
    for (let i = 0; i < rows; i++) {
      matrix[i] = [];
      for (let j = 0; j < cols; j++) {
        matrix[i][j] = parseFloat(MathUtils.randomWeight(rows, cols).toFixed(4));
      }
    }
    return matrix;
  },

  // Initialize bias vector
  initBiasVector(size) {
    const bias = [];
    for (let i = 0; i < size; i++) {
      bias[i] = MathUtils.randomBias();
    }
    return bias;
  },

  // Matrix multiply: A (m×n) × B (n×p) = C (m×p)
  matMul(A, B) {
    const m = A.length;
    const n = A[0].length;
    const p = B[0].length;
    const C = [];
    for (let i = 0; i < m; i++) {
      C[i] = [];
      for (let j = 0; j < p; j++) {
        let sum = 0;
        for (let k = 0; k < n; k++) {
          sum += A[i][k] * B[k][j];
        }
        C[i][j] = parseFloat(sum.toFixed(6));
      }
    }
    return C;
  },

  // Vector dot product
  dotProduct(a, b) {
    let sum = 0;
    for (let i = 0; i < a.length; i++) {
      sum += a[i] * b[i];
    }
    return sum;
  },

  // Add bias to each row of matrix
  addBias(matrix, bias) {
    return matrix.map(row =>
      row.map((val, j) => parseFloat((val + bias[j]).toFixed(6)))
    );
  },

  // Apply sigmoid element-wise to matrix
  applySigmoid(matrix) {
    return matrix.map(row =>
      row.map(val => parseFloat(MathUtils.sigmoid(val).toFixed(6)))
    );
  },

  // Transpose matrix
  transpose(matrix) {
    const rows = matrix.length;
    const cols = matrix[0].length;
    const result = [];
    for (let j = 0; j < cols; j++) {
      result[j] = [];
      for (let i = 0; i < rows; i++) {
        result[j][i] = matrix[i][j];
      }
    }
    return result;
  },

  // Element-wise multiplication
  elementMul(A, B) {
    return A.map((row, i) =>
      row.map((val, j) => parseFloat((val * B[i][j]).toFixed(6)))
    );
  },

  // Element-wise subtraction
  elementSub(A, B) {
    return A.map((row, i) =>
      row.map((val, j) => parseFloat((val - B[i][j]).toFixed(6)))
    );
  },

  // Scalar multiply matrix
  scalarMul(matrix, scalar) {
    return matrix.map(row =>
      row.map(val => parseFloat((val * scalar).toFixed(6)))
    );
  },

  // Binary Cross Entropy Loss
  binaryCrossEntropy(yTrue, yPred) {
    const eps = 1e-7;
    const clipped = Math.max(eps, Math.min(1 - eps, yPred));
    return -(yTrue * Math.log(clipped) + (1 - yTrue) * Math.log(1 - clipped));
  },

  // Format number for display
  formatNum(n, decimals = 4) {
    return parseFloat(n).toFixed(decimals);
  },

  // Create identity-like structure for display
  matrixToString(matrix) {
    return matrix.map(row => '[' + row.map(v => MathUtils.formatNum(v)).join(', ') + ']').join('\n');
  }
};
