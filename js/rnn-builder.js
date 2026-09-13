/* ============================================
   RNN Builder - Recurrent Neural Network Configuration & State
   ============================================ */

class RNNBuilder {
  constructor() {
    this.config = {
      inputFeatures: 1,         // Number of input features per time step
      hiddenUnits: 4,           // Number of hidden units
      timeSteps: 4,             // Number of time steps (sequence length)
      numLayers: 1,             // Number of RNN layers (1-3)
      cellType: 'simple',       // 'simple', 'lstm', 'gru'
      activation: 'tanh',       // 'tanh' or 'relu'
      outputMode: 'many-to-one', // 'many-to-one' or 'many-to-many'
      outputSize: 1,            // Output dimension
      optimizer: 'adam',
      learningRate: 0.01,
      epochs: 100,
      batchSize: 8,
      clipGradient: 5,          // Gradient clipping threshold
    };
    this.network = null;
    this.data = null;
    this.isBuilt = false;
  }

  // ── Configuration Setters ──

  setInputFeatures(n) {
    this.config.inputFeatures = Math.max(1, Math.min(12, n));
  }

  setHiddenUnits(n) {
    this.config.hiddenUnits = Math.max(1, Math.min(20, n));
  }

  setTimeSteps(n) {
    this.config.timeSteps = Math.max(2, Math.min(10, n));
  }

  setNumLayers(n) {
    this.config.numLayers = Math.max(1, Math.min(3, n));
  }

  setCellType(type) {
    this.config.cellType = type;
  }

  setActivation(act) {
    this.config.activation = act;
  }

  setOutputMode(mode) {
    this.config.outputMode = mode;
  }

  setOptimizer(opt) {
    this.config.optimizer = opt;
  }

  setLearningRate(lr) {
    this.config.learningRate = Math.max(0.0001, Math.min(1, lr));
  }

  setEpochs(n) {
    this.config.epochs = Math.max(1, Math.min(500, n));
  }

  setBatchSize(n) {
    this.config.batchSize = Math.max(1, Math.min(64, n));
  }

  setClipGradient(val) {
    this.config.clipGradient = Math.max(0.1, Math.min(50, val));
  }

  // ── Math Utilities ──

  static _tanh(x) {
    if (x > 20) return 1;
    if (x < -20) return -1;
    const ep = Math.exp(2 * x);
    return (ep - 1) / (ep + 1);
  }

  static _tanhDeriv(tanhVal) {
    return 1 - tanhVal * tanhVal;
  }

  static _relu(x) {
    return Math.max(0, x);
  }

  static _reluDeriv(x) {
    return x > 0 ? 1 : 0;
  }

  static _sigmoid(x) {
    if (x > 20) return 1;
    if (x < -20) return 0;
    return 1 / (1 + Math.exp(-x));
  }

  static _sigmoidDeriv(sigVal) {
    return sigVal * (1 - sigVal);
  }

  activate(x) {
    if (this.config.activation === 'relu') return RNNBuilder._relu(x);
    return RNNBuilder._tanh(x);
  }

  activateDeriv(val) {
    if (this.config.activation === 'relu') return RNNBuilder._reluDeriv(val);
    return RNNBuilder._tanhDeriv(val);
  }

  // ── Weight Initialization (Xavier) ──

  static _xavierInit(fanIn, fanOut) {
    const limit = Math.sqrt(6 / (fanIn + fanOut));
    return (Math.random() * 2 - 1) * limit;
  }

  static _createMatrix(rows, cols, fanIn, fanOut) {
    const m = [];
    for (let i = 0; i < rows; i++) {
      m[i] = [];
      for (let j = 0; j < cols; j++) {
        m[i][j] = RNNBuilder._xavierInit(fanIn || rows, fanOut || cols);
      }
    }
    return m;
  }

  static _createVector(size, val = 0) {
    return new Array(size).fill(val);
  }

  // ── Matrix Ops ──

  static _matVecMul(mat, vec) {
    // mat: [rows][cols], vec: [cols] → result: [rows]
    const result = [];
    for (let i = 0; i < mat.length; i++) {
      let sum = 0;
      for (let j = 0; j < vec.length; j++) {
        sum += mat[i][j] * vec[j];
      }
      result.push(sum);
    }
    return result;
  }

  static _vecAdd(a, b) {
    return a.map((v, i) => v + (b[i] || 0));
  }

  static _vecSub(a, b) {
    return a.map((v, i) => v - (b[i] || 0));
  }

  static _vecMul(a, b) {
    return a.map((v, i) => v * (b[i] || 0));
  }

  static _vecScale(vec, s) {
    return vec.map(v => v * s);
  }

  static _outerProduct(a, b) {
    // a: [m], b: [n] → result: [m][n]
    const result = [];
    for (let i = 0; i < a.length; i++) {
      result[i] = [];
      for (let j = 0; j < b.length; j++) {
        result[i][j] = a[i] * b[j];
      }
    }
    return result;
  }

  static _matAdd(a, b) {
    return a.map((row, i) => row.map((v, j) => v + b[i][j]));
  }

  static _matScale(mat, s) {
    return mat.map(row => row.map(v => v * s));
  }

  static _transpose(mat) {
    const rows = mat.length;
    const cols = mat[0].length;
    const result = [];
    for (let j = 0; j < cols; j++) {
      result[j] = [];
      for (let i = 0; i < rows; i++) {
        result[j][i] = mat[i][j];
      }
    }
    return result;
  }

  static _deepCopy(obj) {
    return JSON.parse(JSON.stringify(obj));
  }

  // ── Build Network ──

  build() {
    // Auto-set output size to match input features (target data is same dimension)
    this.config.outputSize = this.config.inputFeatures;
    
    const { inputFeatures, hiddenUnits, outputSize, outputMode, timeSteps } = this.config;

    // Simple RNN weights (shared across time steps)
    const weights = {
      // Input → Hidden: [hiddenUnits x inputFeatures]
      W_xh: RNNBuilder._createMatrix(hiddenUnits, inputFeatures),
      // Hidden → Hidden: [hiddenUnits x hiddenUnits]
      W_hh: RNNBuilder._createMatrix(hiddenUnits, hiddenUnits),
      // Hidden bias: [hiddenUnits]
      b_h: RNNBuilder._createVector(hiddenUnits),
      // Hidden → Output: [outputSize x hiddenUnits]
      W_hy: RNNBuilder._createMatrix(outputSize, hiddenUnits),
      // Output bias: [outputSize]
      b_y: RNNBuilder._createVector(outputSize),
    };

    // For LSTM: additional gate weights
    if (this.config.cellType === 'lstm') {
      // Forget gate
      weights.W_f_x = RNNBuilder._createMatrix(hiddenUnits, inputFeatures);
      weights.W_f_h = RNNBuilder._createMatrix(hiddenUnits, hiddenUnits);
      weights.b_f = new Array(hiddenUnits).fill(1); // Initialize forget gate bias to 1
      // Input gate
      weights.W_i_x = RNNBuilder._createMatrix(hiddenUnits, inputFeatures);
      weights.W_i_h = RNNBuilder._createMatrix(hiddenUnits, hiddenUnits);
      weights.b_i = RNNBuilder._createVector(hiddenUnits);
      // Cell candidate
      weights.W_c_x = RNNBuilder._createMatrix(hiddenUnits, inputFeatures);
      weights.W_c_h = RNNBuilder._createMatrix(hiddenUnits, hiddenUnits);
      weights.b_c = RNNBuilder._createVector(hiddenUnits);
      // Output gate
      weights.W_o_x = RNNBuilder._createMatrix(hiddenUnits, inputFeatures);
      weights.W_o_h = RNNBuilder._createMatrix(hiddenUnits, hiddenUnits);
      weights.b_o = RNNBuilder._createVector(hiddenUnits);
    }

    // For GRU: gate weights
    if (this.config.cellType === 'gru') {
      // Reset gate
      weights.W_r_x = RNNBuilder._createMatrix(hiddenUnits, inputFeatures);
      weights.W_r_h = RNNBuilder._createMatrix(hiddenUnits, hiddenUnits);
      weights.b_r = RNNBuilder._createVector(hiddenUnits);
      // Update gate
      weights.W_z_x = RNNBuilder._createMatrix(hiddenUnits, inputFeatures);
      weights.W_z_h = RNNBuilder._createMatrix(hiddenUnits, hiddenUnits);
      weights.b_z = RNNBuilder._createVector(hiddenUnits);
      // Candidate
      weights.W_n_x = RNNBuilder._createMatrix(hiddenUnits, inputFeatures);
      weights.W_n_h = RNNBuilder._createMatrix(hiddenUnits, hiddenUnits);
      weights.b_n = RNNBuilder._createVector(hiddenUnits);
    }

    // Layer 2 weights (if numLayers === 2)
    if (this.config.numLayers === 2) {
      weights.W_xh2 = RNNBuilder._createMatrix(hiddenUnits, hiddenUnits);
      weights.W_hh2 = RNNBuilder._createMatrix(hiddenUnits, hiddenUnits);
      weights.b_h2 = RNNBuilder._createVector(hiddenUnits);

      if (this.config.cellType === 'lstm') {
        weights.W_f_x2 = RNNBuilder._createMatrix(hiddenUnits, hiddenUnits);
        weights.W_f_h2 = RNNBuilder._createMatrix(hiddenUnits, hiddenUnits);
        weights.b_f2 = new Array(hiddenUnits).fill(1);
        
        weights.W_i_x2 = RNNBuilder._createMatrix(hiddenUnits, hiddenUnits);
        weights.W_i_h2 = RNNBuilder._createMatrix(hiddenUnits, hiddenUnits);
        weights.b_i2 = RNNBuilder._createVector(hiddenUnits);
        
        weights.W_c_x2 = RNNBuilder._createMatrix(hiddenUnits, hiddenUnits);
        weights.W_c_h2 = RNNBuilder._createMatrix(hiddenUnits, hiddenUnits);
        weights.b_c2 = RNNBuilder._createVector(hiddenUnits);
        
        weights.W_o_x2 = RNNBuilder._createMatrix(hiddenUnits, hiddenUnits);
        weights.W_o_h2 = RNNBuilder._createMatrix(hiddenUnits, hiddenUnits);
        weights.b_o2 = RNNBuilder._createVector(hiddenUnits);
      }

      if (this.config.cellType === 'gru') {
        weights.W_r_x2 = RNNBuilder._createMatrix(hiddenUnits, hiddenUnits);
        weights.W_r_h2 = RNNBuilder._createMatrix(hiddenUnits, hiddenUnits);
        weights.b_r2 = RNNBuilder._createVector(hiddenUnits);
        
        weights.W_z_x2 = RNNBuilder._createMatrix(hiddenUnits, hiddenUnits);
        weights.W_z_h2 = RNNBuilder._createMatrix(hiddenUnits, hiddenUnits);
        weights.b_z2 = RNNBuilder._createVector(hiddenUnits);
        
        weights.W_n_x2 = RNNBuilder._createMatrix(hiddenUnits, hiddenUnits);
        weights.W_n_h2 = RNNBuilder._createMatrix(hiddenUnits, hiddenUnits);
        weights.b_n2 = RNNBuilder._createVector(hiddenUnits);
      }
    }

    this.network = {
      weights: weights,
      initialWeights: RNNBuilder._deepCopy(weights),
      config: { ...this.config },
    };

    // Generate training data
    this.data = this._generateSequenceData();
    this.isBuilt = true;

    return this.network;
  }

  // ── Generate Sequential Training Data ──

  _generateSequenceData() {
    const { inputFeatures, timeSteps } = this.config;
    const numSamples = 30;
    const sequences = [];
    const targets = [];

    // Generate sine-wave based sequences
    for (let s = 0; s < numSamples; s++) {
      const phase = Math.random() * Math.PI * 2;
      const freq = 0.5 + Math.random() * 1.5;
      const seq = [];

      for (let t = 0; t < timeSteps + 1; t++) {
        const step = [];
        for (let f = 0; f < inputFeatures; f++) {
          step.push(parseFloat((Math.sin(phase + freq * t * (f + 1) * 0.5) * 0.5 + 0.5).toFixed(4)));
        }
        seq.push(step);
      }

      // Input: first timeSteps
      sequences.push(seq.slice(0, timeSteps));

      if (this.config.outputMode === 'many-to-one') {
        // Target: last step values (predict next value)
        targets.push(seq[timeSteps]);
      } else {
        // Target: shifted sequence (predict next for each step)
        targets.push(seq.slice(1, timeSteps + 1));
      }
    }

    return { sequences, targets };
  }

  // ── Forward Pass (Single Sequence) ──

  forward(sequence) {
    const { hiddenUnits } = this.config;
    const w = this.network.weights;
    const T = sequence.length;

    // Store all intermediate values for visualization & backprop
    const states = {
      inputs: [],      // x_t for each time step
      hiddens: [],     // h_t for each time step (after activation)
      hiddens2: [],    // h²_t for layer 2
      rawHiddens: [],  // z_t (pre-activation) for each step
      rawHiddens2: [], // z²_t for layer 2
      outputs: [],     // y_t for each time step
      cells: [],       // c_t for LSTM
      cells2: [],      // c²_t for LSTM layer 2
      gates: [],       // gate values for LSTM/GRU
      gates2: [],      // gate values for layer 2
    };

    let h_prev = RNNBuilder._createVector(hiddenUnits, 0);
    let c_prev = RNNBuilder._createVector(hiddenUnits, 0); // For LSTM layer 1
    
    let h_prev2 = RNNBuilder._createVector(hiddenUnits, 0);
    let c_prev2 = RNNBuilder._createVector(hiddenUnits, 0); // For LSTM layer 2

    states.hiddens.push([...h_prev]); // h_0
    if (this.config.numLayers === 2) {
      states.hiddens2.push([...h_prev2]);
    }

    for (let t = 0; t < T; t++) {
      const x_t = sequence[t];
      states.inputs.push([...x_t]);

      // --- Layer 1 ---
      if (this.config.cellType === 'lstm') {
        const result = this._lstmStep(x_t, h_prev, c_prev, w);
        states.hiddens.push([...result.h]);
        states.cells.push([...result.c]);
        states.gates.push(result.gates);
        states.rawHiddens.push(result.rawHidden);
        h_prev = result.h;
        c_prev = result.c;
      } else if (this.config.cellType === 'gru') {
        const result = this._gruStep(x_t, h_prev, w);
        states.hiddens.push([...result.h]);
        states.gates.push(result.gates);
        states.rawHiddens.push(result.rawHidden);
        h_prev = result.h;
      } else {
        // Simple RNN
        const wx = RNNBuilder._matVecMul(w.W_xh, x_t);
        const wh = RNNBuilder._matVecMul(w.W_hh, h_prev);
        const z_t = RNNBuilder._vecAdd(RNNBuilder._vecAdd(wx, wh), w.b_h);
        states.rawHiddens.push([...z_t]);
        const h_t = z_t.map(v => this.activate(v));
        states.hiddens.push([...h_t]);
        h_prev = h_t;
      }

      let final_h = h_prev;

      // --- Layer 2 ---
      if (this.config.numLayers === 2) {
        const x_t2 = h_prev; // Layer 1 output is layer 2 input
        
        if (this.config.cellType === 'lstm') {
          const w2 = {
            W_f_x: w.W_f_x2, W_f_h: w.W_f_h2, b_f: w.b_f2,
            W_i_x: w.W_i_x2, W_i_h: w.W_i_h2, b_i: w.b_i2,
            W_c_x: w.W_c_x2, W_c_h: w.W_c_h2, b_c: w.b_c2,
            W_o_x: w.W_o_x2, W_o_h: w.W_o_h2, b_o: w.b_o2
          };
          const result2 = this._lstmStep(x_t2, h_prev2, c_prev2, w2);
          states.hiddens2.push([...result2.h]);
          states.cells2.push([...result2.c]);
          states.gates2.push(result2.gates);
          states.rawHiddens2.push(result2.rawHidden);
          h_prev2 = result2.h;
          c_prev2 = result2.c;
        } else if (this.config.cellType === 'gru') {
          const w2 = {
            W_r_x: w.W_r_x2, W_r_h: w.W_r_h2, b_r: w.b_r2,
            W_z_x: w.W_z_x2, W_z_h: w.W_z_h2, b_z: w.b_z2,
            W_n_x: w.W_n_x2, W_n_h: w.W_n_h2, b_n: w.b_n2
          };
          const result2 = this._gruStep(x_t2, h_prev2, w2);
          states.hiddens2.push([...result2.h]);
          states.gates2.push(result2.gates);
          states.rawHiddens2.push(result2.rawHidden);
          h_prev2 = result2.h;
        } else {
          const wx = RNNBuilder._matVecMul(w.W_xh2, x_t2);
          const wh = RNNBuilder._matVecMul(w.W_hh2, h_prev2);
          const z_t2 = RNNBuilder._vecAdd(RNNBuilder._vecAdd(wx, wh), w.b_h2);
          states.rawHiddens2.push([...z_t2]);
          const h_t2 = z_t2.map(v => this.activate(v));
          states.hiddens2.push([...h_t2]);
          h_prev2 = h_t2;
        }
        final_h = h_prev2;
      }

      // Output: y_t = W_hy * final_h + b_y
      const y_t = RNNBuilder._vecAdd(
        RNNBuilder._matVecMul(w.W_hy, final_h),
        w.b_y
      );
      states.outputs.push([...y_t]);
    }

    return states;
  }

  _lstmStep(x_t, h_prev, c_prev, w) {
    const H = this.config.hiddenUnits;

    // Forget gate: f_t = σ(W_f_x * x_t + W_f_h * h_{t-1} + b_f)
    const f_raw = RNNBuilder._vecAdd(
      RNNBuilder._vecAdd(RNNBuilder._matVecMul(w.W_f_x, x_t), RNNBuilder._matVecMul(w.W_f_h, h_prev)),
      w.b_f
    );
    const f_t = f_raw.map(v => RNNBuilder._sigmoid(v));

    // Input gate: i_t = σ(W_i_x * x_t + W_i_h * h_{t-1} + b_i)
    const i_raw = RNNBuilder._vecAdd(
      RNNBuilder._vecAdd(RNNBuilder._matVecMul(w.W_i_x, x_t), RNNBuilder._matVecMul(w.W_i_h, h_prev)),
      w.b_i
    );
    const i_t = i_raw.map(v => RNNBuilder._sigmoid(v));

    // Cell candidate: c̃_t = tanh(W_c_x * x_t + W_c_h * h_{t-1} + b_c)
    const c_cand_raw = RNNBuilder._vecAdd(
      RNNBuilder._vecAdd(RNNBuilder._matVecMul(w.W_c_x, x_t), RNNBuilder._matVecMul(w.W_c_h, h_prev)),
      w.b_c
    );
    const c_cand = c_cand_raw.map(v => RNNBuilder._tanh(v));

    // Cell state: c_t = f_t ⊙ c_{t-1} + i_t ⊙ c̃_t
    const c_t = RNNBuilder._vecAdd(RNNBuilder._vecMul(f_t, c_prev), RNNBuilder._vecMul(i_t, c_cand));

    // Output gate: o_t = σ(W_o_x * x_t + W_o_h * h_{t-1} + b_o)
    const o_raw = RNNBuilder._vecAdd(
      RNNBuilder._vecAdd(RNNBuilder._matVecMul(w.W_o_x, x_t), RNNBuilder._matVecMul(w.W_o_h, h_prev)),
      w.b_o
    );
    const o_t = o_raw.map(v => RNNBuilder._sigmoid(v));

    // Hidden state: h_t = o_t ⊙ tanh(c_t)
    const h_t = RNNBuilder._vecMul(o_t, c_t.map(v => RNNBuilder._tanh(v)));

    return {
      h: h_t,
      c: c_t,
      rawHidden: c_cand_raw,
      gates: { f: f_t, i: i_t, c_cand, o: o_t, f_raw, i_raw, o_raw, c_cand_raw },
    };
  }

  _gruStep(x_t, h_prev, w) {
    // Reset gate: r_t = σ(W_r_x * x_t + W_r_h * h_{t-1} + b_r)
    const r_raw = RNNBuilder._vecAdd(
      RNNBuilder._vecAdd(RNNBuilder._matVecMul(w.W_r_x, x_t), RNNBuilder._matVecMul(w.W_r_h, h_prev)),
      w.b_r
    );
    const r_t = r_raw.map(v => RNNBuilder._sigmoid(v));

    // Update gate: z_t = σ(W_z_x * x_t + W_z_h * h_{t-1} + b_z)
    const z_raw = RNNBuilder._vecAdd(
      RNNBuilder._vecAdd(RNNBuilder._matVecMul(w.W_z_x, x_t), RNNBuilder._matVecMul(w.W_z_h, h_prev)),
      w.b_z
    );
    const z_t = z_raw.map(v => RNNBuilder._sigmoid(v));

    // Candidate: n_t = tanh(W_n_x * x_t + W_n_h * (r_t ⊙ h_{t-1}) + b_n)
    const rh = RNNBuilder._vecMul(r_t, h_prev);
    const n_raw = RNNBuilder._vecAdd(
      RNNBuilder._vecAdd(RNNBuilder._matVecMul(w.W_n_x, x_t), RNNBuilder._matVecMul(w.W_n_h, rh)),
      w.b_n
    );
    const n_t = n_raw.map(v => RNNBuilder._tanh(v));

    // h_t = (1 - z_t) ⊙ n_t + z_t ⊙ h_{t-1}
    const h_t = [];
    for (let i = 0; i < h_prev.length; i++) {
      h_t.push((1 - z_t[i]) * n_t[i] + z_t[i] * h_prev[i]);
    }

    return {
      h: h_t,
      rawHidden: n_raw,
      gates: { r: r_t, z: z_t, n: n_t, r_raw, z_raw, n_raw },
    };
  }

  // ── Loss Computation (MSE) ──

  computeLoss(predicted, target) {
    let loss = 0;
    if (this.config.outputMode === 'many-to-one') {
      // predicted: outputs array, target: single array
      const lastOutput = predicted[predicted.length - 1];
      for (let i = 0; i < target.length; i++) {
        loss += Math.pow(lastOutput[i] - target[i], 2);
      }
      loss /= target.length;
    } else {
      // many-to-many
      let count = 0;
      for (let t = 0; t < predicted.length; t++) {
        for (let i = 0; i < target[t].length; i++) {
          loss += Math.pow(predicted[t][i] - target[t][i], 2);
          count++;
        }
      }
      loss /= count;
    }
    return loss;
  }

  // ── Render Data Table ──

  renderDataTable() {
    if (!this.data) return '<p style="padding:2rem;text-align:center;color:var(--text-muted)">قم ببناء الشبكة...</p>';

    const { sequences, targets } = this.data;
    const { timeSteps, inputFeatures, outputMode } = this.config;
    const maxShow = Math.min(10, sequences.length);

    let html = '<table class="data-table"><thead><tr>';
    html += '<th>#</th>';
    for (let t = 0; t < timeSteps; t++) {
      for (let f = 0; f < inputFeatures; f++) {
        html += `<th>X<sub>t${t},f${f}</sub></th>`;
      }
    }
    if (outputMode === 'many-to-one') {
      for (let f = 0; f < inputFeatures; f++) {
        html += `<th style="color:var(--accent)">Y<sub>f${f}</sub></th>`;
      }
    } else {
      html += `<th style="color:var(--accent)">Target</th>`;
    }
    html += '</tr></thead><tbody>';

    for (let s = 0; s < maxShow; s++) {
      html += '<tr>';
      html += `<td>${s + 1}</td>`;
      for (let t = 0; t < timeSteps; t++) {
        for (let f = 0; f < inputFeatures; f++) {
          html += `<td>${sequences[s][t][f].toFixed(3)}</td>`;
        }
      }
      if (outputMode === 'many-to-one') {
        const tgt = targets[s];
        for (let f = 0; f < tgt.length; f++) {
          html += `<td style="color:var(--accent)">${tgt[f].toFixed(3)}</td>`;
        }
      } else {
        html += `<td style="color:var(--accent);font-size:0.7rem">Seq[${targets[s].length}]</td>`;
      }
      html += '</tr>';
    }

    html += '</tbody></table>';
    if (sequences.length > maxShow) {
      html += `<p style="text-align:center;color:var(--text-muted);font-size:0.8rem;padding:0.5rem">... و ${sequences.length - maxShow} تسلسلات أخرى</p>`;
    }
    return html;
  }
}
