/* ============================================
   CNN Builder - Convolutional Neural Network Engine
   ============================================ */

class CNNBuilder {
  constructor() {
    this.config = {
      inputSize: 5,            // 5×5 image
      inputChannels: 1,        // grayscale
      numFilters: 2,           // number of conv filters
      filterSize: 3,           // 3×3 filters
      stride: 1,
      padding: 0,              // 'valid' convolution
      poolSize: 2,             // 2×2 max pooling
      poolType: 'max',         // 'max' or 'average'
      fcNeurons: 10,           // fully connected output (digits 0-9)
    };
    this.network = null;
    this.isBuilt = false;
    this.lastResult = null;    // last feedforward result for visualization
  }

  // ── Configuration Setters ──

  setNumFilters(n) {
    this.config.numFilters = Math.max(1, Math.min(8, n));
  }

  setFilterSize(n) {
    this.config.filterSize = Math.max(2, Math.min(5, n));
  }

  setStride(n) {
    this.config.stride = Math.max(1, Math.min(3, n));
  }

  setPadding(n) {
    this.config.padding = Math.max(0, Math.min(2, n));
  }

  setPoolSize(n) {
    this.config.poolSize = Math.max(2, Math.min(4, n));
  }

  setPoolType(type) {
    this.config.poolType = type; // 'max' or 'average'
  }

  // ── Build Network ──

  build() {
    const { inputSize, numFilters, filterSize, stride, padding, poolSize, fcNeurons } = this.config;

    // Calculate output dimensions
    const convOutSize = Math.floor((inputSize + 2 * padding - filterSize) / stride) + 1;
    const poolOutSize = Math.floor(convOutSize / poolSize);
    const flattenSize = poolOutSize * poolOutSize * numFilters;

    // Initialize filters (random weights)
    const filters = [];
    for (let f = 0; f < numFilters; f++) {
      const filter = [];
      for (let i = 0; i < filterSize; i++) {
        filter[i] = [];
        for (let j = 0; j < filterSize; j++) {
          filter[i][j] = parseFloat((Math.random() * 2 - 1).toFixed(4));
        }
      }
      filters.push(filter);
    }

    // Initialize conv biases
    const convBiases = [];
    for (let f = 0; f < numFilters; f++) {
      convBiases.push(parseFloat((Math.random() * 0.4 - 0.2).toFixed(4)));
    }

    // Initialize FC weights and biases
    const fcWeights = MathUtils.initWeightMatrix(flattenSize, fcNeurons);
    const fcBiases = MathUtils.initBiasVector(fcNeurons);

    this.network = {
      filters,
      convBiases,
      fcWeights,
      fcBiases,
      dims: {
        inputSize,
        convOutSize,
        poolOutSize,
        flattenSize,
        fcNeurons
      },
      layerNames: [
        `الإدخال (${inputSize}×${inputSize})`,
        `التلافيف Conv2D (${filterSize}×${filterSize}×${numFilters})`,
        `التنشيط ReLU`,
        `التجميع MaxPool (${poolSize}×${poolSize})`,
        `التسطيح Flatten (${flattenSize})`,
        `طبقة كاملة FC (${fcNeurons})`,
        `Softmax → الإخراج`
      ]
    };

    this.isBuilt = true;
    return this.network;
  }

  // ── Convolution 2D ──

  convolve2D(input, filter, bias, stride, padding) {
    const inputSize = input.length;
    const filterSize = filter.length;

    // Apply padding
    let padded = input;
    if (padding > 0) {
      const paddedSize = inputSize + 2 * padding;
      padded = [];
      for (let i = 0; i < paddedSize; i++) {
        padded[i] = [];
        for (let j = 0; j < paddedSize; j++) {
          const oi = i - padding;
          const oj = j - padding;
          if (oi >= 0 && oi < inputSize && oj >= 0 && oj < inputSize) {
            padded[i][j] = input[oi][oj];
          } else {
            padded[i][j] = 0;
          }
        }
      }
    }

    const paddedSize = padded.length;
    const outSize = Math.floor((paddedSize - filterSize) / stride) + 1;
    const output = [];
    const steps = []; // For step-by-step visualization

    for (let i = 0; i < outSize; i++) {
      output[i] = [];
      for (let j = 0; j < outSize; j++) {
        let sum = 0;
        const elementProducts = [];
        const inputRegion = [];

        for (let fi = 0; fi < filterSize; fi++) {
          inputRegion[fi] = [];
          for (let fj = 0; fj < filterSize; fj++) {
            const pi = i * stride + fi;
            const pj = j * stride + fj;
            const inputVal = padded[pi][pj];
            const filterVal = filter[fi][fj];
            const product = inputVal * filterVal;
            sum += product;
            elementProducts.push({
              inputVal: parseFloat(inputVal.toFixed(4)),
              filterVal: parseFloat(filterVal.toFixed(4)),
              product: parseFloat(product.toFixed(4)),
              inputPos: { row: pi, col: pj },
              filterPos: { row: fi, col: fj }
            });
            inputRegion[fi][fj] = inputVal;
          }
        }

        sum += bias;
        const result = parseFloat(sum.toFixed(4));
        output[i][j] = result;

        steps.push({
          outputPos: { row: i, col: j },
          inputTopLeft: { row: i * stride, col: j * stride },
          inputRegion,
          elementProducts,
          bias: parseFloat(bias.toFixed(4)),
          sumBeforeBias: parseFloat((sum - bias).toFixed(4)),
          result
        });
      }
    }

    return { output, steps };
  }

  // ── ReLU Activation ──

  applyReLU(matrix) {
    const before = JSON.parse(JSON.stringify(matrix));
    const after = [];
    for (let i = 0; i < matrix.length; i++) {
      after[i] = [];
      for (let j = 0; j < matrix[i].length; j++) {
        after[i][j] = Math.max(0, parseFloat(matrix[i][j].toFixed(4)));
      }
    }
    return { before, after };
  }

  // ── Max Pooling ──

  maxPool(matrix, poolSize) {
    const inputSize = matrix.length;
    const outSize = Math.floor(inputSize / poolSize);
    const output = [];
    const steps = [];

    for (let i = 0; i < outSize; i++) {
      output[i] = [];
      for (let j = 0; j < outSize; j++) {
        let maxVal = -Infinity;
        let maxPos = { row: 0, col: 0 };
        const region = [];
        const allValues = [];

        for (let pi = 0; pi < poolSize; pi++) {
          region[pi] = [];
          for (let pj = 0; pj < poolSize; pj++) {
            const ri = i * poolSize + pi;
            const rj = j * poolSize + pj;
            const val = ri < inputSize && rj < inputSize ? matrix[ri][rj] : 0;
            region[pi][pj] = val;
            allValues.push({ val, row: ri, col: rj });
            if (val > maxVal) {
              maxVal = val;
              maxPos = { row: ri, col: rj };
            }
          }
        }

        output[i][j] = parseFloat(maxVal.toFixed(4));
        steps.push({
          outputPos: { row: i, col: j },
          inputTopLeft: { row: i * poolSize, col: j * poolSize },
          region,
          allValues,
          maxVal: parseFloat(maxVal.toFixed(4)),
          maxPos,
          result: parseFloat(maxVal.toFixed(4))
        });
      }
    }

    return { output, steps };
  }

  // ── Average Pooling ──

  avgPool(matrix, poolSize) {
    const inputSize = matrix.length;
    const outSize = Math.floor(inputSize / poolSize);
    const output = [];
    const steps = [];

    for (let i = 0; i < outSize; i++) {
      output[i] = [];
      for (let j = 0; j < outSize; j++) {
        let sum = 0;
        let count = 0;
        const region = [];
        const allValues = [];

        for (let pi = 0; pi < poolSize; pi++) {
          region[pi] = [];
          for (let pj = 0; pj < poolSize; pj++) {
            const ri = i * poolSize + pi;
            const rj = j * poolSize + pj;
            const val = ri < inputSize && rj < inputSize ? matrix[ri][rj] : 0;
            region[pi][pj] = val;
            allValues.push({ val, row: ri, col: rj });
            sum += val;
            count++;
          }
        }

        const avg = parseFloat((sum / count).toFixed(4));
        output[i][j] = avg;
        steps.push({
          outputPos: { row: i, col: j },
          inputTopLeft: { row: i * poolSize, col: j * poolSize },
          region,
          allValues,
          avgVal: avg,
          result: avg
        });
      }
    }

    return { output, steps };
  }

  // ── Flatten ──

  flatten(matrices) {
    const vector = [];
    const mapping = []; // tracks which matrix[i][j] maps to vector[k]
    
    for (let f = 0; f < matrices.length; f++) {
      const mat = matrices[f];
      for (let i = 0; i < mat.length; i++) {
        for (let j = 0; j < mat[i].length; j++) {
          mapping.push({
            filterIdx: f,
            row: i,
            col: j,
            vectorIdx: vector.length,
            value: mat[i][j]
          });
          vector.push(mat[i][j]);
        }
      }
    }

    return { vector, mapping };
  }

  // ── Fully Connected + Softmax ──

  fullyConnected(flatVector) {
    const { fcWeights, fcBiases } = this.network;
    const numOutputs = fcBiases.length;
    const z = [];
    const details = [];

    for (let j = 0; j < numOutputs; j++) {
      let sum = 0;
      const inputContribs = [];
      for (let i = 0; i < flatVector.length; i++) {
        const contrib = flatVector[i] * fcWeights[i][j];
        sum += contrib;
        inputContribs.push({
          input: flatVector[i],
          weight: fcWeights[i][j],
          contrib: parseFloat(contrib.toFixed(6))
        });
      }
      sum += fcBiases[j];
      z.push(parseFloat(sum.toFixed(4)));
      details.push({
        neuronIndex: j,
        linearSum: parseFloat(sum.toFixed(4)),
        bias: fcBiases[j],
        inputContribs
      });
    }

    const probabilities = MathUtils.softmax(z).map(v => parseFloat(v.toFixed(6)));
    const predictedClass = probabilities.indexOf(Math.max(...probabilities));

    return { z, probabilities, predictedClass, details };
  }

  // ── Full Feedforward Pipeline ──

  feedforward(inputImage) {
    if (!this.isBuilt) return null;

    const { filters, convBiases } = this.network;
    const { stride, padding, poolSize, poolType } = this.config;

    // Step 1: Convolution for each filter
    const convResults = [];
    const convFeatureMaps = [];
    for (let f = 0; f < filters.length; f++) {
      const conv = this.convolve2D(inputImage, filters[f], convBiases[f], stride, padding);
      convResults.push(conv);
      convFeatureMaps.push(conv.output);
    }

    // Step 2: ReLU activation
    const reluResults = [];
    const reluFeatureMaps = [];
    for (let f = 0; f < convFeatureMaps.length; f++) {
      const relu = this.applyReLU(convFeatureMaps[f]);
      reluResults.push(relu);
      reluFeatureMaps.push(relu.after);
    }

    // Step 3: Pooling
    const poolResults = [];
    const poolFeatureMaps = [];
    for (let f = 0; f < reluFeatureMaps.length; f++) {
      let pool;
      if (poolType === 'max') {
        pool = this.maxPool(reluFeatureMaps[f], poolSize);
      } else {
        pool = this.avgPool(reluFeatureMaps[f], poolSize);
      }
      poolResults.push(pool);
      poolFeatureMaps.push(pool.output);
    }

    // Step 4: Flatten
    const flattenResult = this.flatten(poolFeatureMaps);

    // Step 5: FC + Softmax
    const fcResult = this.fullyConnected(flattenResult.vector);

    this.lastResult = {
      input: inputImage,
      convResults,
      convFeatureMaps,
      reluResults,
      reluFeatureMaps,
      poolResults,
      poolFeatureMaps,
      flattenResult,
      fcResult,
      prediction: fcResult.predictedClass,
      probabilities: fcResult.probabilities
    };

    return this.lastResult;
  }

  // ── Pre-built Sample Digits (5×5 binary) ──

  static getSampleDigits() {
    return {
      0: [
        [0, 1, 1, 1, 0],
        [1, 0, 0, 0, 1],
        [1, 0, 0, 0, 1],
        [1, 0, 0, 0, 1],
        [0, 1, 1, 1, 0]
      ],
      1: [
        [0, 0, 1, 0, 0],
        [0, 1, 1, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 1, 1, 1, 0]
      ],
      2: [
        [0, 1, 1, 1, 0],
        [1, 0, 0, 0, 1],
        [0, 0, 1, 1, 0],
        [0, 1, 0, 0, 0],
        [1, 1, 1, 1, 1]
      ],
      3: [
        [1, 1, 1, 1, 0],
        [0, 0, 0, 0, 1],
        [0, 1, 1, 1, 0],
        [0, 0, 0, 0, 1],
        [1, 1, 1, 1, 0]
      ],
      4: [
        [1, 0, 0, 1, 0],
        [1, 0, 0, 1, 0],
        [1, 1, 1, 1, 1],
        [0, 0, 0, 1, 0],
        [0, 0, 0, 1, 0]
      ],
      5: [
        [1, 1, 1, 1, 1],
        [1, 0, 0, 0, 0],
        [1, 1, 1, 1, 0],
        [0, 0, 0, 0, 1],
        [1, 1, 1, 1, 0]
      ],
      6: [
        [0, 1, 1, 1, 0],
        [1, 0, 0, 0, 0],
        [1, 1, 1, 1, 0],
        [1, 0, 0, 0, 1],
        [0, 1, 1, 1, 0]
      ],
      7: [
        [1, 1, 1, 1, 1],
        [0, 0, 0, 1, 0],
        [0, 0, 1, 0, 0],
        [0, 1, 0, 0, 0],
        [0, 1, 0, 0, 0]
      ],
      8: [
        [0, 1, 1, 1, 0],
        [1, 0, 0, 0, 1],
        [0, 1, 1, 1, 0],
        [1, 0, 0, 0, 1],
        [0, 1, 1, 1, 0]
      ],
      9: [
        [0, 1, 1, 1, 0],
        [1, 0, 0, 0, 1],
        [0, 1, 1, 1, 1],
        [0, 0, 0, 0, 1],
        [0, 1, 1, 1, 0]
      ]
    };
  }

  // ── Get Current Input as 5×5 from Drawing Canvas ──

  static canvasToMatrix(drawGrid) {
    // drawGrid is a flat array or 2D array of 0/1
    if (Array.isArray(drawGrid[0])) return drawGrid;
    const size = Math.sqrt(drawGrid.length);
    const matrix = [];
    for (let i = 0; i < size; i++) {
      matrix[i] = [];
      for (let j = 0; j < size; j++) {
        matrix[i][j] = drawGrid[i * size + j];
      }
    }
    return matrix;
  }

  // ── Network Summary ──

  getSummary() {
    if (!this.network) return null;
    const d = this.network.dims;
    const totalFilterParams = this.config.numFilters * (this.config.filterSize * this.config.filterSize + 1);
    const totalFCParams = d.flattenSize * d.fcNeurons + d.fcNeurons;
    return {
      pipeline: `${d.inputSize}×${d.inputSize} → Conv(${this.config.filterSize}×${this.config.filterSize}×${this.config.numFilters}) → ReLU → Pool(${this.config.poolSize}×${this.config.poolSize}) → Flatten(${d.flattenSize}) → FC(${d.fcNeurons}) → Softmax`,
      dims: d,
      totalParams: totalFilterParams + totalFCParams,
      filterParams: totalFilterParams,
      fcParams: totalFCParams
    };
  }
}
