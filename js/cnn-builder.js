/* ============================================
   CNN Builder - Convolutional Neural Network Engine
   ============================================ */

class CNNBuilder {
  constructor() {
    this.config = {
      inputSize: 5,            // default 5×5 image
      inputChannels: 1,        // grayscale
    };
    
    // Default architecture
    this.layers = [
      { type: 'conv2d', filters: 2, size: 3, stride: 1, padding: 0, activation: 'relu' },
      { type: 'pool2d', poolType: 'max', size: 2, stride: 2 },
      { type: 'flatten' },
      { type: 'fc', neurons: 10, activation: 'softmax' }
    ];
    
    this.network = null;
    this.isBuilt = false;
    this.lastResult = null;
  }

  setInputSize(size) {
    this.config.inputSize = size;
  }

  addLayer(type, config = {}) {
    if (this.layers.length >= 10) {
        alert('تم الوصول للحد الأقصى للطبقات (10)');
        return;
    }
    
    let layer = { type };
    if (type === 'conv2d') {
      layer = { ...layer, filters: 2, size: 3, stride: 1, padding: 0, activation: 'relu', ...config };
    } else if (type === 'pool2d') {
      layer = { ...layer, poolType: 'max', size: 2, stride: 2, ...config };
    } else if (type === 'fc') {
      layer = { ...layer, neurons: 10, activation: 'relu', ...config };
    } else if (type === 'flatten') {
      layer = { ...layer };
    }
    this.layers.push(layer);
  }

  removeLayer(index) {
    if (index >= 0 && index < this.layers.length) {
      this.layers.splice(index, 1);
    }
  }

  moveLayer(index, dir) {
    if (dir === 'up' && index > 0) {
      const temp = this.layers[index];
      this.layers[index] = this.layers[index - 1];
      this.layers[index - 1] = temp;
    } else if (dir === 'down' && index < this.layers.length - 1) {
      const temp = this.layers[index];
      this.layers[index] = this.layers[index + 1];
      this.layers[index + 1] = temp;
    }
  }

  updateLayer(index, config) {
    if (index >= 0 && index < this.layers.length) {
      this.layers[index] = { ...this.layers[index], ...config };
    }
  }

  // ── Educational Predefined Filters ──
  getEducationalFilter(index, size) {
    const filter = [];
    let name = `فلتر عشوائي (${size}×${size})`;
    
    for(let i=0; i<size; i++) {
        filter[i] = new Array(size).fill(0);
    }
    
    if (index === 0) {
        name = 'كاشف الحواف الرأسية (Vertical Edge)';
        for(let i=0; i<size; i++) {
            filter[i][0] = 1;
            filter[i][size-1] = -1;
        }
    } else if (index === 1) {
        name = 'كاشف الحواف الأفقية (Horizontal Edge)';
        for(let j=0; j<size; j++) {
            filter[0][j] = 1;
            filter[size-1][j] = -1;
        }
    } else if (index === 2) {
        name = 'كاشف الحواف القطرية (Main Diagonal)';
        for(let i=0; i<size; i++) {
            filter[i][i] = 1;
            if (i < size - 1) filter[i][i+1] = -1;
            if (i > 0) filter[i][i-1] = -1;
        }
    } else if (index === 3) {
        name = 'كاشف الحواف القطرية العكسية (Anti-Diagonal)';
        for(let i=0; i<size; i++) {
            filter[size - 1 - i][i] = 1;
            if (i < size - 1) filter[size - 1 - i - 1][i] = -1;
        }
    } else if (index === 4) {
        name = 'كاشف النقاط / الزوايا (Point/Corner)';
        let center = Math.floor(size/2);
        for(let i=0; i<size; i++) {
            for(let j=0; j<size; j++) {
                filter[i][j] = -1;
            }
        }
        filter[center][center] = size * size - 1;
    } else if (index === 5) {
        name = 'فلتر التوضيح (Sharpen)';
        let center = Math.floor(size/2);
        for(let i=0; i<size; i++) {
            for(let j=0; j<size; j++) {
                filter[i][j] = (i === center || j === center) ? -1 : 0;
            }
        }
        filter[center][center] = size * 2 - 1;
    } else {
        name = `فلتر عشوائي (${index + 1})`;
        for(let i=0; i<size; i++) {
            for(let j=0; j<size; j++) {
                filter[i][j] = parseFloat((Math.random() * 2 - 1).toFixed(4));
            }
        }
    }
    return { matrix: filter, name };
  }

  // ── Build Network ──
  build() {
    this.network = {
      layersData: [], // Store weights, biases, etc. per layer
      layerNames: [`الإدخال (${this.config.inputSize}×${this.config.inputSize})`],
      totalParams: 0,
      filterParams: 0,
      fcParams: 0
    };

    let currentShape = { type: '2d', size: this.config.inputSize, channels: this.config.inputChannels };
    
    for (let l = 0; l < this.layers.length; l++) {
      const layer = this.layers[l];
      const layerData = { config: layer, inShape: { ...currentShape } };

      if (layer.type === 'conv2d') {
        if (currentShape.type !== '2d') throw new Error('Conv2D must receive 2D input');
        
        const outSize = Math.floor((currentShape.size + 2 * layer.padding - layer.size) / layer.stride) + 1;
        
        const filters = [];
        const filterNames = [];
        for (let f = 0; f < layer.filters; f++) {
          const edFilter = this.getEducationalFilter(f, layer.size);
          filters.push(edFilter.matrix);
          filterNames.push(edFilter.name);
        }
        const biases = [];
        for (let f = 0; f < layer.filters; f++) {
          biases.push(parseFloat((Math.random() * 0.4 - 0.2).toFixed(4)));
        }

        layerData.filters = filters;
        layerData.filterNames = filterNames;
        layerData.biases = biases;
        
        const paramsCount = layer.filters * (layer.size * layer.size * currentShape.channels + 1);
        this.network.totalParams += paramsCount;
        this.network.filterParams += paramsCount;
        
        currentShape = { type: '2d', size: outSize, channels: layer.filters };
        layerData.outShape = { ...currentShape };
        
        this.network.layerNames.push(`Conv2D (${layer.size}×${layer.size}×${layer.filters})`);
        
      } else if (layer.type === 'pool2d') {
        if (currentShape.type !== '2d') throw new Error('Pool2D must receive 2D input');
        
        const outSize = Math.floor(currentShape.size / layer.stride); // usually poolSize == stride
        currentShape = { type: '2d', size: outSize, channels: currentShape.channels };
        layerData.outShape = { ...currentShape };
        
        this.network.layerNames.push(`${layer.poolType === 'max' ? 'MaxPool' : 'AvgPool'} (${layer.size}×${layer.size})`);
        
      } else if (layer.type === 'flatten') {
        if (currentShape.type !== '2d') throw new Error('Flatten must receive 2D input');
        
        const flatSize = currentShape.size * currentShape.size * currentShape.channels;
        currentShape = { type: '1d', length: flatSize };
        layerData.outShape = { ...currentShape };
        
        this.network.layerNames.push(`Flatten (${flatSize})`);
        
      } else if (layer.type === 'fc') {
        if (currentShape.type !== '1d') {
           // Auto-flatten if needed
           if(currentShape.type === '2d') {
              const flatSize = currentShape.size * currentShape.size * currentShape.channels;
              currentShape = { type: '1d', length: flatSize };
              layerData.inShape = { ...currentShape }; // update inShape for this layer
              this.network.layerNames.push(`Flatten (${flatSize}) [Auto]`);
           } else {
              throw new Error('FC must receive 1D input');
           }
        }
        
        const weights = MathUtils.initWeightMatrix(currentShape.length, layer.neurons);
        const biases = MathUtils.initBiasVector(layer.neurons);
        
        layerData.weights = weights;
        layerData.biases = biases;
        
        const paramsCount = currentShape.length * layer.neurons + layer.neurons;
        this.network.totalParams += paramsCount;
        this.network.fcParams += paramsCount;
        
        currentShape = { type: '1d', length: layer.neurons };
        layerData.outShape = { ...currentShape };
        
        this.network.layerNames.push(`FC (${layer.neurons})`);
      }
      
      this.network.layersData.push(layerData);
    }
    
    // Output layer name based on last layer activation if it exists
    const lastLayer = this.layers[this.layers.length-1];
    if(lastLayer && lastLayer.type === 'fc' && lastLayer.activation) {
        this.network.layerNames.push(`${lastLayer.activation} → الإخراج`);
    }

    this.isBuilt = true;
    return this.network;
  }

  // ── Convolution 2D ──
  convolve2D(inputChannels, filters, biases, stride, padding) {
      const inChannels = inputChannels.length;
      const inputSize = inputChannels[0].length;
      const filterSize = filters[0].length;
      const numFilters = filters.length;
      
      // Apply padding to all input channels
      let paddedChannels = [];
      for(let c = 0; c < inChannels; c++) {
          let padded = inputChannels[c];
          if (padding > 0) {
            const paddedSize = inputSize + 2 * padding;
            padded = [];
            for (let i = 0; i < paddedSize; i++) {
              padded[i] = [];
              for (let j = 0; j < paddedSize; j++) {
                const oi = i - padding;
                const oj = j - padding;
                if (oi >= 0 && oi < inputSize && oj >= 0 && oj < inputSize) {
                  padded[i][j] = inputChannels[c][oi][oj];
                } else {
                  padded[i][j] = 0;
                }
              }
            }
          }
          paddedChannels.push(padded);
      }

      const paddedSize = paddedChannels[0].length;
      const outSize = Math.floor((paddedSize - filterSize) / stride) + 1;
      
      const outputMaps = [];
      const steps = [];

      for(let f = 0; f < numFilters; f++) {
          const filter = filters[f];
          const bias = biases[f];
          const outMap = [];
          
          for (let i = 0; i < outSize; i++) {
            outMap[i] = [];
            for (let j = 0; j < outSize; j++) {
              let sum = 0;
              const elementProducts = [];
              const inputRegion = []; // Visualizer expects 2D region, we'll just take channel 0 for visualizer if multi-channel

              for (let c = 0; c < inChannels; c++) {
                  for (let fi = 0; fi < filterSize; fi++) {
                    if (c === 0) inputRegion[fi] = [];
                    for (let fj = 0; fj < filterSize; fj++) {
                      const pi = i * stride + fi;
                      const pj = j * stride + fj;
                      const inputVal = paddedChannels[c][pi][pj];
                      const filterVal = filter[fi][fj];
                      const product = inputVal * filterVal;
                      sum += product;
                      
                      if (c === 0) { // store first channel for visualizer
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
                  }
              }

              sum += bias;
              const result = parseFloat(sum.toFixed(4));
              outMap[i][j] = result;

              // Store steps for visualizer
              let stepFilterIndex = f;
              if (steps.findIndex(s => s.filterIndex === stepFilterIndex && s.outputPos.row === i && s.outputPos.col === j) === -1) {
                  steps.push({
                      filterIndex: f,
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
          }
          outputMaps.push(outMap);
      }
      return { outputMaps, steps };
  }

  // ── Activation ──
  applyActivation(featureMaps, type) {
    if (!type || type === 'none') return { beforeMaps: featureMaps, afterMaps: featureMaps };
    
    const beforeMaps = JSON.parse(JSON.stringify(featureMaps));
    const afterMaps = [];
    for (let m = 0; m < featureMaps.length; m++) {
      const map = featureMaps[m];
      const after = [];
      for (let i = 0; i < map.length; i++) {
        after[i] = [];
        for (let j = 0; j < map[i].length; j++) {
            let val = parseFloat(map[i][j].toFixed(4));
            if(type === 'relu') val = Math.max(0, val);
            else if(type === 'sigmoid') val = 1 / (1 + Math.exp(-val));
            else if(type === 'tanh') val = Math.tanh(val);
            after[i][j] = val;
        }
      }
      afterMaps.push(after);
    }
    return { beforeMaps, afterMaps };
  }

  // ── Pooling ──
  pool2D(featureMaps, poolSize, stride, poolType) {
      const outputMaps = [];
      const steps = [];

      for(let f=0; f<featureMaps.length; f++) {
          const matrix = featureMaps[f];
          const inputSize = matrix.length;
          const outSize = Math.floor((inputSize - poolSize) / stride) + 1;
          const outMap = [];
          
          for (let i = 0; i < outSize; i++) {
            outMap[i] = [];
            for (let j = 0; j < outSize; j++) {
              let maxVal = -Infinity;
              let sumVal = 0;
              let maxPos = { row: 0, col: 0 };
              const region = [];
              const allValues = [];
              let count = 0;

              for (let pi = 0; pi < poolSize; pi++) {
                region[pi] = [];
                for (let pj = 0; pj < poolSize; pj++) {
                  const ri = i * stride + pi;
                  const rj = j * stride + pj;
                  const val = (ri < inputSize && rj < inputSize) ? matrix[ri][rj] : 0;
                  region[pi][pj] = val;
                  allValues.push({ val, row: ri, col: rj });
                  
                  if (val > maxVal) {
                    maxVal = val;
                    maxPos = { row: ri, col: rj };
                  }
                  sumVal += val;
                  count++;
                }
              }

              const resultVal = poolType === 'max' ? parseFloat(maxVal.toFixed(4)) : parseFloat((sumVal/count).toFixed(4));
              outMap[i][j] = resultVal;
              
              steps.push({
                filterIndex: f,
                outputPos: { row: i, col: j },
                inputTopLeft: { row: i * stride, col: j * stride },
                region,
                allValues,
                maxVal: parseFloat(maxVal.toFixed(4)),
                avgVal: parseFloat((sumVal/count).toFixed(4)),
                maxPos,
                result: resultVal
              });
            }
          }
          outputMaps.push(outMap);
      }
      return { outputMaps, steps };
  }

  // ── Flatten ──
  flatten(matrices) {
    const vector = [];
    const mapping = [];
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

  // ── Fully Connected ──
  fullyConnected(flatVector, weights, biases) {
    const numOutputs = biases.length;
    const z = [];
    const details = [];

    for (let j = 0; j < numOutputs; j++) {
      let sum = 0;
      const inputContribs = [];
      for (let i = 0; i < flatVector.length; i++) {
        const contrib = flatVector[i] * weights[i][j];
        sum += contrib;
        inputContribs.push({
          input: flatVector[i],
          weight: weights[i][j],
          contrib: parseFloat(contrib.toFixed(6))
        });
      }
      sum += biases[j];
      z.push(parseFloat(sum.toFixed(4)));
      details.push({
        neuronIndex: j,
        linearSum: parseFloat(sum.toFixed(4)),
        bias: biases[j],
        inputContribs
      });
    }
    return { z, details };
  }

  // ── Full Feedforward Pipeline ──
  feedforward(inputImage) {
    if (!this.isBuilt) return null;

    let currentData = [inputImage]; // Array of feature maps (channels)
    let isFlattened = false;
    let flatData = [];
    const pipelineResults = [];

    for (let l = 0; l < this.network.layersData.length; l++) {
        const layerMeta = this.network.layersData[l];
        const config = layerMeta.config;
        
        if (config.type === 'conv2d') {
            const conv = this.convolve2D(currentData, layerMeta.filters, layerMeta.biases, config.stride, config.padding);
            const activated = this.applyActivation(conv.outputMaps, config.activation);
            
            pipelineResults.push({
                layerIndex: l,
                type: 'conv2d',
                config,
                meta: layerMeta,
                convSteps: conv.steps,
                featureMaps: conv.outputMaps,
                activatedMaps: activated.afterMaps,
                activationType: config.activation
            });
            currentData = activated.afterMaps;
            
        } else if (config.type === 'pool2d') {
            const pool = this.pool2D(currentData, config.size, config.stride, config.poolType);
            pipelineResults.push({
                layerIndex: l,
                type: 'pool2d',
                config,
                meta: layerMeta,
                poolSteps: pool.steps,
                inputMaps: currentData, // before pooling
                featureMaps: pool.outputMaps
            });
            currentData = pool.outputMaps;
            
        } else if (config.type === 'flatten') {
            const flat = this.flatten(currentData);
            pipelineResults.push({
                layerIndex: l,
                type: 'flatten',
                config,
                meta: layerMeta,
                inputMaps: currentData,
                flatVector: flat.vector,
                mapping: flat.mapping
            });
            isFlattened = true;
            flatData = flat.vector;
            
        } else if (config.type === 'fc') {
            if (!isFlattened) {
                // Auto-flatten
                const flat = this.flatten(currentData);
                isFlattened = true;
                flatData = flat.vector;
            }
            
            const fc = this.fullyConnected(flatData, layerMeta.weights, layerMeta.biases);
            let activatedZ = fc.z;
            let probabilities = null;
            let prediction = -1;
            
            if (config.activation === 'softmax') {
                probabilities = MathUtils.softmax(fc.z).map(v => parseFloat(v.toFixed(6)));
                prediction = probabilities.indexOf(Math.max(...probabilities));
            } else if (config.activation === 'relu') {
                activatedZ = fc.z.map(v => Math.max(0, v));
            } else if (config.activation === 'sigmoid') {
                activatedZ = fc.z.map(v => 1 / (1 + Math.exp(-v)));
            }

            pipelineResults.push({
                layerIndex: l,
                type: 'fc',
                config,
                meta: layerMeta,
                flatVector: flatData,
                z: fc.z,
                activatedZ: activatedZ,
                probabilities,
                prediction,
                details: fc.details
            });
            
            flatData = activatedZ;
        }
    }

    this.lastResult = {
      input: inputImage,
      pipelineResults
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

  // ── Get Current Input from canvas/array ──
  static canvasToMatrix(drawGrid, size = 5) {
    if (Array.isArray(drawGrid[0])) return drawGrid;
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
    let pipeline = `${this.config.inputSize}×${this.config.inputSize}`;
    
    this.layers.forEach(l => {
       if(l.type === 'conv2d') pipeline += ` → Conv(${l.size}×${l.size}×${l.filters})`;
       else if(l.type === 'pool2d') pipeline += ` → Pool(${l.size}×${l.size})`;
       else if(l.type === 'flatten') pipeline += ` → Flatten`;
       else if(l.type === 'fc') pipeline += ` → FC(${l.neurons})`;
    });

    return {
      pipeline: pipeline,
      totalParams: this.network.totalParams,
      filterParams: this.network.filterParams,
      fcParams: this.network.fcParams
    };
  }
}
