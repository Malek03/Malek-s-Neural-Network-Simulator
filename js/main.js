/* ============================================
   Main Application Logic
   ============================================ */

document.addEventListener('DOMContentLoaded', () => {
  // 1. Initialize Hero Animation
  const heroAnim = new HeroAnimation('heroCanvas');

  // 2. Navigation Scroll Effect & Mobile Toggle
  const navbar = document.querySelector('.navbar');
  window.addEventListener('scroll', () => {
    if (window.scrollY > 50) {
      navbar.classList.add('scrolled');
    } else {
      navbar.classList.remove('scrolled');
    }
  });

  const mobileToggle = document.querySelector('.nav-mobile-toggle');
  const navLinks = document.querySelector('.nav-links');
  if (mobileToggle && navLinks) {
    mobileToggle.addEventListener('click', () => {
      navLinks.classList.toggle('active');
    });
    // Close on link click
    navLinks.querySelectorAll('a').forEach(link => {
      link.addEventListener('click', () => {
        navLinks.classList.remove('active');
      });
    });
  }

  // 3. Timeline Scroll Animation
  const observerOptions = {
    root: null,
    rootMargin: '0px',
    threshold: 0.1
  };
  
  const observer = new IntersectionObserver((entries, observer) => {
    entries.forEach(entry => {
      if (entry.isIntersecting) {
        entry.target.classList.add('visible');
        observer.unobserve(entry.target);
      }
    });
  }, observerOptions);

  document.querySelectorAll('.timeline-item').forEach(item => {
    observer.observe(item);
  });

  // 4. Bio Section Canvas
  initBioCanvas();

  // 5. Simulator Initialization
  initSimulator();

  // 6. DNN Simulator Initialization
  initDNNSimulator();

  // 7. CNN Simulator Initialization
  initCNNSimulator();

  // 8. RNN Simulator Initialization
  initRNNSimulator();

  // 9. Transformer Simulator Initialization
  if (typeof initTransformerSimulator === 'function') {
    initTransformerSimulator();
  }

  // 10. Concept Cards Interactive Visualizations
  if (typeof ConceptCards !== 'undefined') {
    ConceptCards.init();
  }
});

function initBioCanvas() {
  const canvas = document.getElementById('bioCanvas');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  
  const resize = () => {
    const parent = canvas.parentElement;
    canvas.width = parent.clientWidth;
    canvas.height = parent.clientHeight;
    drawBio();
  };
  
  const drawBio = () => {
    const w = canvas.width;
    const h = canvas.height;
    ctx.clearRect(0, 0, w, h);
    
    // Draw Biological Neuron representation
    ctx.strokeStyle = '#6C63FF';
    ctx.lineWidth = 2;
    
    // Soma (Body)
    ctx.beginPath();
    ctx.arc(w/2, h/2, 40, 0, Math.PI * 2);
    ctx.fillStyle = 'rgba(108, 99, 255, 0.2)';
    ctx.fill();
    ctx.stroke();
    
    // Nucleus
    ctx.beginPath();
    ctx.arc(w/2, h/2, 15, 0, Math.PI * 2);
    ctx.fillStyle = '#6C63FF';
    ctx.fill();
    
    // Dendrites
    const numDendrites = 5;
    for (let i = 0; i < numDendrites; i++) {
      const angle = (i * Math.PI * 2) / numDendrites;
      if (angle > -0.5 && angle < 0.5) continue; // Leave space for axon
      
      ctx.beginPath();
      ctx.moveTo(w/2 + Math.cos(angle)*40, h/2 + Math.sin(angle)*40);
      
      // Branching
      const endX = w/2 + Math.cos(angle)*100;
      const endY = h/2 + Math.sin(angle)*100;
      
      ctx.quadraticCurveTo(
        w/2 + Math.cos(angle)*70 + (Math.random()*20-10),
        h/2 + Math.sin(angle)*70 + (Math.random()*20-10),
        endX, endY
      );
      ctx.stroke();
    }
    
    // Axon
    ctx.beginPath();
    ctx.moveTo(w/2 + 40, h/2);
    ctx.lineTo(w - 20, h/2);
    ctx.lineWidth = 4;
    ctx.stroke();
    
    // Myelin sheaths
    ctx.fillStyle = '#00E5FF';
    for (let x = w/2 + 50; x < w - 40; x += 30) {
      ctx.fillRect(x, h/2 - 8, 20, 16);
    }
    
    // Axon Terminals
    ctx.beginPath();
    ctx.moveTo(w-20, h/2);
    ctx.lineTo(w, h/2 - 20);
    ctx.moveTo(w-20, h/2);
    ctx.lineTo(w, h/2 + 20);
    ctx.lineWidth = 2;
    ctx.stroke();
  };
  
  window.addEventListener('resize', resize);
  resize();
}

function initSimulator() {
  const annCard = document.getElementById('ann-card');
  const simOverlay = document.getElementById('simulator-overlay');
  const closeBtn = document.getElementById('close-sim');
  const buildBtn = document.getElementById('btn-build');
  
  if (!annCard || !simOverlay) return;

  // Simulator State
  const builder = new NetworkBuilder();
  const renderer = new NetworkRenderer('network-canvas');
  const ffUI = new FeedForwardUI(builder, renderer);
  const bpUI = new BackPropUI(builder, renderer);
  
  // Show/Hide Simulator
  annCard.addEventListener('click', () => {
    simOverlay.classList.add('active');
    document.body.style.overflow = 'hidden'; // Prevent background scrolling
    updateConfigUI();
  });
  
  closeBtn.addEventListener('click', () => {
    simOverlay.classList.remove('active');
    document.body.style.overflow = '';
  });

  // Config Sliders
  const inputsSlider = document.getElementById('config-inputs');
  const inputsVal = document.getElementById('val-inputs');
  const layersSlider = document.getElementById('config-layers');
  const layersVal = document.getElementById('val-layers');
  const hiddenConfigContainer = document.getElementById('hidden-layers-config');
  
  inputsSlider.addEventListener('input', (e) => {
    const val = parseInt(e.target.value);
    inputsVal.innerText = val;
    builder.setInputNodes(val);
  });
  
  layersSlider.addEventListener('input', (e) => {
    const val = parseInt(e.target.value);
    layersVal.innerText = val;
    builder.setHiddenLayerCount(val);
    updateHiddenLayersUI();
  });

  function updateHiddenLayersUI() {
    hiddenConfigContainer.innerHTML = '';
    const layers = builder.config.hiddenLayers;
    
    layers.forEach((nodes, idx) => {
      const div = document.createElement('div');
      div.className = 'hidden-layer-item';
      div.innerHTML = `
        <label>طبقة ${idx + 1}</label>
        <input type="range" class="config-slider" min="1" max="10" value="${nodes}" data-idx="${idx}">
        <span class="config-value" id="val-hl-${idx}">${nodes}</span>
      `;
      hiddenConfigContainer.appendChild(div);
      
      const slider = div.querySelector('input');
      const valSpan = div.querySelector('span');
      slider.addEventListener('input', (e) => {
        const val = parseInt(e.target.value);
        valSpan.innerText = val;
        builder.setHiddenLayerNodes(idx, val);
      });
    });
  }

  function updateConfigUI() {
    inputsSlider.value = builder.config.inputNodes;
    inputsVal.innerText = builder.config.inputNodes;
    layersSlider.value = builder.config.hiddenLayers.length;
    layersVal.innerText = builder.config.hiddenLayers.length;
    updateHiddenLayersUI();
  }

  // Build Network
  buildBtn.addEventListener('click', () => {
    const network = builder.build();
    renderer.render(network, true);
    
    // Render Data Table
    const dataContainer = document.getElementById('data-table-container');
    dataContainer.innerHTML = DataGenerator.renderTable(builder.data);
    
    // Setup Action Buttons
    setupActionButtons();
  });

  function setupActionButtons() {
    const actionContainer = document.getElementById('action-buttons-container');
    actionContainer.innerHTML = `
      <button class="btn btn-secondary" id="btn-math-explanation">
        <i class="fas fa-book-open"></i> شرح المعادلات
      </button>
      <button class="btn btn-primary" id="btn-ff-select">
        <i class="fas fa-forward"></i> FeedForward
      </button>
      <button class="btn btn-danger" id="btn-bp-start" disabled>
        <i class="fas fa-backward"></i> BackPropagation
      </button>
    `;
    
    document.getElementById('btn-math-explanation').addEventListener('click', () => {
      document.getElementById('math-explanation-modal').classList.add('active');
    });
    
    document.getElementById('btn-ff-select').addEventListener('click', showComputationModal);
    
    // Row selection logic
    const rows = document.querySelectorAll('#dataTable tbody tr');
    let selectedRowIndex = 0; // Default select first row
    
    function highlightRow(idx) {
      rows.forEach(r => r.classList.remove('active-row'));
      if(rows[idx]) rows[idx].classList.add('active-row');
      selectedRowIndex = idx;
    }
    
    highlightRow(0); // init
    
    rows.forEach((row, idx) => {
      row.addEventListener('click', () => {
        highlightRow(idx);
        document.getElementById('btn-bp-start').disabled = true; // disable bp on new selection
      });
    });
  }

  // Computation Modal
  const compModal = document.getElementById('comp-modal');
  
  function showComputationModal() {
    compModal.classList.add('active');
  }

  document.getElementById('btn-ff-node').addEventListener('click', () => {
    compModal.classList.remove('active');
    startFF('node');
  });

  document.getElementById('btn-ff-matrix').addEventListener('click', () => {
    compModal.classList.remove('active');
    startFF('matrix');
  });
  
  function startFF(mode) {
    const activeRow = document.querySelector('#dataTable tbody tr.active-row');
    if (!activeRow) return alert('الرجاء تحديد صف بيانات');
    
    const rowIndex = parseInt(activeRow.dataset.row);
    const dataRow = builder.data.features[rowIndex];
    const yTrue = builder.data.labels[rowIndex];
    
    if (mode === 'node') {
      ffUI.startNodeByNode(rowIndex, dataRow);
    } else {
      ffUI.startMatrix(rowIndex, dataRow);
    }
    
    // Enable BP for this row
    const bpBtn = document.getElementById('btn-bp-start');
    bpBtn.disabled = false;
    
    // Clear old listeners to avoid multiple fires
    const newBpBtn = bpBtn.cloneNode(true);
    bpBtn.parentNode.replaceChild(newBpBtn, bpBtn);
    
    newBpBtn.addEventListener('click', () => {
      bpUI.start(rowIndex, dataRow, yTrue);
    });
  }

  // Hook FF UI Controls
  document.getElementById('ff-next-btn').addEventListener('click', () => ffUI.next());
  document.getElementById('ff-prev-btn').addEventListener('click', () => ffUI.prev());
  document.getElementById('ff-play-btn').addEventListener('click', () => ffUI.togglePlay());
  document.getElementById('close-ff').addEventListener('click', () => ffUI.close());

  // Hook BP UI Controls
  document.getElementById('bp-next-btn').addEventListener('click', () => bpUI.next());
  document.getElementById('bp-prev-btn').addEventListener('click', () => bpUI.prev());
  document.getElementById('bp-play-btn').addEventListener('click', () => bpUI.togglePlay());
  document.getElementById('close-bp').addEventListener('click', () => bpUI.close());
}

function initDNNSimulator() {
  const dnnCard = document.getElementById('dnn-card');
  const dnnOverlay = document.getElementById('dnn-simulator-overlay');
  const dnnCloseBtn = document.getElementById('dnn-close-sim');
  const dnnBuildBtn = document.getElementById('dnn-btn-build');

  if (!dnnCard || !dnnOverlay) return;

  const dnnBuilder = new DNNBuilder();
  const dnnRenderer = new NetworkRenderer('dnn-network-canvas');
  let dnnTrainer = null;

  // ── Open/Close DNN Simulator ──
  dnnCard.addEventListener('click', () => {
    dnnOverlay.classList.add('active');
    document.body.style.overflow = 'hidden';
    updateDNNConfigUI();
  });

  dnnCloseBtn.addEventListener('click', () => {
    dnnOverlay.classList.remove('active');
    document.body.style.overflow = '';
    if (dnnTrainer && dnnTrainer.isTraining) {
      dnnTrainer.stop();
    }
  });

  // ── Tab Switching ──
  const tabs = dnnOverlay.querySelectorAll('.workspace-tab');
  const archTab = document.getElementById('dnn-arch-tab');
  const trainTab = document.getElementById('dnn-train-tab');

  tabs.forEach(tab => {
    tab.addEventListener('click', () => {
      tabs.forEach(t => t.classList.remove('active'));
      tab.classList.add('active');

      if (tab.dataset.tab === 'arch') {
        archTab.classList.add('active');
        trainTab.classList.remove('active');
      } else {
        trainTab.classList.add('active');
        archTab.classList.remove('active');
        
        // The canvas was hidden (display:none) so its width/height were 0.
        // We must redraw it now that it's visible to get correct dimensions.
        if (dnnTrainer && dnnTrainer.history && dnnTrainer.history.loss.length > 0) {
          DNNTrainer.drawLossChart('dnn-loss-chart', dnnTrainer.history);
        }
      }
    });
  });

  // ── Config Controls ──
  const inputsSlider = document.getElementById('dnn-config-inputs');
  const inputsVal = document.getElementById('dnn-val-inputs');
  const layersSlider = document.getElementById('dnn-config-layers');
  const layersVal = document.getElementById('dnn-val-layers');
  const hiddenConfig = document.getElementById('dnn-hidden-layers-config');
  const outputTypeSelect = document.getElementById('dnn-output-type');
  const classesGroup = document.getElementById('dnn-classes-group');
  const classesSlider = document.getElementById('dnn-config-classes');
  const classesVal = document.getElementById('dnn-val-classes');
  const optimizerSelect = document.getElementById('dnn-optimizer');
  const lrSlider = document.getElementById('dnn-config-lr');
  const lrVal = document.getElementById('dnn-val-lr');
  const epochsSlider = document.getElementById('dnn-config-epochs');
  const epochsVal = document.getElementById('dnn-val-epochs');
  const batchSlider = document.getElementById('dnn-config-batch');
  const batchVal = document.getElementById('dnn-val-batch');
  const dropoutSlider = document.getElementById('dnn-config-dropout');
  const dropoutVal = document.getElementById('dnn-val-dropout');
  const dropoutHint = document.getElementById('dnn-dropout-hint');
  const earlyStopToggle = document.getElementById('dnn-early-stop-toggle');
  const earlyStopConfig = document.getElementById('dnn-early-stop-config');
  const patienceSlider = document.getElementById('dnn-config-patience');
  const patienceVal = document.getElementById('dnn-val-patience');

  inputsSlider.addEventListener('input', (e) => {
    const val = parseInt(e.target.value);
    inputsVal.innerText = val;
    dnnBuilder.setInputNodes(val);
  });

  layersSlider.addEventListener('input', (e) => {
    const val = parseInt(e.target.value);
    layersVal.innerText = val;
    dnnBuilder.setHiddenLayerCount(val);
    updateHiddenLayersUI();
  });

  outputTypeSelect.addEventListener('change', (e) => {
    dnnBuilder.setOutputType(e.target.value);
    classesGroup.style.display = e.target.value === 'multiclass' ? 'flex' : 'none';
  });

  classesSlider.addEventListener('input', (e) => {
    const val = parseInt(e.target.value);
    classesVal.innerText = val;
    dnnBuilder.setOutputClasses(val);
  });

  optimizerSelect.addEventListener('change', (e) => {
    dnnBuilder.setOptimizer(e.target.value);
  });

  lrSlider.addEventListener('input', (e) => {
    const val = parseInt(e.target.value) / 1000;
    lrVal.innerText = val.toFixed(4);
    dnnBuilder.setLearningRate(val);
  });

  epochsSlider.addEventListener('input', (e) => {
    const val = parseInt(e.target.value);
    epochsVal.innerText = val;
    dnnBuilder.setEpochs(val);
  });

  batchSlider.addEventListener('input', (e) => {
    const val = parseInt(e.target.value);
    batchVal.innerText = val;
    dnnBuilder.setBatchSize(val);
  });

  dropoutSlider.addEventListener('input', (e) => {
    const val = parseInt(e.target.value);
    const rate = val / 100;
    dropoutVal.innerText = val + '%';
    dnnBuilder.setDropoutRate(rate);
    // Update hint text based on value
    if (val === 0) {
      dropoutHint.innerText = '0% = بدون إسقاط. القيم الموصى بها: 20% - 50%';
      dropoutHint.style.color = 'var(--text-muted)';
    } else if (val <= 50) {
      dropoutHint.innerText = `سيتم إسقاط ${val}% من الخلايا عشوائياً أثناء التدريب`;
      dropoutHint.style.color = '#E040FB';
    } else {
      dropoutHint.innerText = `⚠ نسبة عالية! قد تبطئ التعلم بشكل كبير`;
      dropoutHint.style.color = 'var(--warning)';
    }
  });

  earlyStopToggle.addEventListener('change', (e) => {
    const enabled = e.target.checked;
    dnnBuilder.setEarlyStopEnabled(enabled);
    earlyStopConfig.style.display = enabled ? 'flex' : 'none';
  });

  patienceSlider.addEventListener('input', (e) => {
    const val = parseInt(e.target.value);
    patienceVal.innerText = val;
    dnnBuilder.setEarlyStopPatience(val);
  });

  function updateHiddenLayersUI() {
    hiddenConfig.innerHTML = '';
    const layers = dnnBuilder.config.hiddenLayers;
    const activations = dnnBuilder.config.activations;

    layers.forEach((nodes, idx) => {
      const div = document.createElement('div');
      div.className = 'dnn-layer-config-item';
      div.innerHTML = `
        <div class="dnn-layer-config-row">
          <label>طبقة ${idx + 1}</label>
          <input type="range" class="config-slider" min="1" max="255" value="${nodes}" data-idx="${idx}">
          <span class="config-value">${nodes}</span>
        </div>
        <div class="dnn-layer-config-row">
          <label>التفعيل</label>
          <select data-idx="${idx}">
            <option value="relu" ${activations[idx] === 'relu' ? 'selected' : ''}>ReLU</option>
            <option value="sigmoid" ${activations[idx] === 'sigmoid' ? 'selected' : ''}>Sigmoid</option>
            <option value="tanh" ${activations[idx] === 'tanh' ? 'selected' : ''}>Tanh</option>
            <option value="leaky_relu" ${activations[idx] === 'leaky_relu' ? 'selected' : ''}>Leaky ReLU</option>
          </select>
        </div>
      `;
      hiddenConfig.appendChild(div);

      const slider = div.querySelector('input[type="range"]');
      const valSpan = div.querySelector('.config-value');
      slider.addEventListener('input', (e) => {
        const val = parseInt(e.target.value);
        valSpan.innerText = val;
        dnnBuilder.setHiddenLayerNodes(idx, val);
      });

      const actSelect = div.querySelector('select');
      actSelect.addEventListener('change', (e) => {
        dnnBuilder.setLayerActivation(idx, e.target.value);
      });
    });
  }

  function updateDNNConfigUI() {
    inputsSlider.value = dnnBuilder.config.inputNodes;
    inputsVal.innerText = dnnBuilder.config.inputNodes;
    layersSlider.value = dnnBuilder.config.hiddenLayers.length;
    layersVal.innerText = dnnBuilder.config.hiddenLayers.length;
    outputTypeSelect.value = dnnBuilder.config.outputType;
    classesGroup.style.display = dnnBuilder.config.outputType === 'multiclass' ? 'flex' : 'none';
    classesSlider.value = dnnBuilder.config.outputClasses;
    classesVal.innerText = dnnBuilder.config.outputClasses;
    optimizerSelect.value = dnnBuilder.config.optimizer;
    const lrDisplay = dnnBuilder.config.learningRate;
    lrSlider.value = Math.round(lrDisplay * 1000);
    lrVal.innerText = lrDisplay.toFixed(4);
    epochsSlider.value = dnnBuilder.config.epochs;
    epochsVal.innerText = dnnBuilder.config.epochs;
    batchSlider.value = dnnBuilder.config.batchSize;
    batchVal.innerText = dnnBuilder.config.batchSize;
    // Dropout
    const dropoutPercent = Math.round(dnnBuilder.config.dropoutRate * 100);
    dropoutSlider.value = dropoutPercent;
    dropoutVal.innerText = dropoutPercent + '%';
    // Early Stopping
    earlyStopToggle.checked = dnnBuilder.config.earlyStopEnabled;
    earlyStopConfig.style.display = dnnBuilder.config.earlyStopEnabled ? 'flex' : 'none';
    patienceSlider.value = dnnBuilder.config.earlyStopPatience;
    patienceVal.innerText = dnnBuilder.config.earlyStopPatience;
    updateHiddenLayersUI();
  }

  // ── Build Network ──
  dnnBuildBtn.addEventListener('click', () => {
    const network = dnnBuilder.build();

    // Render network on canvas (reuse NetworkRenderer with custom network object)
    dnnRenderer.render(network, true);

    // Render data table
    const dataContainer = document.getElementById('dnn-data-table-container');
    dataContainer.innerHTML = dnnBuilder.renderDataTable();

    // Reset training state
    resetTrainingUI();

    // Create trainer
    dnnTrainer = new DNNTrainer(dnnBuilder);

    // Initialize Weights Viewer
    const layerSelect = document.getElementById('dnn-weights-layer-select');
    if (layerSelect && network.weights) {
      layerSelect.innerHTML = '';
      for (let i = 0; i < network.weights.length; i++) {
        const option = document.createElement('option');
        option.value = i;
        option.innerText = `الطبقة ${i + 1} (${network.layerNames[i]} → ${network.layerNames[i+1]})`;
        layerSelect.appendChild(option);
      }
      
      const updateWeightsView = () => {
        const idx = parseInt(layerSelect.value);
        if (isNaN(idx)) return;
        
        const initWeights = network.initialWeights[idx];
        const currWeights = network.weights[idx];
        
        document.getElementById('dnn-initial-weights-table').innerHTML = renderWeightsTableHTML(initWeights);
        document.getElementById('dnn-current-weights-table').innerHTML = renderWeightsTableHTML(currWeights);
      };
      
      layerSelect.removeEventListener('change', window._dnnWeightsHandler);
      window._dnnWeightsHandler = updateWeightsView;
      layerSelect.addEventListener('change', window._dnnWeightsHandler);
      
      updateWeightsView();
    }
  });

  function renderWeightsTableHTML(matrix) {
    if (!matrix || matrix.length === 0) return '';
    let html = '<thead><tr><th></th>';
    for (let j = 0; j < matrix[0].length; j++) {
      html += `<th>N${j+1}</th>`;
    }
    html += '</tr></thead><tbody>';
    for (let i = 0; i < matrix.length; i++) {
      html += `<tr><th>N${i+1}</th>`;
      for (let j = 0; j < matrix[i].length; j++) {
        html += `<td>${matrix[i][j].toFixed(4)}</td>`;
      }
      html += '</tr>';
    }
    html += '</tbody>';
    return html;
  }

  // ── Training Controls ──
  const trainBtn = document.getElementById('dnn-btn-train');
  const stopBtn = document.getElementById('dnn-btn-stop');
  const resetBtn = document.getElementById('dnn-btn-reset');

  trainBtn.addEventListener('click', async () => {
    if (!dnnBuilder.isBuilt) {
      alert('الرجاء بناء الشبكة أولاً');
      return;
    }

    if (!dnnTrainer || dnnTrainer.isTraining) return;

    trainBtn.disabled = true;
    stopBtn.disabled = false;

    const statusEl = document.getElementById('dnn-training-status');
    statusEl.classList.add('dnn-training-active');

    document.getElementById('dnn-opt-display').innerText = dnnBuilder.config.optimizer.toUpperCase();

    // Show dropout info if active
    const dropoutRate = dnnBuilder.config.dropoutRate;
    const dropoutStat = document.getElementById('dnn-dropout-stat');
    if (dropoutRate > 0) {
      dropoutStat.style.display = 'block';
      document.getElementById('dnn-dropout-display').innerText = Math.round(dropoutRate * 100) + '%';
      // Generate initial dropout masks to show visually
      dnnBuilder.generateDropoutMasks();
      dnnRenderer.setDropoutMasks(dnnBuilder.dropoutMasks, dnnBuilder.network.layers);
      dnnRenderer.draw();
    } else {
      dropoutStat.style.display = 'none';
      dnnRenderer.setDropoutMasks(null, null);
    }

    // Hook up animation for the first sample pass
    dnnTrainer.onPassAnimation = async (layerOutputs, deltas) => {
      const passIndicator = document.getElementById('dnn-pass-indicator');
      const passLabel = document.getElementById('dnn-pass-label');
      
      if (passIndicator && passLabel) {
        passIndicator.style.display = 'flex';
        passIndicator.className = 'dnn-pass-indicator forward';
        passLabel.innerText = 'إنتشار أمامي (Forward Pass)';
      }
      
      await dnnRenderer.animateForwardPass(layerOutputs, 200);
      
      if (passIndicator && passLabel) {
        passIndicator.className = 'dnn-pass-indicator backward';
        passLabel.innerText = 'إنتشار خلفي (Backward Pass)';
      }
      
      await dnnRenderer.animateBackwardPass(deltas, 200);
      
      if (passIndicator) {
        passIndicator.style.display = 'none';
      }
      
      dnnRenderer.setAllValues(layerOutputs);
    };

    dnnTrainer.onEpochEnd = (epoch, loss, accuracy) => {
      const totalEpochs = dnnBuilder.config.epochs;
      document.getElementById('dnn-epoch-display').innerText = `${epoch + 1} / ${totalEpochs}`;
      document.getElementById('dnn-loss-display').innerText = loss.toFixed(4);
      document.getElementById('dnn-acc-display').innerText = (accuracy * 100).toFixed(1) + '%';
      document.getElementById('dnn-train-progress').style.width = `${((epoch + 1) / totalEpochs) * 100}%`;

      // Update chart every few epochs to avoid too many redraws
      if (epoch % Math.max(1, Math.floor(totalEpochs / 100)) === 0 || epoch === totalEpochs - 1) {
        DNNTrainer.drawLossChart('dnn-loss-chart', dnnTrainer.history);
      }
      
      // Live update the current weights table
      if (window._dnnWeightsHandler) {
        window._dnnWeightsHandler();
      }

      // Update dropout visualization dynamically
      if (dnnBuilder.config.dropoutRate > 0) {
        dnnRenderer.setDropoutMasks(dnnBuilder.dropoutMasks, dnnBuilder.network.layers);
        dnnRenderer.draw();
      }
    };

    dnnTrainer.onTrainingEnd = (history) => {
      trainBtn.disabled = false;
      stopBtn.disabled = true;
      statusEl.classList.remove('dnn-training-active');
      DNNTrainer.drawLossChart('dnn-loss-chart', history);
      // Clear dropout visual
      dnnRenderer.setDropoutMasks(null, null);
      dnnRenderer.draw();
    };

    dnnTrainer.onEarlyStop = (epoch, bestLoss) => {
      const statusEl = document.getElementById('dnn-training-status');
      // Show early stop notification
      const passIndicator = document.getElementById('dnn-pass-indicator');
      const passLabel = document.getElementById('dnn-pass-label');
      if (passIndicator && passLabel) {
        passIndicator.style.display = 'flex';
        passIndicator.className = 'dnn-pass-indicator early-stop';
        passLabel.innerHTML = `<i class="fas fa-hand-paper"></i> توقف مبكر عند Epoch ${epoch + 1} — أفضل Loss: ${bestLoss.toFixed(4)}`;
        // Auto-hide after 5 seconds
        setTimeout(() => {
          passIndicator.style.display = 'none';
        }, 5000);
      }
    };

    await dnnTrainer.train();
  });

  stopBtn.addEventListener('click', () => {
    if (dnnTrainer) {
      dnnTrainer.stop();
    }
    trainBtn.disabled = false;
    stopBtn.disabled = true;
    document.getElementById('dnn-training-status').classList.remove('dnn-training-active');
  });

  resetBtn.addEventListener('click', () => {
    if (dnnTrainer && dnnTrainer.isTraining) {
      dnnTrainer.stop();
    }
    // Rebuild network with fresh weights
    if (dnnBuilder.isBuilt) {
      dnnBuildBtn.click();
    }
    resetTrainingUI();
  });

  function resetTrainingUI() {
    document.getElementById('dnn-epoch-display').innerText = '0 / 0';
    document.getElementById('dnn-loss-display').innerText = '—';
    document.getElementById('dnn-acc-display').innerText = '—';
    document.getElementById('dnn-opt-display').innerText = '—';
    document.getElementById('dnn-train-progress').style.width = '0%';
    document.getElementById('dnn-training-status').classList.remove('dnn-training-active');
    document.getElementById('dnn-dropout-stat').style.display = 'none';

    // Hide pass indicator (early stop notification)
    const passIndicator = document.getElementById('dnn-pass-indicator');
    if (passIndicator) passIndicator.style.display = 'none';

    trainBtn.disabled = false;
    stopBtn.disabled = true;

    // Clear dropout visual
    dnnRenderer.setDropoutMasks(null, null);

    // Clear chart
    const canvas = document.getElementById('dnn-loss-chart');
    if (canvas) {
      const ctx = canvas.getContext('2d');
      ctx.clearRect(0, 0, canvas.width, canvas.height);
    }
  }

  // Automatically build the initial network so the data table populates
  setTimeout(() => dnnBuildBtn.click(), 100);
}

/* ============================================
   CNN Simulator Initialization
   ============================================ */

function initCNNSimulator() {
  const cnnCard = document.getElementById('cnn-card');
  const cnnOverlay = document.getElementById('cnn-simulator-overlay');
  const cnnCloseBtn = document.getElementById('cnn-close-sim');
  const cnnRunBtn = document.getElementById('cnn-btn-run');

  if (!cnnCard || !cnnOverlay) return;

  const cnnBuilder = new CNNBuilder();
  const cnnVisualizer = new CNNVisualizer('cnn-viewport');
  window._cnnVisualizer = cnnVisualizer; // for pipeline node clicks

  // CNN Architectures
  const cnnArchitectures = new CNNArchitectures('cnn-arch-modal');
  window._cnnArchitectures = cnnArchitectures; // for onclick in rendered HTML

  const archBtn = document.getElementById('cnn-arch-btn');
  const archCloseBtn = document.getElementById('cnn-arch-close');

  if (archBtn) {
    archBtn.addEventListener('click', (e) => {
      e.stopPropagation();
      cnnArchitectures.open();
    });
  }

  if (archCloseBtn) {
    archCloseBtn.addEventListener('click', () => {
      cnnArchitectures.close();
    });
  }

  // Close arch modal on overlay click
  const archModal = document.getElementById('cnn-arch-modal');
  if (archModal) {
    archModal.addEventListener('click', (e) => {
      if (e.target === archModal) {
        cnnArchitectures.close();
      }
    });
  }

  // ── Tabs ──
  const tabInput = document.getElementById('cnn-tab-input');
  const tabArch = document.getElementById('cnn-tab-arch');
  const contentInput = document.getElementById('cnn-content-input');
  const contentArch = document.getElementById('cnn-content-arch');

  if(tabInput) {
      tabInput.addEventListener('click', () => {
          tabInput.classList.add('active');
          tabArch.classList.remove('active');
          contentInput.classList.add('active');
          contentArch.classList.remove('active');
      });
  }
  if(tabArch) {
      tabArch.addEventListener('click', () => {
          tabArch.classList.add('active');
          tabInput.classList.remove('active');
          contentArch.classList.add('active');
          contentInput.classList.remove('active');
      });
  }

  // ── Grid Size & Drawing state ──
  let gridSize = 5;
  let drawGrid = new Array(gridSize * gridSize).fill(0);
  let isDrawing = false;
  let drawMode = 1; // 1 = paint, 0 = erase
  let inputMode = 'draw'; // 'draw' or 'image'
  let uploadedImageMatrix = null;

  const gridSizeSelect = document.getElementById('cnn-grid-size-select');
  if(gridSizeSelect) {
      gridSizeSelect.addEventListener('change', (e) => {
          gridSize = parseInt(e.target.value);
          cnnBuilder.setInputSize(gridSize);
          drawGrid = new Array(gridSize * gridSize).fill(0);
          inputMode = 'draw';
          uploadedImageMatrix = null;
          document.getElementById('cnn-image-preview-container').style.display = 'none';
          initDrawingCanvas();
      });
  }

  // ── Image Upload ──
  const imageUpload = document.getElementById('cnn-image-upload');
  const imagePreview = document.getElementById('cnn-image-preview-canvas');
  const imagePreviewContainer = document.getElementById('cnn-image-preview-container');
  const thresholdSlider = document.getElementById('cnn-threshold-slider');
  const thresholdVal = document.getElementById('cnn-threshold-val');
  let currentImageSrc = null;

  if (imageUpload) {
      imageUpload.addEventListener('change', (e) => {
          const file = e.target.files[0];
          if (!file) return;
          const reader = new FileReader();
          reader.onload = function(event) {
              const img = new Image();
              img.onload = function() {
                  currentImageSrc = img;
                  processUploadedImage();
              }
              img.src = event.target.result;
          }
          reader.readAsDataURL(file);
      });
  }

  if (thresholdSlider) {
      thresholdSlider.addEventListener('input', (e) => {
          thresholdVal.innerText = e.target.value;
          if (currentImageSrc) processUploadedImage();
      });
  }

  function processUploadedImage() {
      if(!currentImageSrc) return;
      
      const ctx = imagePreview.getContext('2d');
      imagePreview.width = gridSize;
      imagePreview.height = gridSize;
      
      // Draw image scaled down
      ctx.drawImage(currentImageSrc, 0, 0, gridSize, gridSize);
      
      const imageData = ctx.getImageData(0, 0, gridSize, gridSize);
      const data = imageData.data;
      const threshold = parseInt(thresholdSlider.value);
      
      uploadedImageMatrix = [];
      let flatIndex = 0;
      
      for (let i = 0; i < gridSize; i++) {
          uploadedImageMatrix[i] = [];
          for (let j = 0; j < gridSize; j++) {
              const r = data[(i * gridSize + j) * 4];
              const g = data[(i * gridSize + j) * 4 + 1];
              const b = data[(i * gridSize + j) * 4 + 2];
              // grayscale
              const gray = 0.299*r + 0.587*g + 0.114*b;
              // binarize (invert so dark ink is 1)
              const val = gray < threshold ? 1 : 0;
              
              uploadedImageMatrix[i][j] = val;
              
              // update preview to show binarized result
              const c = val === 1 ? 0 : 255;
              data[(i * gridSize + j) * 4] = c;
              data[(i * gridSize + j) * 4 + 1] = c;
              data[(i * gridSize + j) * 4 + 2] = c;
              data[(i * gridSize + j) * 4 + 3] = 255; // alpha
              
              drawGrid[flatIndex++] = val;
          }
      }
      
      ctx.putImageData(imageData, 0, 0);
      imagePreviewContainer.style.display = 'block';
      inputMode = 'image';
      
      // Update draw grid visually
      const cells = document.querySelectorAll('.cnn-draw-cell');
      if(cells.length === gridSize*gridSize) {
          cells.forEach((cell, idx) => {
              if(drawGrid[idx] === 1) cell.classList.add('active');
              else cell.classList.remove('active');
          });
      }
  }

  // ── Open / Close ──
  cnnCard.addEventListener('click', () => {
    cnnOverlay.classList.add('active');
    document.body.style.overflow = 'hidden';
    initDrawingCanvas();
    initSampleDigits();
    renderArchBuilder();
  });

  cnnCloseBtn.addEventListener('click', () => {
    cnnOverlay.classList.remove('active');
    document.body.style.overflow = '';
    cnnVisualizer.stopPlay();
  });

  // ── Drawing Canvas ──
  function initDrawingCanvas() {
    const grid = document.getElementById('cnn-draw-grid');
    if (!grid) return;
    
    grid.innerHTML = '';
    grid.className = `cnn-draw-grid cnn-grid-${gridSize}`;
    
    for (let i = 0; i < gridSize * gridSize; i++) {
      const cell = document.createElement('div');
      cell.className = 'cnn-draw-cell';
      if(drawGrid[i] === 1) cell.classList.add('active');
      cell.dataset.idx = i;

      cell.addEventListener('mousedown', (e) => {
        e.preventDefault();
        inputMode = 'draw';
        isDrawing = true;
        drawMode = drawGrid[i] === 1 ? 0 : 1;
        drawGrid[i] = drawMode;
        updateDrawCell(cell, drawMode);
      });

      cell.addEventListener('mouseenter', () => {
        if (isDrawing) {
          drawGrid[parseInt(cell.dataset.idx)] = drawMode;
          updateDrawCell(cell, drawMode);
        }
      });

      // Touch support
      cell.addEventListener('touchstart', (e) => {
        e.preventDefault();
        inputMode = 'draw';
        isDrawing = true;
        drawMode = drawGrid[i] === 1 ? 0 : 1;
        drawGrid[i] = drawMode;
        updateDrawCell(cell, drawMode);
      });

      grid.appendChild(cell);
    }

    document.addEventListener('mouseup', () => { isDrawing = false; });
    document.addEventListener('touchend', () => { isDrawing = false; });

    // Handle touch move for drawing
    grid.addEventListener('touchmove', (e) => {
      e.preventDefault();
      if (!isDrawing) return;
      const touch = e.touches[0];
      const target = document.elementFromPoint(touch.clientX, touch.clientY);
      if (target && target.classList.contains('cnn-draw-cell')) {
        const idx = parseInt(target.dataset.idx);
        drawGrid[idx] = drawMode;
        updateDrawCell(target, drawMode);
      }
    });

    // Clear button
    const clearBtn = document.getElementById('cnn-clear-canvas');
    if (clearBtn) {
        // remove old listeners by replacing node
        const newClearBtn = clearBtn.cloneNode(true);
        clearBtn.parentNode.replaceChild(newClearBtn, clearBtn);
        
        newClearBtn.addEventListener('click', () => {
          drawGrid.fill(0);
          grid.querySelectorAll('.cnn-draw-cell').forEach(c => c.classList.remove('active'));
          document.querySelectorAll('.cnn-sample-btn').forEach(b => b.classList.remove('active'));
          inputMode = 'draw';
          uploadedImageMatrix = null;
          document.getElementById('cnn-image-preview-container').style.display = 'none';
          currentImageSrc = null;
          if(imageUpload) imageUpload.value = '';
        });
    }
  }

  function updateDrawCell(cell, val) {
    if (val === 1) cell.classList.add('active');
    else cell.classList.remove('active');
  }

  function setDrawGrid(matrix) {
    // Only works if grid is same size, assume samples are 5x5
    if (gridSize !== 5) {
        gridSizeSelect.value = "5";
        gridSizeSelect.dispatchEvent(new Event('change'));
    }
    
    inputMode = 'draw';
    setTimeout(() => {
        const grid = document.getElementById('cnn-draw-grid');
        if (!grid) return;
        const cells = grid.querySelectorAll('.cnn-draw-cell');
        for (let i = 0; i < 5; i++) {
          for (let j = 0; j < 5; j++) {
            const idx = i * 5 + j;
            drawGrid[idx] = matrix[i][j];
            if (cells[idx]) updateDrawCell(cells[idx], matrix[i][j]);
          }
        }
    }, 50);
  }

  function getDrawMatrix() {
    if (inputMode === 'image' && uploadedImageMatrix) {
        return uploadedImageMatrix;
    }
    const matrix = [];
    for (let i = 0; i < gridSize; i++) {
      matrix[i] = [];
      for (let j = 0; j < gridSize; j++) {
        matrix[i][j] = drawGrid[i * gridSize + j];
      }
    }
    return matrix;
  }

  // ── Sample Digits ──
  function initSampleDigits() {
    const samplesGrid = document.getElementById('cnn-samples-grid');
    if (!samplesGrid || samplesGrid.children.length > 0) return;

    const digits = CNNBuilder.getSampleDigits();
    for (let d = 0; d <= 9; d++) {
      const btn = document.createElement('button');
      btn.className = 'cnn-sample-btn';
      btn.innerText = d;
      btn.dataset.digit = d;

      btn.addEventListener('click', () => {
        document.querySelectorAll('.cnn-sample-btn').forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        setDrawGrid(digits[d]);
      });

      samplesGrid.appendChild(btn);
    }
  }

  // ── Architecture Builder ──
  function renderArchBuilder() {
     const list = document.getElementById('cnn-arch-builder-list');
     if(!list) return;
     list.innerHTML = '';

     cnnBuilder.layers.forEach((layer, index) => {
        const card = document.createElement('div');
        card.className = 'cnn-layer-card';
        
        let titleHtml = '';
        let settingsHtml = '';

        if(layer.type === 'conv2d') {
            titleHtml = `<div class="cnn-layer-title conv2d"><i class="fas fa-asterisk"></i> Conv2D</div>`;
            settingsHtml = `
               <div class="cnn-layer-setting-group">Filters: <input type="number" class="cnn-layer-input" value="${layer.filters}" min="1" max="16" onchange="window._cnnBuilderUpdate(${index}, 'filters', this.value)"></div>
               <div class="cnn-layer-setting-group">Size: <input type="number" class="cnn-layer-input" value="${layer.size}" min="2" max="5" onchange="window._cnnBuilderUpdate(${index}, 'size', this.value)"></div>
               <div class="cnn-layer-setting-group">Act: 
                  <select class="cnn-layer-select" onchange="window._cnnBuilderUpdate(${index}, 'activation', this.value)">
                     <option value="none" ${layer.activation==='none'?'selected':''}>None</option>
                     <option value="relu" ${layer.activation==='relu'?'selected':''}>ReLU</option>
                     <option value="sigmoid" ${layer.activation==='sigmoid'?'selected':''}>Sigmoid</option>
                  </select>
               </div>
            `;
        } else if(layer.type === 'pool2d') {
            titleHtml = `<div class="cnn-layer-title pool2d"><i class="fas fa-compress-arrows-alt"></i> Pool2D</div>`;
            settingsHtml = `
               <div class="cnn-layer-setting-group">Size: <input type="number" class="cnn-layer-input" value="${layer.size}" min="2" max="4" onchange="window._cnnBuilderUpdate(${index}, 'size', this.value)"></div>
               <div class="cnn-layer-setting-group">Type: 
                  <select class="cnn-layer-select" onchange="window._cnnBuilderUpdate(${index}, 'poolType', this.value)">
                     <option value="max" ${layer.poolType==='max'?'selected':''}>Max</option>
                     <option value="average" ${layer.poolType==='average'?'selected':''}>Avg</option>
                  </select>
               </div>
            `;
        } else if(layer.type === 'flatten') {
            titleHtml = `<div class="cnn-layer-title flatten"><i class="fas fa-arrows-alt-h"></i> Flatten</div>`;
            settingsHtml = `<div style="font-size:0.75rem; color:var(--text-muted)">تحويل المصفوفة إلى متجه</div>`;
        } else if(layer.type === 'fc') {
            titleHtml = `<div class="cnn-layer-title fc"><i class="fas fa-project-diagram"></i> Fully Connected</div>`;
            settingsHtml = `
               <div class="cnn-layer-setting-group">Neurons: <input type="number" class="cnn-layer-input" value="${layer.neurons}" min="2" max="128" onchange="window._cnnBuilderUpdate(${index}, 'neurons', this.value)"></div>
               <div class="cnn-layer-setting-group">Act: 
                  <select class="cnn-layer-select" onchange="window._cnnBuilderUpdate(${index}, 'activation', this.value)">
                     <option value="none" ${layer.activation==='none'?'selected':''}>None</option>
                     <option value="relu" ${layer.activation==='relu'?'selected':''}>ReLU</option>
                     <option value="softmax" ${layer.activation==='softmax'?'selected':''}>Softmax</option>
                  </select>
               </div>
            `;
        }

        card.innerHTML = `
          <div class="cnn-layer-header">
             ${titleHtml}
             <div class="cnn-layer-controls">
                <button class="cnn-layer-btn" onclick="window._cnnBuilderMove(${index}, 'up')" ${index===0?'disabled':''}><i class="fas fa-chevron-up"></i></button>
                <button class="cnn-layer-btn" onclick="window._cnnBuilderMove(${index}, 'down')" ${index===cnnBuilder.layers.length-1?'disabled':''}><i class="fas fa-chevron-down"></i></button>
                <button class="cnn-layer-btn delete" onclick="window._cnnBuilderRemove(${index})"><i class="fas fa-times"></i></button>
             </div>
          </div>
          <div class="cnn-layer-settings">${settingsHtml}</div>
        `;
        list.appendChild(card);
     });
     updateCNNSummaryUI();
  }

  // Global functions for inline HTML handlers
  window._cnnBuilderUpdate = function(index, key, val) {
      const v = (key === 'activation' || key === 'poolType') ? val : parseInt(val);
      cnnBuilder.updateLayer(index, { [key]: v });
      updateCNNSummaryUI();
  };
  window._cnnBuilderMove = function(index, dir) {
      cnnBuilder.moveLayer(index, dir);
      renderArchBuilder();
  };
  window._cnnBuilderRemove = function(index) {
      cnnBuilder.removeLayer(index);
      renderArchBuilder();
  };

  document.getElementById('cnn-add-conv')?.addEventListener('click', () => { cnnBuilder.addLayer('conv2d'); renderArchBuilder(); });
  document.getElementById('cnn-add-pool')?.addEventListener('click', () => { cnnBuilder.addLayer('pool2d'); renderArchBuilder(); });
  document.getElementById('cnn-add-fc')?.addEventListener('click', () => { cnnBuilder.addLayer('fc'); renderArchBuilder(); });

  function updateCNNSummaryUI() {
      // Just a mock build to get summary
      try {
         cnnBuilder.build();
         const summary = cnnBuilder.getSummary();
         const summaryText = document.getElementById('cnn-summary-text');
         if (summaryText && summary) {
           summaryText.innerHTML = `
             <div style="margin-bottom: 0.3rem; font-size: 0.8rem;"><b>المسار:</b></div>
             <div style="direction: ltr; text-align: left; margin-bottom: 0.5rem; font-size: 0.75rem; word-break: break-word;">${summary.pipeline}</div>
             <div style="font-size: 0.8rem;"><b>المعاملات:</b> ${summary.totalParams}</div>
           `;
         }
      } catch(e) {
         const summaryText = document.getElementById('cnn-summary-text');
         if (summaryText) summaryText.innerHTML = `<div style="color:var(--error); font-size:0.8rem;">خطأ في الهيكل: ${e.message}</div>`;
      }
  }

  // ── Run CNN ──
  cnnRunBtn.addEventListener('click', () => {
    const inputMatrix = getDrawMatrix();

    // Check if anything is drawn
    const hasPixels = drawGrid.some(v => v === 1);
    if (!hasPixels) {
      alert('الرجاء رسم رقم أو اختيار مثال أولاً');
      return;
    }

    // Build network
    cnnBuilder.build();

    // Update summary
    const summary = cnnBuilder.getSummary();
    const summaryText = document.getElementById('cnn-summary-text');
    if (summaryText && summary) {
      summaryText.innerHTML = `
        <div style="margin-bottom: 0.3rem;"><b>Pipeline:</b></div>
        <div style="direction: ltr; text-align: left; margin-bottom: 0.5rem;">${summary.pipeline}</div>
        <div><b>المعاملات:</b> ${summary.totalParams} (فلاتر: ${summary.filterParams} + FC: ${summary.fcParams})</div>
      `;
    }

    // Init visualizer
    cnnVisualizer.init(cnnBuilder);

    // Run pipeline
    cnnVisualizer.run(inputMatrix);

    // Render pipeline overview
    cnnVisualizer.renderPipelineOverview('cnn-pipeline-bar');

    // Create step dots
    const dotsContainer = document.getElementById('cnn-step-dots');
    if (dotsContainer) {
      dotsContainer.innerHTML = '';
      for (let i = 0; i < cnnVisualizer.totalSteps; i++) {
        const dot = document.createElement('div');
        dot.className = 'cnn-step-dot' + (i === 0 ? ' active' : '');
        dot.title = `الخطوة ${i + 1}`;
        dot.addEventListener('click', () => {
          cnnVisualizer.goToStep(i);
        });
        dotsContainer.appendChild(dot);
      }
    }
  });

  // ── Step Navigation ──
  document.getElementById('cnn-prev-btn').addEventListener('click', () => {
    cnnVisualizer.prev();
  });

  document.getElementById('cnn-next-btn').addEventListener('click', () => {
    cnnVisualizer.next();
  });

  document.getElementById('cnn-play-btn').addEventListener('click', () => {
    cnnVisualizer.togglePlay();
  });
}

/* ============================================
   RNN SIMULATOR INITIALIZATION
   ============================================ */
function initRNNSimulator() {
  const rnnCard = document.getElementById('rnn-card');
  const rnnOverlay = document.getElementById('rnn-simulator-overlay');
  const rnnCloseBtn = document.getElementById('rnn-close-sim');
  const rnnBuildBtn = document.getElementById('rnn-btn-build');

  if (!rnnCard || !rnnOverlay) return;

  const rnnBuilder = new RNNBuilder();
  let rnnTrainer = null;
  let rnnStates = null; // Stores forward pass states for visualization
  let animationStep = -1;
  let animTimer = null;

  // ── Open/Close RNN Simulator ──
  rnnCard.addEventListener('click', () => {
    rnnOverlay.classList.add('active');
    document.body.style.overflow = 'hidden';
    updateRNNConfigUI();
  });

  rnnCloseBtn.addEventListener('click', () => {
    rnnOverlay.classList.remove('active');
    document.body.style.overflow = '';
    if (rnnTrainer && rnnTrainer.isTraining) {
      rnnTrainer.stop();
    }
    if (animTimer) {
      clearInterval(animTimer);
      animTimer = null;
    }
  });

  // ── Tab Switching ──
  const tabs = rnnOverlay.querySelectorAll('.workspace-tab');
  const archTab = document.getElementById('rnn-arch-tab');
  const trainTab = document.getElementById('rnn-train-tab');

  tabs.forEach(tab => {
    tab.addEventListener('click', () => {
      tabs.forEach(t => t.classList.remove('active'));
      tab.classList.add('active');

      if (tab.dataset.tab === 'rnn-arch') {
        archTab.classList.add('active');
        trainTab.classList.remove('active');
      } else {
        trainTab.classList.add('active');
        archTab.classList.remove('active');

        // Redraw chart since canvas may have been hidden
        if (rnnTrainer && rnnTrainer.history && rnnTrainer.history.loss.length > 0) {
          RNNTrainer.drawLossChart('rnn-loss-chart', rnnTrainer.history);
        }
      }
    });
  });

  // ── Config Controls ──
  const inputsSlider = document.getElementById('rnn-config-inputs');
  const inputsVal = document.getElementById('rnn-val-inputs');
  const hiddenSlider = document.getElementById('rnn-config-hidden');
  const hiddenVal = document.getElementById('rnn-val-hidden');
  const timestepsSlider = document.getElementById('rnn-config-timesteps');
  const timestepsVal = document.getElementById('rnn-val-timesteps');
  const cellTypeSelect = document.getElementById('rnn-cell-type');
  const activationSelect = document.getElementById('rnn-activation');
  const activationGroup = document.getElementById('rnn-activation-group');
  const outputModeSelect = document.getElementById('rnn-output-mode');
  const optimizerSelect = document.getElementById('rnn-optimizer');
  const lrSlider = document.getElementById('rnn-config-lr');
  const lrVal = document.getElementById('rnn-val-lr');
  const epochsSlider = document.getElementById('rnn-config-epochs');
  const epochsVal = document.getElementById('rnn-val-epochs');
  const clipSlider = document.getElementById('rnn-config-clip');
  const clipVal = document.getElementById('rnn-val-clip');

  function autoRebuild() {
    // Only auto-rebuild if not actively training
    if (!rnnTrainer || !rnnTrainer.isTraining) {
      rnnBuildBtn.click();
    }
  }

  inputsSlider.addEventListener('input', (e) => {
    const val = parseInt(e.target.value);
    inputsVal.innerText = val;
    rnnBuilder.setInputFeatures(val);
    autoRebuild();
  });

  hiddenSlider.addEventListener('input', (e) => {
    const val = parseInt(e.target.value);
    hiddenVal.innerText = val;
    rnnBuilder.setHiddenUnits(val);
    autoRebuild();
  });

  const layersSlider = document.getElementById('rnn-config-layers');
  const layersVal = document.getElementById('rnn-val-layers');
  
  if(layersSlider) {
    layersSlider.addEventListener('input', (e) => {
      const val = parseInt(e.target.value);
      layersVal.innerText = val;
      rnnBuilder.setNumLayers(val);
      autoRebuild();
    });
  }

  timestepsSlider.addEventListener('input', (e) => {
    const val = parseInt(e.target.value);
    timestepsVal.innerText = val;
    rnnBuilder.setTimeSteps(val);
    autoRebuild();
  });

  cellTypeSelect.addEventListener('change', (e) => {
    rnnBuilder.setCellType(e.target.value);
    // Hide activation dropdown for LSTM/GRU (they use fixed gates)
    if (e.target.value === 'lstm' || e.target.value === 'gru') {
      activationGroup.style.display = 'none';
    } else {
      activationGroup.style.display = '';
    }
    autoRebuild();
  });

  activationSelect.addEventListener('change', (e) => {
    rnnBuilder.setActivation(e.target.value);
    autoRebuild();
  });

  outputModeSelect.addEventListener('change', (e) => {
    rnnBuilder.setOutputMode(e.target.value);
    autoRebuild();
  });

  optimizerSelect.addEventListener('change', (e) => {
    rnnBuilder.setOptimizer(e.target.value);
  });

  lrSlider.addEventListener('input', (e) => {
    const val = parseInt(e.target.value) / 1000;
    lrVal.innerText = val.toFixed(3);
    rnnBuilder.setLearningRate(val);
  });

  epochsSlider.addEventListener('input', (e) => {
    const val = parseInt(e.target.value);
    epochsVal.innerText = val;
    rnnBuilder.setEpochs(val);
  });

  clipSlider.addEventListener('input', (e) => {
    const val = parseInt(e.target.value);
    clipVal.innerText = val;
    rnnBuilder.setClipGradient(val);
  });

  function updateRNNConfigUI() {
    inputsSlider.value = rnnBuilder.config.inputFeatures;
    inputsVal.innerText = rnnBuilder.config.inputFeatures;
    hiddenSlider.value = rnnBuilder.config.hiddenUnits;
    hiddenVal.innerText = rnnBuilder.config.hiddenUnits;
    if (layersSlider) {
      layersSlider.value = rnnBuilder.config.numLayers;
      layersVal.innerText = rnnBuilder.config.numLayers;
    }
    timestepsSlider.value = rnnBuilder.config.timeSteps;
    timestepsVal.innerText = rnnBuilder.config.timeSteps;
    cellTypeSelect.value = rnnBuilder.config.cellType;
    activationSelect.value = rnnBuilder.config.activation;
    outputModeSelect.value = rnnBuilder.config.outputMode;
    optimizerSelect.value = rnnBuilder.config.optimizer;
    lrSlider.value = rnnBuilder.config.learningRate * 1000;
    lrVal.innerText = rnnBuilder.config.learningRate.toFixed(3);
    epochsSlider.value = rnnBuilder.config.epochs;
    epochsVal.innerText = rnnBuilder.config.epochs;
    clipSlider.value = rnnBuilder.config.clipGradient;
    clipVal.innerText = rnnBuilder.config.clipGradient;

    // Show/hide activation group
    if (rnnBuilder.config.cellType !== 'simple') {
      activationGroup.style.display = 'none';
    } else {
      activationGroup.style.display = '';
    }
  }

  // ── Build Network ──
  rnnBuildBtn.addEventListener('click', () => {
    const network = rnnBuilder.build();

    // Run forward pass on first sample to get states for visualization
    if (rnnBuilder.data && rnnBuilder.data.sequences.length > 0) {
      rnnStates = rnnBuilder.forward(rnnBuilder.data.sequences[0]);
    }

    // Draw unfolded RNN
    drawRNNArchitecture(-1);

    // Update data table
    const dataContainer = document.getElementById('rnn-data-table-container');
    dataContainer.innerHTML = rnnBuilder.renderDataTable();

    // Update weights tables
    RNNTrainer.renderWeightsTable(network.initialWeights, 'rnn-initial-weights-table');
    RNNTrainer.renderWeightsTable(network.weights, 'rnn-current-weights-table');

    // Update status
    document.getElementById('rnn-epoch-display').innerText = `0/${rnnBuilder.config.epochs}`;
    document.getElementById('rnn-loss-display').innerText = '—';
    document.getElementById('rnn-cell-display').innerText = rnnBuilder.config.cellType.toUpperCase();
    document.getElementById('rnn-opt-display').innerText = rnnBuilder.config.optimizer.toUpperCase();
    document.getElementById('rnn-train-progress').style.width = '0%';
  });

  function drawRNNArchitecture(highlight) {
    RNNTrainer.drawUnfoldedRNN('rnn-network-canvas', rnnBuilder.config, rnnStates, highlight);
  }

  function startDataFlowAnimation() {
    if (animTimer) clearInterval(animTimer);
    animationStep = 0;

    animTimer = setInterval(() => {
      drawRNNArchitecture(animationStep);
      animationStep = (animationStep + 1) % rnnBuilder.config.timeSteps;
    }, 800);
  }

  // ── Training Controls ──
  const trainBtn = document.getElementById('rnn-btn-train');
  const stopBtn = document.getElementById('rnn-btn-stop');
  const resetBtn = document.getElementById('rnn-btn-reset');

  trainBtn.addEventListener('click', async () => {
    if (!rnnBuilder.isBuilt) {
      // Auto-build the network first (synchronous)
      const network = rnnBuilder.build();
      
      if (rnnBuilder.data && rnnBuilder.data.sequences.length > 0) {
        rnnStates = rnnBuilder.forward(rnnBuilder.data.sequences[0]);
      }
      drawRNNArchitecture(-1);
      
      const dataContainer = document.getElementById('rnn-data-table-container');
      dataContainer.innerHTML = rnnBuilder.renderDataTable();
      
      RNNTrainer.renderWeightsTable(network.initialWeights, 'rnn-initial-weights-table');
      RNNTrainer.renderWeightsTable(network.weights, 'rnn-current-weights-table');
      
      document.getElementById('rnn-cell-display').innerText = rnnBuilder.config.cellType.toUpperCase();
      document.getElementById('rnn-opt-display').innerText = rnnBuilder.config.optimizer.toUpperCase();
    }

    if (animTimer) {
      clearInterval(animTimer);
      animTimer = null;
    }

    rnnTrainer = new RNNTrainer(rnnBuilder);

    trainBtn.disabled = true;
    stopBtn.disabled = false;
    rnnBuildBtn.disabled = true;

    document.getElementById('rnn-epoch-display').innerText = `0/${rnnBuilder.config.epochs}`;
    document.getElementById('rnn-train-progress').style.width = '0%';

    const passIndicator = document.getElementById('rnn-pass-indicator');
    const passLabel = document.getElementById('rnn-pass-label');
    const T = rnnBuilder.config.timeSteps;

    // ── Animation callback: Forward Through Time + BPTT ──
    rnnTrainer.onPassAnimation = async (states, sequence) => {
      if (rnnTrainer.shouldStop) return;

      rnnStates = states;

      // ── Forward Through Time Animation ──
      if (passIndicator && passLabel) {
        passIndicator.style.display = 'flex';
        passIndicator.className = 'dnn-pass-indicator forward';
        passLabel.innerHTML = '<i class="fas fa-arrow-left"></i> إنتشار أمامي عبر الزمن (Forward Through Time)';
      }

      for (let t = 0; t < T; t++) {
        if (rnnTrainer.shouldStop) break;
        drawRNNArchitecture(t);
        await new Promise(r => setTimeout(r, 250));
      }

      if (rnnTrainer.shouldStop) {
        if (passIndicator) passIndicator.style.display = 'none';
        return;
      }

      // ── BPTT Animation ──
      if (passIndicator && passLabel) {
        passIndicator.className = 'dnn-pass-indicator backward';
        passLabel.innerHTML = '<i class="fas fa-arrow-right"></i> إنتشار خلفي عبر الزمن (BPTT)';
      }

      for (let t = T - 1; t >= 0; t--) {
        if (rnnTrainer.shouldStop) break;
        drawRNNArchitecture(t);
        await new Promise(r => setTimeout(r, 250));
      }

      // Hide indicator after animation
      if (passIndicator) {
        passIndicator.style.display = 'none';
      }

      if (!rnnTrainer.shouldStop) {
        // Show final state after animation
        drawRNNArchitecture(-1);
      }
    };

    await rnnTrainer.train(
      // onEpoch callback
      (epoch, loss) => {
        document.getElementById('rnn-epoch-display').innerText = `${epoch}/${rnnBuilder.config.epochs}`;
        document.getElementById('rnn-loss-display').innerText = loss.toFixed(6);
        const progress = (epoch / rnnBuilder.config.epochs) * 100;
        document.getElementById('rnn-train-progress').style.width = `${progress}%`;

        // Update loss chart every few epochs
        if (epoch % Math.max(1, Math.floor(rnnBuilder.config.epochs / 100)) === 0 || epoch === rnnBuilder.config.epochs) {
          RNNTrainer.drawLossChart('rnn-loss-chart', rnnTrainer.history);
        }

        // Update weights table periodically
        if (epoch % 10 === 0 || epoch === 1) {
          RNNTrainer.renderWeightsTable(rnnBuilder.network.weights, 'rnn-current-weights-table');
        }

        // Redraw network with updated states every 5 epochs
        if (epoch % 5 === 0 || epoch === 1) {
          if (rnnBuilder.data && rnnBuilder.data.sequences.length > 0) {
            rnnStates = rnnBuilder.forward(rnnBuilder.data.sequences[0]);
            drawRNNArchitecture(-1);
          }
        }
      },
      // onComplete callback
      (history) => {
        trainBtn.disabled = false;
        stopBtn.disabled = true;
        rnnBuildBtn.disabled = false;

        // Hide pass indicator
        if (passIndicator) {
          passIndicator.style.display = 'none';
        }

        // Final forward pass
        if (rnnBuilder.data && rnnBuilder.data.sequences.length > 0) {
          rnnStates = rnnBuilder.forward(rnnBuilder.data.sequences[0]);
          drawRNNArchitecture(-1);
        }

        // Update weights table
        RNNTrainer.renderWeightsTable(rnnBuilder.network.weights, 'rnn-current-weights-table');

        // Draw loss chart
        RNNTrainer.drawLossChart('rnn-loss-chart', history);

        // Restart animation only if training completed naturally
        if (!rnnTrainer.shouldStop) {
          startDataFlowAnimation();
        }
      }
    );
  });

  stopBtn.addEventListener('click', () => {
    if (rnnTrainer) {
      rnnTrainer.stop();
    }
    trainBtn.disabled = false;
    stopBtn.disabled = true;
    rnnBuildBtn.disabled = false;

    // Hide pass indicator immediately
    const passIndicator = document.getElementById('rnn-pass-indicator');
    if (passIndicator) {
      passIndicator.style.display = 'none';
    }

    // Stop the idle animation if it was running
    if (animTimer) {
      clearInterval(animTimer);
      animTimer = null;
    }

    // Draw static architecture
    drawRNNArchitecture(-1);

    // Draw chart so far
    if (rnnTrainer && rnnTrainer.history.loss.length > 0) {
      RNNTrainer.drawLossChart('rnn-loss-chart', rnnTrainer.history);
    }
  });

  resetBtn.addEventListener('click', () => {
    if (rnnTrainer && rnnTrainer.isTraining) {
      rnnTrainer.stop();
    }
    if (animTimer) {
      clearInterval(animTimer);
      animTimer = null;
    }

    rnnBuilder.network = null;
    rnnBuilder.data = null;
    rnnBuilder.isBuilt = false;
    rnnStates = null;
    rnnTrainer = null;

    // Clear canvas
    const canvas = document.getElementById('rnn-network-canvas');
    if (canvas) {
      const ctx = canvas.getContext('2d');
      ctx.clearRect(0, 0, canvas.width, canvas.height);
    }

    // Reset UI
    document.getElementById('rnn-data-table-container').innerHTML =
      '<p style="padding: 2rem; text-align:center; color: var(--text-muted)">قم ببناء الشبكة لتوليد البيانات...</p>';
    document.getElementById('rnn-epoch-display').innerText = '0/0';
    document.getElementById('rnn-loss-display').innerText = '—';
    document.getElementById('rnn-cell-display').innerText = '—';
    document.getElementById('rnn-opt-display').innerText = '—';
    document.getElementById('rnn-train-progress').style.width = '0%';

    document.getElementById('rnn-initial-weights-table').innerHTML =
      '<tbody><tr><td style="padding: 2rem; color: var(--text-muted);">قم ببناء الشبكة...</td></tr></tbody>';
    document.getElementById('rnn-current-weights-table').innerHTML =
      '<tbody><tr><td style="padding: 2rem; color: var(--text-muted);">قم ببناء الشبكة...</td></tr></tbody>';

    trainBtn.disabled = false;
    stopBtn.disabled = true;
    rnnBuildBtn.disabled = false;
  });

  // ── Handle window resize for canvas ──
  let resizeTimeout;
  window.addEventListener('resize', () => {
    clearTimeout(resizeTimeout);
    resizeTimeout = setTimeout(() => {
      if (rnnBuilder.isBuilt && rnnOverlay.classList.contains('active')) {
        drawRNNArchitecture(animationStep >= 0 ? animationStep : -1);
      }
    }, 200);
  });
}
