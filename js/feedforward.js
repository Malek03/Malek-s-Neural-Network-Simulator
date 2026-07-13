/* ============================================
   FeedForward Visualization
   ============================================ */

class FeedForwardUI {
  constructor(networkBuilder, networkRenderer) {
    this.builder = networkBuilder;
    this.renderer = networkRenderer;
    this.currentDataRow = -1;
    this.currentStep = 0;
    this.steps = [];
    this.isPlaying = false;
    this.playInterval = null;
    
    // UI Elements
    this.panel = document.getElementById('ff-panel');
    this.stepsContainer = document.getElementById('ff-steps');
    this.progressBar = document.getElementById('ff-progress');
    this.progressText = document.getElementById('ff-progress-text');
    this.matrixDisplay = document.getElementById('matrix-display');
  }

  /**
   * Start Node-by-Node visualization
   */
  startNodeByNode(rowIndex, dataRow) {
    this.mode = 'node';
    this.currentDataRow = rowIndex;
    this.inputData = dataRow;
    document.getElementById('data-table-section').style.display = 'none';
    this.panel.classList.add('active');
    this.stepsContainer.innerHTML = '';
    this.matrixDisplay.innerHTML = '';
    
    // Generate steps
    this.generateNodeSteps();
    this.currentStep = 0;
    this.renderCurrentStep();
    this.updateProgress();
  }

  /**
   * Start Matrix visualization
   */
  startMatrix(rowIndex, dataRow) {
    this.mode = 'matrix';
    this.currentDataRow = rowIndex;
    this.inputData = dataRow;
    document.getElementById('data-table-section').style.display = 'none';
    this.panel.classList.add('active');
    this.stepsContainer.innerHTML = '';
    this.matrixDisplay.innerHTML = '';
    
    this.steps = this.builder.feedforwardMatrix(dataRow).steps;
    this.currentStep = 0;
    this.renderCurrentMatrixStep();
    this.updateProgress();
  }

  generateNodeSteps() {
    const { network } = this.builder;
    const { weights, biases, layers } = network;
    this.steps = [];
    
    let currentInput = this.inputData;
    
    for (let l = 0; l < weights.length; l++) {
      const W = weights[l];
      const b = biases[l];
      const nextInput = [];
      
      for (let j = 0; j < W[0].length; j++) {
        // Step 1: Weights & Inputs
        const nodeWeights = W.map(row => row[j]);
        this.steps.push({
          type: 'weights',
          layer: l,
          nodeIdx: j,
          inputs: [...currentInput],
          weights: nodeWeights,
          bias: b[j],
          desc: `تجهيز المدخلات والأوزان للعقدة ${j + 1} في ${network.layerNames[l + 1]}`
        });

        // Step 2: Linear Equation
        let sum = 0;
        let equation = 'z = ';
        for (let i = 0; i < currentInput.length; i++) {
          sum += currentInput[i] * nodeWeights[i];
          equation += `(${currentInput[i].toFixed(4)} × ${nodeWeights[i].toFixed(4)}) + `;
        }
        sum += b[j];
        equation += `${b[j].toFixed(4)} = <span class="result">${sum.toFixed(4)}</span>`;
        
        this.steps.push({
          type: 'linear',
          layer: l,
          nodeIdx: j,
          z: sum,
          equation,
          desc: `حساب المعادلة الخطية (مجموع الجداءات + البايس)`
        });

        // Step 3: Activation
        const a = MathUtils.sigmoid(sum);
        nextInput.push(a);
        
        this.steps.push({
          type: 'activation',
          layer: l,
          nodeIdx: j,
          z: sum,
          a: a,
          equation: `σ(${sum.toFixed(4)}) = <span class="result">${a.toFixed(4)}</span>`,
          desc: `تطبيق دالة التنشيط (Sigmoid)`
        });
        
        // Step 4: Output
        this.steps.push({
          type: 'output',
          layer: l,
          nodeIdx: j,
          value: a,
          desc: `المخرج النهائي للعقدة يرسل للطبقة التالية`
        });
      }
      currentInput = nextInput;
    }
  }

  renderCurrentStep() {
    if (this.mode === 'matrix') {
      this.renderCurrentMatrixStep();
      return;
    }

    const step = this.steps[this.currentStep];
    if (!step) return;

    this.stepsContainer.innerHTML = '';
    const stepEl = document.createElement('div');
    stepEl.className = 'step-container';

    let contentHtml = '';

    switch (step.type) {
      case 'weights':
        contentHtml = `
          <div class="step-header">
            <div class="step-number"><span class="num">${this.currentStep + 1}</span> تجهيز الأوزان</div>
            <span class="step-badge weights">Weights</span>
          </div>
          <div class="step-content">
            <p>${step.desc}</p>
            <div class="formula">
              Inputs (X): [${step.inputs.map(x => `<span class="highlight">${x.toFixed(4)}</span>`).join(', ')}]<br>
              Weights (W): [${step.weights.map(w => `<span class="highlight">${w.toFixed(4)}</span>`).join(', ')}]<br>
              Bias (b): <span class="highlight">${step.bias.toFixed(4)}</span>
            </div>
          </div>
        `;
        this.renderer.highlightNodeConnections(step.layer + 1, step.nodeIdx);
        break;

      case 'linear':
        contentHtml = `
          <div class="step-header">
            <div class="step-number"><span class="num">${this.currentStep + 1}</span> المعادلة الخطية</div>
            <span class="step-badge linear">Z = X·W + b</span>
          </div>
          <div class="step-content">
            <p>${step.desc}</p>
            <div class="formula">${step.equation}</div>
          </div>
        `;
        this.renderer.highlightNode(step.layer + 1, step.nodeIdx);
        break;

      case 'activation':
        contentHtml = `
          <div class="step-header">
            <div class="step-number"><span class="num">${this.currentStep + 1}</span> دالة التنشيط</div>
            <span class="step-badge activation">Sigmoid</span>
          </div>
          <div class="step-content">
            <p>${step.desc}</p>
            <div class="formula">${step.equation}</div>
            <div class="explanation">تقوم الدالة بضغط القيمة ${step.z.toFixed(4)} لتصبح بين 0 و 1 (النتيجة: ${step.a.toFixed(4)})</div>
          </div>
        `;
        break;

      case 'output':
        contentHtml = `
          <div class="step-header">
            <div class="step-number"><span class="num">${this.currentStep + 1}</span> المخرج</div>
            <span class="step-badge output">Output</span>
          </div>
          <div class="step-content">
            <p>${step.desc}</p>
            <div class="formula">a = <span class="result">${step.value.toFixed(4)}</span></div>
          </div>
        `;
        // Update the value on the node in canvas
        let nodeValues = new Array(this.builder.network.layers[step.layer + 1]).fill(null);
        nodeValues[step.nodeIdx] = step.value;
        this.renderer.setLayerValues(step.layer + 1, nodeValues);
        this.renderer.clearHighlights();
        break;
    }

    stepEl.innerHTML = contentHtml;
    this.stepsContainer.appendChild(stepEl);
  }

  renderCurrentMatrixStep() {
    const step = this.steps[this.currentStep];
    if (!step) return;

    this.stepsContainer.innerHTML = '';
    
    // Header
    const stepEl = document.createElement('div');
    stepEl.className = 'step-container';
    stepEl.innerHTML = `
      <div class="step-header">
        <div class="step-number"><span class="num">${this.currentStep + 1}</span> حساب الطبقة كاملة بالمصفوفات</div>
        <span class="step-badge linear">Matrix Ops</span>
      </div>
      <div class="step-content">
        <p>الطبقة: ${this.builder.network.layerNames[step.layerIndex + 1]}</p>
        <div class="formula">A = σ(X · W + B)</div>
      </div>
    `;
    this.stepsContainer.appendChild(stepEl);

    // Matrix Display
    let matrixHtml = `<div class="matrix-container">`;
    
    // X
    matrixHtml += `<div class="matrix-wrapper">
      <div class="matrix-label">X (Inputs)</div>
      <div class="matrix">
        ${this.buildMatrixHtml(step.X)}
      </div>
    </div>`;

    matrixHtml += `<div class="matrix-op">×</div>`;

    // W
    matrixHtml += `<div class="matrix-wrapper">
      <div class="matrix-label">W (Weights)</div>
      <div class="matrix">
        ${this.buildMatrixHtml(step.W)}
      </div>
    </div>`;

    matrixHtml += `<div class="matrix-op">+</div>`;

    // B
    matrixHtml += `<div class="matrix-wrapper">
      <div class="matrix-label">B (Biases)</div>
      <div class="matrix">
        ${this.buildMatrixHtml(step.B)}
      </div>
    </div>`;

    matrixHtml += `<div class="matrix-op">=</div>`;

    // Z
    matrixHtml += `<div class="matrix-wrapper">
      <div class="matrix-label">Z (Linear)</div>
      <div class="matrix">
        ${this.buildMatrixHtml(step.Z)}
      </div>
    </div>`;

    matrixHtml += `</div>`; // end container
    
    // Activation Container
    matrixHtml += `<div class="matrix-container mt-2">`;
    matrixHtml += `<div class="matrix-op">σ(Z) = </div>`;
    
    // A
    matrixHtml += `<div class="matrix-wrapper">
      <div class="matrix-label">A (Activation / Output)</div>
      <div class="matrix">
        ${this.buildMatrixHtml(step.A, true)}
      </div>
    </div>`;
    matrixHtml += `</div>`;

    this.matrixDisplay.innerHTML = matrixHtml;
    
    // Update canvas with all values for this layer
    this.renderer.setLayerValues(step.layerIndex + 1, step.A[0]);
  }

  buildMatrixHtml(matrix, isResult = false) {
    let html = '';
    matrix.forEach(row => {
      html += `<div class="matrix-row">`;
      row.forEach(val => {
        html += `<div class="matrix-cell ${isResult ? 'result' : ''}">${val.toFixed(3)}</div>`;
      });
      html += `</div>`;
    });
    return html;
  }

  next() {
    if (this.currentStep < this.steps.length - 1) {
      this.currentStep++;
      this.renderCurrentStep();
      this.updateProgress();
    } else {
      this.pause();
      // Completed FF
      this.showCompletion();
    }
  }

  prev() {
    if (this.currentStep > 0) {
      this.currentStep--;
      this.renderCurrentStep();
      this.updateProgress();
    }
  }

  play() {
    if (this.currentStep >= this.steps.length - 1) {
      this.currentStep = 0;
    }
    this.isPlaying = true;
    this.playInterval = setInterval(() => {
      if (this.currentStep < this.steps.length - 1) {
        this.next();
      } else {
        this.pause();
      }
    }, this.mode === 'matrix' ? 2000 : 1500);
    document.getElementById('ff-play-btn').innerHTML = '<i class="fas fa-pause"></i>';
  }

  pause() {
    this.isPlaying = false;
    clearInterval(this.playInterval);
    document.getElementById('ff-play-btn').innerHTML = '<i class="fas fa-play"></i>';
  }

  togglePlay() {
    if (this.isPlaying) this.pause();
    else this.play();
  }

  updateProgress() {
    const percent = ((this.currentStep + 1) / this.steps.length) * 100;
    this.progressBar.style.width = `${percent}%`;
    this.progressText.innerText = `${this.currentStep + 1} / ${this.steps.length}`;
  }
  
  showCompletion() {
    const finalVal = this.mode === 'matrix' 
      ? this.steps[this.steps.length - 1].A[0][0] 
      : this.steps[this.steps.length - 1].value;
      
    this.stepsContainer.innerHTML += `
      <div class="step-container" style="border-color: var(--success); box-shadow: 0 0 15px rgba(0, 230, 118, 0.2);">
        <div class="step-header">
          <div class="step-number" style="color: var(--success)"><i class="fas fa-check-circle"></i> اكتملت عملية FeedForward</div>
        </div>
        <div class="step-content">
          <div class="formula">التوقع النهائي (ŷ) = <span class="result" style="font-size: 1.2rem">${finalVal.toFixed(4)}</span></div>
        </div>
      </div>
    `;
    
    // Update main data table if needed
    const dataRows = document.querySelectorAll('#dataTable tbody tr');
    if (dataRows[this.currentDataRow]) {
      dataRows[this.currentDataRow].classList.remove('active-row');
    }
  }

  close() {
    this.pause();
    this.panel.classList.remove('active');
    document.getElementById('data-table-section').style.display = 'block';
    this.renderer.clearHighlights();
  }
}
