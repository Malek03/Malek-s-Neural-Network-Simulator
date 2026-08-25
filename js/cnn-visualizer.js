/* ============================================
   CNN Visualizer - Interactive Visual Engine
   ============================================ */

class CNNVisualizer {
  constructor(containerId) {
    this.container = document.getElementById(containerId);
    this.cnnBuilder = null;
    this.currentStep = 0;
    this.totalSteps = 0;
    this.pipelineData = null;
    this.isPlaying = false;
    this.playTimer = null;
    this.currentFilterView = 0;

    // Color palette for heatmap
    this.heatmapColors = {
      negative: { r: 255, g: 82, b: 82 },    // red for negative
      zero:     { r: 20, g: 24, b: 50 },      // dark bg for zero
      positive: { r: 0, g: 230, b: 118 },     // green for positive
      active:   { r: 108, g: 99, b: 255 },    // primary purple for active cells
      maxPool:  { r: 0, g: 229, b: 255 },     // accent cyan for max selected
    };
  }

  // ── Initialize with builder ──
  init(cnnBuilder) {
    this.cnnBuilder = cnnBuilder;
    this.currentStep = 0;
  }

  // ── Run Pipeline & Build Visualization ──
  run(inputImage) {
    if (!this.cnnBuilder || !this.cnnBuilder.isBuilt) return;

    const result = this.cnnBuilder.feedforward(inputImage);
    this.pipelineData = result;

    // Build all steps
    this.steps = this.buildSteps(result);
    this.totalSteps = this.steps.length;
    this.currentStep = 0;

    this.renderCurrentStep();
    this.updateStepCounter();
  }

  // ── Build Step Array ──
  buildSteps(result) {
    const steps = [];
    let stepNumber = 1;

    // Step 0: Input image
    steps.push({
      type: 'input',
      title: `الخطوة ${stepNumber++}: صورة الإدخال (Input Image)`,
      subtitle: `مصفوفة ${this.cnnBuilder.config.inputSize}×${this.cnnBuilder.config.inputSize}`,
      data: result.input
    });

    result.pipelineResults.forEach(pr => {
      if (pr.type === 'conv2d') {
        // Convolution for each filter
        for (let f = 0; f < pr.config.filters; f++) {
          steps.push({
            type: 'convolution',
            title: `الخطوة ${stepNumber++}: التلافيف (Conv2D) — فلتر ${f + 1}`,
            subtitle: `انزلاق الفلتر ${pr.config.size}×${pr.config.size} على الإدخال مع حساب حاصل ضرب العناصر وجمعها`,
            data: pr,
            filterIndex: f,
            filter: pr.meta.filters[f],
            filterName: pr.meta.filterNames[f],
            bias: pr.meta.biases[f],
            featureMap: pr.featureMaps[f]
          });
        }
        if (pr.activationType && pr.activationType !== 'none') {
           steps.push({
             type: 'activation',
             title: `الخطوة ${stepNumber++}: دالة التنشيط (${pr.activationType.toUpperCase()})`,
             subtitle: 'تطبيق دالة التنشيط على خرائط الميزات',
             data: pr,
             beforeMaps: pr.featureMaps,
             afterMaps: pr.activatedMaps,
             activationType: pr.activationType
           });
        }
      } else if (pr.type === 'pool2d') {
         steps.push({
           type: 'pooling',
           title: `الخطوة ${stepNumber++}: التجميع (${pr.config.poolType === 'max' ? 'Max Pooling' : 'Average Pooling'})`,
           subtitle: `تقسيم كل Feature Map إلى مربعات واختيار القيمة الممثلة`,
           data: pr,
           inputMaps: pr.inputMaps,
           outputMaps: pr.featureMaps
         });
      } else if (pr.type === 'flatten') {
         steps.push({
           type: 'flatten',
           title: `الخطوة ${stepNumber++}: التسطيح (Flatten)`,
           subtitle: `تحويل الخرائط إلى متجه أحادي البعد بطول ${pr.flatVector.length}`,
           data: pr,
           inputMaps: pr.inputMaps
         });
      } else if (pr.type === 'fc') {
         steps.push({
           type: 'fc',
           title: `الخطوة ${stepNumber++}: الطبقة المتصلة بالكامل (FC)`,
           subtitle: `حساب النتيجة للخلايا المتصلة بالكامل`,
           data: pr,
           flatVector: pr.flatVector
         });
      }
    });

    // Final prediction (find the last FC with softmax)
    const lastFC = result.pipelineResults.slice().reverse().find(pr => pr.type === 'fc' && pr.config.activation === 'softmax');
    if (lastFC) {
      steps.push({
        type: 'prediction',
        title: `الخطوة ${stepNumber++}: النتيجة النهائية (Prediction)`,
        subtitle: 'التصنيف المتوقع بناءً على أعلى احتمال',
        data: lastFC
      });
    }

    return steps;
  }

  // ── Navigation ──
  next() {
    if (this.currentStep < this.totalSteps - 1) {
      this.currentStep++;
      this.renderCurrentStep();
      this.updateStepCounter();
    }
  }

  prev() {
    if (this.currentStep > 0) {
      this.currentStep--;
      this.renderCurrentStep();
      this.updateStepCounter();
    }
  }

  goToStep(idx) {
    if (idx >= 0 && idx < this.totalSteps) {
      this.currentStep = idx;
      this.renderCurrentStep();
      this.updateStepCounter();
    }
  }

  togglePlay() {
    if (this.isPlaying) {
      this.stopPlay();
    } else {
      this.startPlay();
    }
  }

  startPlay() {
    this.isPlaying = true;
    const playBtn = document.getElementById('cnn-play-btn');
    if (playBtn) {
      playBtn.innerHTML = '<i class="fas fa-pause"></i>';
      playBtn.classList.add('playing');
    }
    this.playTimer = setInterval(() => {
      if (this.currentStep < this.totalSteps - 1) {
        this.next();
      } else {
        this.stopPlay();
      }
    }, 2000);
  }

  stopPlay() {
    this.isPlaying = false;
    clearInterval(this.playTimer);
    const playBtn = document.getElementById('cnn-play-btn');
    if (playBtn) {
      playBtn.innerHTML = '<i class="fas fa-play"></i>';
      playBtn.classList.remove('playing');
    }
  }

  updateStepCounter() {
    const counter = document.getElementById('cnn-step-counter');
    if (counter) {
      counter.innerText = `${this.currentStep + 1} / ${this.totalSteps}`;
    }
    // Update prev/next button states
    const prevBtn = document.getElementById('cnn-prev-btn');
    const nextBtn = document.getElementById('cnn-next-btn');
    if (prevBtn) prevBtn.disabled = this.currentStep === 0;
    if (nextBtn) nextBtn.disabled = this.currentStep === this.totalSteps - 1;

    // Update step indicators
    const indicators = document.querySelectorAll('.cnn-step-dot');
    indicators.forEach((dot, idx) => {
      dot.classList.toggle('active', idx === this.currentStep);
      dot.classList.toggle('completed', idx < this.currentStep);
    });
  }

  // ── Render Current Step ──
  renderCurrentStep() {
    if (!this.container || !this.steps || this.currentStep >= this.steps.length) return;

    const step = this.steps[this.currentStep];
    const viewport = document.getElementById('cnn-viewport');
    if (!viewport) return;

    // Clear viewport
    viewport.innerHTML = '';

    // Add step header
    const header = document.createElement('div');
    header.className = 'cnn-step-header';
    header.innerHTML = `
      <h3 class="cnn-step-title">${step.title}</h3>
      <p class="cnn-step-subtitle">${step.subtitle}</p>
    `;
    viewport.appendChild(header);

    // Render based on type
    const content = document.createElement('div');
    content.className = 'cnn-step-content';

    switch (step.type) {
      case 'input':
        this.renderInputStep(content, step);
        break;
      case 'convolution':
        this.renderConvolutionStep(content, step);
        break;
      case 'activation':
        this.renderActivationStep(content, step);
        break;
      case 'pooling':
        this.renderPoolingStep(content, step);
        break;
      case 'flatten':
        this.renderFlattenStep(content, step);
        break;
      case 'fc':
        this.renderFCStep(content, step);
        break;
      case 'prediction':
        this.renderPredictionStep(content, step);
        break;
    }

    viewport.appendChild(content);

    // Animate entry
    viewport.classList.remove('cnn-fade-in');
    void viewport.offsetWidth; // reflow
    viewport.classList.add('cnn-fade-in');
  }

  // ── Step Renderers ──
  renderInputStep(container, step) {
    const gridHtml = this.buildGridHTML(step.data, 'input');
    container.innerHTML = `
      <div class="cnn-visual-center">
        <div class="cnn-matrix-block">
          <div class="cnn-matrix-label">صورة الإدخال (Input)</div>
          ${gridHtml}
          <div class="cnn-matrix-dims">${step.data.length} × ${step.data[0].length}</div>
        </div>
      </div>
      <div class="cnn-info-box">
        <i class="fas fa-info-circle"></i>
        <span>عرض قيم البكسلات في الإدخال</span>
      </div>
    `;
  }

  renderConvolutionStep(container, step) {
    const { filter, bias, featureMap, filterName, data, filterIndex } = step;
    const filterGrid = this.buildGridHTML(filter, 'filter');
    const featureGrid = this.buildGridHTML(featureMap, 'heatmap');

    // Build convolution detail table
    let stepsDetailHTML = '<div class="cnn-conv-steps-scroll"><table class="cnn-conv-table"><thead><tr>';
    stepsDetailHTML += '<th>الموقع</th><th>منطقة الإدخال</th><th>× الفلتر</th><th>المجموع</th><th>+ الانحياز</th><th>النتيجة</th>';
    stepsDetailHTML += '</tr></thead><tbody>';

    const convSteps = data.convSteps.filter(s => s.filterIndex === filterIndex);
    
    // Limits steps if too many
    const showSteps = convSteps.slice(0, 100);

    showSteps.forEach((s, idx) => {
      const regionStr = s.inputRegion.map(r => '[' + r.map(v => v.toFixed(1)).join(', ') + ']').join('<br>');
      const filterStr = filter.map(r => '[' + r.map(v => v.toFixed(2)).join(', ') + ']').join('<br>');
      stepsDetailHTML += `
        <tr class="cnn-conv-step-row" data-step="${idx}">
          <td class="cnn-mono">(${s.outputPos.row}, ${s.outputPos.col})</td>
          <td class="cnn-mini-matrix">${regionStr}</td>
          <td class="cnn-mini-matrix">${filterStr}</td>
          <td class="cnn-mono cnn-highlight-val">${s.sumBeforeBias}</td>
          <td class="cnn-mono" style="color: var(--warning)">${s.bias}</td>
          <td class="cnn-mono cnn-result-val">${s.result}</td>
        </tr>
      `;
    });
    stepsDetailHTML += '</tbody></table></div>';

    container.innerHTML = `
      <div class="cnn-conv-layout">
        <div class="cnn-conv-visual">
          <div class="cnn-conv-operator">
            <i class="fas fa-asterisk"></i>
            <span>Conv2D</span>
          </div>
          <div class="cnn-matrix-block cnn-filter-block">
            <div class="cnn-matrix-label" style="color: var(--warning)">الفلتر ${filterIndex + 1}</div>
            ${filterGrid}
            <div class="cnn-matrix-dims">Bias: ${bias.toFixed(4)}</div>
            <div style="font-size: 0.75rem; color: var(--warning); margin-top: 5px; font-weight: 700;">${filterName}</div>
          </div>
          <div class="cnn-conv-operator">
            <i class="fas fa-equals"></i>
          </div>
          <div class="cnn-matrix-block cnn-feature-block">
            <div class="cnn-matrix-label" style="color: var(--accent)">Feature Map ${filterIndex + 1}</div>
            ${featureGrid}
            <div class="cnn-matrix-dims">${featureMap.length} × ${featureMap[0].length}</div>
          </div>
        </div>
        <div class="cnn-conv-formula">
          <code>Output[i,j] = Σ(Input_region × Filter) + Bias</code>
        </div>
        ${stepsDetailHTML}
      </div>
    `;
  }

  renderActivationStep(container, step) {
    let mapsHTML = '';
    const MAX_MAPS = 4;
    const toShow = Math.min(step.data.config.filters, MAX_MAPS);
    
    for (let f = 0; f < toShow; f++) {
      const beforeGrid = this.buildGridHTML(step.beforeMaps[f], 'heatmap');
      const afterGrid = this.buildGridHTML(step.afterMaps[f], 'heatmap');

      mapsHTML += `
        <div class="cnn-relu-pair">
          <div class="cnn-matrix-block">
            <div class="cnn-matrix-label">قبل (فلتر ${f + 1})</div>
            ${beforeGrid}
          </div>
          <div class="cnn-relu-arrow">
            <i class="fas fa-arrow-left"></i>
            <code>${step.activationType}</code>
          </div>
          <div class="cnn-matrix-block cnn-relu-after">
            <div class="cnn-matrix-label" style="color: var(--success)">بعد (فلتر ${f + 1})</div>
            ${afterGrid}
          </div>
        </div>
      `;
    }

    container.innerHTML = `
      <div class="cnn-relu-layout">
        ${mapsHTML}
      </div>
    `;
  }

  renderPoolingStep(container, step) {
    const poolType = step.data.config.poolType;
    let mapsHTML = '';
    const MAX_MAPS = 4;
    const toShow = Math.min(step.inputMaps.length, MAX_MAPS);

    for (let f = 0; f < toShow; f++) {
      const inputGrid = this.buildGridHTML(step.inputMaps[f], 'heatmap');
      const outputGrid = this.buildGridHTML(step.outputMaps[f], 'heatmap');

      // Build pool detail
      let poolDetail = '<div class="cnn-pool-detail">';
      const fSteps = step.data.poolSteps.filter(s => s.filterIndex === f).slice(0,10);
      fSteps.forEach((s) => {
        const regionVals = s.allValues.map(v => v.val.toFixed(2));
        const resultVal = poolType === 'max' ? s.maxVal : s.avgVal;
        poolDetail += `
          <div class="cnn-pool-region-card">
            <div class="cnn-pool-region-vals">[${regionVals.join(', ')}]</div>
            <div class="cnn-pool-region-arrow">→</div>
            <div class="cnn-pool-region-result">${resultVal.toFixed(4)}</div>
          </div>
        `;
      });
      poolDetail += '</div>';

      mapsHTML += `
        <div class="cnn-pool-pair">
          <div class="cnn-matrix-block">
            <div class="cnn-matrix-label">Feature Map ${f + 1}</div>
            ${inputGrid}
          </div>
          <div class="cnn-relu-arrow">
            <i class="fas fa-compress-arrows-alt"></i>
            <code>${poolType === 'max' ? 'Max' : 'Avg'}</code>
          </div>
          <div class="cnn-matrix-block">
            <div class="cnn-matrix-label" style="color: var(--accent)">بعد ${poolType === 'max' ? 'MaxPool' : 'AvgPool'}</div>
            ${outputGrid}
          </div>
        </div>
        ${poolDetail}
      `;
    }

    container.innerHTML = `
      <div class="cnn-pool-layout">
        ${mapsHTML}
      </div>
    `;
  }

  renderFlattenStep(container, step) {
    const { vector, mapping } = step.data;
    
    // Visual: show maps being unfolded
    let mapsHTML = '';
    const MAX_MAPS = 4;
    const toShow = Math.min(step.inputMaps.length, MAX_MAPS);
    
    for (let f = 0; f < toShow; f++) {
      const grid = this.buildGridHTML(step.inputMaps[f], 'heatmap');
      mapsHTML += `
        <div class="cnn-matrix-block cnn-flatten-source">
          <div class="cnn-matrix-label">Map ${f + 1}</div>
          ${grid}
        </div>
      `;
    }

    // Build vector display
    let vectorHTML = '<div class="cnn-flatten-vector">';
    const showVec = vector.slice(0, 500); // limit to 500 for perf
    showVec.forEach((val, idx) => {
      const color = this.getHeatmapColor(val, -1, 1);
      vectorHTML += `<div class="cnn-flatten-cell" style="background:${color}" title="Index ${idx} = ${val.toFixed(4)}">${val.toFixed(2)}</div>`;
    });
    if(vector.length > 500) {
        vectorHTML += '<div style="padding:10px; color:#fff;">...</div>';
    }
    vectorHTML += '</div>';

    container.innerHTML = `
      <div class="cnn-flatten-layout">
        <div class="cnn-flatten-maps">
          ${mapsHTML}
        </div>
        <div class="cnn-flatten-arrow-big">
          <i class="fas fa-arrow-down"></i>
          <span>Flatten</span>
        </div>
        ${vectorHTML}
        <div class="cnn-matrix-dims" style="text-align:center; margin-top: 0.5rem;">
          المتجه الناتج: ${vector.length} قيمة
        </div>
      </div>
    `;
  }

  renderFCStep(container, step) {
    const { z, probabilities, prediction, details, config } = step.data;
    const isSoftmax = config.activation === 'softmax';

    let html = '';
    
    if (isSoftmax) {
        const labels = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'];

        // Build probability bars
        let barsHTML = '';
        const maxProb = Math.max(...probabilities);
        
        for (let i = 0; i < probabilities.length; i++) {
          const prob = probabilities[i];
          const pct = (prob * 100).toFixed(1);
          const isMax = i === prediction;
          const barColor = isMax ? 'var(--success)' : 'var(--primary)';
          const opacity = isMax ? 1 : 0.4 + (prob / maxProb) * 0.6;

          barsHTML += `
            <div class="cnn-prob-row ${isMax ? 'cnn-prob-max' : ''}">
              <div class="cnn-prob-label">${labels[i] || i}</div>
              <div class="cnn-prob-bar-bg">
                <div class="cnn-prob-bar-fill" style="width: ${pct}%; background: ${barColor}; opacity: ${opacity}"></div>
              </div>
              <div class="cnn-prob-value">${pct}%</div>
            </div>
          `;
        }
        
        // Show FC computation summary
        let fcSummaryHTML = '<div class="cnn-fc-summary"><table class="cnn-conv-table"><thead><tr>';
        fcSummaryHTML += '<th>الخلية</th><th>القيمة الخطية (Z)</th><th>Softmax (P)</th>';
        fcSummaryHTML += '</tr></thead><tbody>';
        for (let i = 0; i < z.length; i++) {
          const isMax = i === prediction;
          fcSummaryHTML += `<tr class="${isMax ? 'cnn-row-highlight' : ''}">
            <td>Class ${labels[i] || i}</td>
            <td class="cnn-mono">${z[i].toFixed(4)}</td>
            <td class="cnn-mono" style="color: ${isMax ? 'var(--success)' : 'var(--text-muted)'}">${probabilities[i].toFixed(6)}</td>
          </tr>`;
        }
        fcSummaryHTML += '</tbody></table></div>';
        
        html = `
          <div class="cnn-fc-layout">
            <div class="cnn-prob-chart">
              <div class="cnn-prob-chart-title">توزيع الاحتمالات (Softmax Output)</div>
              ${barsHTML}
            </div>
            ${fcSummaryHTML}
          </div>
        `;
    } else {
        // Just standard FC
        let fcSummaryHTML = '<div class="cnn-fc-summary"><table class="cnn-conv-table"><thead><tr>';
        fcSummaryHTML += '<th>Neuron</th><th>Linear (Z)</th><th>Activation</th>';
        fcSummaryHTML += '</tr></thead><tbody>';
        
        const activatedZ = step.data.activatedZ;
        const toShow = Math.min(z.length, 20); // max 20 neurons to show
        for (let i = 0; i < toShow; i++) {
          fcSummaryHTML += `<tr>
            <td>Neuron ${i}</td>
            <td class="cnn-mono">${z[i].toFixed(4)}</td>
            <td class="cnn-mono" style="color: var(--success)">${activatedZ[i].toFixed(4)}</td>
          </tr>`;
        }
        fcSummaryHTML += '</tbody></table></div>';
        
        html = `
          <div class="cnn-fc-layout">
             <p>Fully Connected Layer with ${config.neurons} neurons. Activation: ${config.activation}</p>
             ${fcSummaryHTML}
          </div>
        `;
    }

    container.innerHTML = html;
  }

  renderPredictionStep(container, step) {
    const { prediction, probabilities } = step.data;
    const confidence = (probabilities[prediction] * 100).toFixed(1);

    // Build mini probability bars
    let miniBarsHTML = '';
    for (let i = 0; i < probabilities.length; i++) {
      const pct = (probabilities[i] * 100).toFixed(1);
      const isMax = i === prediction;
      miniBarsHTML += `
        <div class="cnn-mini-prob ${isMax ? 'active' : ''}">
          <span class="cnn-mini-prob-label">${i}</span>
          <div class="cnn-mini-prob-bar" style="height: ${Math.max(4, probabilities[i] * 100)}%; background: ${isMax ? 'var(--success)' : 'var(--primary)'}"></div>
          <span class="cnn-mini-prob-val">${pct}%</span>
        </div>
      `;
    }

    container.innerHTML = `
      <div class="cnn-prediction-layout">
        <div class="cnn-prediction-big">
          <div class="cnn-prediction-digit">${prediction}</div>
          <div class="cnn-prediction-label">الرقم المتوقع</div>
          <div class="cnn-prediction-confidence">
            <span class="cnn-confidence-value">${confidence}%</span>
            <span class="cnn-confidence-label">نسبة الثقة</span>
          </div>
        </div>
        <div class="cnn-prediction-chart">
          ${miniBarsHTML}
        </div>
      </div>
    `;
  }

  // ── Grid HTML Builder ──
  buildGridHTML(matrix, type = 'input') {
    if (!matrix || matrix.length === 0) return '<div class="cnn-grid-empty">لا توجد بيانات</div>';

    const rows = matrix.length;
    const cols = matrix[0].length;
    const cellSize = rows <= 5 ? 'large' : (rows <= 10 ? 'medium' : 'small');

    let html = `<div class="cnn-grid cnn-grid-${cellSize}" style="grid-template-columns: repeat(${cols}, 1fr);">`;

    // Find min/max for heatmap
    let minVal = Infinity, maxVal = -Infinity;
    for (let i = 0; i < rows; i++) {
      for (let j = 0; j < cols; j++) {
        if (matrix[i][j] < minVal) minVal = matrix[i][j];
        if (matrix[i][j] > maxVal) maxVal = matrix[i][j];
      }
    }

    for (let i = 0; i < rows; i++) {
      for (let j = 0; j < cols; j++) {
        const val = matrix[i][j];
        let bgColor, textColor;

        if (type === 'input') {
          bgColor = val > 0.5 ? 'rgba(0, 230, 118, 0.8)' : 'rgba(20, 24, 50, 0.8)';
          textColor = val > 0.5 ? '#0A0E27' : '#6B6F8D';
        } else if (type === 'filter') {
          bgColor = this.getHeatmapColor(val, -1, 1);
          textColor = Math.abs(val) > 0.5 ? '#fff' : '#A0A3BD';
        } else {
          bgColor = this.getHeatmapColor(val, minVal, maxVal);
          textColor = '#fff';
        }

        const displayVal = type === 'input' ? val.toFixed(1) : val.toFixed(2);
        
        // Dont render values if grid is very large
        const content = rows > 14 ? '' : displayVal;
        
        html += `<div class="cnn-cell" style="background:${bgColor}; color:${textColor}" title="[${i},${j}] = ${val.toFixed(4)}">${content}</div>`;
      }
    }

    html += '</div>';
    return html;
  }

  // ── Heatmap Color ──
  getHeatmapColor(val, minVal, maxVal) {
    const range = maxVal - minVal || 1;
    const norm = (val - minVal) / range; // 0..1

    if (val < 0) {
      const t = Math.min(1, Math.abs(val) / (Math.abs(minVal) || 1));
      return `rgba(255, 82, 82, ${0.2 + t * 0.7})`;
    } else if (val === 0) {
      return 'rgba(20, 24, 50, 0.8)';
    } else {
      const t = Math.min(1, val / (maxVal || 1));
      return `rgba(0, 230, 118, ${0.15 + t * 0.75})`;
    }
  }

  // ── Render Full Pipeline Overview ──
  renderPipelineOverview(containerId) {
    const el = document.getElementById(containerId);
    if (!el || !this.pipelineData) return;

    let html = '<div class="cnn-pipeline-overview">';
    
    // Add input node
    html += `
      <div class="cnn-pipeline-node">
        <div class="cnn-pipeline-icon" style="color: var(--success); border-color: var(--success)"><i class="fas fa-image"></i></div>
        <div class="cnn-pipeline-label">Input</div>
        <div class="cnn-pipeline-dims">${this.cnnBuilder.config.inputSize}×${this.cnnBuilder.config.inputSize}</div>
      </div>
      <div class="cnn-pipeline-arrow"><i class="fas fa-chevron-left"></i></div>
    `;

    this.pipelineData.pipelineResults.forEach((pr, idx) => {
        let icon, color, label, dims;
        if (pr.type === 'conv2d') {
            icon = 'fa-asterisk'; color = 'var(--primary-light)'; label = 'Conv2D'; dims = `${pr.meta.outShape.size}×${pr.meta.outShape.size}×${pr.meta.outShape.channels}`;
        } else if (pr.type === 'pool2d') {
            icon = 'fa-compress-arrows-alt'; color = 'var(--accent)'; label = 'Pool'; dims = `${pr.meta.outShape.size}×${pr.meta.outShape.size}×${pr.meta.outShape.channels}`;
        } else if (pr.type === 'flatten') {
            icon = 'fa-arrows-alt-h'; color = '#E040FB'; label = 'Flatten'; dims = `${pr.meta.outShape.length}`;
        } else if (pr.type === 'fc') {
            icon = 'fa-project-diagram'; color = 'var(--error)'; label = 'FC'; dims = `${pr.meta.outShape.length}`;
        }

        html += `
          <div class="cnn-pipeline-node">
            <div class="cnn-pipeline-icon" style="color: ${color}; border-color: ${color}"><i class="fas ${icon}"></i></div>
            <div class="cnn-pipeline-label">${label}</div>
            <div class="cnn-pipeline-dims">${dims}</div>
          </div>
        `;
        if (idx < this.pipelineData.pipelineResults.length - 1) {
          html += '<div class="cnn-pipeline-arrow"><i class="fas fa-chevron-left"></i></div>';
        }
    });

    html += '</div>';
    el.innerHTML = html;
  }
}
