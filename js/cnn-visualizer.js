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

    // Step 0: Input image
    steps.push({
      type: 'input',
      title: 'الخطوة 1: صورة الإدخال (Input Image)',
      subtitle: `مصفوفة ${this.cnnBuilder.config.inputSize}×${this.cnnBuilder.config.inputSize} — القيم: 0 (أسود) و 1 (أبيض)`,
      data: result.input
    });

    // Steps 1..N: Convolution for each filter
    for (let f = 0; f < result.convResults.length; f++) {
      steps.push({
        type: 'convolution',
        title: `الخطوة 2: التلافيف (Convolution) — فلتر ${f + 1}`,
        subtitle: `انزلاق الفلتر ${this.cnnBuilder.config.filterSize}×${this.cnnBuilder.config.filterSize} على الصورة مع حساب حاصل ضرب العناصر وجمعها`,
        data: result.convResults[f],
        filter: this.cnnBuilder.network.filters[f],
        bias: this.cnnBuilder.network.convBiases[f],
        filterIndex: f,
        featureMap: result.convFeatureMaps[f],
        inputImage: result.input
      });
    }

    // ReLU step
    steps.push({
      type: 'relu',
      title: 'الخطوة 3: دالة التنشيط (ReLU)',
      subtitle: 'تحويل القيم السالبة إلى صفر مع الإبقاء على القيم الموجبة — f(x) = max(0, x)',
      data: result.reluResults,
      beforeMaps: result.convFeatureMaps,
      afterMaps: result.reluFeatureMaps
    });

    // MaxPooling step
    steps.push({
      type: 'pooling',
      title: `الخطوة 4: التجميع ${this.cnnBuilder.config.poolType === 'max' ? 'الأقصى (Max Pooling)' : 'المتوسط (Average Pooling)'}`,
      subtitle: `تقسيم كل Feature Map إلى مربعات ${this.cnnBuilder.config.poolSize}×${this.cnnBuilder.config.poolSize} واختيار ${this.cnnBuilder.config.poolType === 'max' ? 'القيمة الأكبر' : 'المتوسط'}`,
      data: result.poolResults,
      inputMaps: result.reluFeatureMaps,
      outputMaps: result.poolFeatureMaps
    });

    // Flatten step
    steps.push({
      type: 'flatten',
      title: 'الخطوة 5: التسطيح (Flatten)',
      subtitle: `تحويل جميع Feature Maps من مصفوفات ثنائية الأبعاد إلى متجه أحادي البعد بطول ${result.flattenResult.vector.length}`,
      data: result.flattenResult,
      inputMaps: result.poolFeatureMaps
    });

    // FC + Softmax step
    steps.push({
      type: 'fc_softmax',
      title: 'الخطوة 6: الطبقة المتصلة بالكامل (FC) + Softmax',
      subtitle: 'حساب الاحتمالات لكل رقم (0-9) باستخدام طبقة كاملة الاتصال ودالة Softmax',
      data: result.fcResult,
      flatVector: result.flattenResult.vector
    });

    // Final prediction
    steps.push({
      type: 'prediction',
      title: 'الخطوة 7: النتيجة النهائية (Prediction)',
      subtitle: 'الرقم المتوقع بناءً على أعلى احتمال من Softmax',
      data: {
        prediction: result.prediction,
        probabilities: result.probabilities
      }
    });

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
      case 'relu':
        this.renderReLUStep(content, step);
        break;
      case 'pooling':
        this.renderPoolingStep(content, step);
        break;
      case 'flatten':
        this.renderFlattenStep(content, step);
        break;
      case 'fc_softmax':
        this.renderFCSoftmaxStep(content, step);
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
        <span>كل خلية تمثل بكسل واحد. القيمة <code>1</code> (أخضر) = بكسل مفعّل، <code>0</code> (داكن) = خلفية فارغة.</span>
      </div>
    `;
  }

  renderConvolutionStep(container, step) {
    const { filter, bias, featureMap, inputImage, data } = step;
    const filterGrid = this.buildGridHTML(filter, 'filter');
    const inputGrid = this.buildGridHTML(inputImage, 'input');
    const featureGrid = this.buildGridHTML(featureMap, 'heatmap');

    // Build convolution detail table
    let stepsDetailHTML = '<div class="cnn-conv-steps-scroll"><table class="cnn-conv-table"><thead><tr>';
    stepsDetailHTML += '<th>الموقع</th><th>منطقة الإدخال</th><th>× الفلتر</th><th>المجموع</th><th>+ الانحياز</th><th>النتيجة</th>';
    stepsDetailHTML += '</tr></thead><tbody>';

    data.steps.forEach((s, idx) => {
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
          <div class="cnn-matrix-block">
            <div class="cnn-matrix-label">الإدخال (Input)</div>
            ${inputGrid}
          </div>
          <div class="cnn-conv-operator">
            <i class="fas fa-asterisk"></i>
            <span>Conv2D</span>
          </div>
          <div class="cnn-matrix-block cnn-filter-block">
            <div class="cnn-matrix-label" style="color: var(--warning)">الفلتر ${step.filterIndex + 1}</div>
            ${filterGrid}
            <div class="cnn-matrix-dims">Bias: ${bias.toFixed(4)}</div>
          </div>
          <div class="cnn-conv-operator">
            <i class="fas fa-equals"></i>
          </div>
          <div class="cnn-matrix-block cnn-feature-block">
            <div class="cnn-matrix-label" style="color: var(--accent)">Feature Map ${step.filterIndex + 1}</div>
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

  renderReLUStep(container, step) {
    let mapsHTML = '';
    for (let f = 0; f < step.data.length; f++) {
      const beforeGrid = this.buildGridHTML(step.beforeMaps[f], 'heatmap');
      const afterGrid = this.buildGridHTML(step.afterMaps[f], 'heatmap');

      mapsHTML += `
        <div class="cnn-relu-pair">
          <div class="cnn-matrix-block">
            <div class="cnn-matrix-label">قبل ReLU (فلتر ${f + 1})</div>
            ${beforeGrid}
          </div>
          <div class="cnn-relu-arrow">
            <i class="fas fa-arrow-left"></i>
            <code>max(0, x)</code>
          </div>
          <div class="cnn-matrix-block cnn-relu-after">
            <div class="cnn-matrix-label" style="color: var(--success)">بعد ReLU (فلتر ${f + 1})</div>
            ${afterGrid}
          </div>
        </div>
      `;
    }

    container.innerHTML = `
      <div class="cnn-relu-layout">
        ${mapsHTML}
      </div>
      <div class="cnn-info-box">
        <i class="fas fa-lightbulb" style="color: var(--warning)"></i>
        <span>دالة <b>ReLU</b> تحوّل جميع القيم السالبة إلى <code>0</code> (الخلايا الحمراء تتحول لداكنة)، وتترك القيم الموجبة كما هي.</span>
      </div>
    `;
  }

  renderPoolingStep(container, step) {
    const poolType = this.cnnBuilder.config.poolType;
    let mapsHTML = '';

    for (let f = 0; f < step.data.length; f++) {
      const inputGrid = this.buildGridHTML(step.inputMaps[f], 'heatmap');
      const outputGrid = this.buildGridHTML(step.outputMaps[f], 'heatmap');

      // Build pool detail
      let poolDetail = '<div class="cnn-pool-detail">';
      step.data[f].steps.forEach((s) => {
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
            <div class="cnn-matrix-label">Feature Map ${f + 1} (بعد ReLU)</div>
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
    for (let f = 0; f < step.inputMaps.length; f++) {
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
    vector.forEach((val, idx) => {
      const map = mapping[idx];
      const color = this.getHeatmapColor(val, -1, 1);
      vectorHTML += `<div class="cnn-flatten-cell" style="background:${color}" title="Filter ${map.filterIdx + 1} [${map.row},${map.col}] = ${val.toFixed(4)}">${val.toFixed(2)}</div>`;
    });
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
      <div class="cnn-info-box">
        <i class="fas fa-info-circle"></i>
        <span>يتم تحويل جميع خرائط الميزات (Feature Maps) إلى متجه واحد لتمريره إلى الطبقة المتصلة بالكامل.</span>
      </div>
    `;
  }

  renderFCSoftmaxStep(container, step) {
    const { probabilities, predictedClass, z } = step.data;
    const labels = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'];

    // Build probability bars
    let barsHTML = '';
    const maxProb = Math.max(...probabilities);
    
    for (let i = 0; i < probabilities.length; i++) {
      const prob = probabilities[i];
      const pct = (prob * 100).toFixed(1);
      const isMax = i === predictedClass;
      const barColor = isMax ? 'var(--success)' : 'var(--primary)';
      const opacity = isMax ? 1 : 0.4 + (prob / maxProb) * 0.6;

      barsHTML += `
        <div class="cnn-prob-row ${isMax ? 'cnn-prob-max' : ''}">
          <div class="cnn-prob-label">${labels[i]}</div>
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
      const isMax = i === predictedClass;
      fcSummaryHTML += `<tr class="${isMax ? 'cnn-row-highlight' : ''}">
        <td>Class ${labels[i]}</td>
        <td class="cnn-mono">${z[i].toFixed(4)}</td>
        <td class="cnn-mono" style="color: ${isMax ? 'var(--success)' : 'var(--text-muted)'}">${probabilities[i].toFixed(6)}</td>
      </tr>`;
    }
    fcSummaryHTML += '</tbody></table></div>';

    container.innerHTML = `
      <div class="cnn-fc-layout">
        <div class="cnn-prob-chart">
          <div class="cnn-prob-chart-title">توزيع الاحتمالات (Softmax Output)</div>
          ${barsHTML}
        </div>
        <div class="cnn-conv-formula">
          <code>P(class_i) = e^(z_i) / Σ e^(z_k)</code>
        </div>
        ${fcSummaryHTML}
      </div>
    `;
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
      <div class="cnn-info-box" style="border-color: var(--success); background: rgba(0,230,118,0.05);">
        <i class="fas fa-check-circle" style="color: var(--success)"></i>
        <span>الشبكة تتوقع أن الصورة المدخلة هي الرقم <b>${prediction}</b> بنسبة ثقة <b>${confidence}%</b>.</span>
      </div>
      <div class="cnn-info-box">
        <i class="fas fa-exclamation-triangle" style="color: var(--warning)"></i>
        <span>ملاحظة: الأوزان عشوائية (غير مدربة)، لذا التوقع لن يكون دقيقاً. الهدف هو فهم <b>آلية العمل</b> وليس دقة التصنيف.</span>
      </div>
    `;
  }

  // ── Grid HTML Builder ──

  buildGridHTML(matrix, type = 'input') {
    if (!matrix || matrix.length === 0) return '<div class="cnn-grid-empty">لا توجد بيانات</div>';

    const rows = matrix.length;
    const cols = matrix[0].length;
    const cellSize = rows <= 5 ? 'large' : (rows <= 8 ? 'medium' : 'small');

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

        const displayVal = type === 'input' ? val.toFixed(0) : val.toFixed(2);
        html += `<div class="cnn-cell" style="background:${bgColor}; color:${textColor}" title="[${i},${j}] = ${val.toFixed(4)}">${displayVal}</div>`;
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

    const d = this.cnnBuilder.network.dims;
    const steps = [
      { icon: 'fa-image', label: 'Input', dims: `${d.inputSize}×${d.inputSize}×1`, color: 'var(--success)' },
      { icon: 'fa-asterisk', label: 'Conv2D', dims: `${d.convOutSize}×${d.convOutSize}×${this.cnnBuilder.config.numFilters}`, color: 'var(--primary-light)' },
      { icon: 'fa-wave-square', label: 'ReLU', dims: `${d.convOutSize}×${d.convOutSize}×${this.cnnBuilder.config.numFilters}`, color: 'var(--warning)' },
      { icon: 'fa-compress-arrows-alt', label: 'MaxPool', dims: `${d.poolOutSize}×${d.poolOutSize}×${this.cnnBuilder.config.numFilters}`, color: 'var(--accent)' },
      { icon: 'fa-arrows-alt-h', label: 'Flatten', dims: `${d.flattenSize}`, color: '#E040FB' },
      { icon: 'fa-project-diagram', label: 'FC+Softmax', dims: `${d.fcNeurons}`, color: 'var(--error)' },
    ];

    let html = '<div class="cnn-pipeline-overview">';
    steps.forEach((s, idx) => {
      html += `
        <div class="cnn-pipeline-node" data-step="${idx}" onclick="window._cnnVisualizer && window._cnnVisualizer.goToStep(${idx})">
          <div class="cnn-pipeline-icon" style="color: ${s.color}; border-color: ${s.color}">
            <i class="fas ${s.icon}"></i>
          </div>
          <div class="cnn-pipeline-label">${s.label}</div>
          <div class="cnn-pipeline-dims">${s.dims}</div>
        </div>
      `;
      if (idx < steps.length - 1) {
        html += '<div class="cnn-pipeline-arrow"><i class="fas fa-chevron-left"></i></div>';
      }
    });
    html += '</div>';

    el.innerHTML = html;
  }
}
