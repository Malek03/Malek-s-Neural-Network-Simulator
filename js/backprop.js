/* ============================================
   BackPropagation Visualization
   ============================================ */

class BackPropUI {
  constructor(networkBuilder, networkRenderer) {
    this.builder = networkBuilder;
    this.renderer = networkRenderer;
    this.currentDataRow = -1;
    this.currentStep = 0;
    this.steps = [];
    this.isPlaying = false;
    this.playInterval = null;
    
    // UI Elements
    this.panel = document.getElementById('bp-panel');
    this.stepsContainer = document.getElementById('bp-steps');
    this.progressBar = document.getElementById('bp-progress');
    this.progressText = document.getElementById('bp-progress-text');
  }

  /**
   * Start BackPropagation visualization
   */
  start(rowIndex, dataRow, yTrue) {
    this.currentDataRow = rowIndex;
    document.getElementById('data-table-section').style.display = 'none';
    this.panel.classList.add('active');
    this.stepsContainer.innerHTML = '';
    
    // Run BP and get steps
    const result = this.builder.backpropagate(dataRow, yTrue, 0.1);
    this.steps = result.steps;
    
    this.currentStep = 0;
    this.renderCurrentStep();
    this.updateProgress();
  }

  renderCurrentStep() {
    const step = this.steps[this.currentStep];
    if (!step) return;

    this.stepsContainer.innerHTML = '';
    const stepEl = document.createElement('div');
    stepEl.className = 'step-container';

    let contentHtml = '';

    switch (step.type) {
      case 'loss':
        contentHtml = `
          <div class="step-header">
            <div class="step-number"><span class="num">${this.currentStep + 1}</span> حساب الخطأ (Loss)</div>
            <span class="step-badge loss">Binary Cross Entropy</span>
          </div>
          <div class="step-content">
            <p>نحسب مقدار الخطأ بين التوقع والقيمة الحقيقية</p>
            <div class="formula">
              y (الحقيقي) = <span class="highlight">${step.yTrue}</span><br>
              ŷ (التوقع) = <span class="highlight">${step.yPred}</span><br>
              Loss = -(y·log(ŷ) + (1-y)·log(1-ŷ)) = <span class="result">${step.loss}</span>
            </div>
          </div>
        `;
        this.renderer.clearHighlights();
        break;

      case 'output_delta':
        contentHtml = `
          <div class="step-header">
            <div class="step-number"><span class="num">${this.currentStep + 1}</span> مشتقة طبقة الإخراج</div>
            <span class="step-badge gradient">Output Delta (δ)</span>
          </div>
          <div class="step-content">
            <p>حساب مقدار التغير المطلوب لطبقة الإخراج</p>
            <div class="formula">
              a (المخرج الحالي) = [${step.output.join(', ')}]<br>
              y (الحقيقي) = ${step.yTrue}<br>
              δ = a - y = [${step.delta.map(d => `<span class="result">${d}</span>`).join(', ')}]
            </div>
          </div>
        `;
        // Highlight output layer
        this.renderer.highlightNode(step.layerIndex, 0);
        break;

      case 'hidden_delta':
        contentHtml = `
          <div class="step-header">
            <div class="step-number"><span class="num">${this.currentStep + 1}</span> مشتقة الطبقة المخفية</div>
            <span class="step-badge gradient">Hidden Delta (δ)</span>
          </div>
          <div class="step-content">
            <p>نقل الخطأ للطبقة السابقة (Backpropagate)</p>
            <div class="formula">
              δ_hidden = (δ_next · W^T) ⊙ (a · (1 - a))<br>
              δ = [${step.delta.map(d => `<span class="result">${d}</span>`).join(', ')}]
            </div>
          </div>
        `;
        // Highlight current layer nodes
        this.renderer.clearHighlights();
        step.delta.forEach((_, idx) => {
          this.renderer.highlightNode(step.layerIndex, idx);
        });
        break;

      case 'weight_update':
        contentHtml = `
          <div class="step-header">
            <div class="step-number"><span class="num">${this.currentStep + 1}</span> تحديث الأوزان</div>
            <span class="step-badge update">Update Weights</span>
          </div>
          <div class="step-content">
            <p>تحديث الأوزان والبايس باستخدام Gradient Descent (LR = ${step.learningRate})</p>
            <div class="formula">W_new = W_old - LR × Gradient</div>
            <div class="summary-table-wrapper">
              <table class="summary-table">
                <thead>
                  <tr>
                    <th>الوزن القديم</th>
                    <th>الـ Gradient</th>
                    <th>الوزن الجديد</th>
                  </tr>
                </thead>
                <tbody>
                  ${this.generateWeightUpdateRows(step)}
                </tbody>
              </table>
            </div>
          </div>
        `;
        this.renderer.clearHighlights();
        // Just flash the layer connections
        setTimeout(() => this.renderer.draw(), 500);
        break;
    }

    stepEl.innerHTML = contentHtml;
    this.stepsContainer.appendChild(stepEl);
  }

  generateWeightUpdateRows(step) {
    let rows = '';
    // Limit to max 5 weights for display
    let count = 0;
    const maxRows = 5;

    for (let i = 0; i < step.oldWeights.length && count < maxRows; i++) {
      for (let j = 0; j < step.oldWeights[i].length && count < maxRows; j++) {
        rows += `
          <tr>
            <td>${step.oldWeights[i][j].toFixed(4)}</td>
            <td style="color: var(--error)">${step.weightGradients[i][j].toFixed(4)}</td>
            <td style="color: var(--success); font-weight: bold">${step.newWeights[i][j].toFixed(4)}</td>
          </tr>
        `;
        count++;
      }
    }
    if (count === maxRows && (step.oldWeights.length * step.oldWeights[0].length) > maxRows) {
      rows += `<tr><td colspan="3">... والمزيد</td></tr>`;
    }
    return rows;
  }

  next() {
    if (this.currentStep < this.steps.length - 1) {
      this.currentStep++;
      this.renderCurrentStep();
      this.updateProgress();
    } else {
      this.pause();
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
    }, 1500);
    document.getElementById('bp-play-btn').innerHTML = '<i class="fas fa-pause"></i>';
  }

  pause() {
    this.isPlaying = false;
    clearInterval(this.playInterval);
    document.getElementById('bp-play-btn').innerHTML = '<i class="fas fa-play"></i>';
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
    this.stepsContainer.innerHTML += `
      <div class="step-container" style="border-color: var(--success); box-shadow: 0 0 15px rgba(0, 230, 118, 0.2);">
        <div class="step-header">
          <div class="step-number" style="color: var(--success)"><i class="fas fa-check-circle"></i> اكتملت عملية BackPropagation</div>
        </div>
        <div class="step-content">
          <p>تم تحديث أوزان الشبكة بنجاح لتقليل نسبة الخطأ في المرات القادمة.</p>
        </div>
      </div>
    `;
    
    // Re-render network with new weights
    this.renderer.render(this.builder.network, false);
  }

  close() {
    this.pause();
    this.panel.classList.remove('active');
    document.getElementById('data-table-section').style.display = 'block';
    this.renderer.clearHighlights();
  }
}
