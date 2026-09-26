/* ============================================
   GAN Simulator - Interactive Step-by-Step
   Generative Adversarial Networks
   ============================================ */

function initGANSimulator() {
  const card = document.getElementById('gan-card');
  const overlay = document.getElementById('gan-simulator-overlay');
  const closeBtn = document.getElementById('gan-close-sim');
  
  if (!card || !overlay) return;

  card.addEventListener('click', () => {
    overlay.classList.add('active');
    document.body.style.overflow = 'hidden';
    GANSim.init();
  });

  closeBtn.addEventListener('click', () => {
    overlay.classList.remove('active');
    document.body.style.overflow = '';
    GANSim.stopTraining();
  });

  // Tab switching
  overlay.querySelectorAll('.gan-nav-tab').forEach(tab => {
    tab.addEventListener('click', () => {
      const tabId = tab.dataset.tab;
      overlay.querySelectorAll('.gan-nav-tab').forEach(t => t.classList.remove('active'));
      overlay.querySelectorAll('.gan-tab-content').forEach(c => c.classList.remove('active'));
      tab.classList.add('active');
      document.getElementById(tabId + '-tab').classList.add('active');
      GANSim.activateTab(tabId);
    });
  });
}

/* ============================================
   GANSim - Main Controller
   ============================================ */
const GANSim = {
  initialized: false,
  currentTab: 'gan-overview',
  isTraining: false,
  trainInterval: null,
  
  init() {
    if (this.initialized) {
      this.activateTab(this.currentTab);
      return;
    }
    this.initialized = true;
    GANOverview.init();
  },

  activateTab(tabId) {
    this.currentTab = tabId;
    switch(tabId) {
      case 'gan-overview': GANOverview.init(); break;
      case 'gan-generator': GANGenerator.init(); break;
      case 'gan-discriminator': GANDiscriminator.init(); break;
      case 'gan-loss': GANLossView.init(); break;
      case 'gan-training': GANTraining.init(); break;
      case 'gan-applications': GANApplications.init(); break;
    }
  },

  stopTraining() {
    this.isTraining = false;
    if (this.trainInterval) {
      clearInterval(this.trainInterval);
      this.trainInterval = null;
    }
  }
};

/* ============================================
   Helper Utilities
   ============================================ */
const GANUtils = {
  colors: {
    generator: '#4ECDC4',
    discriminator: '#FF6584',
    noise: '#FFD93D',
    real: '#00E676',
    fake: '#FF5252',
    accent: '#6C63FF',
    bg: '#0a0e27',
    border: 'rgba(255,255,255,0.08)'
  },

  // Generate random noise vector
  randomNoise(size) {
    const noise = [];
    for (let i = 0; i < size; i++) {
      noise.push(Math.random() * 2 - 1);
    }
    return noise;
  },

  // Simple sigmoid
  sigmoid(x) {
    return 1 / (1 + Math.exp(-x));
  },

  // LeakyReLU
  leakyReLU(x) {
    return x > 0 ? x : 0.01 * x;
  },

  // Tanh
  tanh(x) {
    return Math.tanh(x);
  },

  // Generate a simple 8x8 pattern (digit-like)
  generatePattern(type) {
    const grid = Array(8).fill(null).map(() => Array(8).fill(0));
    switch(type) {
      case 0: // Circle
        for (let y = 0; y < 8; y++) {
          for (let x = 0; x < 8; x++) {
            const dx = x - 3.5, dy = y - 3.5;
            if (Math.sqrt(dx*dx + dy*dy) >= 2 && Math.sqrt(dx*dx + dy*dy) <= 3.2) {
              grid[y][x] = 0.9 + Math.random() * 0.1;
            }
          }
        }
        break;
      case 1: // Cross
        for (let i = 1; i < 7; i++) {
          grid[i][3] = 0.9 + Math.random() * 0.1;
          grid[i][4] = 0.9 + Math.random() * 0.1;
          grid[3][i] = 0.9 + Math.random() * 0.1;
          grid[4][i] = 0.9 + Math.random() * 0.1;
        }
        break;
      case 2: // Square
        for (let i = 1; i < 7; i++) {
          grid[1][i] = 0.9 + Math.random() * 0.1;
          grid[6][i] = 0.9 + Math.random() * 0.1;
          grid[i][1] = 0.9 + Math.random() * 0.1;
          grid[i][6] = 0.9 + Math.random() * 0.1;
        }
        break;
      case 3: // Triangle
        for (let y = 0; y < 7; y++) {
          const left = 4 - Math.floor(y * 3 / 6);
          const right = 4 + Math.floor(y * 3 / 6);
          grid[y + 1][left] = 0.9 + Math.random() * 0.1;
          grid[y + 1][right] = 0.9 + Math.random() * 0.1;
          if (y === 6) {
            for (let x = left; x <= right; x++) {
              grid[y + 1][x] = 0.9 + Math.random() * 0.1;
            }
          }
        }
        break;
    }
    return grid;
  },

  // Generate fake image from noise (simulated generator output)
  generateFakeImage(noise, quality) {
    const grid = Array(8).fill(null).map(() => Array(8).fill(0));
    const patternIdx = Math.abs(Math.floor(noise[0] * 4)) % 4;
    const realPattern = this.generatePattern(patternIdx);
    
    for (let y = 0; y < 8; y++) {
      for (let x = 0; x < 8; x++) {
        const noiseVal = noise[(y * 8 + x) % noise.length] * (1 - quality);
        const signalVal = realPattern[y][x] * quality;
        grid[y][x] = Math.max(0, Math.min(1, signalVal + noiseVal * 0.3 + Math.random() * (1 - quality) * 0.2));
      }
    }
    return grid;
  },

  // Draw 8x8 grid on canvas
  drawGrid(canvas, grid, borderColor) {
    const ctx = canvas.getContext('2d');
    const cellW = canvas.width / 8;
    const cellH = canvas.height / 8;
    
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    
    for (let y = 0; y < 8; y++) {
      for (let x = 0; x < 8; x++) {
        const v = Math.floor(grid[y][x] * 255);
        ctx.fillStyle = `rgb(${v}, ${v}, ${v})`;
        ctx.fillRect(x * cellW, y * cellH, cellW - 1, cellH - 1);
      }
    }
    
    if (borderColor) {
      ctx.strokeStyle = borderColor;
      ctx.lineWidth = 2;
      ctx.strokeRect(0, 0, canvas.width, canvas.height);
    }
  },

  // Draw loss chart
  drawLossChart(canvas, genLosses, discLosses) {
    const ctx = canvas.getContext('2d');
    const w = canvas.width;
    const h = canvas.height;
    const pad = { top: 30, right: 30, bottom: 40, left: 50 };
    
    ctx.clearRect(0, 0, w, h);
    ctx.fillStyle = 'rgba(0,0,0,0.3)';
    ctx.fillRect(0, 0, w, h);
    
    if (genLosses.length < 2) return;
    
    const allLosses = [...genLosses, ...discLosses];
    const maxL = Math.max(...allLosses, 1);
    const minL = Math.min(...allLosses, 0);
    const range = maxL - minL || 1;
    
    const plotW = w - pad.left - pad.right;
    const plotH = h - pad.top - pad.bottom;
    
    // Grid
    ctx.strokeStyle = 'rgba(255,255,255,0.05)';
    ctx.lineWidth = 1;
    for (let i = 0; i <= 5; i++) {
      const y = pad.top + (plotH / 5) * i;
      ctx.beginPath();
      ctx.moveTo(pad.left, y);
      ctx.lineTo(w - pad.right, y);
      ctx.stroke();
      
      const val = (maxL - (range / 5) * i).toFixed(2);
      ctx.fillStyle = '#666';
      ctx.font = '10px JetBrains Mono, monospace';
      ctx.textAlign = 'right';
      ctx.fillText(val, pad.left - 5, y + 4);
    }
    
    // Draw line function
    function drawLine(data, color) {
      ctx.strokeStyle = color;
      ctx.lineWidth = 2;
      ctx.beginPath();
      
      const step = plotW / (data.length - 1);
      data.forEach((val, i) => {
        const x = pad.left + i * step;
        const y = pad.top + plotH - ((val - minL) / range) * plotH;
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      });
      ctx.stroke();
    }
    
    drawLine(genLosses, '#4ECDC4');
    drawLine(discLosses, '#FF6584');
    
    // Labels
    ctx.fillStyle = '#666';
    ctx.font = '11px Cairo, sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText('Epoch', w / 2, h - 5);
    
    ctx.save();
    ctx.translate(12, h / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.fillText('Loss', 0, 0);
    ctx.restore();
  }
};


/* ============================================
   1. Overview Tab
   ============================================ */
const GANOverview = {
  initialized: false,

  init() {
    if (this.initialized) return;
    this.initialized = true;
    
    // Make arch blocks clickable to switch tabs
    document.querySelectorAll('#gan-overview-tab .gan-arch-block[data-tab]').forEach(block => {
      block.addEventListener('click', () => {
        const tabId = block.dataset.tab;
        const overlay = document.getElementById('gan-simulator-overlay');
        
        overlay.querySelectorAll('.gan-nav-tab').forEach(t => {
          t.classList.toggle('active', t.dataset.tab === tabId);
        });
        overlay.querySelectorAll('.gan-tab-content').forEach(c => c.classList.remove('active'));
        document.getElementById(tabId + '-tab').classList.add('active');
        GANSim.activateTab(tabId);
      });
    });

    // Animate data flow
    this.startFlowAnimation();
  },

  startFlowAnimation() {
    const blocks = document.querySelectorAll('#gan-overview-tab .gan-arch-block');
    let idx = 0;
    
    const animate = () => {
      blocks.forEach(b => b.style.boxShadow = '');
      if (idx < blocks.length) {
        const b = blocks[idx];
        const color = b.classList.contains('gan-arch-gen-layer') || b.classList.contains('gan-arch-gen-output')
          ? 'rgba(78, 205, 196, 0.3)' 
          : b.classList.contains('gan-arch-disc-layer') || b.classList.contains('gan-arch-disc-output')
          ? 'rgba(255, 101, 132, 0.3)' 
          : 'rgba(255, 214, 0, 0.3)';
        b.style.boxShadow = `0 0 20px ${color}`;
      }
      idx = (idx + 1) % (blocks.length + 2);
    };
    
    setInterval(animate, 800);
  }
};


/* ============================================
   2. Generator Tab
   ============================================ */
const GANGenerator = {
  initialized: false,
  currentStep: 0,
  totalSteps: 5,

  steps: [
    // Step 1: What is a Generator?
    () => `
      <div class="gan-info-card generator-card">
        <h3><i class="fas fa-magic"></i> ما هو المولد (Generator)؟</h3>
        <p>
          المولد هو شبكة عصبية مهمتها <b>توليد بيانات جديدة تبدو حقيقية</b>. يأخذ متجه عشوائي (Noise Vector) كمدخل ويحوّله إلى صورة أو بيانات تشبه البيانات الحقيقية.
          <br><br>
          فكّر فيه كـ<b>فنان مزيّف</b> يحاول رسم لوحات تبدو كأنها من فنان حقيقي!
        </p>
      </div>
      <div class="gan-highlight-box tip">
        <b>💡 الهدف:</b> خداع المميز (Discriminator) ليعتقد أن المخرجات حقيقية.
      </div>
      <div style="text-align:center; padding: 2rem; color: rgba(255,255,255,0.2);">
        <i class="fas fa-magic" style="font-size: 4rem; color: rgba(78, 205, 196, 0.3);"></i>
        <p style="margin-top: 1rem; font-size: 0.9rem;">G(z) → صورة مزيفة</p>
      </div>
    `,

    // Step 2: Noise Vector Input
    () => {
      return `
        <div class="gan-info-card generator-card">
          <h3><i class="fas fa-random"></i> الخطوة 1: متجه الضوضاء (Noise Vector - z)</h3>
          <p>
            يبدأ المولد بأخذ متجه من أرقام عشوائية. هذا المتجه يُسمى <b>متجه الضوضاء (Latent Vector)</b> ويتم أخذه عادة من توزيع طبيعي (Gaussian Distribution).
          </p>
        </div>
        <div class="gan-formula">
          <div class="formula-label">توليد متجه الضوضاء</div>
          z ~ N(0, 1) &nbsp;&nbsp; → &nbsp;&nbsp; z = [z₁, z₂, ..., z_n]
        </div>
        <div id="gan-gen-noise-display"></div>
        <div style="text-align:center; margin-top: 1rem;">
          <button class="btn btn-secondary btn-sm" id="gan-gen-noise-btn">
            <i class="fas fa-dice"></i> توليد ضوضاء جديدة
          </button>
        </div>
      `;
    },

    // Step 3: Generator Architecture
    () => `
      <div class="gan-info-card generator-card">
        <h3><i class="fas fa-layer-group"></i> الخطوة 2: بنية المولد</h3>
        <p>
          المولد هو شبكة عصبية عميقة (عادة <b>Deconvolutional</b>) تحتوي على طبقات متعددة. كل طبقة تزيد أبعاد البيانات تدريجياً حتى تصل للحجم المطلوب.
        </p>
      </div>
      <div class="gan-flow">
        <div class="gan-flow-step active">
          <div class="step-num">1</div>
          <div><b>Noise Vector (z)</b><br><small style="color:#888;">حجم: 100 × 1</small></div>
        </div>
        <div class="gan-flow-arrow-down"><i class="fas fa-arrow-down"></i></div>
        <div class="gan-flow-step">
          <div class="step-num">2</div>
          <div><b>Dense Layer + Reshape</b><br><small style="color:#888;">حجم: 4 × 4 × 256</small></div>
        </div>
        <div class="gan-flow-arrow-down"><i class="fas fa-arrow-down"></i></div>
        <div class="gan-flow-step">
          <div class="step-num">3</div>
          <div><b>TransConv2D + BatchNorm + ReLU</b><br><small style="color:#888;">حجم: 8 × 8 × 128</small></div>
        </div>
        <div class="gan-flow-arrow-down"><i class="fas fa-arrow-down"></i></div>
        <div class="gan-flow-step">
          <div class="step-num">4</div>
          <div><b>TransConv2D + BatchNorm + ReLU</b><br><small style="color:#888;">حجم: 16 × 16 × 64</small></div>
        </div>
        <div class="gan-flow-arrow-down"><i class="fas fa-arrow-down"></i></div>
        <div class="gan-flow-step" style="border-color: #4ECDC4;">
          <div class="step-num" style="background: rgba(78,205,196,0.2); color: #4ECDC4;">5</div>
          <div><b>TransConv2D + Tanh</b><br><small style="color:#888;">حجم: 28 × 28 × 1 (صورة مولدة)</small></div>
        </div>
      </div>
      <div class="gan-highlight-box info">
        <b>ℹ️</b> دالة التنشيط <b>Tanh</b> في الطبقة الأخيرة تضمن أن قيم البكسل تقع بين -1 و 1.
      </div>
    `,

    // Step 4: Generator Output Visualization
    () => {
      return `
        <div class="gan-info-card generator-card">
          <h3><i class="fas fa-image"></i> الخطوة 3: مخرجات المولد</h3>
          <p>
            شاهد كيف يتحسن إخراج المولد مع مرور وقت التدريب. في البداية، ينتج ضوضاء عشوائية ثم يتعلم تدريجياً توليد أنماط ذات معنى.
          </p>
        </div>
        <div class="gan-slider-group">
          <div class="gan-slider-label">
            <span>جودة التوليد (مستوى التدريب)</span>
            <span class="value" id="gan-gen-quality-val">0%</span>
          </div>
          <input type="range" class="gan-slider" id="gan-gen-quality" min="0" max="100" value="0">
        </div>
        <div style="display: flex; gap: 1rem; justify-content: center; flex-wrap: wrap; margin: 1rem 0;">
          <div id="gan-gen-output-grid"></div>
        </div>
        <div class="gan-highlight-box warning">
          <b>⚠️</b> حرّك المنزلق لمشاهدة تطور جودة الصور المولدة من الضوضاء إلى أنماط واضحة.
        </div>
      `;
    },

    // Step 5: Generator Loss
    () => `
      <div class="gan-info-card generator-card">
        <h3><i class="fas fa-chart-line"></i> الخطوة 4: دالة خسارة المولد</h3>
        <p>
          المولد يريد تعظيم احتمالية أن يصنف المميز مخرجاته على أنها "حقيقية". يتم ذلك عبر تقليل دالة الخسارة التالية:
        </p>
      </div>
      <div class="gan-formula">
        <div class="formula-label">دالة خسارة المولد (Generator Loss)</div>
        L_G = -log(D(G(z)))
      </div>
      <div class="gan-two-col">
        <div class="gan-info-card" style="border-color: rgba(0,230,118,0.2);">
          <h3 style="color: #00E676;"><i class="fas fa-check-circle"></i> عندما ينجح المولد</h3>
          <p>
            إذا صنّف المميز الصورة المزيفة كحقيقية:<br>
            <code style="color:#4ECDC4;">D(G(z)) ≈ 1</code> → <code style="color:#00E676;">L_G ≈ 0</code> (خسارة قليلة)
          </p>
        </div>
        <div class="gan-info-card" style="border-color: rgba(255,82,82,0.2);">
          <h3 style="color: #FF5252;"><i class="fas fa-times-circle"></i> عندما يفشل المولد</h3>
          <p>
            إذا كشف المميز أن الصورة مزيفة:<br>
            <code style="color:#FF6584;">D(G(z)) ≈ 0</code> → <code style="color:#FF5252;">L_G → ∞</code> (خسارة كبيرة)
          </p>
        </div>
      </div>
    `
  ],

  init() {
    this.render();
  },

  render() {
    const content = document.getElementById('gan-generator-content');
    const stepIndicator = document.getElementById('gan-gen-step');
    
    content.innerHTML = typeof this.steps[this.currentStep] === 'function' 
      ? this.steps[this.currentStep]() 
      : this.steps[this.currentStep];
    stepIndicator.textContent = `${this.currentStep + 1}/${this.totalSteps}`;
    
    // Post-render setup
    this.postRender();
  },

  postRender() {
    const step = this.currentStep;
    
    if (step === 1) {
      // Noise visualization
      this.renderNoiseVector();
      const btn = document.getElementById('gan-gen-noise-btn');
      if (btn) btn.addEventListener('click', () => this.renderNoiseVector());
    }
    
    if (step === 3) {
      // Quality slider
      const slider = document.getElementById('gan-gen-quality');
      const valDisplay = document.getElementById('gan-gen-quality-val');
      
      this.renderGeneratorOutput(0);
      
      if (slider) {
        slider.addEventListener('input', () => {
          const q = parseInt(slider.value);
          valDisplay.textContent = q + '%';
          this.renderGeneratorOutput(q / 100);
        });
      }
    }

    // Bind navigation
    document.getElementById('gan-gen-prev').onclick = () => {
      if (this.currentStep > 0) { this.currentStep--; this.render(); }
    };
    document.getElementById('gan-gen-next').onclick = () => {
      if (this.currentStep < this.totalSteps - 1) { this.currentStep++; this.render(); }
    };
    document.getElementById('gan-gen-reset').onclick = () => {
      this.currentStep = 0; this.render();
    };
  },

  renderNoiseVector() {
    const container = document.getElementById('gan-gen-noise-display');
    if (!container) return;
    
    const noise = GANUtils.randomNoise(100);
    let html = '<div class="gan-noise-grid">';
    for (let i = 0; i < 100; i++) {
      const val = (noise[i] + 1) / 2;
      const r = Math.floor(255 * val);
      const g = Math.floor(214 * val);
      const b = Math.floor(61 * val);
      html += `<div class="gan-noise-cell" style="background: rgb(${r},${g},${b});" title="z[${i}] = ${noise[i].toFixed(3)}"></div>`;
    }
    html += '</div>';
    html += `<p style="text-align:center; color:#888; font-size: 0.8rem; margin-top: 0.5rem; font-family: monospace;">100 قيمة عشوائية من التوزيع الطبيعي N(0,1)</p>`;
    container.innerHTML = html;
  },

  renderGeneratorOutput(quality) {
    const container = document.getElementById('gan-gen-output-grid');
    if (!container) return;
    
    let html = '<div class="gan-gallery" style="grid-template-columns: repeat(4, 1fr); max-width: 350px; margin: 0 auto;">';
    for (let i = 0; i < 8; i++) {
      const noise = GANUtils.randomNoise(64);
      noise[0] = (i % 4) / 2 - 0.5; // Bias toward different patterns
      const grid = GANUtils.generateFakeImage(noise, quality);
      const canvasId = `gan-gen-out-${i}`;
      html += `<div class="gan-gallery-item"><canvas id="${canvasId}" width="64" height="64"></canvas></div>`;
    }
    html += '</div>';
    container.innerHTML = html;
    
    // Render each
    for (let i = 0; i < 8; i++) {
      const noise = GANUtils.randomNoise(64);
      noise[0] = (i % 4) / 2 - 0.5;
      const grid = GANUtils.generateFakeImage(noise, quality);
      const canvas = document.getElementById(`gan-gen-out-${i}`);
      if (canvas) GANUtils.drawGrid(canvas, grid, GANUtils.colors.generator);
    }
  }
};


/* ============================================
   3. Discriminator Tab
   ============================================ */
const GANDiscriminator = {
  initialized: false,
  currentStep: 0,
  totalSteps: 5,

  steps: [
    // Step 1: What is a Discriminator?
    () => `
      <div class="gan-info-card discriminator-card">
        <h3><i class="fas fa-search"></i> ما هو المميز (Discriminator)؟</h3>
        <p>
          المميز هو شبكة عصبية مهمتها <b>التمييز بين البيانات الحقيقية والمزيفة</b>. يأخذ صورة كمدخل ويخرج احتمالاً بين 0 و 1 يشير إلى مدى "حقيقية" هذه الصورة.
          <br><br>
          فكّر فيه كـ<b>محقق فني</b> يحاول كشف اللوحات المزيفة!
        </p>
      </div>
      <div class="gan-highlight-box tip">
        <b>💡 الهدف:</b> التعرف بدقة على الصور الحقيقية (إخراج 1) والمزيفة (إخراج 0).
      </div>
      <div style="text-align:center; padding: 2rem; color: rgba(255,255,255,0.2);">
        <i class="fas fa-search" style="font-size: 4rem; color: rgba(255, 101, 132, 0.3);"></i>
        <p style="margin-top: 1rem; font-size: 0.9rem;">D(x) → احتمال [0, 1]</p>
      </div>
    `,

    // Step 2: Discriminator Architecture
    () => `
      <div class="gan-info-card discriminator-card">
        <h3><i class="fas fa-layer-group"></i> بنية المميز</h3>
        <p>
          المميز هو شبكة <b>التفافية (CNN)</b> تقلل أبعاد الصورة تدريجياً حتى تصل لقرار واحد: حقيقي أم مزيف.
        </p>
      </div>
      <div class="gan-flow">
        <div class="gan-flow-step" style="border-color: #FF8A5C;">
          <div class="step-num" style="background: rgba(255,138,92,0.2); color: #FF8A5C;">1</div>
          <div><b>الصورة المدخلة (Real / Fake)</b><br><small style="color:#888;">حجم: 28 × 28 × 1</small></div>
        </div>
        <div class="gan-flow-arrow-down"><i class="fas fa-arrow-down"></i></div>
        <div class="gan-flow-step">
          <div class="step-num">2</div>
          <div><b>Conv2D + LeakyReLU</b><br><small style="color:#888;">حجم: 14 × 14 × 64</small></div>
        </div>
        <div class="gan-flow-arrow-down"><i class="fas fa-arrow-down"></i></div>
        <div class="gan-flow-step">
          <div class="step-num">3</div>
          <div><b>Conv2D + BatchNorm + LeakyReLU</b><br><small style="color:#888;">حجم: 7 × 7 × 128</small></div>
        </div>
        <div class="gan-flow-arrow-down"><i class="fas fa-arrow-down"></i></div>
        <div class="gan-flow-step">
          <div class="step-num">4</div>
          <div><b>Flatten + Dense</b><br><small style="color:#888;">حجم: 1024</small></div>
        </div>
        <div class="gan-flow-arrow-down"><i class="fas fa-arrow-down"></i></div>
        <div class="gan-flow-step" style="border-color: #FF6584;">
          <div class="step-num" style="background: rgba(255,101,132,0.2); color: #FF6584;">5</div>
          <div><b>Dense + Sigmoid</b><br><small style="color:#888;">حجم: 1 (احتمال: حقيقي/مزيف)</small></div>
        </div>
      </div>
      <div class="gan-highlight-box info">
        <b>ℹ️</b> يستخدم <b>LeakyReLU</b> بدلاً من ReLU لتجنب "موت" الخلايا العصبية أثناء التدريب.
      </div>
    `,

    // Step 3: Real vs Fake Classification
    () => {
      return `
        <div class="gan-info-card discriminator-card">
          <h3><i class="fas fa-balance-scale"></i> التمييز بين الحقيقي والمزيف</h3>
          <p>
            يستقبل المميز صوراً حقيقية وأخرى مولدة ويحاول تصنيف كل منها. اختبر المميز بنفسك!
          </p>
        </div>
        <div id="gan-disc-test-area"></div>
        <div style="text-align:center; margin: 1rem 0;">
          <button class="btn btn-secondary btn-sm" id="gan-disc-test-btn">
            <i class="fas fa-random"></i> اختبار جديد
          </button>
        </div>
      `;
    },

    // Step 4: Discriminator Loss
    () => `
      <div class="gan-info-card discriminator-card">
        <h3><i class="fas fa-chart-line"></i> دالة خسارة المميز</h3>
        <p>
          المميز يريد <b>تعظيم</b> قدرته على التصنيف الصحيح. دالة الخسارة تجمع بين خطأ تصنيف الصور الحقيقية وخطأ تصنيف الصور المزيفة.
        </p>
      </div>
      <div class="gan-formula">
        <div class="formula-label">دالة خسارة المميز (Discriminator Loss)</div>
        L_D = -[log(D(x)) + log(1 - D(G(z)))]
      </div>
      <div class="gan-two-col">
        <div class="gan-info-card" style="border-color: rgba(0,230,118,0.2);">
          <h3 style="color: #00E676;"><i class="fas fa-check"></i> للبيانات الحقيقية</h3>
          <p>
            المميز يريد <code style="color:#4ECDC4;">D(x) → 1</code><br>
            فيقلل: <code style="color:#00E676;">-log(D(x))</code>
          </p>
        </div>
        <div class="gan-info-card" style="border-color: rgba(255,82,82,0.2);">
          <h3 style="color: #FF5252;"><i class="fas fa-times"></i> للبيانات المزيفة</h3>
          <p>
            المميز يريد <code style="color:#FF6584;">D(G(z)) → 0</code><br>
            فيقلل: <code style="color:#FF5252;">-log(1 - D(G(z)))</code>
          </p>
        </div>
      </div>
    `,

    // Step 5: Discriminator vs Generator Game
    () => `
      <div class="gan-info-card">
        <h3 style="color: #FFD93D;"><i class="fas fa-gamepad"></i> اللعبة التنافسية (Adversarial Game)</h3>
        <p>
          التدريب في GAN هو لعبة <b>Minimax</b> بين المولد والمميز. كل طرف يحاول التغلب على الآخر:
        </p>
      </div>
      <div class="gan-formula">
        <div class="formula-label">دالة القيمة (Value Function) - Minimax Game</div>
        min_G max_D V(D,G) = E[log(D(x))] + E[log(1 - D(G(z)))]
      </div>
      <div class="gan-flow">
        <div class="gan-flow-step" style="border-color: #4ECDC4;">
          <div class="step-num" style="background: rgba(78,205,196,0.2); color: #4ECDC4;">G</div>
          <div>المولد يتحسن في <b>خداع</b> المميز → الصور المولدة تصبح أكثر واقعية</div>
        </div>
        <div class="gan-flow-arrow-down"><i class="fas fa-arrows-alt-v" style="color: #FFD93D;"></i></div>
        <div class="gan-flow-step" style="border-color: #FF6584;">
          <div class="step-num" style="background: rgba(255,101,132,0.2); color: #FF6584;">D</div>
          <div>المميز يتحسن في <b>كشف</b> التزييف → يصبح أكثر دقة في التمييز</div>
        </div>
        <div class="gan-flow-arrow-down"><i class="fas fa-arrows-alt-v" style="color: #FFD93D;"></i></div>
        <div class="gan-flow-step" style="border-color: #FFD93D;">
          <div class="step-num" style="background: rgba(255,214,0,0.2); color: #FFD93D;">⚖</div>
          <div><b>نقطة التوازن (Nash Equilibrium):</b> المولد ينتج صوراً لا يمكن تمييزها → D(G(z)) ≈ 0.5</div>
        </div>
      </div>
    `
  ],

  init() {
    this.render();
  },

  render() {
    const content = document.getElementById('gan-discriminator-content');
    const stepIndicator = document.getElementById('gan-disc-step');
    
    content.innerHTML = typeof this.steps[this.currentStep] === 'function'
      ? this.steps[this.currentStep]()
      : this.steps[this.currentStep];
    stepIndicator.textContent = `${this.currentStep + 1}/${this.totalSteps}`;
    
    this.postRender();
  },

  postRender() {
    if (this.currentStep === 2) {
      this.renderDiscriminatorTest();
      const btn = document.getElementById('gan-disc-test-btn');
      if (btn) btn.addEventListener('click', () => this.renderDiscriminatorTest());
    }

    document.getElementById('gan-disc-prev').onclick = () => {
      if (this.currentStep > 0) { this.currentStep--; this.render(); }
    };
    document.getElementById('gan-disc-next').onclick = () => {
      if (this.currentStep < this.totalSteps - 1) { this.currentStep++; this.render(); }
    };
    document.getElementById('gan-disc-reset').onclick = () => {
      this.currentStep = 0; this.render();
    };
  },

  renderDiscriminatorTest() {
    const area = document.getElementById('gan-disc-test-area');
    if (!area) return;
    
    // Generate 6 images: mix of real and fake
    const images = [];
    for (let i = 0; i < 6; i++) {
      const isReal = Math.random() > 0.5;
      if (isReal) {
        images.push({ grid: GANUtils.generatePattern(Math.floor(Math.random() * 4)), isReal: true });
      } else {
        const noise = GANUtils.randomNoise(64);
        const quality = 0.3 + Math.random() * 0.5;
        images.push({ grid: GANUtils.generateFakeImage(noise, quality), isReal: false });
      }
    }
    
    let html = '<div class="gan-gallery" style="grid-template-columns: repeat(3, 1fr); max-width: 400px; margin: 0 auto;">';
    images.forEach((img, i) => {
      const score = img.isReal ? (0.75 + Math.random() * 0.2) : (0.05 + Math.random() * 0.3);
      const label = img.isReal ? 'حقيقي' : 'مزيف';
      const labelColor = img.isReal ? '#00E676' : '#FF5252';
      html += `
        <div style="text-align:center;">
          <div class="gan-gallery-item" style="border-color: ${labelColor};">
            <canvas id="gan-disc-img-${i}" width="64" height="64"></canvas>
          </div>
          <div style="font-size: 0.75rem; margin-top: 4px;">
            <span style="color: ${labelColor};">${label}</span><br>
            <span style="color: #888; font-family: monospace;">D(x)=${score.toFixed(2)}</span>
          </div>
        </div>
      `;
    });
    html += '</div>';
    area.innerHTML = html;
    
    // Draw images
    images.forEach((img, i) => {
      const canvas = document.getElementById(`gan-disc-img-${i}`);
      if (canvas) {
        const borderColor = img.isReal ? GANUtils.colors.real : GANUtils.colors.fake;
        GANUtils.drawGrid(canvas, img.grid, borderColor);
      }
    });
  }
};


/* ============================================
   4. Loss Functions Tab
   ============================================ */
const GANLossView = {
  initialized: false,
  currentStep: 0,
  totalSteps: 4,

  steps: [
    // Step 1: Overview of GAN Losses
    () => `
      <div class="gan-info-card">
        <h3 style="color: #FFD93D;"><i class="fas fa-calculator"></i> دوال الخسارة في GAN</h3>
        <p>
          في GAN، نستخدم <b>Binary Cross-Entropy (BCE)</b> كدالة خسارة أساسية. ولكن كل شبكة لها هدف مختلف:
        </p>
      </div>
      <div class="gan-formula">
        <div class="formula-label">Binary Cross-Entropy Loss</div>
        BCE(y, ŷ) = -[y·log(ŷ) + (1-y)·log(1-ŷ)]
      </div>
      <div class="gan-two-col">
        <div class="gan-info-card generator-card">
          <h3><i class="fas fa-magic"></i> المولد (G)</h3>
          <p>يريد <b>تقليل</b> الخسارة:<br><code style="color:#4ECDC4;">L_G = BCE(1, D(G(z)))</code></p>
        </div>
        <div class="gan-info-card discriminator-card">
          <h3><i class="fas fa-search"></i> المميز (D)</h3>
          <p>يريد <b>تقليل</b> الخسارة:<br><code style="color:#FF6584;">L_D = BCE(1,D(x)) + BCE(0,D(G(z)))</code></p>
        </div>
      </div>
    `,

    // Step 2: Interactive BCE visualization
    () => {
      return `
        <div class="gan-info-card">
          <h3 style="color: #FFD93D;"><i class="fas fa-chart-area"></i> تصوّر BCE بصرياً</h3>
          <p>
            حرّك المنزلق لرؤية كيف تتغير قيمة الخسارة مع تغير مخرج المميز. لاحظ كيف تزداد الخسارة بشكل حاد عند الخطأ.
          </p>
        </div>
        <div class="gan-canvas-container" style="height: 250px;">
          <canvas id="gan-bce-canvas" width="600" height="250"></canvas>
        </div>
        <div class="gan-slider-group">
          <div class="gan-slider-label">
            <span>مخرج المميز D(x)</span>
            <span class="value" id="gan-bce-val">0.50</span>
          </div>
          <input type="range" class="gan-slider" id="gan-bce-slider" min="1" max="99" value="50">
        </div>
        <div class="gan-two-col">
          <div class="gan-stat-card">
            <div class="gan-stat-label">خسارة (y=1) "حقيقي"</div>
            <div class="gan-stat-value gen" id="gan-bce-real">0.693</div>
          </div>
          <div class="gan-stat-card">
            <div class="gan-stat-label">خسارة (y=0) "مزيف"</div>
            <div class="gan-stat-value disc" id="gan-bce-fake">0.693</div>
          </div>
        </div>
      `;
    },

    // Step 3: Training Dynamics
    () => `
      <div class="gan-info-card">
        <h3 style="color: #FFD93D;"><i class="fas fa-sync-alt"></i> ديناميكيات التدريب</h3>
        <p>
          تدريب GAN يتبع نمطاً تبادلياً: نحدّث المميز أولاً ثم نحدّث المولد. هذا التبادل هو سر نجاح GAN.
        </p>
      </div>
      <div class="gan-flow">
        <div class="gan-flow-step" style="border-color: #FF6584;">
          <div class="step-num" style="background: rgba(255,101,132,0.2); color: #FF6584;">1</div>
          <div>
            <b>تحديث المميز (D):</b><br>
            <small style="color:#888;">أعط المميز بيانات حقيقية (y=1) ومزيفة (y=0)، واحسب الخسارة ثم حدّث أوزانه.</small>
          </div>
        </div>
        <div class="gan-flow-arrow-down"><i class="fas fa-arrow-down"></i></div>
        <div class="gan-flow-step" style="border-color: #4ECDC4;">
          <div class="step-num" style="background: rgba(78,205,196,0.2); color: #4ECDC4;">2</div>
          <div>
            <b>تحديث المولد (G):</b><br>
            <small style="color:#888;">ولّد صوراً مزيفة، مرّرها عبر المميز، واحسب الخسارة ثم حدّث أوزان المولد فقط.</small>
          </div>
        </div>
        <div class="gan-flow-arrow-down"><i class="fas fa-arrow-down"></i></div>
        <div class="gan-flow-step" style="border-color: #FFD93D;">
          <div class="step-num" style="background: rgba(255,214,0,0.2); color: #FFD93D;">3</div>
          <div>
            <b>تكرار:</b><br>
            <small style="color:#888;">كرر الخطوتين السابقتين حتى يصل النظام لنقطة توازن أو عدد Epochs محدد.</small>
          </div>
        </div>
      </div>
      <div class="gan-highlight-box warning">
        <b>⚠️ مشكلة شائعة:</b> إذا أصبح المميز أقوى بكثير من المولد مبكراً، قد يعاني المولد من <b>تلاشي الانحدار (Vanishing Gradients)</b> ولن يتعلم شيئاً.
      </div>
    `,

    // Step 4: Common Problems
    () => `
      <div class="gan-info-card">
        <h3 style="color: #FFD93D;"><i class="fas fa-exclamation-triangle"></i> مشاكل تدريب GAN الشائعة</h3>
        <p>تدريب GAN ليس سهلاً! إليك أبرز المشاكل وحلولها:</p>
      </div>
      <div style="display: flex; flex-direction: column; gap: 1rem;">
        <div class="gan-info-card" style="border-color: rgba(255,82,82,0.2);">
          <h3 style="color: #FF5252;"><i class="fas fa-clone"></i> 1. انهيار النمط (Mode Collapse)</h3>
          <p>
            المولد يولّد نفس الصورة باستمرار بدلاً من تنوع المخرجات.<br>
            <b>الحل:</b> استخدام Minibatch Discrimination أو تقنيات Wasserstein GAN.
          </p>
        </div>
        <div class="gan-info-card" style="border-color: rgba(255,214,0,0.2);">
          <h3 style="color: #FFD93D;"><i class="fas fa-balance-scale-left"></i> 2. عدم التوازن (Imbalanced Training)</h3>
          <p>
            أحد الشبكتين تتفوق بشكل كبير على الأخرى.<br>
            <b>الحل:</b> استخدام Two-Timescale Update Rule (TTUR) أو تقليل معدل التعلم.
          </p>
        </div>
        <div class="gan-info-card" style="border-color: rgba(108,99,255,0.2);">
          <h3 style="color: #6C63FF;"><i class="fas fa-ghost"></i> 3. تلاشي الانحدار (Vanishing Gradients)</h3>
          <p>
            عندما يكون المميز مثالياً تماماً، لا يحصل المولد على تدرجات مفيدة.<br>
            <b>الحل:</b> استخدام Wasserstein Loss بدلاً من BCE أو تدريب المميز أقل.
          </p>
        </div>
      </div>
    `
  ],

  init() {
    this.render();
  },

  render() {
    const content = document.getElementById('gan-loss-content');
    const stepIndicator = document.getElementById('gan-loss-step');
    
    content.innerHTML = typeof this.steps[this.currentStep] === 'function'
      ? this.steps[this.currentStep]()
      : this.steps[this.currentStep];
    stepIndicator.textContent = `${this.currentStep + 1}/${this.totalSteps}`;
    
    this.postRender();
  },

  postRender() {
    if (this.currentStep === 1) {
      this.drawBCEChart(0.5);
      const slider = document.getElementById('gan-bce-slider');
      if (slider) {
        slider.addEventListener('input', () => {
          const val = parseInt(slider.value) / 100;
          document.getElementById('gan-bce-val').textContent = val.toFixed(2);
          this.drawBCEChart(val);
        });
      }
    }

    document.getElementById('gan-loss-prev').onclick = () => {
      if (this.currentStep > 0) { this.currentStep--; this.render(); }
    };
    document.getElementById('gan-loss-next').onclick = () => {
      if (this.currentStep < this.totalSteps - 1) { this.currentStep++; this.render(); }
    };
    document.getElementById('gan-loss-reset').onclick = () => {
      this.currentStep = 0; this.render();
    };
  },

  drawBCEChart(currentVal) {
    const canvas = document.getElementById('gan-bce-canvas');
    if (!canvas) return;
    
    const ctx = canvas.getContext('2d');
    const w = canvas.width;
    const h = canvas.height;
    const pad = { top: 20, right: 20, bottom: 35, left: 50 };
    
    ctx.clearRect(0, 0, w, h);
    
    const plotW = w - pad.left - pad.right;
    const plotH = h - pad.top - pad.bottom;
    
    // Grid
    ctx.strokeStyle = 'rgba(255,255,255,0.05)';
    ctx.lineWidth = 1;
    for (let i = 0; i <= 5; i++) {
      const y = pad.top + (plotH / 5) * i;
      ctx.beginPath();
      ctx.moveTo(pad.left, y);
      ctx.lineTo(w - pad.right, y);
      ctx.stroke();
    }
    
    // BCE curves
    const maxLoss = 5;
    
    // -log(x) curve (y=1, real)
    ctx.strokeStyle = '#4ECDC4';
    ctx.lineWidth = 2;
    ctx.beginPath();
    for (let px = 0; px < plotW; px++) {
      const x = (px / plotW) * 0.98 + 0.01;
      const loss = Math.min(-Math.log(x), maxLoss);
      const screenX = pad.left + px;
      const screenY = pad.top + plotH - (loss / maxLoss) * plotH;
      if (px === 0) ctx.moveTo(screenX, screenY);
      else ctx.lineTo(screenX, screenY);
    }
    ctx.stroke();
    
    // -log(1-x) curve (y=0, fake)
    ctx.strokeStyle = '#FF6584';
    ctx.lineWidth = 2;
    ctx.beginPath();
    for (let px = 0; px < plotW; px++) {
      const x = (px / plotW) * 0.98 + 0.01;
      const loss = Math.min(-Math.log(1 - x), maxLoss);
      const screenX = pad.left + px;
      const screenY = pad.top + plotH - (loss / maxLoss) * plotH;
      if (px === 0) ctx.moveTo(screenX, screenY);
      else ctx.lineTo(screenX, screenY);
    }
    ctx.stroke();
    
    // Current value marker
    const markerX = pad.left + currentVal * plotW;
    ctx.strokeStyle = 'rgba(255,255,255,0.5)';
    ctx.lineWidth = 1;
    ctx.setLineDash([4, 4]);
    ctx.beginPath();
    ctx.moveTo(markerX, pad.top);
    ctx.lineTo(markerX, h - pad.bottom);
    ctx.stroke();
    ctx.setLineDash([]);
    
    // Marker dots
    const realLoss = Math.min(-Math.log(currentVal), maxLoss);
    const fakeLoss = Math.min(-Math.log(1 - currentVal), maxLoss);
    
    const realY = pad.top + plotH - (realLoss / maxLoss) * plotH;
    const fakeY = pad.top + plotH - (fakeLoss / maxLoss) * plotH;
    
    ctx.beginPath();
    ctx.arc(markerX, realY, 6, 0, Math.PI * 2);
    ctx.fillStyle = '#4ECDC4';
    ctx.fill();
    
    ctx.beginPath();
    ctx.arc(markerX, fakeY, 6, 0, Math.PI * 2);
    ctx.fillStyle = '#FF6584';
    ctx.fill();
    
    // Labels
    ctx.fillStyle = '#666';
    ctx.font = '11px Cairo, sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText('D(x)', w / 2, h - 5);
    
    ctx.textAlign = 'right';
    ctx.fillText('0', pad.left - 5, h - pad.bottom + 4);
    ctx.textAlign = 'left';
    ctx.fillText('1', w - pad.right + 5, h - pad.bottom + 4);
    
    // Legend
    ctx.font = '10px JetBrains Mono, monospace';
    ctx.textAlign = 'right';
    ctx.fillStyle = '#4ECDC4';
    ctx.fillText('-log(D(x))', w - pad.right, pad.top + 15);
    ctx.fillStyle = '#FF6584';
    ctx.fillText('-log(1-D(x))', w - pad.right, pad.top + 30);
    
    // Update stat values
    const realDisplay = document.getElementById('gan-bce-real');
    const fakeDisplay = document.getElementById('gan-bce-fake');
    if (realDisplay) realDisplay.textContent = realLoss.toFixed(3);
    if (fakeDisplay) fakeDisplay.textContent = fakeLoss.toFixed(3);
  }
};


/* ============================================
   5. Interactive Training Tab
   ============================================ */
const GANTraining = {
  initialized: false,
  epoch: 0,
  maxEpochs: 200,
  genLosses: [],
  discLosses: [],
  quality: 0,
  genWeights: null,
  discWeights: null,
  lr: 0.01,

  init() {
    if (!this.initialized) {
      this.initialized = true;
      this.setupUI();
    }
    this.renderGallery();
    this.renderChart();
    this.updateStats();
  },

  setupUI() {
    // Train button
    const trainBtn = document.getElementById('gan-train-btn');
    const stopBtn = document.getElementById('gan-stop-btn');
    const resetBtn = document.getElementById('gan-reset-btn');
    const lrSlider = document.getElementById('gan-lr-slider');
    
    if (trainBtn) trainBtn.addEventListener('click', () => this.startTraining());
    if (stopBtn) stopBtn.addEventListener('click', () => this.stopTraining());
    if (resetBtn) resetBtn.addEventListener('click', () => this.reset());
    if (lrSlider) {
      lrSlider.addEventListener('input', () => {
        this.lr = parseFloat(lrSlider.value);
        document.getElementById('gan-lr-val').textContent = this.lr.toFixed(3);
      });
    }
  },

  startTraining() {
    if (GANSim.isTraining) return;
    GANSim.isTraining = true;
    
    document.getElementById('gan-train-btn').disabled = true;
    document.getElementById('gan-stop-btn').disabled = false;
    
    GANSim.trainInterval = setInterval(() => {
      if (this.epoch >= this.maxEpochs) {
        this.stopTraining();
        return;
      }
      this.trainStep();
      this.epoch++;
      this.updateStats();
      
      if (this.epoch % 5 === 0) {
        this.renderGallery();
        this.renderChart();
      }
    }, 100);
  },

  stopTraining() {
    GANSim.isTraining = false;
    if (GANSim.trainInterval) {
      clearInterval(GANSim.trainInterval);
      GANSim.trainInterval = null;
    }
    
    const trainBtn = document.getElementById('gan-train-btn');
    const stopBtn = document.getElementById('gan-stop-btn');
    if (trainBtn) trainBtn.disabled = false;
    if (stopBtn) stopBtn.disabled = true;
  },

  reset() {
    this.stopTraining();
    this.epoch = 0;
    this.genLosses = [];
    this.discLosses = [];
    this.quality = 0;
    this.updateStats();
    this.renderGallery();
    this.renderChart();
  },

  trainStep() {
    // Simulated training dynamics
    const t = this.epoch / this.maxEpochs;
    
    // Generator loss: starts high, decreases with oscillation
    const glBase = 2.5 * (1 - t) + 0.3;
    const glNoise = Math.sin(t * 20) * 0.15 * (1 - t * 0.5) + (Math.random() - 0.5) * 0.2;
    const genLoss = Math.max(0.1, glBase + glNoise);
    
    // Discriminator loss: starts low, increases then stabilizes
    const dlBase = 0.3 + t * 0.5;
    const dlNoise = Math.cos(t * 25) * 0.1 * (1 - t * 0.3) + (Math.random() - 0.5) * 0.15;
    const discLoss = Math.max(0.1, Math.min(1.2, dlBase + dlNoise));
    
    this.genLosses.push(genLoss);
    this.discLosses.push(discLoss);
    
    // Quality improves over training
    this.quality = Math.min(0.95, t * 1.1);
  },

  updateStats() {
    const epochDisplay = document.getElementById('gan-epoch-val');
    const genLossDisplay = document.getElementById('gan-gloss-val');
    const discLossDisplay = document.getElementById('gan-dloss-val');
    const progressBar = document.getElementById('gan-train-progress');
    
    if (epochDisplay) epochDisplay.textContent = `${this.epoch}/${this.maxEpochs}`;
    if (genLossDisplay) genLossDisplay.textContent = this.genLosses.length > 0 
      ? this.genLosses[this.genLosses.length - 1].toFixed(3) : '—';
    if (discLossDisplay) discLossDisplay.textContent = this.discLosses.length > 0 
      ? this.discLosses[this.discLosses.length - 1].toFixed(3) : '—';
    if (progressBar) progressBar.style.width = `${(this.epoch / this.maxEpochs) * 100}%`;
  },

  renderGallery() {
    const container = document.getElementById('gan-training-gallery');
    if (!container) return;
    
    let html = '';
    for (let i = 0; i < 16; i++) {
      const noise = GANUtils.randomNoise(64);
      noise[0] = (i % 4) / 2 - 0.5;
      const grid = GANUtils.generateFakeImage(noise, this.quality);
      const canvasId = `gan-train-img-${i}`;
      html += `<div class="gan-gallery-item"><canvas id="${canvasId}" width="64" height="64"></canvas></div>`;
    }
    container.innerHTML = html;
    
    for (let i = 0; i < 16; i++) {
      const noise = GANUtils.randomNoise(64);
      noise[0] = (i % 4) / 2 - 0.5;
      const grid = GANUtils.generateFakeImage(noise, this.quality);
      const canvas = document.getElementById(`gan-train-img-${i}`);
      if (canvas) GANUtils.drawGrid(canvas, grid, 'rgba(78, 205, 196, 0.5)');
    }
  },

  renderChart() {
    const canvas = document.getElementById('gan-loss-chart');
    if (!canvas || this.genLosses.length < 2) return;
    
    canvas.width = canvas.parentElement.offsetWidth;
    canvas.height = 200;
    GANUtils.drawLossChart(canvas, this.genLosses, this.discLosses);
  }
};


/* ============================================
   6. Applications Tab
   ============================================ */
const GANApplications = {
  initialized: false,
  currentStep: 0,
  totalSteps: 4,

  steps: [
    // Step 1: GAN Variants
    () => `
      <div class="gan-info-card">
        <h3 style="color: #6C63FF;"><i class="fas fa-project-diagram"></i> أنواع GAN المشهورة</h3>
        <p>ظهرت عشرات الأنواع المحسنة من GAN الأصلي. إليك أبرزها:</p>
      </div>
      <div style="display: flex; flex-direction: column; gap: 0.8rem;">
        <div class="gan-info-card" style="border-color: rgba(78,205,196,0.2);">
          <h3 style="color: #4ECDC4;"><i class="fas fa-image"></i> DCGAN (2015)</h3>
          <p>أول GAN يستخدم طبقات التفاف عميقة (Deep Convolutional). أثبت أن GAN يمكنها توليد صور عالية الجودة مع بنية مستقرة.</p>
        </div>
        <div class="gan-info-card" style="border-color: rgba(255,214,0,0.2);">
          <h3 style="color: #FFD93D;"><i class="fas fa-exchange-alt"></i> Pix2Pix (2016)</h3>
          <p>GAN مشروط يحول صورة من نوع لآخر (مثل: رسم → صورة حقيقية، نهار → ليل).</p>
        </div>
        <div class="gan-info-card" style="border-color: rgba(108,99,255,0.2);">
          <h3 style="color: #6C63FF;"><i class="fas fa-sync-alt"></i> CycleGAN (2017)</h3>
          <p>يحول الصور بين مجالين دون الحاجة لأزواج متطابقة (مثل: حصان ↔ حمار وحشي).</p>
        </div>
        <div class="gan-info-card" style="border-color: rgba(255,101,132,0.2);">
          <h3 style="color: #FF6584;"><i class="fas fa-crown"></i> StyleGAN (2018-2021)</h3>
          <p>من Nvidia. ينتج وجوهاً بشرية واقعية بشكل مذهل مع التحكم في الأسلوب (العمر، الشعر، النظارات...).</p>
        </div>
      </div>
    `,

    // Step 2: Image Generation
    () => `
      <div class="gan-info-card">
        <h3 style="color: #4ECDC4;"><i class="fas fa-paint-brush"></i> توليد الصور</h3>
        <p>
          أبرز تطبيقات GAN هو <b>توليد صور واقعية</b> لم تكن موجودة من قبل. من الوجوه البشرية إلى المناظر الطبيعية.
        </p>
      </div>
      <div class="gan-highlight-box tip">
        <b>💡 مثال:</b> موقع <b>thispersondoesnotexist.com</b> يستخدم StyleGAN لتوليد وجوه بشرية كاملة لأشخاص غير موجودين.
      </div>
      <div class="gan-flow">
        <div class="gan-flow-step" style="border-color: #4ECDC4;">
          <div class="step-num" style="background: rgba(78,205,196,0.2); color: #4ECDC4;">1</div>
          <div><b>توليد الوجوه:</b> StyleGAN ينتج وجوهاً واقعية بدقة 1024×1024</div>
        </div>
        <div class="gan-flow-step" style="border-color: #FFD93D;">
          <div class="step-num" style="background: rgba(255,214,0,0.2); color: #FFD93D;">2</div>
          <div><b>تحسين الصور:</b> Super-Resolution GAN يرفع دقة الصور المنخفضة</div>
        </div>
        <div class="gan-flow-step" style="border-color: #FF6584;">
          <div class="step-num" style="background: rgba(255,101,132,0.2); color: #FF6584;">3</div>
          <div><b>تلوين الصور:</b> تحويل الصور الأبيض والأسود إلى ملونة</div>
        </div>
        <div class="gan-flow-step" style="border-color: #6C63FF;">
          <div class="step-num" style="background: rgba(108,99,255,0.2); color: #6C63FF;">4</div>
          <div><b>إكمال الصور:</b> ملء الأجزاء المفقودة من الصور (Inpainting)</div>
        </div>
      </div>
    `,

    // Step 3: Other Applications
    () => `
      <div class="gan-info-card">
        <h3 style="color: #FF6584;"><i class="fas fa-rocket"></i> تطبيقات أخرى لـ GAN</h3>
        <p>GAN ليست مقتصرة على الصور فقط! إليك تطبيقات متنوعة:</p>
      </div>
      <div class="gan-two-col">
        <div class="gan-info-card" style="border-color: rgba(78,205,196,0.2);">
          <h3 style="color: #4ECDC4;"><i class="fas fa-music"></i> توليد الموسيقى</h3>
          <p>MuseGAN و WaveGAN ينتجان مقاطع موسيقية وأصوات جديدة.</p>
        </div>
        <div class="gan-info-card" style="border-color: rgba(255,214,0,0.2);">
          <h3 style="color: #FFD93D;"><i class="fas fa-pills"></i> اكتشاف الأدوية</h3>
          <p>MolGAN يولّد تراكيب جزيئية جديدة لتسريع اكتشاف أدوية.</p>
        </div>
        <div class="gan-info-card" style="border-color: rgba(255,101,132,0.2);">
          <h3 style="color: #FF6584;"><i class="fas fa-video"></i> توليد الفيديو</h3>
          <p>DVD-GAN و MoCoGAN ينتجان مقاطع فيديو واقعية.</p>
        </div>
        <div class="gan-info-card" style="border-color: rgba(108,99,255,0.2);">
          <h3 style="color: #6C63FF;"><i class="fas fa-cube"></i> نماذج ثلاثية الأبعاد</h3>
          <p>3D-GAN ينتج نماذج ثلاثية الأبعاد من الصفر.</p>
        </div>
      </div>
    `,

    // Step 4: GAN vs Diffusion
    () => `
      <div class="gan-info-card">
        <h3 style="color: #FFD93D;"><i class="fas fa-balance-scale"></i> GAN مقابل نماذج الانتشار (Diffusion)</h3>
        <p>
          في السنوات الأخيرة، ظهرت نماذج الانتشار (مثل Stable Diffusion و DALL-E) كبديل قوي لـ GAN. ما الفرق؟
        </p>
      </div>
      <div style="overflow-x: auto;">
        <table style="width:100%; border-collapse: collapse; font-size: 0.85rem;">
          <thead>
            <tr style="border-bottom: 2px solid rgba(255,255,255,0.1);">
              <th style="padding: 0.8rem; text-align: right; color: #888;">المعيار</th>
              <th style="padding: 0.8rem; text-align: center; color: #FF5252;">GAN</th>
              <th style="padding: 0.8rem; text-align: center; color: #6C63FF;">Diffusion Models</th>
            </tr>
          </thead>
          <tbody style="color: #ccc;">
            <tr style="border-bottom: 1px solid rgba(255,255,255,0.05);">
              <td style="padding: 0.6rem 0.8rem;">سرعة التوليد</td>
              <td style="padding: 0.6rem; text-align:center; color: #00E676;">⚡ سريع جداً</td>
              <td style="padding: 0.6rem; text-align:center; color: #FF5252;">🐢 بطيء</td>
            </tr>
            <tr style="border-bottom: 1px solid rgba(255,255,255,0.05);">
              <td style="padding: 0.6rem 0.8rem;">جودة الصورة</td>
              <td style="padding: 0.6rem; text-align:center;">ممتازة</td>
              <td style="padding: 0.6rem; text-align:center; color: #00E676;">ممتازة+</td>
            </tr>
            <tr style="border-bottom: 1px solid rgba(255,255,255,0.05);">
              <td style="padding: 0.6rem 0.8rem;">استقرار التدريب</td>
              <td style="padding: 0.6rem; text-align:center; color: #FF5252;">صعب وغير مستقر</td>
              <td style="padding: 0.6rem; text-align:center; color: #00E676;">مستقر</td>
            </tr>
            <tr style="border-bottom: 1px solid rgba(255,255,255,0.05);">
              <td style="padding: 0.6rem 0.8rem;">تنوع المخرجات</td>
              <td style="padding: 0.6rem; text-align:center; color: #FFD93D;">قد يعاني Mode Collapse</td>
              <td style="padding: 0.6rem; text-align:center; color: #00E676;">تنوع عالي</td>
            </tr>
            <tr>
              <td style="padding: 0.6rem 0.8rem;">التحكم بالنص</td>
              <td style="padding: 0.6rem; text-align:center; color: #FFD93D;">محدود</td>
              <td style="padding: 0.6rem; text-align:center; color: #00E676;">ممتاز (Text-to-Image)</td>
            </tr>
          </tbody>
        </table>
      </div>
      <div class="gan-highlight-box info" style="margin-top: 1rem;">
        <b>ℹ️</b> رغم ظهور Diffusion Models، لا تزال GAN مستخدمة في التطبيقات التي تتطلب <b>سرعة توليد عالية</b> مثل ألعاب الفيديو والتطبيقات التفاعلية.
      </div>
    `
  ],

  init() {
    this.render();
  },

  render() {
    const content = document.getElementById('gan-applications-content');
    const stepIndicator = document.getElementById('gan-app-step');
    
    content.innerHTML = typeof this.steps[this.currentStep] === 'function'
      ? this.steps[this.currentStep]()
      : this.steps[this.currentStep];
    stepIndicator.textContent = `${this.currentStep + 1}/${this.totalSteps}`;
    
    this.postRender();
  },

  postRender() {
    document.getElementById('gan-app-prev').onclick = () => {
      if (this.currentStep > 0) { this.currentStep--; this.render(); }
    };
    document.getElementById('gan-app-next').onclick = () => {
      if (this.currentStep < this.totalSteps - 1) { this.currentStep++; this.render(); }
    };
    document.getElementById('gan-app-reset').onclick = () => {
      this.currentStep = 0; this.render();
    };
  }
};
