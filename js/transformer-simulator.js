/* ============================================
   Transformer Simulator - Interactive Step-by-Step
   Inspired by "Attention Is All You Need" paper
   ============================================ */

function initTransformerSimulator() {
  const card = document.getElementById('transformer-card');
  const overlay = document.getElementById('transformer-simulator-overlay');
  const closeBtn = document.getElementById('transformer-close-sim');
  
  if (!card || !overlay) return;

  card.addEventListener('click', () => {
    overlay.classList.add('active');
    document.body.style.overflow = 'hidden';
    TransformerSim.init();
  });

  closeBtn.addEventListener('click', () => {
    overlay.classList.remove('active');
    document.body.style.overflow = '';
  });

  // Tab switching
  overlay.querySelectorAll('.transformer-tab').forEach(tab => {
    tab.addEventListener('click', () => {
      const tabId = tab.dataset.tab;
      overlay.querySelectorAll('.transformer-tab').forEach(t => t.classList.remove('active'));
      overlay.querySelectorAll('.transformer-tab-content').forEach(c => c.classList.remove('active'));
      tab.classList.add('active');
      document.getElementById(tabId + '-tab').classList.add('active');
      TransformerSim.activateTab(tabId);
    });
  });
}

/* ============================================
   TransformerSim - Main Controller
   ============================================ */
const TransformerSim = {
  initialized: false,
  currentTab: 'tf-overview',
  
  init() {
    if (this.initialized) {
      this.activateTab(this.currentTab);
      return;
    }
    this.initialized = true;
    OverviewView.init();
  },

  activateTab(tabId) {
    this.currentTab = tabId;
    switch(tabId) {
      case 'tf-overview': OverviewView.init(); break;
      case 'tf-embedding': EmbeddingView.init(); break;
      case 'tf-selfattn': SelfAttentionView.init(); break;
      case 'tf-multihead': MultiHeadView.init(); break;
      case 'tf-ffn': FFNView.init(); break;
      case 'tf-addnorm': AddNormView.init(); break;
      case 'tf-decoder': DecoderView.init(); break;
      case 'tf-training': TrainingView.init(); break;
    }
  }
};

/* ============================================
   Helper Utilities
   ============================================ */
const TFUtils = {
  // Color palette for transformer
  colors: {
    encoder: '#6C63FF',
    decoder: '#FF6584',
    attention: '#FFD93D',
    ffn: '#4ECDC4',
    embedding: '#45B7D1',
    norm: '#96CEB4',
    output: '#FF8A5C',
    bg: '#0a0e27',
    glow: 'rgba(108, 99, 255, 0.3)',
    text: '#e0e0e0',
    muted: '#888',
    grid: 'rgba(108, 99, 255, 0.05)'
  },

  softmax(arr) {
    const max = Math.max(...arr);
    const exps = arr.map(x => Math.exp(x - max));
    const sum = exps.reduce((a,b) => a + b, 0);
    return exps.map(x => x / sum);
  },

  matMul(a, b) {
    const rows = a.length, cols = b[0].length, n = b.length;
    const result = Array.from({length: rows}, () => new Array(cols).fill(0));
    for (let i = 0; i < rows; i++)
      for (let j = 0; j < cols; j++)
        for (let k = 0; k < n; k++)
          result[i][j] += a[i][k] * b[k][j];
    return result;
  },

  transpose(m) {
    return m[0].map((_, i) => m.map(row => row[i]));
  },

  layerNorm(arr) {
    const mean = arr.reduce((a,b) => a+b, 0) / arr.length;
    const variance = arr.reduce((a,b) => a + (b-mean)**2, 0) / arr.length;
    const std = Math.sqrt(variance + 1e-6);
    return arr.map(x => (x - mean) / std);
  },

  relu(x) { return Math.max(0, x); },

  randomMatrix(rows, cols, scale=0.5) {
    return Array.from({length: rows}, () =>
      Array.from({length: cols}, () => (Math.random() - 0.5) * scale * 2)
    );
  },

  formatNum(n) {
    return typeof n === 'number' ? n.toFixed(3) : n;
  },

  drawRoundedRect(ctx, x, y, w, h, r, fill, stroke) {
    ctx.beginPath();
    ctx.roundRect(x, y, w, h, r);
    if (fill) { ctx.fillStyle = fill; ctx.fill(); }
    if (stroke) { ctx.strokeStyle = stroke; ctx.lineWidth = 1.5; ctx.stroke(); }
  },

  drawGlowRect(ctx, x, y, w, h, r, color, glowSize=15) {
    ctx.save();
    ctx.shadowColor = color;
    ctx.shadowBlur = glowSize;
    ctx.fillStyle = color + '22';
    ctx.beginPath();
    ctx.roundRect(x, y, w, h, r);
    ctx.fill();
    ctx.restore();
    ctx.strokeStyle = color;
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.roundRect(x, y, w, h, r);
    ctx.stroke();
  },

  drawArrow(ctx, fromX, fromY, toX, toY, color='#555', dashed=false) {
    ctx.save();
    ctx.strokeStyle = color;
    ctx.fillStyle = color;
    ctx.lineWidth = 2;
    if (dashed) ctx.setLineDash([5, 5]);
    ctx.beginPath();
    ctx.moveTo(fromX, fromY);
    ctx.lineTo(toX, toY);
    ctx.stroke();
    // Arrowhead
    const angle = Math.atan2(toY - fromY, toX - fromX);
    const headLen = 8;
    ctx.beginPath();
    ctx.moveTo(toX, toY);
    ctx.lineTo(toX - headLen * Math.cos(angle - Math.PI/6), toY - headLen * Math.sin(angle - Math.PI/6));
    ctx.lineTo(toX - headLen * Math.cos(angle + Math.PI/6), toY - headLen * Math.sin(angle + Math.PI/6));
    ctx.closePath();
    ctx.fill();
    ctx.restore();
  },

  drawText(ctx, text, x, y, color='#e0e0e0', size=13, align='center', font='Cairo') {
    ctx.fillStyle = color;
    ctx.font = `${size}px '${font}', sans-serif`;
    ctx.textAlign = align;
    ctx.textBaseline = 'middle';
    ctx.fillText(text, x, y);
  }
};


/* ============================================
   1. OVERVIEW VIEW - Full Architecture
   ============================================ */
const OverviewView = {
  init() {
    const blocks = document.querySelectorAll('.tf-arch-block');
    blocks.forEach(block => {
      block.addEventListener('click', (e) => {
        const targetTab = block.getAttribute('data-tab');
        if (targetTab) {
          const tabBtn = document.querySelector('.transformer-tab[data-tab="' + targetTab + '"]');
          if (tabBtn) tabBtn.click();
        }
      });
    });
  },

  destroy() {
    // No cleanup needed for simple DOM listeners
  }
};

/* ============================================
   2. EMBEDDING VIEW
   ============================================ */
const EmbeddingView = {
  step: 0,
  maxSteps: 5,
  sentence: ['أنا', 'أحب', 'التعلم', 'العميق'],
  tokenIds: [42, 187, 301, 156],
  dModel: 4,
  embeddings: null,
  posEncodings: null,

  init() {
    this.step = 0;
    this.generateData();
    this.render();
    this.setupButtons();
  },

  generateData() {
    // Random embedding vectors
    this.embeddings = this.tokenIds.map(() => 
      Array.from({length: this.dModel}, () => +(Math.random() * 2 - 1).toFixed(3))
    );
    // Positional encodings using sin/cos
    this.posEncodings = this.sentence.map((_, pos) =>
      Array.from({length: this.dModel}, (_, i) => {
        const angle = pos / Math.pow(10000, (2 * Math.floor(i/2)) / this.dModel);
        return +(i % 2 === 0 ? Math.sin(angle) : Math.cos(angle)).toFixed(3);
      })
    );
  },

  setupButtons() {
    const next = document.getElementById('tf-emb-next');
    const prev = document.getElementById('tf-emb-prev');
    const reset = document.getElementById('tf-emb-reset');
    if (next) next.onclick = () => { if (this.step < this.maxSteps) { this.step++; this.render(); }};
    if (prev) prev.onclick = () => { if (this.step > 0) { this.step--; this.render(); }};
    if (reset) reset.onclick = () => { this.step = 0; this.render(); };
  },

  render() {
    const container = document.getElementById('tf-embedding-content');
    if (!container) return;
    
    const steps = [
      this.renderStep0(),
      this.renderStep1(),
      this.renderStep2(),
      this.renderStep3(),
      this.renderStep4(),
      this.renderStep5()
    ];
    
    container.innerHTML = steps[this.step];
    this.updateStepIndicator();
  },

  updateStepIndicator() {
    const ind = document.getElementById('tf-emb-step');
    if (ind) ind.textContent = `${this.step + 1} / ${this.maxSteps + 1}`;
  },

  renderStep0() {
    return `
      <div class="tf-step-card">
        <div class="tf-step-title">📝 الخطوة 1: النص الخام (Raw Text)</div>
        <div class="tf-step-desc">نبدأ بجملة نصية يريد النموذج فهمها:</div>
        <div class="tf-sentence-display">
          ${this.sentence.map(w => `<span class="tf-word">${w}</span>`).join('<span class="tf-arrow-inline">→</span>')}
        </div>
        <div class="tf-info-box">
          <i class="fas fa-info-circle"></i>
          الكمبيوتر لا يفهم النصوص مباشرة! يجب تحويل كل كلمة إلى أرقام (vectors) يمكن للشبكة العصبية معالجتها.
        </div>
      </div>`;
  },

  renderStep1() {
    return `
      <div class="tf-step-card">
        <div class="tf-step-title">🔢 الخطوة 2: التحويل إلى رموز (Tokenization)</div>
        <div class="tf-step-desc">كل كلمة تُعطى رقم تعريفي فريد من القاموس:</div>
        <div class="tf-matrix-display">
          <table class="tf-matrix">
            <tr><th>الكلمة</th><th>Token ID</th></tr>
            ${this.sentence.map((w, i) => `
              <tr><td class="tf-word-cell">${w}</td><td class="tf-num-cell">${this.tokenIds[i]}</td></tr>
            `).join('')}
          </table>
        </div>
        <div class="tf-info-box">
          <i class="fas fa-lightbulb"></i>
          القاموس يحتوي عادةً على 30,000 - 100,000 كلمة/رمز. كل كلمة لها رقم واحد فقط.
        </div>
      </div>`;
  },

  renderStep2() {
    return `
      <div class="tf-step-card">
        <div class="tf-step-title">📊 الخطوة 3: تحويل لمتجهات (Word Embedding)</div>
        <div class="tf-step-desc">كل Token ID يتحول إلى متجه (vector) بحجم d_model = ${this.dModel}:</div>
        <div class="tf-matrix-display">
          <table class="tf-matrix">
            <tr>
              <th>الكلمة</th>
              ${Array.from({length: this.dModel}, (_, i) => `<th>d<sub>${i+1}</sub></th>`).join('')}
            </tr>
            ${this.sentence.map((w, i) => `
              <tr>
                <td class="tf-word-cell">${w}</td>
                ${this.embeddings[i].map(v => `<td class="tf-num-cell ${v >= 0 ? 'positive' : 'negative'}">${v}</td>`).join('')}
              </tr>
            `).join('')}
          </table>
        </div>
        <div class="tf-info-box">
          <i class="fas fa-info-circle"></i>
          هذه المتجهات تتعلم أثناء التدريب! الكلمات المتشابهة في المعنى تكون متجهاتها قريبة في الفضاء.
          <br>في GPT-3 مثلاً، d_model = 12288 (أي كل كلمة = 12288 رقم!)
        </div>
      </div>`;
  },

  renderStep3() {
    return `
      <div class="tf-step-card">
        <div class="tf-step-title">📍 الخطوة 4: ترميز الموضع (Positional Encoding)</div>
        <div class="tf-step-desc">Transformer لا يعرف ترتيب الكلمات! لذلك نضيف معلومات الموضع:</div>
        <div class="tf-formula">
          PE<sub>(pos, 2i)</sub> = sin(pos / 10000<sup>2i/d</sup>)
          <br>
          PE<sub>(pos, 2i+1)</sub> = cos(pos / 10000<sup>2i/d</sup>)
        </div>
        <div class="tf-matrix-display">
          <table class="tf-matrix">
            <tr>
              <th>الموضع</th>
              ${Array.from({length: this.dModel}, (_, i) => `<th>PE<sub>${i+1}</sub></th>`).join('')}
            </tr>
            ${this.sentence.map((w, i) => `
              <tr>
                <td class="tf-word-cell">pos=${i}</td>
                ${this.posEncodings[i].map(v => `<td class="tf-num-cell pe-cell">${v}</td>`).join('')}
              </tr>
            `).join('')}
          </table>
        </div>
        <div class="tf-info-box">
          <i class="fas fa-lightbulb"></i>
          نستخدم دوال sin و cos بترددات مختلفة حتى يتمكن النموذج من التمييز بين المواضع. كل موضع له "بصمة" فريدة!
        </div>
      </div>`;
  },

  renderStep4() {
    const result = this.embeddings.map((emb, i) =>
      emb.map((v, j) => +(v + this.posEncodings[i][j]).toFixed(3))
    );
    return `
      <div class="tf-step-card">
        <div class="tf-step-title">➕ الخطوة 5: الجمع (Embedding + Position)</div>
        <div class="tf-step-desc">نجمع متجه الكلمة مع متجه الموضع:</div>
        <div class="tf-addition-visual">
          <div class="tf-add-label">Embedding</div>
          <div class="tf-add-op">+</div>
          <div class="tf-add-label">Positional</div>
          <div class="tf-add-op">=</div>
          <div class="tf-add-label">المخرج النهائي</div>
        </div>
        <div class="tf-matrix-display">
          <table class="tf-matrix result-matrix">
            <tr>
              <th>الكلمة</th>
              ${Array.from({length: this.dModel}, (_, i) => `<th>x<sub>${i+1}</sub></th>`).join('')}
            </tr>
            ${this.sentence.map((w, i) => `
              <tr>
                <td class="tf-word-cell">${w}</td>
                ${result[i].map(v => `<td class="tf-num-cell result-cell">${v}</td>`).join('')}
              </tr>
            `).join('')}
          </table>
        </div>
        <div class="tf-info-box success">
          <i class="fas fa-check-circle"></i>
          الآن كل كلمة لها تمثيل رقمي يحمل معناها <strong>وموضعها</strong> في الجملة. هذا هو المدخل لطبقات Encoder!
        </div>
      </div>`;
  },

  renderStep5() {
    return `
      <div class="tf-step-card">
        <div class="tf-step-title">🎯 ملخص Embedding + Positional Encoding</div>
        <div class="tf-summary-flow">
          <div class="tf-flow-item"><span class="tf-flow-icon">📝</span><span>نص</span></div>
          <div class="tf-flow-arrow">→</div>
          <div class="tf-flow-item"><span class="tf-flow-icon">🔢</span><span>Token IDs</span></div>
          <div class="tf-flow-arrow">→</div>
          <div class="tf-flow-item"><span class="tf-flow-icon">📊</span><span>Embedding</span></div>
          <div class="tf-flow-arrow">+</div>
          <div class="tf-flow-item"><span class="tf-flow-icon">📍</span><span>Position</span></div>
          <div class="tf-flow-arrow">→</div>
          <div class="tf-flow-item highlight"><span class="tf-flow-icon">✨</span><span>المدخل النهائي</span></div>
        </div>
        <div class="tf-info-box success">
          <i class="fas fa-arrow-left"></i>
          انتقل للتبويب التالي <strong>"Self-Attention"</strong> لنرى كيف يفهم النموذج العلاقات بين الكلمات!
        </div>
      </div>`;
  }
};

/* ============================================
   3. SELF-ATTENTION VIEW
   ============================================ */
const SelfAttentionView = {
  step: 0,
  maxSteps: 7,
  words: ['I', 'love', 'deep', 'learning'],
  dModel: 4,
  dK: 4,
  X: null, Wq: null, Wk: null, Wv: null,
  Q: null, K: null, V: null,
  scores: null, attnWeights: null, output: null,

  init() {
    this.step = 0;
    this.generateData();
    this.render();
    this.setupButtons();
  },

  generateData() {
    this.X = TFUtils.randomMatrix(4, this.dModel, 1);
    this.Wq = TFUtils.randomMatrix(this.dModel, this.dK, 0.5);
    this.Wk = TFUtils.randomMatrix(this.dModel, this.dK, 0.5);
    this.Wv = TFUtils.randomMatrix(this.dModel, this.dK, 0.5);
    this.Q = TFUtils.matMul(this.X, this.Wq);
    this.K = TFUtils.matMul(this.X, this.Wk);
    this.V = TFUtils.matMul(this.X, this.Wv);
    // Scores = Q * K^T / sqrt(dK)
    const KT = TFUtils.transpose(this.K);
    const raw = TFUtils.matMul(this.Q, KT);
    const scale = Math.sqrt(this.dK);
    this.scores = raw.map(row => row.map(v => v / scale));
    this.attnWeights = this.scores.map(row => TFUtils.softmax(row));
    this.output = TFUtils.matMul(this.attnWeights, this.V);
  },

  setupButtons() {
    const next = document.getElementById('tf-attn-next');
    const prev = document.getElementById('tf-attn-prev');
    const reset = document.getElementById('tf-attn-reset');
    if (next) next.onclick = () => { if (this.step < this.maxSteps) { this.step++; this.render(); }};
    if (prev) prev.onclick = () => { if (this.step > 0) { this.step--; this.render(); }};
    if (reset) reset.onclick = () => { this.step = 0; this.render(); };
  },

  render() {
    const container = document.getElementById('tf-selfattn-content');
    if (!container) return;
    const steps = [
      this.stepIntro(),
      this.stepInputMatrix(),
      this.stepQKV(),
      this.stepComputeQ(),
      this.stepScores(),
      this.stepSoftmax(),
      this.stepWeightedSum(),
      this.stepSummary()
    ];
    container.innerHTML = steps[this.step];
    this.updateStepIndicator();
  },

  updateStepIndicator() {
    const ind = document.getElementById('tf-attn-step');
    if (ind) ind.textContent = `${this.step + 1} / ${this.maxSteps + 1}`;
  },

  matrixHTML(matrix, labels, colLabels, className='') {
    return `<table class="tf-matrix ${className}">
      <tr><th></th>${colLabels.map(c => `<th>${c}</th>`).join('')}</tr>
      ${matrix.map((row, i) => `<tr><td class="tf-word-cell">${labels[i]}</td>${row.map(v => `<td class="tf-num-cell">${TFUtils.formatNum(v)}</td>`).join('')}</tr>`).join('')}
    </table>`;
  },

  stepIntro() {
    return `<div class="tf-step-card">
      <div class="tf-step-title">🎯 ما هو Self-Attention؟</div>
      <div class="tf-step-desc">
        Self-Attention هو الآلية التي تسمح لكل كلمة بـ"النظر" إلى كل الكلمات الأخرى في الجملة لفهم السياق.
      </div>
      <div class="tf-example-box">
        <div class="tf-example-sentence">"The <span class="tf-highlight">animal</span> didn't cross the street because <span class="tf-highlight">it</span> was too tired"</div>
        <div class="tf-example-explain">هنا كلمة "it" تحتاج أن "تنتبه" لكلمة "animal" لتعرف أنها تشير إليها!</div>
      </div>
      <div class="tf-formula">
        Attention(Q, K, V) = softmax(QK<sup>T</sup> / √d<sub>k</sub>) · V
      </div>
      <div class="tf-info-box">
        <i class="fas fa-lightbulb"></i>
        <strong>Q</strong> = Query (السؤال) | <strong>K</strong> = Key (المفتاح) | <strong>V</strong> = Value (القيمة)
        <br>كل كلمة تسأل: "أي كلمات أخرى مهمة لي؟"
      </div>
    </div>`;
  },

  stepInputMatrix() {
    const cols = Array.from({length: this.dModel}, (_, i) => `d${i+1}`);
    return `<div class="tf-step-card">
      <div class="tf-step-title">📊 الخطوة 1: مصفوفة المدخلات X</div>
      <div class="tf-step-desc">المدخلات هي المتجهات الناتجة من طبقة Embedding + Position:</div>
      <div class="tf-matrix-display">
        ${this.matrixHTML(this.X, this.words, cols)}
      </div>
      <div class="tf-info-box">
        <i class="fas fa-info-circle"></i>
        كل صف = تمثيل رقمي لكلمة واحدة (4 أبعاد في مثالنا، 512+ في الواقع)
      </div>
    </div>`;
  },

  stepQKV() {
    const cols = Array.from({length: this.dK}, (_, i) => `d${i+1}`);
    return `<div class="tf-step-card">
      <div class="tf-step-title">🔑 الخطوة 2: مصفوفات الأوزان W<sub>Q</sub>, W<sub>K</sub>, W<sub>V</sub></div>
      <div class="tf-step-desc">ثلاث مصفوفات أوزان (تتعلم أثناء التدريب) تحول X إلى Q, K, V:</div>
      <div class="tf-three-matrices">
        <div class="tf-mat-group">
          <div class="tf-mat-label" style="color: ${TFUtils.colors.attention}">W<sub>Q</sub></div>
          ${this.matrixHTML(this.Wq, ['w1','w2','w3','w4'], cols, 'q-matrix')}
        </div>
        <div class="tf-mat-group">
          <div class="tf-mat-label" style="color: ${TFUtils.colors.encoder}">W<sub>K</sub></div>
          ${this.matrixHTML(this.Wk, ['w1','w2','w3','w4'], cols, 'k-matrix')}
        </div>
        <div class="tf-mat-group">
          <div class="tf-mat-label" style="color: ${TFUtils.colors.ffn}">W<sub>V</sub></div>
          ${this.matrixHTML(this.Wv, ['w1','w2','w3','w4'], cols, 'v-matrix')}
        </div>
      </div>
      <div class="tf-formula">Q = X · W<sub>Q</sub> &nbsp; | &nbsp; K = X · W<sub>K</sub> &nbsp; | &nbsp; V = X · W<sub>V</sub></div>
    </div>`;
  },

  stepComputeQ() {
    const cols = Array.from({length: this.dK}, (_, i) => `d${i+1}`);
    return `<div class="tf-step-card">
      <div class="tf-step-title">🧮 الخطوة 3: حساب Q, K, V</div>
      <div class="tf-step-desc">نضرب مصفوفة المدخلات X بكل مصفوفة وزن:</div>
      <div class="tf-three-matrices">
        <div class="tf-mat-group">
          <div class="tf-mat-label" style="color: ${TFUtils.colors.attention}">Q = X · W<sub>Q</sub></div>
          ${this.matrixHTML(this.Q, this.words, cols, 'q-matrix')}
        </div>
        <div class="tf-mat-group">
          <div class="tf-mat-label" style="color: ${TFUtils.colors.encoder}">K = X · W<sub>K</sub></div>
          ${this.matrixHTML(this.K, this.words, cols, 'k-matrix')}
        </div>
        <div class="tf-mat-group">
          <div class="tf-mat-label" style="color: ${TFUtils.colors.ffn}">V = X · W<sub>V</sub></div>
          ${this.matrixHTML(this.V, this.words, cols, 'v-matrix')}
        </div>
      </div>
      <div class="tf-info-box">
        <i class="fas fa-lightbulb"></i>
        <strong>Q</strong>: ماذا أبحث عنه؟ | <strong>K</strong>: ماذا أملك؟ | <strong>V</strong>: ما هي المعلومات الفعلية؟
      </div>
    </div>`;
  },

  stepScores() {
    return `<div class="tf-step-card">
      <div class="tf-step-title">📐 الخطوة 4: حساب درجات الانتباه (Attention Scores)</div>
      <div class="tf-formula">Scores = Q · K<sup>T</sup> / √d<sub>k</sub> = Q · K<sup>T</sup> / √${this.dK} = Q · K<sup>T</sup> / ${Math.sqrt(this.dK).toFixed(2)}</div>
      <div class="tf-step-desc">كل خلية تمثل مدى "تشابه" سؤال كلمة مع مفتاح كلمة أخرى:</div>
      <div class="tf-matrix-display">
        ${this.matrixHTML(this.scores, this.words, this.words, 'scores-matrix')}
      </div>
      <div class="tf-info-box">
        <i class="fas fa-info-circle"></i>
        نقسم على √d<sub>k</sub> لمنع القيم من أن تصبح كبيرة جداً (مما يجعل Softmax تركز على قيمة واحدة فقط).
      </div>
    </div>`;
  },

  stepSoftmax() {
    return `<div class="tf-step-card">
      <div class="tf-step-title">📊 الخطوة 5: تطبيق Softmax</div>
      <div class="tf-step-desc">نحول الدرجات إلى احتمالات (مجموع كل صف = 1):</div>
      <div class="tf-formula">Attention Weights = softmax(Scores)</div>
      <div class="tf-matrix-display">
        <table class="tf-matrix attn-heatmap">
          <tr><th></th>${this.words.map(w => `<th>${w}</th>`).join('')}</tr>
          ${this.attnWeights.map((row, i) => `
            <tr>
              <td class="tf-word-cell">${this.words[i]}</td>
              ${row.map(v => {
                const intensity = Math.floor(v * 255);
                const bg = `rgba(255, 217, 61, ${v * 0.8})`;
                return `<td class="tf-num-cell" style="background:${bg}; color: ${v > 0.4 ? '#000' : '#fff'}">${(v).toFixed(3)}</td>`;
              }).join('')}
            </tr>
          `).join('')}
        </table>
      </div>
      <div class="tf-info-box">
        <i class="fas fa-eye"></i>
        الخلايا الأغمق = انتباه أعلى. كل كلمة "تنظر" للكلمات الأخرى بنسب مختلفة!
      </div>
    </div>`;
  },

  stepWeightedSum() {
    const cols = Array.from({length: this.dK}, (_, i) => `d${i+1}`);
    return `<div class="tf-step-card">
      <div class="tf-step-title">✨ الخطوة 6: المجموع الموزون (Weighted Sum)</div>
      <div class="tf-formula">Output = Attention Weights · V</div>
      <div class="tf-step-desc">نضرب أوزان الانتباه بمصفوفة القيم V للحصول على المخرج:</div>
      <div class="tf-matrix-display">
        ${this.matrixHTML(this.output, this.words, cols, 'result-matrix')}
      </div>
      <div class="tf-info-box success">
        <i class="fas fa-check-circle"></i>
        كل متجه مخرج هو مزيج ذكي من معلومات كل الكلمات، مرجح حسب أهميتها! هذا هو "الانتباه الذاتي".
      </div>
    </div>`;
  },

  stepSummary() {
    return `<div class="tf-step-card">
      <div class="tf-step-title">🎯 ملخص Self-Attention</div>
      <div class="tf-summary-flow">
        <div class="tf-flow-item"><span class="tf-flow-icon">📊</span><span>X</span></div>
        <div class="tf-flow-arrow">→</div>
        <div class="tf-flow-item" style="border-color:${TFUtils.colors.attention}"><span class="tf-flow-icon">🔑</span><span>Q, K, V</span></div>
        <div class="tf-flow-arrow">→</div>
        <div class="tf-flow-item" style="border-color:${TFUtils.colors.attention}"><span class="tf-flow-icon">📐</span><span>Scores</span></div>
        <div class="tf-flow-arrow">→</div>
        <div class="tf-flow-item"><span class="tf-flow-icon">📊</span><span>Softmax</span></div>
        <div class="tf-flow-arrow">→</div>
        <div class="tf-flow-item highlight"><span class="tf-flow-icon">✨</span><span>Output</span></div>
      </div>
      <div class="tf-info-box success">
        <i class="fas fa-arrow-left"></i>
        انتقل للتبويب التالي <strong>"Multi-Head"</strong> لنرى كيف نستخدم عدة رؤوس انتباه معاً!
      </div>
    </div>`;
  }
};


/* ============================================
   4. MULTI-HEAD ATTENTION VIEW
   ============================================ */
const MultiHeadView = {
  step: 0,
  maxSteps: 4,
  numHeads: 2,

  init() {
    this.step = 0;
    this.render();
    this.setupButtons();
  },

  setupButtons() {
    const next = document.getElementById('tf-mh-next');
    const prev = document.getElementById('tf-mh-prev');
    const reset = document.getElementById('tf-mh-reset');
    if (next) next.onclick = () => { if (this.step < this.maxSteps) { this.step++; this.render(); }};
    if (prev) prev.onclick = () => { if (this.step > 0) { this.step--; this.render(); }};
    if (reset) reset.onclick = () => { this.step = 0; this.render(); };
  },

  render() {
    const container = document.getElementById('tf-multihead-content');
    if (!container) return;
    const steps = [this.step0(), this.step1(), this.step2(), this.step3(), this.step4()];
    container.innerHTML = steps[this.step];
    const ind = document.getElementById('tf-mh-step');
    if (ind) ind.textContent = `${this.step + 1} / ${this.maxSteps + 1}`;
  },

  step0() {
    return `<div class="tf-step-card">
      <div class="tf-step-title">🔀 لماذا عدة رؤوس انتباه؟</div>
      <div class="tf-step-desc">
        رأس انتباه واحد يركز على نوع واحد من العلاقات فقط. باستخدام عدة رؤوس، يمكن للنموذج التركيز على أنماط مختلفة:
      </div>
      <div class="tf-heads-visual">
        <div class="tf-head-card" style="border-color: #FF6584;">
          <div class="tf-head-num">Head 1</div>
          <div class="tf-head-desc">قد يركز على العلاقات النحوية (فاعل ← فعل)</div>
        </div>
        <div class="tf-head-card" style="border-color: #4ECDC4;">
          <div class="tf-head-num">Head 2</div>
          <div class="tf-head-desc">قد يركز على العلاقات الدلالية (صفة ← موصوف)</div>
        </div>
        <div class="tf-head-card" style="border-color: #FFD93D;">
          <div class="tf-head-num">Head 3</div>
          <div class="tf-head-desc">قد يركز على الموضع القريب (الكلمة المجاورة)</div>
        </div>
        <div class="tf-head-card" style="border-color: #6C63FF;">
          <div class="tf-head-num">Head 8</div>
          <div class="tf-head-desc">قد يركز على المراجع البعيدة (ضمائر)</div>
        </div>
      </div>
      <div class="tf-info-box"><i class="fas fa-brain"></i> في GPT-3: 96 رأس انتباه! كل رأس يتعلم نمط مختلف تلقائياً.</div>
    </div>`;
  },

  step1() {
    return `<div class="tf-step-card">
      <div class="tf-step-title">✂️ الخطوة 1: تقسيم المتجهات</div>
      <div class="tf-step-desc">نقسم d_model على عدد الرؤوس. إذا d_model=512 و heads=8:</div>
      <div class="tf-formula">d<sub>k</sub> = d<sub>model</sub> / h = 512 / 8 = 64</div>
      <div class="tf-split-visual">
        <div class="tf-split-source">
          <div class="tf-split-label">المتجه الأصلي (512)</div>
          <div class="tf-split-bar full"></div>
        </div>
        <div class="tf-split-arrow">✂️ →</div>
        <div class="tf-split-parts">
          ${Array.from({length: 4}, (_, i) => `
            <div class="tf-split-part" style="background: hsl(${i*60 + 200}, 70%, 50%);">
              Head ${i+1} (64)
            </div>
          `).join('')}
          <div class="tf-split-dots">... × 8</div>
        </div>
      </div>
      <div class="tf-info-box"><i class="fas fa-info-circle"></i> كل رأس يحسب Attention بشكل مستقل على جزء مختلف من الأبعاد.</div>
    </div>`;
  },

  step2() {
    return `<div class="tf-step-card">
      <div class="tf-step-title">🧮 الخطوة 2: كل رأس يحسب Attention</div>
      <div class="tf-step-desc">كل رأس يطبق Self-Attention بشكل مستقل:</div>
      <div class="tf-parallel-heads">
        ${Array.from({length: this.numHeads}, (_, i) => {
          const colors = ['#FF6584', '#4ECDC4'];
          const weights = TFUtils.randomMatrix(4, 4, 1).map(row => TFUtils.softmax(row));
          return `<div class="tf-head-computation" style="border-color: ${colors[i]}">
            <div class="tf-head-title" style="color: ${colors[i]}">Head ${i+1}</div>
            <table class="tf-matrix mini-matrix">
              <tr>${['I','love','deep','learn'].map(w => `<th>${w}</th>`).join('')}</tr>
              ${weights.map((row, r) => `<tr>${row.map(v => {
                const bg = `rgba(${i===0?'255,101,132':'78,205,196'}, ${v * 0.8})`;
                return `<td style="background:${bg}; padding:4px; font-size:10px;">${v.toFixed(2)}</td>`;
              }).join('')}</tr>`).join('')}
            </table>
            <div class="tf-head-focus">يركز على: ${i===0?'العلاقات القريبة':'العلاقات البعيدة'}</div>
          </div>`;
        }).join('')}
      </div>
    </div>`;
  },

  step3() {
    return `<div class="tf-step-card">
      <div class="tf-step-title">🔗 الخطوة 3: دمج النتائج (Concatenate + Linear)</div>
      <div class="tf-formula">MultiHead(Q,K,V) = Concat(head<sub>1</sub>, ..., head<sub>h</sub>) · W<sup>O</sup></div>
      <div class="tf-concat-visual">
        <div class="tf-concat-parts">
          ${Array.from({length: 4}, (_, i) => `
            <div class="tf-concat-part" style="background: hsl(${i*60 + 200}, 70%, 40%);">H${i+1}</div>
          `).join('')}
        </div>
        <div class="tf-concat-arrow">Concat →</div>
        <div class="tf-concat-result">
          <div class="tf-concat-full">المتجه المدمج (512)</div>
        </div>
        <div class="tf-concat-arrow">× W<sup>O</sup> →</div>
        <div class="tf-concat-result">
          <div class="tf-concat-final">المخرج النهائي (512)</div>
        </div>
      </div>
      <div class="tf-info-box"><i class="fas fa-info-circle"></i> نضرب بمصفوفة W<sup>O</sup> لدمج "وجهات النظر" المختلفة في تمثيل واحد متماسك.</div>
    </div>`;
  },

  step4() {
    return `<div class="tf-step-card">
      <div class="tf-step-title">🎯 ملخص Multi-Head Attention</div>
      <div class="tf-summary-flow">
        <div class="tf-flow-item"><span class="tf-flow-icon">📊</span><span>X</span></div>
        <div class="tf-flow-arrow">→</div>
        <div class="tf-flow-item" style="border-color:#FF6584"><span class="tf-flow-icon">✂️</span><span>Split</span></div>
        <div class="tf-flow-arrow">→</div>
        <div class="tf-flow-item" style="border-color:${TFUtils.colors.attention}"><span class="tf-flow-icon">🎯</span><span>h × Attention</span></div>
        <div class="tf-flow-arrow">→</div>
        <div class="tf-flow-item"><span class="tf-flow-icon">🔗</span><span>Concat</span></div>
        <div class="tf-flow-arrow">→</div>
        <div class="tf-flow-item highlight"><span class="tf-flow-icon">✨</span><span>W<sup>O</sup></span></div>
      </div>
      <div class="tf-info-box success"><i class="fas fa-arrow-left"></i> انتقل للتبويب التالي <strong>"Feed Forward"</strong>!</div>
    </div>`;
  }
};

/* ============================================
   5. FFN VIEW
   ============================================ */
const FFNView = {
  step: 0,
  maxSteps: 3,

  init() {
    this.step = 0;
    this.render();
    this.setupButtons();
  },

  setupButtons() {
    const next = document.getElementById('tf-ffn-next');
    const prev = document.getElementById('tf-ffn-prev');
    const reset = document.getElementById('tf-ffn-reset');
    if (next) next.onclick = () => { if (this.step < this.maxSteps) { this.step++; this.render(); }};
    if (prev) prev.onclick = () => { if (this.step > 0) { this.step--; this.render(); }};
    if (reset) reset.onclick = () => { this.step = 0; this.render(); };
  },

  render() {
    const container = document.getElementById('tf-ffn-content');
    if (!container) return;
    const steps = [this.step0(), this.step1(), this.step2(), this.step3()];
    container.innerHTML = steps[this.step];
    const ind = document.getElementById('tf-ffn-step');
    if (ind) ind.textContent = `${this.step + 1} / ${this.maxSteps + 1}`;
  },

  step0() {
    return `<div class="tf-step-card">
      <div class="tf-step-title">⚡ شبكة Feed Forward</div>
      <div class="tf-step-desc">بعد Multi-Head Attention، كل كلمة تمر عبر شبكة Feed Forward مستقلة ومتطابقة:</div>
      <div class="tf-formula">FFN(x) = max(0, x·W<sub>1</sub> + b<sub>1</sub>) · W<sub>2</sub> + b<sub>2</sub></div>
      <div class="tf-ffn-visual">
        <div class="tf-ffn-layer">
          <div class="tf-ffn-label">Input (d=512)</div>
          <div class="tf-ffn-nodes">${Array.from({length: 6}, () => '<div class="tf-ffn-node small"></div>').join('')}</div>
        </div>
        <div class="tf-ffn-connections">→ Linear →</div>
        <div class="tf-ffn-layer expand">
          <div class="tf-ffn-label">Hidden (d=2048)</div>
          <div class="tf-ffn-nodes">${Array.from({length: 10}, () => '<div class="tf-ffn-node medium"></div>').join('')}</div>
          <div class="tf-ffn-activation">ReLU</div>
        </div>
        <div class="tf-ffn-connections">→ Linear →</div>
        <div class="tf-ffn-layer">
          <div class="tf-ffn-label">Output (d=512)</div>
          <div class="tf-ffn-nodes">${Array.from({length: 6}, () => '<div class="tf-ffn-node small"></div>').join('')}</div>
        </div>
      </div>
      <div class="tf-info-box"><i class="fas fa-info-circle"></i> الطبقة الوسطى أكبر 4x من المدخل (512→2048→512). هذا يعطي النموذج "مساحة تفكير" أوسع.</div>
    </div>`;
  },

  step1() {
    const input = [0.5, -0.3, 0.8, -0.1];
    const W1 = TFUtils.randomMatrix(4, 8, 0.5);
    const hidden = input.map((_, j) => {
      let sum = 0;
      for (let k = 0; k < 4; k++) sum += input[k] * W1[k][j % 8];
      return TFUtils.relu(sum);
    });
    return `<div class="tf-step-card">
      <div class="tf-step-title">🧮 الخطوة 1: الطبقة الأولى + ReLU</div>
      <div class="tf-step-desc">نضرب المدخل بـ W<sub>1</sub> ونضيف bias ثم نطبق ReLU:</div>
      <div class="tf-computation">
        <div class="tf-comp-row">
          <span class="tf-comp-label">Input x:</span>
          <span class="tf-comp-values">[${input.join(', ')}]</span>
        </div>
        <div class="tf-comp-row">
          <span class="tf-comp-label">بعد x·W₁+b₁:</span>
          <span class="tf-comp-values">[${hidden.map(v => v.toFixed(3)).join(', ')}]</span>
        </div>
        <div class="tf-comp-row">
          <span class="tf-comp-label">بعد ReLU:</span>
          <span class="tf-comp-values">[${hidden.map(v => Math.max(0, v).toFixed(3)).join(', ')}]</span>
        </div>
      </div>
      <div class="tf-info-box"><i class="fas fa-lightbulb"></i> ReLU(x) = max(0, x) — تحول القيم السالبة إلى صفر.</div>
    </div>`;
  },

  step2() {
    return `<div class="tf-step-card">
      <div class="tf-step-title">🧮 الخطوة 2: الطبقة الثانية (الإسقاط)</div>
      <div class="tf-step-desc">نضرب الناتج بـ W<sub>2</sub> لإرجاع الحجم إلى d_model:</div>
      <div class="tf-formula">Output = ReLU(x·W<sub>1</sub> + b<sub>1</sub>) · W<sub>2</sub> + b<sub>2</sub></div>
      <div class="tf-dimension-flow">
        <div class="tf-dim-box">512</div>
        <div class="tf-dim-arrow">→ W₁ →</div>
        <div class="tf-dim-box big">2048</div>
        <div class="tf-dim-arrow">→ ReLU →</div>
        <div class="tf-dim-box big">2048</div>
        <div class="tf-dim-arrow">→ W₂ →</div>
        <div class="tf-dim-box">512</div>
      </div>
      <div class="tf-info-box"><i class="fas fa-info-circle"></i> هذه الشبكة تُطبّق على كل كلمة بشكل مستقل (نفس الأوزان لكل الكلمات).</div>
    </div>`;
  },

  step3() {
    return `<div class="tf-step-card">
      <div class="tf-step-title">🎯 لماذا FFN مهم؟</div>
      <div class="tf-step-desc">
        <ul class="tf-feature-list">
          <li>🔹 Self-Attention يمزج المعلومات بين الكلمات</li>
          <li>🔹 FFN يعالج المعلومات داخل كل كلمة</li>
          <li>🔹 التوسيع (512→2048) يعطي "قدرة تمثيلية" أعلى</li>
          <li>🔹 بدون FFN، الشبكة ستكون مجرد عمليات خطية!</li>
        </ul>
      </div>
      <div class="tf-info-box success"><i class="fas fa-arrow-left"></i> انتقل لتبويب <strong>"Add & Norm"</strong>!</div>
    </div>`;
  }
};

/* ============================================
   6. ADD & NORM VIEW
   ============================================ */
const AddNormView = {
  step: 0,
  maxSteps: 3,

  init() { this.step = 0; this.render(); this.setupButtons(); },

  setupButtons() {
    const next = document.getElementById('tf-an-next');
    const prev = document.getElementById('tf-an-prev');
    const reset = document.getElementById('tf-an-reset');
    if (next) next.onclick = () => { if (this.step < this.maxSteps) { this.step++; this.render(); }};
    if (prev) prev.onclick = () => { if (this.step > 0) { this.step--; this.render(); }};
    if (reset) reset.onclick = () => { this.step = 0; this.render(); };
  },

  render() {
    const container = document.getElementById('tf-addnorm-content');
    if (!container) return;
    const steps = [this.step0(), this.step1(), this.step2(), this.step3()];
    container.innerHTML = steps[this.step];
    const ind = document.getElementById('tf-an-step');
    if (ind) ind.textContent = `${this.step + 1} / ${this.maxSteps + 1}`;
  },

  step0() {
    return `<div class="tf-step-card">
      <div class="tf-step-title">🔄 Residual Connection + Layer Normalization</div>
      <div class="tf-step-desc">بعد كل طبقة فرعية (Attention أو FFN)، نطبق عمليتين:</div>
      <div class="tf-formula">Output = LayerNorm(x + Sublayer(x))</div>
      <div class="tf-residual-visual">
        <div class="tf-res-input">x (المدخل)</div>
        <div class="tf-res-split">
          <div class="tf-res-path main">
            <div class="tf-res-box">Sublayer (Attention أو FFN)</div>
            <div class="tf-res-result">Sublayer(x)</div>
          </div>
          <div class="tf-res-path skip">
            <div class="tf-res-skip-line">Skip Connection ↓</div>
          </div>
        </div>
        <div class="tf-res-add">➕ Add</div>
        <div class="tf-res-norm">LayerNorm</div>
        <div class="tf-res-output">المخرج</div>
      </div>
    </div>`;
  },

  step1() {
    const x = [1.0, -0.5, 0.3, 0.8];
    const sub = [0.2, 0.7, -0.1, 0.4];
    const added = x.map((v,i) => +(v + sub[i]).toFixed(3));
    return `<div class="tf-step-card">
      <div class="tf-step-title">➕ الخطوة 1: Residual Connection (الجمع)</div>
      <div class="tf-step-desc">نجمع المدخل الأصلي مع مخرج الطبقة الفرعية:</div>
      <div class="tf-computation">
        <div class="tf-comp-row"><span class="tf-comp-label">x (المدخل):</span><span class="tf-comp-values">[${x.join(', ')}]</span></div>
        <div class="tf-comp-row"><span class="tf-comp-label">Sublayer(x):</span><span class="tf-comp-values">[${sub.join(', ')}]</span></div>
        <div class="tf-comp-row result"><span class="tf-comp-label">x + Sublayer(x):</span><span class="tf-comp-values">[${added.join(', ')}]</span></div>
      </div>
      <div class="tf-info-box"><i class="fas fa-lightbulb"></i> <strong>لماذا؟</strong> يحل مشكلة "اختفاء التدرج" في الشبكات العميقة. المعلومات تتدفق مباشرة!</div>
    </div>`;
  },

  step2() {
    const input = [1.2, 0.2, 0.2, 1.2];
    const normed = TFUtils.layerNorm(input);
    const mean = input.reduce((a,b) => a+b, 0) / input.length;
    const variance = input.reduce((a,b) => a + (b-mean)**2, 0) / input.length;
    return `<div class="tf-step-card">
      <div class="tf-step-title">📏 الخطوة 2: Layer Normalization</div>
      <div class="tf-step-desc">نطبع القيم لتكون بمتوسط ≈ 0 وانحراف معياري ≈ 1:</div>
      <div class="tf-formula">LN(x) = (x - μ) / √(σ² + ε) · γ + β</div>
      <div class="tf-computation">
        <div class="tf-comp-row"><span class="tf-comp-label">المدخل:</span><span class="tf-comp-values">[${input.join(', ')}]</span></div>
        <div class="tf-comp-row"><span class="tf-comp-label">المتوسط μ:</span><span class="tf-comp-values">${mean.toFixed(3)}</span></div>
        <div class="tf-comp-row"><span class="tf-comp-label">التباين σ²:</span><span class="tf-comp-values">${variance.toFixed(3)}</span></div>
        <div class="tf-comp-row result"><span class="tf-comp-label">بعد التطبيع:</span><span class="tf-comp-values">[${normed.map(v => v.toFixed(3)).join(', ')}]</span></div>
      </div>
      <div class="tf-info-box"><i class="fas fa-info-circle"></i> γ و β هي معاملات قابلة للتعلم تسمح للنموذج بضبط التطبيع حسب الحاجة.</div>
    </div>`;
  },

  step3() {
    return `<div class="tf-step-card">
      <div class="tf-step-title">🎯 لماذا Add & Norm ضروريان؟</div>
      <div class="tf-feature-list">
        <div class="tf-feature"><span class="tf-feature-icon">🔗</span><strong>Residual:</strong> يمنع فقدان المعلومات في الشبكات العميقة</div>
        <div class="tf-feature"><span class="tf-feature-icon">📏</span><strong>LayerNorm:</strong> يثبت التدريب ويسرّعه</div>
        <div class="tf-feature"><span class="tf-feature-icon">📊</span><strong>معاً:</strong> يسمحان ببناء 96+ طبقة بدون مشاكل!</div>
      </div>
      <div class="tf-info-box success"><i class="fas fa-arrow-left"></i> انتقل لتبويب <strong>"Decoder"</strong>!</div>
    </div>`;
  }
};

/* ============================================
   7. DECODER VIEW
   ============================================ */
const DecoderView = {
  step: 0,
  maxSteps: 4,

  init() { this.step = 0; this.render(); this.setupButtons(); },

  setupButtons() {
    const next = document.getElementById('tf-dec-next');
    const prev = document.getElementById('tf-dec-prev');
    const reset = document.getElementById('tf-dec-reset');
    if (next) next.onclick = () => { if (this.step < this.maxSteps) { this.step++; this.render(); }};
    if (prev) prev.onclick = () => { if (this.step > 0) { this.step--; this.render(); }};
    if (reset) reset.onclick = () => { this.step = 0; this.render(); };
  },

  render() {
    const container = document.getElementById('tf-decoder-content');
    if (!container) return;
    const steps = [this.step0(), this.step1(), this.step2(), this.step3(), this.step4()];
    container.innerHTML = steps[this.step];
    const ind = document.getElementById('tf-dec-step');
    if (ind) ind.textContent = `${this.step + 1} / ${this.maxSteps + 1}`;
  },

  step0() {
    return `<div class="tf-step-card">
      <div class="tf-step-title">🎭 Decoder - فكّ التشفير</div>
      <div class="tf-step-desc">Decoder مشابه لـ Encoder لكن مع إضافات مهمة:</div>
      <div class="tf-decoder-structure">
        <div class="tf-dec-block">
          <div class="tf-dec-layer masked">Masked Self-Attention</div>
          <div class="tf-dec-layer addnorm">Add & Norm</div>
          <div class="tf-dec-layer cross">Cross-Attention</div>
          <div class="tf-dec-layer addnorm">Add & Norm</div>
          <div class="tf-dec-layer ffn">Feed Forward</div>
          <div class="tf-dec-layer addnorm">Add & Norm</div>
        </div>
      </div>
      <div class="tf-info-box"><i class="fas fa-info-circle"></i> الفرق الرئيسي: <strong>Masked Attention</strong> (يمنع النظر للمستقبل) و <strong>Cross-Attention</strong> (يربط بالـ Encoder).</div>
    </div>`;
  },

  step1() {
    return `<div class="tf-step-card">
      <div class="tf-step-title">🎭 Masked Self-Attention</div>
      <div class="tf-step-desc">أثناء التوليد، الكلمة لا يمكنها "رؤية" الكلمات التي لم تُولّد بعد:</div>
      <div class="tf-matrix-display">
        <table class="tf-matrix mask-matrix">
          <tr><th></th><th>كلمة1</th><th>كلمة2</th><th>كلمة3</th><th>كلمة4</th></tr>
          <tr><td>كلمة1</td><td class="tf-visible">✓</td><td class="tf-masked">✗</td><td class="tf-masked">✗</td><td class="tf-masked">✗</td></tr>
          <tr><td>كلمة2</td><td class="tf-visible">✓</td><td class="tf-visible">✓</td><td class="tf-masked">✗</td><td class="tf-masked">✗</td></tr>
          <tr><td>كلمة3</td><td class="tf-visible">✓</td><td class="tf-visible">✓</td><td class="tf-visible">✓</td><td class="tf-masked">✗</td></tr>
          <tr><td>كلمة4</td><td class="tf-visible">✓</td><td class="tf-visible">✓</td><td class="tf-visible">✓</td><td class="tf-visible">✓</td></tr>
        </table>
      </div>
      <div class="tf-info-box"><i class="fas fa-lightbulb"></i> القناع (Mask) يجعل الـ scores = -∞ للمواضع المستقبلية، فبعد Softmax تصبح = 0.</div>
    </div>`;
  },

  step2() {
    return `<div class="tf-step-card">
      <div class="tf-step-title">🔗 Cross-Attention (انتباه متقاطع)</div>
      <div class="tf-step-desc">هنا يأتي الربط بين Encoder و Decoder:</div>
      <div class="tf-cross-visual">
        <div class="tf-cross-side encoder-side">
          <div class="tf-cross-label">Encoder Output</div>
          <div class="tf-cross-provides">يعطي: <strong>K</strong> و <strong>V</strong></div>
        </div>
        <div class="tf-cross-arrow">⟶</div>
        <div class="tf-cross-side decoder-side">
          <div class="tf-cross-label">Decoder</div>
          <div class="tf-cross-provides">يعطي: <strong>Q</strong></div>
        </div>
      </div>
      <div class="tf-formula">CrossAttention(Q<sub>dec</sub>, K<sub>enc</sub>, V<sub>enc</sub>)</div>
      <div class="tf-info-box"><i class="fas fa-lightbulb"></i> Decoder يسأل (Q) و Encoder يجيب (K,V). مثل: "أي أجزاء الجملة المصدر مهمة لتوليد الكلمة التالية؟"</div>
    </div>`;
  },

  step3() {
    return `<div class="tf-step-card">
      <div class="tf-step-title">🔮 التوليد التسلسلي (Autoregressive)</div>
      <div class="tf-step-desc">Decoder يولد كلمة واحدة في كل خطوة:</div>
      <div class="tf-generation-steps">
        <div class="tf-gen-step">
          <div class="tf-gen-input">[START]</div>
          <div class="tf-gen-arrow">→ Decoder →</div>
          <div class="tf-gen-output">مرحبا</div>
        </div>
        <div class="tf-gen-step">
          <div class="tf-gen-input">[START] مرحبا</div>
          <div class="tf-gen-arrow">→ Decoder →</div>
          <div class="tf-gen-output">كيف</div>
        </div>
        <div class="tf-gen-step">
          <div class="tf-gen-input">[START] مرحبا كيف</div>
          <div class="tf-gen-arrow">→ Decoder →</div>
          <div class="tf-gen-output">حالك</div>
        </div>
        <div class="tf-gen-step">
          <div class="tf-gen-input">[START] مرحبا كيف حالك</div>
          <div class="tf-gen-arrow">→ Decoder →</div>
          <div class="tf-gen-output">[END]</div>
        </div>
      </div>
      <div class="tf-info-box"><i class="fas fa-info-circle"></i> كل كلمة جديدة تُضاف للمدخل في الخطوة التالية. هذا ما يسمى "Autoregressive Generation".</div>
    </div>`;
  },

  step4() {
    return `<div class="tf-step-card">
      <div class="tf-step-title">🎯 ملخص Decoder</div>
      <div class="tf-feature-list">
        <div class="tf-feature"><span class="tf-feature-icon">🎭</span><strong>Masked Self-Attention:</strong> يمنع الغش (لا يرى المستقبل)</div>
        <div class="tf-feature"><span class="tf-feature-icon">🔗</span><strong>Cross-Attention:</strong> يربط المخرج بالمدخل الأصلي</div>
        <div class="tf-feature"><span class="tf-feature-icon">🔮</span><strong>Autoregressive:</strong> يولد كلمة تلو الأخرى</div>
        <div class="tf-feature"><span class="tf-feature-icon">📊</span><strong>Linear + Softmax:</strong> يحول المتجه لاحتمالات الكلمة التالية</div>
      </div>
      <div class="tf-info-box success"><i class="fas fa-arrow-left"></i> انتقل لتبويب <strong>"التدريب الكامل"</strong> لرؤية كل شيء يعمل معاً!</div>
    </div>`;
  }
};

/* ============================================
   8. TRAINING VIEW
   ============================================ */
const TrainingView = {
  step: 0,
  maxSteps: 5,
  isTraining: false,
  epoch: 0,
  loss: 2.5,
  lossHistory: [],

  init() { this.step = 0; this.lossHistory = []; this.render(); this.setupButtons(); },

  setupButtons() {
    const next = document.getElementById('tf-train-next');
    const prev = document.getElementById('tf-train-prev');
    const reset = document.getElementById('tf-train-reset');
    const runBtn = document.getElementById('tf-train-run');
    if (next) next.onclick = () => { if (this.step < this.maxSteps) { this.step++; this.render(); }};
    if (prev) prev.onclick = () => { if (this.step > 0) { this.step--; this.render(); }};
    if (reset) reset.onclick = () => { this.step = 0; this.epoch = 0; this.loss = 2.5; this.lossHistory = []; this.isTraining = false; this.render(); };
    if (runBtn) runBtn.onclick = () => this.runTraining();
  },

  runTraining() {
    if (this.isTraining) return;
    this.isTraining = true;
    this.epoch = 0;
    this.loss = 2.5;
    this.lossHistory = [];
    this.step = 4;
    this.render();
    
    const interval = setInterval(() => {
      this.epoch++;
      this.loss = 2.5 * Math.exp(-this.epoch * 0.08) + 0.05 + (Math.random() - 0.5) * 0.05;
      this.lossHistory.push(this.loss);
      this.render();
      if (this.epoch >= 50) {
        clearInterval(interval);
        this.isTraining = false;
        this.step = 5;
        this.render();
      }
    }, 100);
  },

  render() {
    const container = document.getElementById('tf-training-content');
    if (!container) return;
    const steps = [this.step0(), this.step1(), this.step2(), this.step3(), this.step4(), this.step5()];
    container.innerHTML = steps[this.step];
    const ind = document.getElementById('tf-train-step');
    if (ind) ind.textContent = `${this.step + 1} / ${this.maxSteps + 1}`;
    
    if (this.step === 4 && this.lossHistory.length > 0) {
      this.drawLossChart();
    }
  },

  step0() {
    return `<div class="tf-step-card">
      <div class="tf-step-title">🚀 تدريب Transformer الكامل</div>
      <div class="tf-step-desc">مثال: تدريب نموذج ترجمة بسيط (English → Arabic)</div>
      <div class="tf-training-example">
        <div class="tf-train-pair">
          <div class="tf-train-src">Hello World</div>
          <div class="tf-train-arrow">→</div>
          <div class="tf-train-tgt">مرحبا بالعالم</div>
        </div>
        <div class="tf-train-pair">
          <div class="tf-train-src">I love learning</div>
          <div class="tf-train-arrow">→</div>
          <div class="tf-train-tgt">أحب التعلم</div>
        </div>
        <div class="tf-train-pair">
          <div class="tf-train-src">Deep networks</div>
          <div class="tf-train-arrow">→</div>
          <div class="tf-train-tgt">شبكات عميقة</div>
        </div>
      </div>
    </div>`;
  },

  step1() {
    return `<div class="tf-step-card">
      <div class="tf-step-title">📋 خطوات التدريب</div>
      <div class="tf-training-pipeline">
        <div class="tf-pipe-step"><span class="tf-pipe-num">1</span> المدخل يمر عبر Encoder</div>
        <div class="tf-pipe-step"><span class="tf-pipe-num">2</span> الترجمة المتوقعة تمر عبر Decoder</div>
        <div class="tf-pipe-step"><span class="tf-pipe-num">3</span> Decoder يحاول توقع الكلمة التالية</div>
        <div class="tf-pipe-step"><span class="tf-pipe-num">4</span> نقارن التوقع بالإجابة الصحيحة (Cross-Entropy Loss)</div>
        <div class="tf-pipe-step"><span class="tf-pipe-num">5</span> Backpropagation لتحديث الأوزان</div>
        <div class="tf-pipe-step"><span class="tf-pipe-num">6</span> نكرر لآلاف/ملايين المرات!</div>
      </div>
    </div>`;
  },

  step2() {
    const probs = TFUtils.softmax([2.1, 0.5, 3.8, 0.1, 1.2]);
    const words = ['مرحبا', 'أهلا', 'سلام', 'هلا', 'أخرى'];
    return `<div class="tf-step-card">
      <div class="tf-step-title">📊 Linear + Softmax (طبقة المخرج)</div>
      <div class="tf-step-desc">آخر متجه من Decoder يُحوّل لاحتمالات على كل كلمات القاموس:</div>
      <div class="tf-output-probs">
        ${probs.map((p, i) => `
          <div class="tf-prob-bar">
            <div class="tf-prob-word">${words[i]}</div>
            <div class="tf-prob-fill" style="width: ${p * 100}%; background: hsl(${120 * p}, 70%, 50%);">
              ${(p * 100).toFixed(1)}%
            </div>
          </div>
        `).join('')}
      </div>
      <div class="tf-info-box"><i class="fas fa-info-circle"></i> النموذج يختار الكلمة ذات الاحتمال الأعلى (أو يستخدم عشوائية محكومة - Temperature Sampling).</div>
    </div>`;
  },

  step3() {
    return `<div class="tf-step-card">
      <div class="tf-step-title">📉 دالة الخطأ (Cross-Entropy Loss)</div>
      <div class="tf-formula">L = -Σ y<sub>i</sub> · log(ŷ<sub>i</sub>)</div>
      <div class="tf-step-desc">نقيس الفرق بين توقع النموذج والإجابة الصحيحة:</div>
      <div class="tf-loss-example">
        <div class="tf-loss-row"><span>الكلمة الصحيحة:</span> <span class="tf-correct">"مرحبا"</span></div>
        <div class="tf-loss-row"><span>توقع النموذج:</span> <span>P("مرحبا") = 0.35</span></div>
        <div class="tf-loss-row result"><span>Loss = -log(0.35) =</span> <span class="tf-loss-val">1.049</span></div>
      </div>
      <div class="tf-info-box"><i class="fas fa-lightbulb"></i> كلما زاد احتمال الكلمة الصحيحة، قلّ الخطأ. الهدف: تقليل Loss = تحسين التوقعات!</div>
      <button class="btn btn-primary tf-run-btn" id="tf-train-run"><i class="fas fa-play"></i> شغّل التدريب!</button>
    </div>`;
  },

  step4() {
    return `<div class="tf-step-card">
      <div class="tf-step-title">📊 التدريب قيد التشغيل...</div>
      <div class="tf-training-stats">
        <div class="tf-stat"><span class="tf-stat-label">Epoch:</span> <span class="tf-stat-value">${this.epoch}/50</span></div>
        <div class="tf-stat"><span class="tf-stat-label">Loss:</span> <span class="tf-stat-value">${this.loss.toFixed(4)}</span></div>
      </div>
      <div class="tf-chart-wrapper">
        <canvas id="tf-loss-chart" width="600" height="250"></canvas>
      </div>
      <div class="tf-progress-bar-wrapper">
        <div class="tf-progress-fill" style="width: ${(this.epoch / 50) * 100}%"></div>
      </div>
    </div>`;
  },

  step5() {
    return `<div class="tf-step-card">
      <div class="tf-step-title">🎉 التدريب اكتمل!</div>
      <div class="tf-training-stats">
        <div class="tf-stat"><span class="tf-stat-label">Epochs:</span> <span class="tf-stat-value">50</span></div>
        <div class="tf-stat"><span class="tf-stat-label">Final Loss:</span> <span class="tf-stat-value">${this.loss.toFixed(4)}</span></div>
      </div>
      <div class="tf-results-demo">
        <div class="tf-demo-title">🔮 اختبار النموذج:</div>
        <div class="tf-demo-pair">
          <div class="tf-demo-input">Hello World</div>
          <div class="tf-demo-arrow">→</div>
          <div class="tf-demo-output">مرحبا بالعالم ✓</div>
        </div>
      </div>
      <div class="tf-info-box success">
        <i class="fas fa-trophy"></i>
        أحسنت! لقد فهمت كيف يعمل Transformer من الألف إلى الياء!
        <br>هذه هي نفس البنية وراء GPT, BERT, LLaMA, وكل نماذج اللغة الحديثة!
      </div>
    </div>`;
  },

  drawLossChart() {
    const canvas = document.getElementById('tf-loss-chart');
    if (!canvas || this.lossHistory.length === 0) return;
    const ctx = canvas.getContext('2d');
    const w = canvas.width, h = canvas.height;
    const padding = 40;
    
    ctx.clearRect(0, 0, w, h);
    
    // Background
    ctx.fillStyle = '#0d1117';
    ctx.fillRect(0, 0, w, h);
    
    // Grid
    ctx.strokeStyle = '#1a1f2e';
    ctx.lineWidth = 0.5;
    for (let i = 0; i < 5; i++) {
      const y = padding + (h - 2 * padding) * i / 4;
      ctx.beginPath(); ctx.moveTo(padding, y); ctx.lineTo(w - padding, y); ctx.stroke();
    }
    
    // Loss line
    const maxLoss = 2.8;
    const data = this.lossHistory;
    ctx.beginPath();
    ctx.strokeStyle = '#4ECDC4';
    ctx.lineWidth = 2;
    ctx.shadowColor = '#4ECDC4';
    ctx.shadowBlur = 5;
    
    data.forEach((loss, i) => {
      const x = padding + (w - 2 * padding) * i / 49;
      const y = padding + (h - 2 * padding) * (1 - (maxLoss - loss) / maxLoss);
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    ctx.stroke();
    ctx.shadowBlur = 0;
    
    // Labels
    ctx.fillStyle = '#888';
    ctx.font = '11px JetBrains Mono';
    ctx.textAlign = 'center';
    ctx.fillText('Epoch', w/2, h - 5);
    ctx.save();
    ctx.translate(12, h/2);
    ctx.rotate(-Math.PI/2);
    ctx.fillText('Loss', 0, 0);
    ctx.restore();
  }
};
