/* ============================================
   CNN Architectures - Pre-built Famous Architectures
   ============================================ */

class CNNArchitectures {
  constructor(modalId) {
    this.modal = document.getElementById(modalId);
    this.currentView = 'list'; // 'list' or 'detail'
    this.architectures = this.getArchitecturesData();
  }

  // ── Open / Close Modal ──
  open() {
    if (!this.modal) return;
    this.modal.classList.add('active');
    document.body.style.overflow = 'hidden';
    this.showList();
  }

  close() {
    if (!this.modal) return;
    this.modal.classList.remove('active');
    document.body.style.overflow = '';
  }

  // ── Show Architectures List ──
  showList() {
    this.currentView = 'list';
    const container = this.modal.querySelector('.arch-content');
    if (!container) return;

    let html = `
      <div class="arch-list-header">
        <div class="arch-list-title">
          <i class="fas fa-cubes" style="color: var(--accent)"></i>
          اختر هيكل لاستعراضه
        </div>
        <p class="arch-list-desc">أشهر هياكل الشبكات التلافيفية التي غيرت عالم الذكاء الاصطناعي والرؤية الحاسوبية</p>
      </div>
      <div class="arch-grid">
    `;

    this.architectures.forEach((arch, idx) => {
      html += `
        <div class="arch-card" data-arch-idx="${idx}" onclick="window._cnnArchitectures.showDetail(${idx})">
          <div class="arch-card-accent" style="background: ${arch.color}"></div>
          <div class="arch-card-header">
            <div class="arch-card-icon" style="background: ${arch.color}20; color: ${arch.color}">
              <i class="${arch.icon}"></i>
            </div>
            <div class="arch-card-year">${arch.year}</div>
          </div>
          <h3 class="arch-card-name">${arch.name}</h3>
          <div class="arch-card-subtitle">${arch.subtitle}</div>
          <p class="arch-card-desc">${arch.shortDesc}</p>
          <div class="arch-card-stats">
            <div class="arch-card-stat">
              <span class="arch-stat-value">${arch.params}</span>
              <span class="arch-stat-label">معاملات</span>
            </div>
            <div class="arch-card-stat">
              <span class="arch-stat-value">${arch.accuracy}</span>
              <span class="arch-stat-label">دقة Top-5</span>
            </div>
            <div class="arch-card-stat">
              <span class="arch-stat-value">${arch.layers}</span>
              <span class="arch-stat-label">طبقات</span>
            </div>
          </div>
          <div class="arch-card-arrow">
            <i class="fas fa-arrow-left"></i> تفاصيل
          </div>
        </div>
      `;
    });

    html += `</div>`;

    // Comparison table
    html += this.renderComparisonTable();

    container.innerHTML = html;

    // Animate cards entrance
    requestAnimationFrame(() => {
      container.querySelectorAll('.arch-card').forEach((card, i) => {
        card.style.animationDelay = `${i * 0.08}s`;
        card.classList.add('arch-card-enter');
      });
    });
  }

  // ── Show Architecture Detail ──
  showDetail(idx) {
    this.currentView = 'detail';
    const arch = this.architectures[idx];
    const container = this.modal.querySelector('.arch-content');
    if (!container || !arch) return;

    let html = `
      <div class="arch-detail">
        <!-- Back Button -->
        <button class="arch-back-btn" onclick="window._cnnArchitectures.showList()">
          <i class="fas fa-arrow-right"></i> العودة للقائمة
        </button>

        <!-- Hero Header -->
        <div class="arch-detail-hero" style="border-color: ${arch.color}">
          <div class="arch-detail-hero-accent" style="background: linear-gradient(135deg, ${arch.color}30, transparent)"></div>
          <div class="arch-detail-hero-content">
            <div class="arch-detail-icon" style="background: ${arch.color}25; color: ${arch.color}; border: 2px solid ${arch.color}40">
              <i class="${arch.icon}" style="font-size: 2.5rem"></i>
            </div>
            <div class="arch-detail-hero-info">
              <div class="arch-detail-year-badge" style="background: ${arch.color}20; color: ${arch.color}">${arch.year}</div>
              <h2 class="arch-detail-name">${arch.name}</h2>
              <div class="arch-detail-subtitle">${arch.subtitle}</div>
              <div class="arch-detail-authors"><i class="fas fa-users"></i> ${arch.authors}</div>
              <div class="arch-detail-paper"><i class="fas fa-file-alt"></i> ${arch.paper}</div>
            </div>
          </div>
          <div class="arch-detail-quick-stats">
            <div class="arch-quick-stat">
              <div class="arch-quick-stat-value" style="color: ${arch.color}">${arch.params}</div>
              <div class="arch-quick-stat-label">عدد المعاملات</div>
            </div>
            <div class="arch-quick-stat">
              <div class="arch-quick-stat-value" style="color: ${arch.color}">${arch.accuracy}</div>
              <div class="arch-quick-stat-label">دقة Top-5</div>
            </div>
            <div class="arch-quick-stat">
              <div class="arch-quick-stat-value" style="color: ${arch.color}">${arch.layers}</div>
              <div class="arch-quick-stat-label">عدد الطبقات</div>
            </div>
            <div class="arch-quick-stat">
              <div class="arch-quick-stat-value" style="color: ${arch.color}">${arch.inputSize}</div>
              <div class="arch-quick-stat-label">حجم الإدخال</div>
            </div>
          </div>
        </div>

        <!-- Description Section -->
        <div class="arch-section">
          <div class="arch-section-title"><i class="fas fa-info-circle" style="color: var(--accent)"></i> نبذة عن الشبكة</div>
          <p class="arch-section-text">${arch.description}</p>
          <div class="arch-key-innovation">
            <div class="arch-innovation-label"><i class="fas fa-lightbulb" style="color: var(--warning)"></i> الابتكار الرئيسي</div>
            <p>${arch.keyInnovation}</p>
          </div>
        </div>

        <!-- Architecture Diagram -->
        <div class="arch-section">
          <div class="arch-section-title"><i class="fas fa-project-diagram" style="color: var(--primary-light)"></i> بنية الشبكة (Architecture)</div>
          ${this.renderLayersDiagram(arch.layerDetails, arch.color)}
        </div>

        <!-- Training Details -->
        <div class="arch-section">
          <div class="arch-section-title"><i class="fas fa-cogs" style="color: var(--success)"></i> تفاصيل التدريب</div>
          <div class="arch-info-grid">
            ${arch.trainingDetails.map(item => `
              <div class="arch-info-item">
                <div class="arch-info-icon"><i class="${item.icon}"></i></div>
                <div class="arch-info-label">${item.label}</div>
                <div class="arch-info-value">${item.value}</div>
              </div>
            `).join('')}
          </div>
        </div>

        <!-- Dataset Info -->
        <div class="arch-section">
          <div class="arch-section-title"><i class="fas fa-database" style="color: var(--warning)"></i> بيانات التدريب (Dataset)</div>
          <div class="arch-dataset-cards">
            ${arch.datasets.map(ds => `
              <div class="arch-dataset-card">
                <div class="arch-dataset-name"><i class="fas fa-folder-open" style="color: var(--accent)"></i> ${ds.name}</div>
                <div class="arch-dataset-details">
                  ${ds.details.map(d => `<div class="arch-dataset-detail"><i class="fas fa-chevron-left" style="font-size: 0.6rem; color: var(--primary-light)"></i> ${d}</div>`).join('')}
                </div>
              </div>
            `).join('')}
          </div>
        </div>

        <!-- Use Cases -->
        <div class="arch-section">
          <div class="arch-section-title"><i class="fas fa-rocket" style="color: #FF6B6B"></i> الاستخدامات العملية</div>
          <div class="arch-usecases-grid">
            ${arch.useCases.map(uc => `
              <div class="arch-usecase-item">
                <div class="arch-usecase-icon"><i class="${uc.icon}"></i></div>
                <div class="arch-usecase-text">
                  <div class="arch-usecase-title">${uc.title}</div>
                  <div class="arch-usecase-desc">${uc.desc}</div>
                </div>
              </div>
            `).join('')}
          </div>
        </div>

        <!-- Pros & Cons -->
        <div class="arch-section">
          <div class="arch-section-title"><i class="fas fa-balance-scale" style="color: var(--primary-light)"></i> نقاط القوة والضعف</div>
          <div class="arch-proscons">
            <div class="arch-pros">
              <div class="arch-proscons-title" style="color: var(--success)"><i class="fas fa-check-circle"></i> نقاط القوة</div>
              ${arch.pros.map(p => `<div class="arch-proscons-item"><i class="fas fa-plus" style="color: var(--success)"></i> ${p}</div>`).join('')}
            </div>
            <div class="arch-cons">
              <div class="arch-proscons-title" style="color: var(--error)"><i class="fas fa-times-circle"></i> نقاط الضعف</div>
              ${arch.cons.map(c => `<div class="arch-proscons-item"><i class="fas fa-minus" style="color: var(--error)"></i> ${c}</div>`).join('')}
            </div>
          </div>
        </div>

      </div>
    `;

    container.innerHTML = html;
    container.scrollTop = 0;
  }

  // ── Render Layers Diagram ──
  renderLayersDiagram(layers, color) {
    let html = `<div class="arch-layers-diagram">`;

    layers.forEach((layer, i) => {
      const isLast = i === layers.length - 1;
      html += `
        <div class="arch-layer-block" style="--layer-color: ${layer.color || color}">
          <div class="arch-layer-type">${layer.type}</div>
          <div class="arch-layer-name">${layer.name}</div>
          <div class="arch-layer-details">
            ${layer.details.map(d => `<span class="arch-layer-detail">${d}</span>`).join('')}
          </div>
          ${layer.output ? `<div class="arch-layer-output">الخرج: ${layer.output}</div>` : ''}
        </div>
        ${!isLast ? '<div class="arch-layer-arrow"><i class="fas fa-arrow-down"></i></div>' : ''}
      `;
    });

    html += `</div>`;
    return html;
  }

  // ── Comparison Table ──
  renderComparisonTable() {
    let html = `
      <div class="arch-comparison-section">
        <div class="arch-section-title" style="margin-bottom: 1.5rem">
          <i class="fas fa-chart-bar" style="color: var(--accent)"></i> مقارنة بين الهياكل
        </div>
        <div class="arch-comparison-table-wrapper">
          <table class="arch-comparison-table">
            <thead>
              <tr>
                <th>الهيكل</th>
                <th>السنة</th>
                <th>الطبقات</th>
                <th>المعاملات</th>
                <th>Top-5 %</th>
                <th>حجم الإدخال</th>
                <th>الابتكار الرئيسي</th>
              </tr>
            </thead>
            <tbody>
    `;

    this.architectures.forEach(arch => {
      html += `
        <tr>
          <td><span style="color: ${arch.color}; font-weight: 700">${arch.name}</span></td>
          <td>${arch.year}</td>
          <td>${arch.layers}</td>
          <td>${arch.params}</td>
          <td>${arch.accuracy}</td>
          <td>${arch.inputSize}</td>
          <td style="font-size: 0.8rem">${arch.comparisonNote}</td>
        </tr>
      `;
    });

    html += `</tbody></table></div></div>`;
    return html;
  }

  // ══════════════════════════════════════════════
  // ── ARCHITECTURES DATA ──
  // ══════════════════════════════════════════════
  getArchitecturesData() {
    return [
      // ──────── 1. LeNet-5 ────────
      {
        name: 'LeNet-5',
        subtitle: 'الجد الأول للشبكات التلافيفية',
        year: '1998',
        authors: 'Yann LeCun, Léon Bottou, Yoshua Bengio, Patrick Haffner',
        paper: 'Gradient-Based Learning Applied to Document Recognition',
        icon: 'fas fa-pen-nib',
        color: '#4FC3F7',
        params: '60K',
        accuracy: '99.2%',
        layers: '7',
        inputSize: '32×32',
        shortDesc: 'أول شبكة تلافيفية ناجحة استُخدمت في التعرف على الأرقام المكتوبة بخط اليد.',
        comparisonNote: 'أول CNN ناجحة تجارياً',
        description: 'LeNet-5 هي الشبكة التي أسست مفهوم الشبكات التلافيفية كما نعرفها اليوم. صممها Yann LeCun وفريقه في AT&T Bell Labs للتعرف على الأرقام المكتوبة بخط اليد على الشيكات البنكية. كانت أول شبكة تُثبت أن التعلم المباشر من الصور (end-to-end learning) يمكن أن يتفوق على الطرق التقليدية المعتمدة على استخراج الميزات يدوياً.',
        keyInnovation: 'إدخال مفهوم التلافيف (Convolution) والتجميع (Pooling) كبديل لاستخراج الميزات اليدوي، مما أسس لمجال الرؤية الحاسوبية الحديث بالكامل.',
        layerDetails: [
          { type: 'Input', name: 'صورة الإدخال', details: ['32×32×1', 'Grayscale'], output: '32×32×1', color: '#78909C' },
          { type: 'Conv2D', name: 'C1 - التلافيف الأولى', details: ['6 فلاتر', '5×5', 'stride=1'], output: '28×28×6', color: '#4FC3F7' },
          { type: 'AvgPool', name: 'S2 - التجميع', details: ['2×2', 'stride=2'], output: '14×14×6', color: '#4DD0E1' },
          { type: 'Conv2D', name: 'C3 - التلافيف الثانية', details: ['16 فلتر', '5×5'], output: '10×10×16', color: '#4FC3F7' },
          { type: 'AvgPool', name: 'S4 - التجميع', details: ['2×2', 'stride=2'], output: '5×5×16', color: '#4DD0E1' },
          { type: 'FC', name: 'C5 - طبقة كاملة', details: ['120 عقدة'], output: '120', color: '#7E57C2' },
          { type: 'FC', name: 'F6 - طبقة كاملة', details: ['84 عقدة'], output: '84', color: '#7E57C2' },
          { type: 'Output', name: 'الإخراج', details: ['10 فئات', 'Softmax'], output: '10', color: '#66BB6A' },
        ],
        trainingDetails: [
          { icon: 'fas fa-chart-line', label: 'المحسّن (Optimizer)', value: 'SGD (Stochastic Gradient Descent)' },
          { icon: 'fas fa-tachometer-alt', label: 'معدل التعلم (Learning Rate)', value: '0.01 مع تقليل تدريجي' },
          { icon: 'fas fa-redo', label: 'عدد الحقب (Epochs)', value: '20' },
          { icon: 'fas fa-layer-group', label: 'حجم الدفعة (Batch Size)', value: '1 (Online Learning)' },
          { icon: 'fas fa-bolt', label: 'دالة التنشيط', value: 'Tanh (بدلاً من ReLU)' },
          { icon: 'fas fa-calculator', label: 'دالة الخسارة', value: 'MSE (Mean Squared Error)' },
        ],
        datasets: [
          {
            name: 'MNIST',
            details: ['60,000 صورة تدريب + 10,000 اختبار', 'أرقام مكتوبة بخط اليد (0-9)', 'حجم كل صورة: 28×28 بكسل (مع padding إلى 32×32)', 'Grayscale (قناة واحدة)'],
          }
        ],
        useCases: [
          { icon: 'fas fa-money-check-alt', title: 'قراءة الشيكات البنكية', desc: 'التعرف التلقائي على الأرقام المكتوبة بخط اليد على الشيكات' },
          { icon: 'fas fa-envelope', title: 'فرز البريد', desc: 'قراءة الرمز البريدي (ZIP Code) تلقائياً لتصنيف الرسائل' },
          { icon: 'fas fa-graduation-cap', title: 'التعليم', desc: 'نموذج تعليمي مثالي لفهم أساسيات الشبكات التلافيفية' },
        ],
        pros: ['بسيطة وسهلة الفهم', 'سريعة التدريب جداً', 'أساس جميع شبكات CNN الحديثة', 'مثالية للتعلم والتعليم'],
        cons: ['لا تعمل جيداً مع الصور الكبيرة أو الملونة', 'دقة محدودة مقارنة بالشبكات الحديثة', 'عدد طبقات قليل لا يكفي لاستخراج ميزات معقدة'],
      },

      // ──────── 2. AlexNet ────────
      {
        name: 'AlexNet',
        subtitle: 'الشبكة التي فجّرت ثورة Deep Learning',
        year: '2012',
        authors: 'Alex Krizhevsky, Ilya Sutskever, Geoffrey Hinton',
        paper: 'ImageNet Classification with Deep Convolutional Neural Networks',
        icon: 'fas fa-fire',
        color: '#FF7043',
        params: '60M',
        accuracy: '84.7%',
        layers: '8',
        inputSize: '227×227',
        shortDesc: 'فازت بمسابقة ImageNet 2012 بفارق كبير وأثبتت قوة التعلم العميق.',
        comparisonNote: 'فجرت ثورة Deep Learning',
        description: 'AlexNet هي الشبكة التي غيرت مسار تاريخ الذكاء الاصطناعي. فازت بمسابقة ImageNet Large Scale Visual Recognition Challenge (ILSVRC) عام 2012 بفارق ضخم عن المنافسين (خطأ 15.3% مقابل 26.2% للمركز الثاني). هذا الفوز أثبت للعالم أن الشبكات العصبية العميقة مع GPU يمكنها تحقيق نتائج مذهلة في الرؤية الحاسوبية.',
        keyInnovation: 'استخدام ReLU بدلاً من Tanh/Sigmoid (تسريع التدريب 6 مرات)، Dropout لمنع الإفراط في التعلم، وتدريب الشبكة على GPU لأول مرة.',
        layerDetails: [
          { type: 'Input', name: 'صورة الإدخال', details: ['227×227×3', 'RGB ملونة'], output: '227×227×3', color: '#78909C' },
          { type: 'Conv2D', name: 'Conv1', details: ['96 فلتر', '11×11', 'stride=4'], output: '55×55×96', color: '#FF7043' },
          { type: 'MaxPool', name: 'Pool1', details: ['3×3', 'stride=2'], output: '27×27×96', color: '#FF8A65' },
          { type: 'Conv2D', name: 'Conv2', details: ['256 فلتر', '5×5', 'padding=2'], output: '27×27×256', color: '#FF7043' },
          { type: 'MaxPool', name: 'Pool2', details: ['3×3', 'stride=2'], output: '13×13×256', color: '#FF8A65' },
          { type: 'Conv2D', name: 'Conv3', details: ['384 فلتر', '3×3', 'padding=1'], output: '13×13×384', color: '#FF7043' },
          { type: 'Conv2D', name: 'Conv4', details: ['384 فلتر', '3×3', 'padding=1'], output: '13×13×384', color: '#FF7043' },
          { type: 'Conv2D', name: 'Conv5', details: ['256 فلتر', '3×3', 'padding=1'], output: '13×13×256', color: '#FF7043' },
          { type: 'MaxPool', name: 'Pool5', details: ['3×3', 'stride=2'], output: '6×6×256', color: '#FF8A65' },
          { type: 'FC', name: 'FC6', details: ['4096 عقدة', 'Dropout 0.5'], output: '4096', color: '#7E57C2' },
          { type: 'FC', name: 'FC7', details: ['4096 عقدة', 'Dropout 0.5'], output: '4096', color: '#7E57C2' },
          { type: 'Output', name: 'FC8 - الإخراج', details: ['1000 فئة', 'Softmax'], output: '1000', color: '#66BB6A' },
        ],
        trainingDetails: [
          { icon: 'fas fa-chart-line', label: 'المحسّن (Optimizer)', value: 'SGD with Momentum (0.9)' },
          { icon: 'fas fa-tachometer-alt', label: 'معدل التعلم (Learning Rate)', value: '0.01 مع تقليل 10x كل فترة' },
          { icon: 'fas fa-redo', label: 'عدد الحقب (Epochs)', value: '90' },
          { icon: 'fas fa-layer-group', label: 'حجم الدفعة (Batch Size)', value: '128' },
          { icon: 'fas fa-bolt', label: 'دالة التنشيط', value: 'ReLU (أول استخدام في CNN!)' },
          { icon: 'fas fa-calculator', label: 'دالة الخسارة', value: 'Cross-Entropy' },
          { icon: 'fas fa-microchip', label: 'الأجهزة', value: '2× NVIDIA GTX 580 GPU (3GB)' },
          { icon: 'fas fa-clock', label: 'وقت التدريب', value: '5-6 أيام' },
        ],
        datasets: [
          {
            name: 'ImageNet (ILSVRC-2012)',
            details: ['1.2 مليون صورة تدريب', '150,000 صورة اختبار', '1,000 فئة مختلفة', 'صور ملونة بأحجام متفاوتة (مقصوصة إلى 227×227)'],
          }
        ],
        useCases: [
          { icon: 'fas fa-image', title: 'تصنيف الصور', desc: 'تصنيف الصور إلى 1000 فئة مختلفة من الكائنات' },
          { icon: 'fas fa-search', title: 'استخراج الميزات', desc: 'استخدام الطبقات الوسطى كمستخرج ميزات للمهام الأخرى (Transfer Learning)' },
          { icon: 'fas fa-eye', title: 'الرؤية الحاسوبية', desc: 'أساس لمعظم تطبيقات الرؤية الحاسوبية الحديثة' },
        ],
        pros: ['أول من أثبت قوة Deep Learning عملياً', 'أدخلت ReLU و Dropout للمجال', 'أول تدريب على GPU', 'بداية عصر Deep Learning'],
        cons: ['60 مليون معامل (كثيرة جداً)', 'فلاتر كبيرة (11×11) غير فعّالة', 'بنية غير منتظمة', 'تجاوزتها الشبكات الأحدث بكثير'],
      },

      // ──────── 3. VGG16 ────────
      {
        name: 'VGG16',
        subtitle: 'البساطة والعمق - فلاتر 3×3 فقط',
        year: '2014',
        authors: 'Karen Simonyan, Andrew Zisserman (Oxford)',
        paper: 'Very Deep Convolutional Networks for Large-Scale Image Recognition',
        icon: 'fas fa-bars',
        color: '#AB47BC',
        params: '138M',
        accuracy: '92.7%',
        layers: '16',
        inputSize: '224×224',
        shortDesc: 'أثبتت أن استخدام فلاتر صغيرة (3×3) مع طبقات كثيرة أفضل من الفلاتر الكبيرة.',
        comparisonNote: 'فلاتر 3×3 فقط مع عمق كبير',
        description: 'VGG16 من جامعة Oxford أثبتت مبدأ مهماً: العمق يهم! بدلاً من استخدام فلاتر كبيرة مثل AlexNet (11×11)، استخدمت VGG فلاتر 3×3 فقط مكدسة فوق بعضها. فلترين 3×3 متتاليين يعطيان نفس مجال الرؤية (Receptive Field) كفلتر 5×5 واحد، لكن بمعاملات أقل وعدد أكبر من طبقات التنشيط (ReLU) مما يجعل القرار أكثر تمييزاً.',
        keyInnovation: 'إثبات أن تكديس فلاتر 3×3 صغيرة متعددة أفضل من فلاتر كبيرة، مع بنية بسيطة ومنتظمة يسهل فهمها وتطبيقها.',
        layerDetails: [
          { type: 'Input', name: 'صورة الإدخال', details: ['224×224×3', 'RGB'], output: '224×224×3', color: '#78909C' },
          { type: 'Conv×2', name: 'Block 1', details: ['64 فلتر', '3×3', '×2 طبقات'], output: '224×224×64', color: '#AB47BC' },
          { type: 'MaxPool', name: 'Pool 1', details: ['2×2', 'stride=2'], output: '112×112×64', color: '#CE93D8' },
          { type: 'Conv×2', name: 'Block 2', details: ['128 فلتر', '3×3', '×2 طبقات'], output: '112×112×128', color: '#AB47BC' },
          { type: 'MaxPool', name: 'Pool 2', details: ['2×2'], output: '56×56×128', color: '#CE93D8' },
          { type: 'Conv×3', name: 'Block 3', details: ['256 فلتر', '3×3', '×3 طبقات'], output: '56×56×256', color: '#AB47BC' },
          { type: 'MaxPool', name: 'Pool 3', details: ['2×2'], output: '28×28×256', color: '#CE93D8' },
          { type: 'Conv×3', name: 'Block 4', details: ['512 فلتر', '3×3', '×3 طبقات'], output: '28×28×512', color: '#AB47BC' },
          { type: 'MaxPool', name: 'Pool 4', details: ['2×2'], output: '14×14×512', color: '#CE93D8' },
          { type: 'Conv×3', name: 'Block 5', details: ['512 فلتر', '3×3', '×3 طبقات'], output: '14×14×512', color: '#AB47BC' },
          { type: 'MaxPool', name: 'Pool 5', details: ['2×2'], output: '7×7×512', color: '#CE93D8' },
          { type: 'FC', name: 'FC6', details: ['4096 عقدة', 'Dropout'], output: '4096', color: '#7E57C2' },
          { type: 'FC', name: 'FC7', details: ['4096 عقدة', 'Dropout'], output: '4096', color: '#7E57C2' },
          { type: 'Output', name: 'FC8 - الإخراج', details: ['1000 فئة', 'Softmax'], output: '1000', color: '#66BB6A' },
        ],
        trainingDetails: [
          { icon: 'fas fa-chart-line', label: 'المحسّن', value: 'SGD with Momentum (0.9)' },
          { icon: 'fas fa-tachometer-alt', label: 'معدل التعلم', value: '0.01 → تقليل 10x ثلاث مرات' },
          { icon: 'fas fa-redo', label: 'عدد الحقب', value: '74' },
          { icon: 'fas fa-layer-group', label: 'حجم الدفعة', value: '256' },
          { icon: 'fas fa-bolt', label: 'دالة التنشيط', value: 'ReLU' },
          { icon: 'fas fa-calculator', label: 'دالة الخسارة', value: 'Cross-Entropy' },
          { icon: 'fas fa-balance-scale', label: 'Weight Decay', value: '5×10⁻⁴' },
          { icon: 'fas fa-microchip', label: 'الأجهزة', value: '4× NVIDIA Titan Black' },
          { icon: 'fas fa-clock', label: 'وقت التدريب', value: '2-3 أسابيع' },
        ],
        datasets: [
          {
            name: 'ImageNet (ILSVRC-2014)',
            details: ['1.3 مليون صورة تدريب', '50,000 صورة اختبار', '1,000 فئة', 'Data Augmentation: قص عشوائي، انعكاس أفقي، تغيير الألوان'],
          }
        ],
        useCases: [
          { icon: 'fas fa-exchange-alt', title: 'Transfer Learning', desc: 'من أكثر الشبكات استخداماً كقاعدة لنقل التعلم إلى مهام أخرى' },
          { icon: 'fas fa-paint-brush', title: 'نقل الأنماط الفنية', desc: 'Neural Style Transfer - تحويل الصور لأنماط فنية مختلفة' },
          { icon: 'fas fa-search-plus', title: 'كشف الكائنات', desc: 'قاعدة لكثير من أنظمة كشف الكائنات المبكرة' },
          { icon: 'fas fa-graduation-cap', title: 'البحث الأكاديمي', desc: 'مرجع أساسي في آلاف الأوراق البحثية' },
        ],
        pros: ['بنية بسيطة ومنتظمة جداً', 'سهلة الفهم والتعديل', 'ممتازة لنقل التعلم (Transfer Learning)', 'أثبتت أهمية العمق'],
        cons: ['138 مليون معامل (ثقيلة جداً)', 'بطيئة في التدريب والاستدلال', 'تستهلك ذاكرة كبيرة', 'الطبقات FC تحتل 90% من المعاملات'],
      },

      // ──────── 4. GoogLeNet / Inception ────────
      {
        name: 'GoogLeNet',
        subtitle: 'وحدات Inception - التوازي الذكي',
        year: '2014',
        authors: 'Christian Szegedy, Wei Liu وفريق Google',
        paper: 'Going Deeper with Convolutions',
        icon: 'fas fa-network-wired',
        color: '#26A69A',
        params: '6.8M',
        accuracy: '93.3%',
        layers: '22',
        inputSize: '224×224',
        shortDesc: 'أدخلت مفهوم وحدات Inception التي تعالج المدخلات بأحجام فلاتر مختلفة بالتوازي.',
        comparisonNote: 'وحدات Inception المتوازية',
        description: 'GoogLeNet (أو Inception v1) من Google أحدثت نقلة نوعية في تصميم الشبكات. بدلاً من اختيار حجم فلتر واحد لكل طبقة، وحدة Inception تستخدم فلاتر 1×1 و 3×3 و 5×5 بالتوازي ثم تدمج النتائج. هذا يسمح للشبكة بالتقاط ميزات بمقاييس مختلفة في نفس الوقت. والأذكى هو استخدام فلاتر 1×1 لتقليل الأبعاد قبل الفلاتر الكبيرة مما يقلل الحسابات بشكل كبير.',
        keyInnovation: 'وحدات Inception Module: معالجة المدخلات بأحجام فلاتر مختلفة (1×1, 3×3, 5×5) بالتوازي مع استخدام فلاتر 1×1 كعنق زجاجة (Bottleneck) لتقليل الحسابات.',
        layerDetails: [
          { type: 'Input', name: 'صورة الإدخال', details: ['224×224×3', 'RGB'], output: '224×224×3', color: '#78909C' },
          { type: 'Conv2D', name: 'Conv1', details: ['64 فلتر', '7×7', 'stride=2'], output: '112×112×64', color: '#26A69A' },
          { type: 'MaxPool', name: 'Pool1', details: ['3×3', 'stride=2'], output: '56×56×64', color: '#4DB6AC' },
          { type: 'Conv2D', name: 'Conv2', details: ['192 فلتر', '3×3'], output: '56×56×192', color: '#26A69A' },
          { type: 'MaxPool', name: 'Pool2', details: ['3×3', 'stride=2'], output: '28×28×192', color: '#4DB6AC' },
          { type: 'Inception', name: 'Inception 3a + 3b', details: ['1×1 + 3×3 + 5×5 بالتوازي', 'Bottleneck 1×1'], output: '28×28×480', color: '#00897B' },
          { type: 'MaxPool', name: 'Pool3', details: ['3×3', 'stride=2'], output: '14×14×480', color: '#4DB6AC' },
          { type: 'Inception', name: 'Inception 4a-4e', details: ['5 وحدات Inception', 'تعقيد متزايد'], output: '14×14×832', color: '#00897B' },
          { type: 'MaxPool', name: 'Pool4', details: ['3×3', 'stride=2'], output: '7×7×832', color: '#4DB6AC' },
          { type: 'Inception', name: 'Inception 5a + 5b', details: ['2 وحدات Inception'], output: '7×7×1024', color: '#00897B' },
          { type: 'AvgPool', name: 'Global Average Pooling', details: ['7×7', 'بدلاً من FC!'], output: '1×1×1024', color: '#80CBC4' },
          { type: 'Output', name: 'الإخراج', details: ['1000 فئة', 'Softmax'], output: '1000', color: '#66BB6A' },
        ],
        trainingDetails: [
          { icon: 'fas fa-chart-line', label: 'المحسّن', value: 'SGD with Momentum (0.9)' },
          { icon: 'fas fa-tachometer-alt', label: 'معدل التعلم', value: '0.01 مع تقليل 4% كل 8 حقب' },
          { icon: 'fas fa-redo', label: 'عدد الحقب', value: '~100' },
          { icon: 'fas fa-layer-group', label: 'حجم الدفعة', value: '128' },
          { icon: 'fas fa-bolt', label: 'دالة التنشيط', value: 'ReLU' },
          { icon: 'fas fa-calculator', label: 'دالة الخسارة', value: 'Cross-Entropy + Auxiliary Losses' },
          { icon: 'fas fa-microchip', label: 'الأجهزة', value: 'DistBelief على عدة أجهزة' },
        ],
        datasets: [
          {
            name: 'ImageNet (ILSVRC-2014)',
            details: ['1.2 مليون صورة تدريب', '1,000 فئة', 'Data Augmentation متقدمة', 'فازت بالمركز الأول في ILSVRC 2014'],
          }
        ],
        useCases: [
          { icon: 'fas fa-image', title: 'تصنيف الصور المتقدم', desc: 'تصنيف دقيق مع حجم صغير جداً مقارنة بالمنافسين' },
          { icon: 'fas fa-video', title: 'تحليل الفيديو', desc: 'سرعة عالية تمكنها من معالجة الفيديو في الوقت الحقيقي' },
          { icon: 'fas fa-mobile-alt', title: 'التطبيقات المدمجة', desc: 'حجم صغير يسمح بالعمل على أجهزة محدودة الموارد' },
        ],
        pros: ['6.8M معامل فقط (أقل 20x من VGG!)', '22 طبقة بأداء ممتاز', 'وحدات Inception مبتكرة', 'Global Average Pooling بدلاً من FC'],
        cons: ['بنية معقدة وصعبة التنفيذ', 'صعوبة الفهم مقارنة بـ VGG', 'Auxiliary classifiers تزيد التعقيد', 'تحتاج خبرة لتعديل الـ Inception modules'],
      },

      // ──────── 5. ResNet-50 ────────
      {
        name: 'ResNet-50',
        subtitle: 'Skip Connections - حل مشكلة التلاشي',
        year: '2015',
        authors: 'Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun (Microsoft)',
        paper: 'Deep Residual Learning for Image Recognition',
        icon: 'fas fa-link',
        color: '#EF5350',
        params: '25.6M',
        accuracy: '95.5%',
        layers: '50',
        inputSize: '224×224',
        shortDesc: 'حلت مشكلة تلاشي التدرجات عبر Residual Connections مما سمح ببناء شبكات عميقة جداً.',
        comparisonNote: 'Residual/Skip Connections',
        description: 'ResNet (Residual Network) من Microsoft Research حلت واحدة من أكبر مشاكل التعلم العميق: كلما زاد عمق الشبكة، تتلاشى التدرجات (Vanishing Gradients) ويصعب التدريب. الحل الثوري هو "Residual Connections" أو "Skip Connections": بدلاً من أن تتعلم كل طبقة التحويل الكامل F(x)، تتعلم فقط الفرق (البقايا) F(x) = H(x) - x، ثم نضيف المدخل الأصلي. هذا يسمح للتدرجات بالتدفق مباشرة عبر الاتصالات القصيرة.',
        keyInnovation: 'Residual/Skip Connections: إضافة اتصال مباشر يتخطى طبقة أو أكثر (identity shortcut) مما يحل مشكلة تلاشي التدرجات ويسمح بتدريب شبكات بعمق 152+ طبقة.',
        layerDetails: [
          { type: 'Input', name: 'صورة الإدخال', details: ['224×224×3', 'RGB'], output: '224×224×3', color: '#78909C' },
          { type: 'Conv2D', name: 'Conv1', details: ['64 فلتر', '7×7', 'stride=2', 'BatchNorm + ReLU'], output: '112×112×64', color: '#EF5350' },
          { type: 'MaxPool', name: 'Pool1', details: ['3×3', 'stride=2'], output: '56×56×64', color: '#EF9A9A' },
          { type: 'ResBlock', name: 'Conv2_x (×3 blocks)', details: ['Bottleneck: 1×1→3×3→1×1', '64→64→256', 'Skip Connection ↺'], output: '56×56×256', color: '#E53935' },
          { type: 'ResBlock', name: 'Conv3_x (×4 blocks)', details: ['128→128→512', 'stride=2 في أول block', 'Skip Connection ↺'], output: '28×28×512', color: '#E53935' },
          { type: 'ResBlock', name: 'Conv4_x (×6 blocks)', details: ['256→256→1024', 'stride=2 في أول block', 'Skip Connection ↺'], output: '14×14×1024', color: '#E53935' },
          { type: 'ResBlock', name: 'Conv5_x (×3 blocks)', details: ['512→512→2048', 'stride=2 في أول block', 'Skip Connection ↺'], output: '7×7×2048', color: '#E53935' },
          { type: 'AvgPool', name: 'Global Average Pooling', details: ['7×7'], output: '1×1×2048', color: '#FFAB91' },
          { type: 'Output', name: 'FC + Softmax', details: ['1000 فئة'], output: '1000', color: '#66BB6A' },
        ],
        trainingDetails: [
          { icon: 'fas fa-chart-line', label: 'المحسّن', value: 'SGD with Momentum (0.9)' },
          { icon: 'fas fa-tachometer-alt', label: 'معدل التعلم', value: '0.1 → تقليل 10x عند epochs 30, 60, 90' },
          { icon: 'fas fa-redo', label: 'عدد الحقب', value: '90' },
          { icon: 'fas fa-layer-group', label: 'حجم الدفعة', value: '256' },
          { icon: 'fas fa-bolt', label: 'دالة التنشيط', value: 'ReLU مع Batch Normalization' },
          { icon: 'fas fa-calculator', label: 'دالة الخسارة', value: 'Cross-Entropy' },
          { icon: 'fas fa-balance-scale', label: 'Weight Decay', value: '1×10⁻⁴' },
          { icon: 'fas fa-microchip', label: 'الأجهزة', value: '8× GPU' },
        ],
        datasets: [
          {
            name: 'ImageNet (ILSVRC-2015)',
            details: ['1.28 مليون صورة تدريب', '50,000 صورة validation', '1,000 فئة', 'فازت بالمركز الأول في ILSVRC 2015'],
          }
        ],
        useCases: [
          { icon: 'fas fa-search-plus', title: 'كشف الكائنات', desc: 'العمود الفقري لأنظمة Faster R-CNN و YOLO الحديثة' },
          { icon: 'fas fa-cut', title: 'تقسيم الصور (Segmentation)', desc: 'أساس شبكات مثل Mask R-CNN و U-Net++' },
          { icon: 'fas fa-hospital', title: 'التصوير الطبي', desc: 'تحليل صور الأشعة والرنين المغناطيسي' },
          { icon: 'fas fa-car', title: 'القيادة الذاتية', desc: 'كشف المشاة والمركبات وعلامات المرور' },
          { icon: 'fas fa-exchange-alt', title: 'Transfer Learning', desc: 'من أكثر الشبكات استخداماً كنقطة بداية' },
        ],
        pros: ['حلت مشكلة تلاشي التدرجات', 'يمكن بناء شبكات بعمق 152+ طبقة', '25.6M معامل فقط (أقل من VGG)', 'Batch Normalization يسرع التدريب', 'الأكثر استخداماً كـ backbone'],
        cons: ['أبطأ من MobileNet للأجهزة المحمولة', 'تحتاج GPU قوية', 'ResNet-152 قد تكون بطيئة للتطبيقات الحقيقية', 'Bottleneck blocks قد تكون صعبة الفهم للمبتدئين'],
      },

      // ──────── 6. MobileNet ────────
      {
        name: 'MobileNet v2',
        subtitle: 'خفيفة وسريعة - للأجهزة المحمولة',
        year: '2018',
        authors: 'Mark Sandler, Andrew Howard وفريق Google',
        paper: 'MobileNetV2: Inverted Residuals and Linear Bottlenecks',
        icon: 'fas fa-mobile-alt',
        color: '#FFA726',
        params: '3.4M',
        accuracy: '90.0%',
        layers: '53',
        inputSize: '224×224',
        shortDesc: 'مصممة للعمل على الهواتف المحمولة والأجهزة المدمجة بموارد محدودة.',
        comparisonNote: 'Depthwise Separable Conv للأجهزة المحمولة',
        description: 'MobileNet من Google مصممة خصيصاً للأجهزة ذات الموارد المحدودة مثل الهواتف والأجهزة المدمجة. السر هو "Depthwise Separable Convolution": بدلاً من تطبيق فلتر 3×3 على جميع القنوات معاً (مكلف حسابياً)، نقسم العملية إلى خطوتين: (1) Depthwise: فلتر 3×3 لكل قناة منفصلة، (2) Pointwise: فلتر 1×1 لدمج القنوات. هذا يقلل الحسابات 8-9 مرات مع فقدان دقة قليل.',
        keyInnovation: 'Depthwise Separable Convolution + Inverted Residual Blocks + Linear Bottleneck: تقسيم التلافيف إلى خطوتين لتقليل الحسابات 8-9x مع الحفاظ على الدقة.',
        layerDetails: [
          { type: 'Input', name: 'صورة الإدخال', details: ['224×224×3', 'RGB'], output: '224×224×3', color: '#78909C' },
          { type: 'Conv2D', name: 'Conv1', details: ['32 فلتر', '3×3', 'stride=2', 'BatchNorm + ReLU6'], output: '112×112×32', color: '#FFA726' },
          { type: 'InvRes', name: 'Bottleneck ×1', details: ['t=1, c=16', 'stride=1'], output: '112×112×16', color: '#FF9800' },
          { type: 'InvRes', name: 'Bottleneck ×2', details: ['t=6, c=24', 'stride=2'], output: '56×56×24', color: '#FF9800' },
          { type: 'InvRes', name: 'Bottleneck ×3', details: ['t=6, c=32', 'stride=2'], output: '28×28×32', color: '#FF9800' },
          { type: 'InvRes', name: 'Bottleneck ×4', details: ['t=6, c=64', 'stride=2'], output: '14×14×64', color: '#FF9800' },
          { type: 'InvRes', name: 'Bottleneck ×3', details: ['t=6, c=96', 'stride=1'], output: '14×14×96', color: '#FF9800' },
          { type: 'InvRes', name: 'Bottleneck ×3', details: ['t=6, c=160', 'stride=2'], output: '7×7×160', color: '#FF9800' },
          { type: 'InvRes', name: 'Bottleneck ×1', details: ['t=6, c=320', 'stride=1'], output: '7×7×320', color: '#FF9800' },
          { type: 'Conv2D', name: 'Conv2 (1×1)', details: ['1280 فلتر', '1×1'], output: '7×7×1280', color: '#FFA726' },
          { type: 'AvgPool', name: 'Global Average Pooling', details: ['7×7'], output: '1×1×1280', color: '#FFCC80' },
          { type: 'Output', name: 'Conv 1×1 → الإخراج', details: ['1000 فئة'], output: '1000', color: '#66BB6A' },
        ],
        trainingDetails: [
          { icon: 'fas fa-chart-line', label: 'المحسّن', value: 'RMSProp' },
          { icon: 'fas fa-tachometer-alt', label: 'معدل التعلم', value: '0.045 مع تقليل تدريجي (decay)' },
          { icon: 'fas fa-redo', label: 'عدد الحقب', value: '300+' },
          { icon: 'fas fa-layer-group', label: 'حجم الدفعة', value: '96' },
          { icon: 'fas fa-bolt', label: 'دالة التنشيط', value: 'ReLU6 (max(0, min(6, x)))' },
          { icon: 'fas fa-calculator', label: 'دالة الخسارة', value: 'Cross-Entropy + Label Smoothing' },
          { icon: 'fas fa-balance-scale', label: 'Weight Decay', value: '4×10⁻⁵' },
        ],
        datasets: [
          {
            name: 'ImageNet (ILSVRC-2012)',
            details: ['1.28 مليون صورة تدريب', '1,000 فئة', 'Data Augmentation: قص عشوائي، انعكاس أفقي'],
          }
        ],
        useCases: [
          { icon: 'fas fa-mobile-alt', title: 'تطبيقات الهاتف المحمول', desc: 'تصنيف الصور وكشف الكائنات على iOS و Android' },
          { icon: 'fas fa-robot', title: 'الأجهزة المدمجة', desc: 'Raspberry Pi, Arduino, وأجهزة IoT' },
          { icon: 'fas fa-tachometer-alt', title: 'التطبيقات الفورية', desc: 'معالجة الفيديو في الوقت الحقيقي (30+ FPS)' },
          { icon: 'fas fa-camera', title: 'فلاتر الكاميرا', desc: 'Snapchat, Instagram وتطبيقات الواقع المعزز' },
          { icon: 'fas fa-car', title: 'القيادة الذاتية', desc: 'كشف الكائنات على أجهزة السيارات المدمجة' },
        ],
        pros: ['3.4M معامل فقط!', 'سريعة جداً (مناسبة للوقت الحقيقي)', 'تعمل على الهواتف المحمولة', 'Width Multiplier للتحكم بالحجم/الدقة', 'مدعومة في TensorFlow Lite'],
        cons: ['دقة أقل من الشبكات الكبيرة', 'Depthwise Conv قد لا تستغل GPU بكفاءة', 'صعوبة التدريب أحياناً', 'تحتاج ضبط دقيق (fine-tuning)'],
      },

      // ──────── 7. EfficientNet ────────
      {
        name: 'EfficientNet-B0',
        subtitle: 'التوازن المثالي بين الدقة والحجم',
        year: '2019',
        authors: 'Mingxing Tan, Quoc V. Le (Google Brain)',
        paper: 'EfficientNet: Rethinking Model Scaling for CNNs',
        icon: 'fas fa-balance-scale-right',
        color: '#66BB6A',
        params: '5.3M',
        accuracy: '93.3%',
        layers: '237',
        inputSize: '224×224',
        shortDesc: 'طريقة ذكية لتوسيع الشبكة بتوازن بين العرض والعمق ودقة الصورة.',
        comparisonNote: 'Compound Scaling للتوازن الأمثل',
        description: 'EfficientNet من Google Brain أحدثت ثورة في كيفية توسيع (Scaling) الشبكات العصبية. الشبكات السابقة كانت تُوسَّع بطريقة واحدة: إما بزيادة العمق (مثل ResNet) أو العرض (عدد الفلاتر) أو دقة الصورة. EfficientNet اكتشفت أن التوسيع المتوازن في الأبعاد الثلاثة معاً (Compound Scaling) يعطي نتائج أفضل بكثير. باستخدام Neural Architecture Search (NAS) تم تصميم بنية أساسية (B0) ثم توسيعها بمعامل φ للحصول على B1-B7.',
        keyInnovation: 'Compound Scaling: توسيع العمق والعرض ودقة الصورة بنسب متوازنة باستخدام معامل واحد φ، مع بنية أساسية مصممة بـ Neural Architecture Search.',
        layerDetails: [
          { type: 'Input', name: 'صورة الإدخال', details: ['224×224×3 (B0)', 'تتغير مع Scaling'], output: '224×224×3', color: '#78909C' },
          { type: 'Conv2D', name: 'Stem', details: ['32 فلتر', '3×3', 'stride=2', 'BN + Swish'], output: '112×112×32', color: '#66BB6A' },
          { type: 'MBConv', name: 'MBConv1 ×1', details: ['k=3, expand=1', '16 channels'], output: '112×112×16', color: '#43A047' },
          { type: 'MBConv', name: 'MBConv6 ×2', details: ['k=3, expand=6', '24 channels', 'stride=2'], output: '56×56×24', color: '#43A047' },
          { type: 'MBConv', name: 'MBConv6 ×2', details: ['k=5, expand=6', '40 channels', 'stride=2'], output: '28×28×40', color: '#43A047' },
          { type: 'MBConv', name: 'MBConv6 ×3', details: ['k=3, expand=6', '80 channels', 'stride=2'], output: '14×14×80', color: '#43A047' },
          { type: 'MBConv', name: 'MBConv6 ×3', details: ['k=5, expand=6', '112 channels'], output: '14×14×112', color: '#43A047' },
          { type: 'MBConv', name: 'MBConv6 ×4', details: ['k=5, expand=6', '192 channels', 'stride=2'], output: '7×7×192', color: '#43A047' },
          { type: 'MBConv', name: 'MBConv6 ×1', details: ['k=3, expand=6', '320 channels'], output: '7×7×320', color: '#43A047' },
          { type: 'Conv2D', name: 'Head Conv', details: ['1280 فلتر', '1×1'], output: '7×7×1280', color: '#66BB6A' },
          { type: 'AvgPool', name: 'Global Average Pooling', details: ['7×7'], output: '1×1×1280', color: '#A5D6A7' },
          { type: 'Output', name: 'FC → الإخراج', details: ['1000 فئة', 'Softmax'], output: '1000', color: '#66BB6A' },
        ],
        trainingDetails: [
          { icon: 'fas fa-chart-line', label: 'المحسّن', value: 'RMSProp (decay=0.9, momentum=0.9)' },
          { icon: 'fas fa-tachometer-alt', label: 'معدل التعلم', value: '0.256 مع Exponential Decay' },
          { icon: 'fas fa-redo', label: 'عدد الحقب', value: '350' },
          { icon: 'fas fa-layer-group', label: 'حجم الدفعة', value: '2048' },
          { icon: 'fas fa-bolt', label: 'دالة التنشيط', value: 'Swish (x × σ(x)) بدلاً من ReLU' },
          { icon: 'fas fa-calculator', label: 'دالة الخسارة', value: 'Cross-Entropy + Label Smoothing (0.1)' },
          { icon: 'fas fa-random', label: 'تقنيات التنظيم', value: 'Stochastic Depth + Dropout' },
          { icon: 'fas fa-magic', label: 'SE-Net (Squeeze & Excitation)', value: 'موجودة في كل MBConv block' },
        ],
        datasets: [
          {
            name: 'ImageNet (ILSVRC-2012)',
            details: ['1.28 مليون صورة تدريب', '50,000 صورة validation', '1,000 فئة', 'AutoAugment لتحسين البيانات'],
          }
        ],
        useCases: [
          { icon: 'fas fa-expand-arrows-alt', title: 'تطبيقات متعددة الأحجام', desc: 'B0 للأجهزة المحمولة، B7 للخوادم (نفس البنية!)' },
          { icon: 'fas fa-hospital', title: 'التصوير الطبي', desc: 'دقة عالية مع حجم صغير يناسب التطبيقات الطبية' },
          { icon: 'fas fa-satellite', title: 'صور الأقمار الصناعية', desc: 'تصنيف استخدام الأراضي وتحليل الصور الجوية' },
          { icon: 'fas fa-industry', title: 'مراقبة الجودة', desc: 'كشف العيوب في خطوط الإنتاج الصناعية' },
        ],
        pros: ['أفضل توازن بين الدقة والكفاءة', 'عائلة كاملة (B0-B7) لكل الاحتياجات', 'مصممة بـ NAS (تصميم تلقائي)', 'Swish activation أفضل من ReLU', 'SE blocks تحسن الأداء'],
        cons: ['صعوبة فهم التصميم (NAS)', 'التدريب يحتاج موارد كبيرة', 'B7 ضخمة جداً (66M params)', 'Compound Scaling ليس الحل الأمثل دائماً'],
      },

      // ──────── 8. Vision Transformer (ViT) ────────
      {
        name: 'Vision Transformer',
        subtitle: 'ViT - المحولات تغزو عالم الصور',
        year: '2020',
        authors: 'Alexey Dosovitskiy وفريق Google Research',
        paper: 'An Image is Worth 16x16 Words',
        icon: 'fas fa-th',
        color: '#7C4DFF',
        params: '86M',
        accuracy: '90.0%',
        layers: '12',
        inputSize: '224×224',
        shortDesc: 'تطبيق بنية Transformers (المستخدمة في NLP) مباشرة على الصور بتقسيمها لـ patches.',
        comparisonNote: 'Transformers للصور بدون CNN!',
        description: 'Vision Transformer (ViT) من Google أثبت أن بنية Transformers - التي أحدثت ثورة في معالجة اللغات الطبيعية (NLP) - يمكنها أيضاً العمل مع الصور بشكل ممتاز بدون أي طبقات تلافيفية! الفكرة بسيطة: نقسم الصورة إلى رقع (Patches) بحجم 16×16، نحولها إلى تسلسل من المتجهات (Tokens)، ثم نطبق عليها Transformer Encoder تماماً كما يُطبق على الكلمات. هذا يسمح لكل رقعة بالانتباه لجميع الرقع الأخرى (Self-Attention) مما يلتقط العلاقات بعيدة المدى.',
        keyInnovation: 'تقسيم الصورة إلى رقع (16×16 Patches) ومعاملتها كتسلسل من الرموز (Tokens) يُمرر إلى Transformer Encoder، مع إثبات أن Self-Attention يمكنها استبدال التلافيف بالكامل.',
        layerDetails: [
          { type: 'Input', name: 'صورة الإدخال', details: ['224×224×3', 'RGB'], output: '224×224×3', color: '#78909C' },
          { type: 'Patch', name: 'تقسيم إلى رقع (Patches)', details: ['16×16 patch size', '14×14 = 196 رقعة', 'كل رقعة = 768 بُعد'], output: '196 × 768', color: '#7C4DFF' },
          { type: 'Embed', name: 'Patch + Position Embedding', details: ['Linear Projection', 'CLS Token', 'Positional Encoding'], output: '197 × 768', color: '#9C27B0' },
          { type: 'Transformer', name: 'Transformer Encoder ×12', details: ['Multi-Head Self-Attention (12 heads)', 'Layer Norm', 'MLP (3072 hidden)', 'Residual Connections'], output: '197 × 768', color: '#651FFF' },
          { type: 'CLS', name: 'CLS Token Output', details: ['أخذ تمثيل الـ [CLS] token', 'Layer Norm'], output: '768', color: '#B388FF' },
          { type: 'Output', name: 'MLP Head → الإخراج', details: ['1000 فئة', 'Softmax'], output: '1000', color: '#66BB6A' },
        ],
        trainingDetails: [
          { icon: 'fas fa-chart-line', label: 'المحسّن', value: 'Adam (β₁=0.9, β₂=0.999)' },
          { icon: 'fas fa-tachometer-alt', label: 'معدل التعلم', value: 'Linear Warmup + Cosine Decay' },
          { icon: 'fas fa-redo', label: 'عدد الحقب', value: '300 (pre-train) + 20 (fine-tune)' },
          { icon: 'fas fa-layer-group', label: 'حجم الدفعة', value: '4096' },
          { icon: 'fas fa-bolt', label: 'دالة التنشيط', value: 'GELU في MLP blocks' },
          { icon: 'fas fa-calculator', label: 'دالة الخسارة', value: 'Cross-Entropy' },
          { icon: 'fas fa-database', label: 'Pre-training Data', value: 'JFT-300M (300 مليون صورة!)' },
          { icon: 'fas fa-microchip', label: 'الأجهزة', value: 'TPU v3 (عشرات الأجهزة)' },
        ],
        datasets: [
          {
            name: 'JFT-300M (Pre-training)',
            details: ['300 مليون صورة!', '18,291 فئة', 'مجموعة بيانات Google الداخلية', 'ViT يحتاج بيانات ضخمة للتدريب الأولي'],
          },
          {
            name: 'ImageNet (Fine-tuning)',
            details: ['1.28 مليون صورة', '1,000 فئة', 'Fine-tuning بعد Pre-training', 'دقة عالية جداً بعد Pre-training على JFT'],
          }
        ],
        useCases: [
          { icon: 'fas fa-language', title: 'نماذج متعددة الوسائط', desc: 'قاعدة لنماذج مثل CLIP و DALL-E التي تجمع بين النص والصور' },
          { icon: 'fas fa-hospital', title: 'التصوير الطبي المتقدم', desc: 'Self-Attention تلتقط العلاقات بعيدة المدى في الصور الطبية' },
          { icon: 'fas fa-video', title: 'فهم الفيديو', desc: 'Video Vision Transformer (ViViT) لتحليل الفيديو' },
          { icon: 'fas fa-car', title: 'القيادة الذاتية', desc: 'فهم المشهد الكامل عبر Self-Attention' },
          { icon: 'fas fa-robot', title: 'الذكاء الاصطناعي العام', desc: 'جزء من أنظمة Foundation Models مثل GPT-4V' },
        ],
        pros: ['Self-Attention تلتقط العلاقات بعيدة المدى', 'بنية موحدة مع NLP (نفس Transformer)', 'قابلة للتوسيع بسهولة', 'تتفوق على CNN مع بيانات كافية', 'أساس نماذج Foundation Models'],
        cons: ['تحتاج بيانات ضخمة للتدريب (300M+ صورة)', 'بطيئة مع الصور العالية الدقة (O(n²) attention)', 'تحتاج موارد حوسبة هائلة', 'لا تلتقط الأنماط المحلية بكفاءة CNN', 'أقل كفاءة من CNN مع بيانات قليلة'],
      },
    ];
  }
}
