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
