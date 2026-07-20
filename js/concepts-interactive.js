/**
 * Concepts Interactive Visualizations
 * Handles the logic for the 4 educational concept cards on the homepage.
 */

const ConceptCards = {
    init: function() {
        // Activation Functions
        this.activationViz = new ActivationFunctionViz('activation-canvas');
        this.setupActivationTabs();

        // Optimizers
        this.optimizerViz = new OptimizerViz('optimizer-canvas');
        document.getElementById('opt-replay').addEventListener('click', () => {
            this.optimizerViz.reset();
            this.optimizerViz.start();
        });

        // Gradients
        this.gradientViz = new GradientProblemViz('gradient-canvas');
        this.setupGradientTabs();

        // Learning Rate
        this.lrViz = new LearningRateViz('lr-canvas', 'lr-slider', 'lr-indicator', 'lr-status-text');
    },

    toggle: function(headerElement) {
        const card = headerElement.closest('.concept-card');
        const isActive = card.classList.contains('active');
        
        // Close other cards (accordion behavior)
        document.querySelectorAll('.concept-card').forEach(c => {
            c.classList.remove('active');
        });

        if (!isActive) {
            card.classList.add('active');
            
            // Resize and start animation for the specific canvas when opened
            setTimeout(() => {
                const canvasId = card.querySelector('canvas').id;
                if (canvasId === 'activation-canvas') {
                    this.activationViz.resize();
                    this.activationViz.draw();
                } else if (canvasId === 'optimizer-canvas') {
                    this.optimizerViz.resize();
                    this.optimizerViz.reset();
                    this.optimizerViz.start();
                } else if (canvasId === 'gradient-canvas') {
                    this.gradientViz.resize();
                    this.gradientViz.start();
                } else if (canvasId === 'lr-canvas') {
                    this.lrViz.resize();
                    this.lrViz.start();
                }
            }, 300); // Wait for transition
        }
    },

    setupActivationTabs: function() {
        const tabs = document.querySelectorAll('.concept-card:nth-child(1) .concept-tab');
        const formulaEl = document.getElementById('activation-formula');
        const descEl = document.getElementById('activation-desc');

        const info = {
            'sigmoid': { formula: 'f(x) = 1 / (1 + e^(-x))', desc: 'دالة تضغط القيم بين 0 و 1. ممتازة في التصنيف الثنائي وللإجابة بنعم/لا، لكنها تعاني من مشكلة تلاشي الانحدار في الشبكات العميقة.' },
            'relu': { formula: 'f(x) = max(0, x)', desc: 'دالة (Rectified Linear Unit). تُرجع x إذا كان موجباً، و 0 إذا كان سالباً. هي الأكثر استخداماً حالياً لأنها سريعة وتخفف مشكلة تلاشي الانحدار.' },
            'tanh': { formula: 'f(x) = (e^x - e^-x) / (e^x + e^-x)', desc: 'تشبه Sigmoid لكنها تضغط القيم بين -1 و 1. أفضل عادة لأن متوسط المخرجات يكون قريباً من الصفر، مما يسرّع التعلم.' },
            'leaky': { formula: 'f(x) = x (x>0), 0.01x (x<=0)', desc: 'نسخة محسنة من ReLU لا تميت الخلايا العصبية (Dying ReLU)، حيث تسمح بتسرب قيمة صغيرة جداً عندما يكون المدخل سالباً.' }
        };

        tabs.forEach(tab => {
            tab.addEventListener('click', (e) => {
                tabs.forEach(t => t.classList.remove('active'));
                tab.classList.add('active');
                
                const type = tab.dataset.act;
                formulaEl.innerText = info[type].formula;
                descEl.innerText = info[type].desc;
                
                this.activationViz.setFunction(type);
            });
        });
    },

    setupGradientTabs: function() {
        const btnVanishing = document.getElementById('btn-vanishing');
        const btnExploding = document.getElementById('btn-exploding');
        const descEl = document.getElementById('gradient-desc');

        btnVanishing.addEventListener('click', () => {
            btnVanishing.classList.add('active');
            btnExploding.classList.remove('active');
            descEl.innerHTML = '<b>تلاشي الانحدار (Vanishing):</b> عندما تتدفق الإشارات للخلف عبر طبقات عديدة، تصبح الأوزان في الطبقات الأولى صغيرة جداً (قريبة من الصفر)، فتتوقف الشبكة عن التعلم.';
            this.gradientViz.setMode('vanishing');
        });

        btnExploding.addEventListener('click', () => {
            btnExploding.classList.add('active');
            btnVanishing.classList.remove('active');
            descEl.innerHTML = '<b>انفجار الانحدار (Exploding):</b> المشكلة المعاكسة، حيث تتضخم القيم بشكل أسّي أثناء العودة للخلف، مما يؤدي إلى أوزان ضخمة جداً وعدم استقرار في الشبكة (NaN).';
            this.gradientViz.setMode('exploding');
        });
    }
};

/* =========================================================================
   1. Activation Function Visualization
   ========================================================================= */
class ActivationFunctionViz {
    constructor(canvasId) {
        this.canvas = document.getElementById(canvasId);
        this.ctx = this.canvas.getContext('2d');
        this.currentFunc = 'sigmoid';
        this.progress = 1;
        this.animationId = null;
        
        window.addEventListener('resize', () => {
            if (this.canvas.offsetParent !== null) {
                this.resize();
                this.draw();
            }
        });
    }

    resize() {
        const parent = this.canvas.parentElement;
        this.canvas.width = parent.clientWidth;
        this.canvas.height = parent.clientHeight;
    }

    setFunction(type) {
        if (this.currentFunc === type) return;
        this.currentFunc = type;
        this.progress = 0;
        
        const animate = () => {
            this.progress += 0.05;
            if (this.progress >= 1) {
                this.progress = 1;
                this.draw();
                return;
            }
            this.draw();
            this.animationId = requestAnimationFrame(animate);
        };
        
        if (this.animationId) cancelAnimationFrame(this.animationId);
        animate();
    }

    draw() {
        const ctx = this.ctx;
        const w = this.canvas.width;
        const h = this.canvas.height;
        
        ctx.clearRect(0, 0, w, h);
        
        const padding = 20;
        const drawW = w - padding * 2;
        const drawH = h - padding * 2;
        const centerX = w / 2;
        const centerY = h / 2;
        
        // Draw Grid
        ctx.strokeStyle = 'rgba(255, 255, 255, 0.05)';
        ctx.lineWidth = 1;
        ctx.beginPath();
        for (let i = 0; i <= 10; i++) {
            const x = padding + (drawW / 10) * i;
            const y = padding + (drawH / 10) * i;
            ctx.moveTo(x, padding); ctx.lineTo(x, h - padding);
            ctx.moveTo(padding, y); ctx.lineTo(w - padding, y);
        }
        ctx.stroke();

        // Draw Axes
        ctx.strokeStyle = 'rgba(255, 255, 255, 0.2)';
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.moveTo(padding, centerY);
        ctx.lineTo(w - padding, centerY);
        ctx.moveTo(centerX, padding);
        ctx.lineTo(centerX, h - padding);
        ctx.stroke();

        // Draw Function
        ctx.beginPath();
        ctx.strokeStyle = '#00E5FF';
        ctx.lineWidth = 3;
        
        const scaleX = 10; // -5 to 5
        const scaleY = 3;  // -1.5 to 1.5
        
        const totalPoints = drawW;
        const pointsToDraw = Math.floor(totalPoints * this.progress);
        
        for (let i = 0; i < pointsToDraw; i++) {
            const x = (i / drawW) * scaleX - (scaleX / 2);
            let y = 0;
            
            switch (this.currentFunc) {
                case 'sigmoid': y = 1 / (1 + Math.exp(-x)); break;
                case 'relu': y = Math.max(0, x); break;
                case 'tanh': y = Math.tanh(x); break;
                case 'leaky': y = x > 0 ? x : 0.1 * x; break;
            }
            
            const px = padding + i;
            const py = centerY - (y / scaleY) * (drawH / 2);
            
            if (i === 0) ctx.moveTo(px, py);
            else ctx.lineTo(px, py);
        }
        
        ctx.stroke();
        
        // Add glow
        ctx.shadowColor = '#00E5FF';
        ctx.shadowBlur = 10;
        ctx.stroke();
        ctx.shadowBlur = 0;
    }
}

/* =========================================================================
   2. Optimizer Visualization
   ========================================================================= */
class OptimizerViz {
    constructor(canvasId) {
        this.canvas = document.getElementById(canvasId);
        this.ctx = this.canvas.getContext('2d');
        this.animationId = null;
        this.optimizers = [];
        this.isPlaying = false;
        
        window.addEventListener('resize', () => {
            if (this.canvas.offsetParent !== null) {
                this.resize();
                if (!this.isPlaying) this.draw();
            }
        });
    }

    resize() {
        const parent = this.canvas.parentElement;
        this.canvas.width = parent.clientWidth;
        this.canvas.height = parent.clientHeight;
    }

    reset() {
        const w = this.canvas.width;
        const h = this.canvas.height;
        
        const startX = w * 0.1;
        const startY = h * 0.2;
        const targetX = w * 0.8;
        const targetY = h * 0.8;
        
        this.optimizers = [
            { name: 'SGD', color: '#FF5252', x: startX, y: startY, path: [], vx: 0, vy: 0, type: 'sgd', targetX, targetY, completed: false },
            { name: 'RMSProp', color: '#00E676', x: startX, y: startY, path: [], vx: 0, vy: 0, type: 'rmsprop', targetX, targetY, completed: false, cacheX: 0, cacheY: 0 },
            { name: 'Adam', color: '#E040FB', x: startX, y: startY, path: [], vx: 0, vy: 0, type: 'adam', targetX, targetY, completed: false, mX: 0, mY: 0, vX: 0, vY: 0, t: 0 }
        ];
    }

    getGradient(x, y, tx, ty) {
        // Create a fake noisy landscape gradient
        const dx = x - tx;
        const dy = y - ty;
        
        // Base gradient towards center
        let gx = dx * 0.01;
        let gy = dy * 0.05; // Makes it a narrow valley (ravine)
        
        // Add noise/local minima
        gx += Math.sin(x * 0.05) * 0.5;
        gy += Math.cos(y * 0.05) * 0.5;
        
        return { gx, gy };
    }

    start() {
        this.isPlaying = true;
        let allCompleted = false;
        
        const animate = () => {
            if (!this.isPlaying) return;
            
            allCompleted = true;
            
            this.optimizers.forEach(opt => {
                if (opt.completed) return;
                
                const dist = Math.hypot(opt.targetX - opt.x, opt.targetY - opt.y);
                if (dist < 10) {
                    opt.completed = true;
                    return;
                }
                
                allCompleted = false;
                opt.path.push({x: opt.x, y: opt.y});
                
                const { gx, gy } = this.getGradient(opt.x, opt.y, opt.targetX, opt.targetY);
                const lr = 2.0;
                
                if (opt.type === 'sgd') {
                    opt.x -= lr * gx * 0.5;
                    opt.y -= lr * gy * 0.5;
                } 
                else if (opt.type === 'rmsprop') {
                    opt.cacheX = 0.9 * opt.cacheX + 0.1 * gx * gx;
                    opt.cacheY = 0.9 * opt.cacheY + 0.1 * gy * gy;
                    opt.x -= (lr * 1.5) * gx / (Math.sqrt(opt.cacheX) + 1e-8);
                    opt.y -= (lr * 1.5) * gy / (Math.sqrt(opt.cacheY) + 1e-8);
                }
                else if (opt.type === 'adam') {
                    opt.t += 1;
                    opt.mX = 0.9 * opt.mX + 0.1 * gx;
                    opt.mY = 0.9 * opt.mY + 0.1 * gy;
                    opt.vX = 0.999 * opt.vX + 0.001 * gx * gx;
                    opt.vY = 0.999 * opt.vY + 0.001 * gy * gy;
                    
                    const mXHat = opt.mX / (1 - Math.pow(0.9, opt.t));
                    const mYHat = opt.mY / (1 - Math.pow(0.9, opt.t));
                    const vXHat = opt.vX / (1 - Math.pow(0.999, opt.t));
                    const vYHat = opt.vY / (1 - Math.pow(0.999, opt.t));
                    
                    opt.x -= (lr * 2) * mXHat / (Math.sqrt(vXHat) + 1e-8);
                    opt.y -= (lr * 2) * mYHat / (Math.sqrt(vYHat) + 1e-8);
                }
            });
            
            this.draw();
            
            if (!allCompleted) {
                this.animationId = requestAnimationFrame(animate);
            } else {
                this.isPlaying = false;
            }
        };
        
        if (this.animationId) cancelAnimationFrame(this.animationId);
        animate();
    }

    draw() {
        const ctx = this.ctx;
        const w = this.canvas.width;
        const h = this.canvas.height;
        
        ctx.clearRect(0, 0, w, h);
        
        const targetX = w * 0.8;
        const targetY = h * 0.8;
        
        // Draw contour lines (loss surface)
        ctx.lineWidth = 1;
        for (let i = 5; i > 0; i--) {
            ctx.beginPath();
            ctx.ellipse(targetX, targetY, i * 60, i * 20, Math.PI / -4, 0, Math.PI * 2);
            ctx.strokeStyle = `rgba(108, 99, 255, ${0.1 * (6-i)})`;
            ctx.stroke();
            if (i === 1) {
                ctx.fillStyle = 'rgba(108, 99, 255, 0.2)';
                ctx.fill();
                
                // Draw Minima Star
                ctx.fillStyle = '#FFF';
                ctx.beginPath();
                ctx.arc(targetX, targetY, 4, 0, Math.PI*2);
                ctx.fill();
            }
        }
        
        // Draw optimizers paths
        this.optimizers.forEach(opt => {
            if (opt.path.length > 0) {
                ctx.beginPath();
                ctx.moveTo(opt.path[0].x, opt.path[0].y);
                for (let i = 1; i < opt.path.length; i++) {
                    ctx.lineTo(opt.path[i].x, opt.path[i].y);
                }
                ctx.lineTo(opt.x, opt.y);
                ctx.strokeStyle = opt.color;
                ctx.lineWidth = 2;
                ctx.globalAlpha = 0.7;
                ctx.stroke();
                ctx.globalAlpha = 1.0;
            }
            
            // Draw current position
            ctx.beginPath();
            ctx.arc(opt.x, opt.y, 5, 0, Math.PI * 2);
            ctx.fillStyle = opt.color;
            ctx.fill();
            
            ctx.shadowColor = opt.color;
            ctx.shadowBlur = 10;
            ctx.fill();
            ctx.shadowBlur = 0;
        });
    }
}

/* =========================================================================
   3. Vanishing / Exploding Gradients Visualization
   ========================================================================= */
class GradientProblemViz {
    constructor(canvasId) {
        this.canvas = document.getElementById(canvasId);
        this.ctx = this.canvas.getContext('2d');
        this.mode = 'vanishing';
        this.animationId = null;
        this.time = 0;
        
        window.addEventListener('resize', () => {
            if (this.canvas.offsetParent !== null) {
                this.resize();
            }
        });
    }

    resize() {
        const parent = this.canvas.parentElement;
        this.canvas.width = parent.clientWidth;
        this.canvas.height = parent.clientHeight;
    }

    setMode(mode) {
        this.mode = mode;
        this.time = 0;
    }

    start() {
        const animate = () => {
            this.time += 0.02;
            this.draw();
            this.animationId = requestAnimationFrame(animate);
        };
        
        if (this.animationId) cancelAnimationFrame(this.animationId);
        animate();
    }

    draw() {
        const ctx = this.ctx;
        const w = this.canvas.width;
        const h = this.canvas.height;
        
        ctx.clearRect(0, 0, w, h);
        
        const numLayers = 5;
        const nodesPerLayer = 4;
        const layerSpacing = w / (numLayers + 1);
        const nodeSpacing = h / (nodesPerLayer + 1);
        
        // Oscillating factor based on time (0 to 1 back to 0)
        const pulse = (Math.sin(this.time * Math.PI) + 1) / 2;
        
        for (let l = 0; l < numLayers; l++) {
            const x = layerSpacing * (l + 1);
            
            // Calculate gradient magnitude for this layer
            // Backpropagation flows right to left.
            // Layer 4 is output (strongest), Layer 0 is input (weakest in vanishing, strongest in exploding)
            let gradMag = 0;
            
            if (this.mode === 'vanishing') {
                // Diminishes as we go left
                gradMag = Math.pow(0.4, numLayers - 1 - l); 
            } else {
                // Explodes as we go left
                gradMag = Math.pow(1.8, numLayers - 1 - l);
            }
            
            // Draw connections to previous layer
            if (l > 0) {
                const prevX = layerSpacing * l;
                for (let n = 0; n < nodesPerLayer; n++) {
                    const y = nodeSpacing * (n + 1);
                    for (let pn = 0; pn < nodesPerLayer; pn++) {
                        const py = nodeSpacing * (pn + 1);
                        
                        ctx.beginPath();
                        ctx.moveTo(prevX, py);
                        ctx.lineTo(x, y);
                        
                        // Style based on mode and pulse
                        let opacity, lineWidth, color;
                        
                        if (this.mode === 'vanishing') {
                            opacity = 0.1 + (gradMag * 0.9 * pulse);
                            lineWidth = 0.5 + (gradMag * 2 * pulse);
                            color = `rgba(108, 99, 255, ${opacity})`;
                        } else {
                            opacity = Math.min(1, 0.1 + (gradMag * 0.2 * pulse));
                            lineWidth = Math.min(8, 0.5 + (gradMag * 0.5 * pulse));
                            
                            // Color shifts to red/yellow when exploding
                            if (gradMag > 2) {
                                color = `rgba(255, ${Math.max(0, 255 - gradMag*20)}, 0, ${opacity})`;
                            } else {
                                color = `rgba(108, 99, 255, ${opacity})`;
                            }
                        }
                        
                        ctx.strokeStyle = color;
                        ctx.lineWidth = lineWidth;
                        ctx.stroke();
                    }
                }
            }
            
            // Draw Nodes
            for (let n = 0; n < nodesPerLayer; n++) {
                const y = nodeSpacing * (n + 1);
                ctx.beginPath();
                ctx.arc(x, y, 6, 0, Math.PI * 2);
                
                if (this.mode === 'vanishing') {
                    ctx.fillStyle = `rgba(108, 99, 255, ${0.3 + gradMag * 0.7})`;
                } else {
                    if (gradMag > 2 && pulse > 0.5) {
                        ctx.fillStyle = '#FF5252';
                        ctx.shadowColor = '#FF5252';
                        ctx.shadowBlur = gradMag * 2;
                    } else {
                        ctx.fillStyle = '#6C63FF';
                        ctx.shadowBlur = 0;
                    }
                }
                
                ctx.fill();
                ctx.shadowBlur = 0;
            }
            
            // Draw Gradient values text below the layer
            ctx.fillStyle = 'var(--text-secondary)';
            ctx.font = '10px var(--font-mono)';
            ctx.textAlign = 'center';
            let valText = this.mode === 'vanishing' ? 
                (gradMag * pulse).toFixed(4) : 
                (gradMag * pulse).toFixed(1);
                
            if (this.mode === 'exploding' && gradMag * pulse > 10) valText = "NaN!";
            
            ctx.fillText(valText, x, h - 10);
        }
    }
}

/* =========================================================================
   4. Learning Rate Visualization
   ========================================================================= */
class LearningRateViz {
    constructor(canvasId, sliderId, indicatorId, statusTextId) {
        this.canvas = document.getElementById(canvasId);
        this.ctx = this.canvas.getContext('2d');
        this.slider = document.getElementById(sliderId);
        this.indicator = document.getElementById(indicatorId);
        this.statusText = document.getElementById(statusTextId);
        
        this.animationId = null;
        this.lr = 0.1;
        this.x = -4; // Start position
        this.path = [];
        this.frameCount = 0;
        
        this.slider.addEventListener('input', (e) => {
            this.lr = parseFloat(e.target.value);
            this.updateIndicator();
            this.reset();
        });
        
        window.addEventListener('resize', () => {
            if (this.canvas.offsetParent !== null) {
                this.resize();
            }
        });
    }

    updateIndicator() {
        const dot = this.indicator.querySelector('.dot');
        
        if (this.lr < 0.05) {
            this.indicator.style.color = 'var(--warning)';
            this.indicator.style.borderColor = 'rgba(255, 214, 0, 0.3)';
            dot.style.background = 'var(--warning)';
            this.statusText.innerText = `الحالة: تعلم بطيء جداً (${this.lr.toFixed(2)})`;
        } else if (this.lr >= 0.05 && this.lr <= 0.3) {
            this.indicator.style.color = 'var(--success)';
            this.indicator.style.borderColor = 'rgba(0, 230, 118, 0.3)';
            dot.style.background = 'var(--success)';
            this.statusText.innerText = `الحالة: وصول مثالي (${this.lr.toFixed(2)})`;
        } else {
            this.indicator.style.color = 'var(--error)';
            this.indicator.style.borderColor = 'rgba(255, 82, 82, 0.3)';
            dot.style.background = 'var(--error)';
            this.statusText.innerText = `الحالة: تذبذب / غير مستقر (${this.lr.toFixed(2)})`;
        }
    }

    resize() {
        const parent = this.canvas.parentElement;
        this.canvas.width = parent.clientWidth;
        this.canvas.height = parent.clientHeight;
        this.reset();
    }

    reset() {
        this.x = -4;
        this.path = [];
        this.frameCount = 0;
    }

    f(x) {
        return x * x; // Parabola
    }

    df(x) {
        return 2 * x; // Derivative
    }

    start() {
        const animate = () => {
            this.frameCount++;
            
            // Update position every few frames to slow it down visually
            if (this.frameCount % 10 === 0) {
                if (this.path.length === 0 || Math.abs(this.df(this.x)) > 0.01) {
                    this.path.push({x: this.x, y: this.f(this.x)});
                    
                    // Gradient Descent step
                    const grad = this.df(this.x);
                    this.x = this.x - this.lr * grad;
                    
                    // Prevent shooting to infinity if LR is too high
                    if (this.x > 5) this.x = 5;
                    if (this.x < -5) this.x = -5;
                }
            }
            
            this.draw();
            this.animationId = requestAnimationFrame(animate);
        };
        
        if (this.animationId) cancelAnimationFrame(this.animationId);
        animate();
    }

    draw() {
        const ctx = this.ctx;
        const w = this.canvas.width;
        const h = this.canvas.height;
        
        ctx.clearRect(0, 0, w, h);
        
        const scaleX = w / 10; // -5 to 5
        const scaleY = h / 25; // 0 to 25
        
        const toPx = (px, py) => {
            return {
                cx: w/2 + px * scaleX,
                cy: h - 20 - py * scaleY
            };
        };
        
        // Draw Parabola
        ctx.beginPath();
        ctx.strokeStyle = 'rgba(108, 99, 255, 0.3)';
        ctx.lineWidth = 3;
        for (let i = -5; i <= 5; i += 0.1) {
            const pt = toPx(i, this.f(i));
            if (i === -5) ctx.moveTo(pt.cx, pt.cy);
            else ctx.lineTo(pt.cx, pt.cy);
        }
        ctx.stroke();
        
        // Draw Path (Lines)
        if (this.path.length > 1) {
            ctx.beginPath();
            ctx.strokeStyle = 'rgba(255, 255, 255, 0.4)';
            ctx.lineWidth = 2;
            const startPt = toPx(this.path[0].x, this.path[0].y);
            ctx.moveTo(startPt.cx, startPt.cy);
            
            for (let i = 1; i < this.path.length; i++) {
                const pt = toPx(this.path[i].x, this.path[i].y);
                ctx.lineTo(pt.cx, pt.cy);
            }
            ctx.stroke();
        }
        
        // Draw Path (Points)
        for (let i = 0; i < this.path.length; i++) {
            const pt = toPx(this.path[i].x, this.path[i].y);
            ctx.beginPath();
            ctx.arc(pt.cx, pt.cy, 3, 0, Math.PI * 2);
            ctx.fillStyle = 'rgba(255, 255, 255, 0.6)';
            ctx.fill();
        }
        
        // Draw Current Ball
        const currentPt = toPx(this.x, this.f(this.x));
        ctx.beginPath();
        ctx.arc(currentPt.cx, currentPt.cy, 8, 0, Math.PI * 2);
        
        let ballColor = '#00E676';
        if (this.lr < 0.05) ballColor = '#FFD600';
        if (this.lr > 0.3) ballColor = '#FF5252';
        
        ctx.fillStyle = ballColor;
        ctx.fill();
        
        ctx.shadowColor = ballColor;
        ctx.shadowBlur = 10;
        ctx.fill();
        ctx.shadowBlur = 0;
    }
}
