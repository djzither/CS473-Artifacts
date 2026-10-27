/* ============================================================================
   Many Functions, Then Fewer — Gaussian process regression.
   CS 473 module 10 · Murphy, PML §17.2

   A GP puts a distribution over whole functions. With a zero prior mean and
   the RBF kernel

     K(x, x') = exp(−(x − x')² / 2ℓ²)

   and observations y = f(x) + ε, ε ~ N(0, σ²), the posterior over f at test
   inputs X* is Gaussian with

     mean = K*ᵀ (K + σ²I)⁻¹ y
     cov  = K** − K*ᵀ (K + σ²I)⁻¹ K*

   Everything here is about f, the latent function: the band and the samples
   never include the observation noise.

   Layout of this file
     PART 1  Math          pure, DOM-free, exported for the Node test suite
     PART 2  Drawing       the chart: band, samples, true function, mean, data
     PART 3  Controls      buttons, clicking the chart, sliders, toggles

   PART 1 runs under Node (`require('./script.js')` returns the math only); the
   rest is skipped when there is no document. See test-math.js.
   ========================================================================= */

(function (global) {
  'use strict';

  /* ==========================================================================
     PART 1 — MATH
     ====================================================================== */

  var X_MIN = -5, X_MAX = 5;
  var GRID_N = 201;                 // test inputs the curves are drawn on
  var ELL_MIN = 0.2, ELL_MAX = 3;   // length-scale slider (log scale)
  var SIGMA_MAX = 1;
  var MAX_SAMPLES = 10;
  var MAX_OBS = 60;
  var JITTER = 1e-8;                // always added to the diagonal of K + σ²I
  var SAMPLE_JITTER = 1e-9;         // starting jitter for the posterior covariance
  var SNAP = 0.1;                   // a click this close to an existing x reuses that x
  var START = { ell: 1, sigma: 0, samples: 5 };

  // The "Add next observation" sequence: spread out, so each point lands
  // somewhere the GP is still unsure.
  var PRESET_X = [-3.5, 2.5, 0, -1.5, 4.2, 1.2, -4.6, 3.4, -0.7, 1.9];

  function trueF(x) { return Math.sin(x); }

  function rbf(a, b, ell) { var d = a - b; return Math.exp(-d * d / (2 * ell * ell)); }

  function linspace(a, b, n) {
    var out = new Float64Array(n);
    for (var i = 0; i < n; i++) out[i] = a + (b - a) * i / (n - 1);
    return out;
  }
  var GRID = linspace(X_MIN, X_MAX, GRID_N);

  /* Small seeded generator, so the page opens on the same picture every time. */
  function mulberry32(seed) {
    return function () {
      seed = seed + 0x6D2B79F5 | 0;
      var t = Math.imul(seed ^ seed >>> 15, 1 | seed);
      t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t;
      return ((t ^ t >>> 14) >>> 0) / 4294967296;
    };
  }

  /* Standard normal draw (Box–Muller). */
  function gaussian(rand) {
    var u = 0;
    while (u === 0) u = rand();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * rand());
  }

  function normals(rand, n) {
    var z = new Float64Array(n);
    for (var i = 0; i < n; i++) z[i] = gaussian(rand);
    return z;
  }

  // One fixed noise draw per preset point, scaled by whatever σ is set when
  // the point is added.
  var PRESET_Z = normals(mulberry32(473), PRESET_X.length);

  function presetY(i, sigma) { return trueF(PRESET_X[i]) + sigma * PRESET_Z[i]; }

  /* Lower-triangular L with A = L Lᵀ, or null if a pivot isn't positive. */
  function cholesky(A) {
    var n = A.length, L = [], i, j, k;
    for (i = 0; i < n; i++) L.push(new Float64Array(n));
    for (j = 0; j < n; j++) {
      var Lj = L[j], s = A[j][j];
      for (k = 0; k < j; k++) s -= Lj[k] * Lj[k];
      if (!(s > 0)) return null;
      var d = Math.sqrt(s);
      Lj[j] = d;
      for (i = j + 1; i < n; i++) {
        var Li = L[i], t = A[i][j];
        for (k = 0; k < j; k++) t -= Li[k] * Lj[k];
        Li[j] = t / d;
      }
    }
    return L;
  }

  /* Cholesky of A + jI. Starts at j = base and grows it tenfold until the
     factorization succeeds, so repeated inputs (identical rows in K) and
     nearly singular covariances still factor. Reports the jitter it used. */
  function cholJitter(A, base) {
    var j = base;
    for (var tries = 0; tries < 9; tries++, j *= 10) {
      var B = A.map(function (row, i) { var r = Float64Array.from(row); r[i] += j; return r; });
      var L = cholesky(B);
      if (L) return { L: L, jitter: j };
    }
    throw new Error('matrix is not positive semidefinite');
  }

  /* Solve L x = b (forward) and Lᵀ x = b (backward). */
  function forward(L, b) {
    var n = b.length, x = new Float64Array(n);
    for (var i = 0; i < n; i++) {
      var s = b[i], Li = L[i];
      for (var k = 0; k < i; k++) s -= Li[k] * x[k];
      x[i] = s / Li[i];
    }
    return x;
  }

  function backward(L, b) {
    var n = b.length, x = new Float64Array(n);
    for (var i = n - 1; i >= 0; i--) {
      var s = b[i];
      for (var k = i + 1; k < n; k++) s -= L[k][i] * x[k];
      x[i] = s / L[i][i];
    }
    return x;
  }

  /* The GP posterior over f at test inputs xs (default: the plotting grid).
     obs is [{x, y}]. Returns the mean, the full covariance, the pointwise
     standard deviation, and the jitter that K + σ²I needed. */
  function posterior(obs, ell, sigma, xs) {
    xs = xs || GRID;
    var n = obs.length, m = xs.length, a, b, i, k;

    var cov = [];                             // starts as the prior K**
    for (a = 0; a < m; a++) {
      var row = new Float64Array(m);
      for (b = 0; b < m; b++) row[b] = rbf(xs[a], xs[b], ell);
      cov.push(row);
    }
    var mean = new Float64Array(m), jitter = 0;

    if (n) {
      var Ky = obs.map(function (p, r) {
        var out = new Float64Array(n);
        for (var c = 0; c < n; c++) out[c] = rbf(p.x, obs[c].x, ell);
        out[r] += sigma * sigma;
        return out;
      });
      var ch = cholJitter(Ky, JITTER), L = ch.L;
      jitter = ch.jitter;

      var y = Float64Array.from(obs, function (p) { return p.y; });
      var alpha = backward(L, forward(L, y));                // (K + σ²I)⁻¹ y

      var Ks = obs.map(function (p) {                        // K*, n × m
        var out = new Float64Array(m);
        for (var c = 0; c < m; c++) out[c] = rbf(p.x, xs[c], ell);
        return out;
      });

      // V = L⁻¹ K*, so K*ᵀ (K + σ²I)⁻¹ K* = Vᵀ V
      var V = [];
      for (i = 0; i < n; i++) {
        var v = Float64Array.from(Ks[i]);
        for (k = 0; k < i; k++) {
          var lik = L[i][k], Vk = V[k];
          for (a = 0; a < m; a++) v[a] -= lik * Vk[a];
        }
        for (a = 0; a < m; a++) v[a] /= L[i][i];
        V.push(v);
      }

      for (a = 0; a < m; a++) {
        var s = 0;
        for (i = 0; i < n; i++) s += Ks[i][a] * alpha[i];
        mean[a] = s;
      }

      for (a = 0; a < m; a++) {
        for (b = a; b < m; b++) {
          var q = 0;
          for (i = 0; i < n; i++) q += V[i][a] * V[i][b];
          cov[a][b] -= q;
          if (b !== a) cov[b][a] = cov[a][b];
        }
      }
    }

    var sd = new Float64Array(m);
    for (a = 0; a < m; a++) sd[a] = Math.sqrt(Math.max(0, cov[a][a]));
    return { xs: xs, mean: mean, cov: cov, sd: sd, jitter: jitter };
  }

  /* Joint draws from the posterior: f = mean + L z, with L L ᵀ = cov (plus a
     little jitter). Each z is a fixed vector of standard normals, so the same
     z gives a sample that morphs smoothly as the data change. */
  function drawSamples(post, Z) {
    var L = cholJitter(post.cov, SAMPLE_JITTER).L, m = post.mean.length;
    return Z.map(function (z) {
      var f = Float64Array.from(post.mean);
      for (var a = 0; a < m; a++) {
        var s = 0, La = L[a];
        for (var k = 0; k <= a; k++) s += La[k] * z[k];
        f[a] += s;
      }
      return f;
    });
  }

  /* Add an observation. An x within SNAP of an existing input reuses that x
     exactly, so near-duplicates become clean repeats instead of two inputs a
     hair apart that a noise-free model would have to pass through with a
     near-vertical slope. */
  function addObservation(obs, x, y, src) {
    var near = null;
    obs.forEach(function (p) {
      if (Math.abs(p.x - x) < SNAP && (!near || Math.abs(p.x - x) < Math.abs(near.x - x))) near = p;
    });
    var nx = near ? near.x : x;
    return { obs: obs.concat([{ x: nx, y: y, src: src || 'click' }]), repeat: !!near, x: nx };
  }

  /* Share of the grid where the new mean is farther from the true function
     than the old mean was. */
  function worseFraction(oldMean, newMean, xs) {
    xs = xs || GRID;
    var worse = 0;
    for (var a = 0; a < xs.length; a++) {
      var f = trueF(xs[a]);
      if (Math.abs(newMean[a] - f) > Math.abs(oldMean[a] - f) + 1e-6) worse++;
    }
    return worse / xs.length;
  }

  var MATH = {
    X_MIN: X_MIN, X_MAX: X_MAX, GRID: GRID, GRID_N: GRID_N, ELL_MIN: ELL_MIN, ELL_MAX: ELL_MAX,
    SIGMA_MAX: SIGMA_MAX, MAX_SAMPLES: MAX_SAMPLES, JITTER: JITTER, SNAP: SNAP, START: START,
    PRESET_X: PRESET_X, PRESET_Z: PRESET_Z,
    trueF: trueF, rbf: rbf, linspace: linspace, mulberry32: mulberry32, gaussian: gaussian, normals: normals,
    presetY: presetY, cholesky: cholesky, cholJitter: cholJitter, forward: forward, backward: backward,
    posterior: posterior, drawSamples: drawSamples, addObservation: addObservation, worseFraction: worseFraction
  };

  if (typeof module !== 'undefined' && module.exports) module.exports = MATH;
  if (typeof document === 'undefined') return;

  /* ==========================================================================
     PART 2 — DRAWING
     ====================================================================== */

  var NS = 'http://www.w3.org/2000/svg';
  var L = 46, R = 626, T = 14, B = 352;          // plot area in the viewBox
  var Y_MIN = -3, Y_MAX = 3;
  var TWEEN_MS = 450;
  var reduceMotion = global.matchMedia && global.matchMedia('(prefers-reduced-motion: reduce)').matches;

  function $(id) { return document.getElementById(id); }

  function el(name, attrs, parent) {
    var e = document.createElementNS(NS, name);
    for (var k in attrs) e.setAttribute(k, attrs[k]);
    if (parent) parent.appendChild(e);
    return e;
  }

  function sx(x) { return L + (x - X_MIN) / (X_MAX - X_MIN) * (R - L); }
  function sy(y) { return B - (y - Y_MIN) / (Y_MAX - Y_MIN) * (B - T); }
  // Clamped for path data only; the clip path hides anything off the plot.
  function cy(y) { return Math.max(T - 400, Math.min(B + 400, sy(y))); }

  function f2(v) { return v.toFixed(2); }
  function plural(n, w) { return n + ' ' + w + (n === 1 ? '' : 's'); }

  var rand = mulberry32(17);
  var Z = [];
  for (var zi = 0; zi < MAX_SAMPLES; zi++) Z.push(normals(rand, GRID_N));

  var state = {
    obs: [], ell: START.ell, sigma: START.sigma, nSamples: START.samples,
    showTrue: false, showSamples: true, mode: 'click', note: null
  };
  var current = null;   // posterior for the current state
  var shown = null;     // the curves as currently drawn: { mean, sd, samples }
  var tweening = null;

  var svg = $('chart');

  // grid and axes
  var gridG = el('g', { 'class': 'grid' }, svg);
  for (var gx = X_MIN; gx <= X_MAX; gx++) el('line', { x1: sx(gx), y1: T, x2: sx(gx), y2: B }, gridG);
  for (var gy = Y_MIN; gy <= Y_MAX; gy++) el('line', { x1: L, y1: sy(gy), x2: R, y2: sy(gy) }, gridG);
  el('line', { 'class': 'zero', x1: L, y1: sy(0), x2: R, y2: sy(0) }, svg);
  el('line', { 'class': 'axis', x1: L, y1: B, x2: R, y2: B }, svg);
  el('line', { 'class': 'axis', x1: L, y1: T, x2: L, y2: B }, svg);
  for (var tx = X_MIN; tx <= X_MAX; tx++) {
    el('text', { 'class': 'tick', x: sx(tx), y: B + 20, 'text-anchor': 'middle' }, svg).textContent = tx;
  }
  for (var ty = Y_MIN; ty <= Y_MAX; ty++) {
    el('text', { 'class': 'tick', x: L - 9, y: sy(ty) + 4, 'text-anchor': 'end' }, svg).textContent = ty;
  }
  el('text', { 'class': 'axis-title', x: (L + R) / 2, y: B + 42 }, svg).textContent = 'input x';
  el('text', { 'class': 'axis-title', x: 0, y: 0, transform: 'translate(14,' + (T + B) / 2 + ') rotate(-90)' }, svg)
    .textContent = 'f(x)';

  var defs = el('defs', {}, svg);
  var clip = el('clipPath', { id: 'plotClip' }, defs);
  el('rect', { x: L, y: T, width: R - L, height: B - T }, clip);

  // marks, back to front
  var plot = el('g', { 'clip-path': 'url(#plotClip)' }, svg);
  var band = el('path', { 'class': 'band' }, plot);
  var sampleG = el('g', { 'class': 'samples' }, plot);
  var samplePaths = Z.map(function () { return el('path', { 'class': 'sample' }, sampleG); });
  var truth = el('path', { 'class': 'truth' }, plot);
  var meanPath = el('path', { 'class': 'mean' }, plot);
  var sampleLabel = el('text', { 'class': 'sample-label', 'text-anchor': 'end' }, plot);
  sampleLabel.textContent = 'possible functions';
  var dotG = el('g', {}, plot);

  var ghost = el('g', { 'class': 'ghost', visibility: 'hidden' }, svg);
  var ghostLine = el('line', { y1: T, y2: B }, ghost);
  var ghostDot = el('circle', { r: 7 }, ghost);
  var ghostText = el('text', { 'class': 'ghost-text', y: T + 18 }, ghost);

  function linePath(arr) {
    var d = '';
    for (var a = 0; a < GRID_N; a++) d += (a ? 'L' : 'M') + sx(GRID[a]).toFixed(1) + ',' + cy(arr[a]).toFixed(1);
    return d;
  }

  function bandPath(mean, sd) {
    var d = '', a;
    for (a = 0; a < GRID_N; a++) d += (a ? 'L' : 'M') + sx(GRID[a]).toFixed(1) + ',' + cy(mean[a] + 2 * sd[a]).toFixed(1);
    for (a = GRID_N - 1; a >= 0; a--) d += 'L' + sx(GRID[a]).toFixed(1) + ',' + cy(mean[a] - 2 * sd[a]).toFixed(1);
    return d + 'Z';
  }

  truth.setAttribute('d', linePath(GRID.map(trueF)));

  function drawCurves() {
    band.setAttribute('d', bandPath(shown.mean, shown.sd));
    meanPath.setAttribute('d', linePath(shown.mean));
    samplePaths.forEach(function (p, i) {
      var on = state.showSamples && i < state.nSamples;
      p.setAttribute('visibility', on ? 'visible' : 'hidden');
      if (on) p.setAttribute('d', linePath(shown.samples[i]));
    });
    // Label the first sample where the label's box sits farthest from every
    // observation dot, preferring the right edge when nothing is in the way.
    var LW = 124, LH = 16;
    function spot(i) {
      var cx = Math.min(R - 4 - LW / 2, Math.max(L + 4 + LW / 2, sx(GRID[i])));
      var by = Math.max(T + 16, Math.min(B - 8, sy(shown.samples[0][i]) - 8));
      var clear = state.obs.reduce(function (g, p) {
        var px = sx(p.x), py = sy(p.y);
        var dx = Math.max(cx - LW / 2 - px, 0, px - cx - LW / 2), dy = Math.max(by - LH - py, 0, py - by);
        return Math.min(g, Math.hypot(dx, dy));
      }, Infinity);
      return { x: cx, y: by, clear: clear };
    }
    var best = spot(Math.round((GRID_N - 1) * 0.9));
    for (var c = 0.1; c < 0.9; c += 0.04) {
      var sp = spot(Math.round((GRID_N - 1) * c));
      if (best.clear < 14 && sp.clear > best.clear) best = sp;
    }
    sampleLabel.setAttribute('text-anchor', 'middle');
    sampleLabel.setAttribute('x', best.x);
    sampleLabel.setAttribute('y', best.y);
    sampleLabel.setAttribute('visibility', state.showSamples ? 'visible' : 'hidden');
    truth.setAttribute('visibility', state.showTrue ? 'visible' : 'hidden');
  }

  function drawDots() {
    dotG.textContent = '';
    state.obs.forEach(function (p, i) {
      var last = i === state.obs.length - 1;
      el('circle', { 'class': 'obs' + (last && state.justAdded ? ' is-new' : ''), cx: sx(p.x), cy: sy(p.y), r: 6 }, dotG);
    });
    state.justAdded = false;
  }

  function lerp(a, b, t) {
    var out = new Float64Array(a.length);
    for (var i = 0; i < a.length; i++) out[i] = a[i] + (b[i] - a[i]) * t;
    return out;
  }

  /* Move the drawn curves to target, smoothly if animate. */
  function show(target, animate) {
    if (tweening) { cancelAnimationFrame(tweening); tweening = null; }
    if (!animate || !shown || reduceMotion) { shown = target; drawCurves(); return; }
    var from = shown, t0 = null;
    function frame(ts) {
      if (t0 === null) t0 = ts;
      var t = Math.min(1, (ts - t0) / TWEEN_MS);
      var e = t < 0.5 ? 2 * t * t : 1 - Math.pow(-2 * t + 2, 2) / 2;
      shown = {
        mean: lerp(from.mean, target.mean, e),
        sd: lerp(from.sd, target.sd, e),
        samples: target.samples.map(function (s, i) { return lerp(from.samples[i], s, e); })
      };
      drawCurves();
      tweening = t < 1 ? requestAnimationFrame(frame) : null;
    }
    tweening = requestAnimationFrame(frame);
  }

  function presetCount() {
    return state.obs.filter(function (p) { return p.src === 'preset'; }).length;
  }

  function renderText() {
    var n = state.obs.length, post = current;
    var avgBand = 0;
    for (var a = 0; a < GRID_N; a++) avgBand += 4 * post.sd[a];
    avgBand /= GRID_N;

    $('phase').textContent = n ? 'Posterior' : 'Prior';
    $('count').textContent = plural(n, 'observation');
    $('bandW').textContent = f2(avgBand);

    $('lead').textContent = n
      ? 'This is the posterior. The observations update the mean and reduce uncertainty, especially around related inputs.'
      : 'This is the prior. The GP describes functions that are plausible before seeing data.';

    var noise;
    if (!n) {
      noise = 'With no data the mean is 0 everywhere and the band is the same width at every x. Each faint line is one whole function the prior considers plausible.';
    } else if (state.sigma === 0) {
      noise = 'Noise-free (σ = 0): the mean passes through every distinct observed point, and the band pinches to essentially zero there. The model is certain what f is at inputs it has seen.';
    } else {
      noise = 'Noisy (σ = ' + f2(state.sigma) + '): each y is f(x) plus noise, so the mean does not have to pass through every point, and some uncertainty about f generally remains even at observed inputs.';
    }
    $('noiseNote').textContent = noise;

    $('eventNote').hidden = !state.note;
    $('eventNote').textContent = state.note || '';
    $('moreData').hidden = !n;

    var used = presetCount(), addBtn = $('add');
    addBtn.disabled = used >= PRESET_X.length || n >= MAX_OBS;
    $('addCount').textContent = used >= PRESET_X.length ? 'all ' + PRESET_X.length + ' used' : (used + 1) + ' of ' + PRESET_X.length;
    $('back').disabled = !n;

    svg.setAttribute('aria-label', (n ? 'GP posterior after ' + plural(n, 'observation') : 'GP prior with no observations') +
      '. Average width of the ±2 standard deviation band: ' + f2(avgBand) + '.');
  }

  /* Recompute the posterior and redraw everything. */
  function update(animate) {
    current = posterior(state.obs, state.ell, state.sigma);
    show({ mean: current.mean, sd: current.sd, samples: drawSamples(current, Z) }, animate);
    drawDots();
    renderText();
  }

  /* ==========================================================================
     PART 3 — CONTROLS
     ====================================================================== */

  /* Add a point, then say whether the mean got more accurate everywhere. */
  function add(x, y, src) {
    if (state.obs.length >= MAX_OBS) {
      state.note = 'That is ' + MAX_OBS + ' points, the most this page keeps. Use Back or Reset.';
      renderText();
      return;
    }
    var before = current.mean, r = addObservation(state.obs, x, y, src);
    state.obs = r.obs;
    state.justAdded = true;
    update(true);

    var notes = [];
    if (r.repeat) {
      notes.push('Repeated input: this point reuses x = ' + f2(r.x) + '. ' + (state.sigma === 0
        ? 'Noise-free, the model can’t pass through two different values at one x, so the mean there settles on their average. Turn on noise to treat them as two noisy readings of one f(x).'
        : 'With noise, two readings at one x are pooled: the mean there is pulled toward their average, and the band narrows further.'));
    }
    var worse = worseFraction(before, current.mean);
    if (!state.showTrue) {
      notes.push('Reveal the true function to check whether this point made the mean more accurate everywhere.');
    } else if (worse > 0.005) {
      notes.push('Checked against the hidden function: this point made the mean less accurate on ' +
        Math.round(worse * 100) + '% of the axis, even though the band got no wider anywhere.');
    } else {
      notes.push('Checked against the hidden function: this time the mean got no less accurate anywhere. That doesn’t always happen.');
    }
    state.note = notes.join(' ');
    renderText();
  }

  $('add').addEventListener('click', function () {
    var i = presetCount();
    if (i < PRESET_X.length) add(PRESET_X[i], presetY(i, state.sigma), 'preset');
  });

  $('back').addEventListener('click', function () {
    state.obs = state.obs.slice(0, -1);
    state.note = null;
    update(true);
  });

  // Reset: no data, starting ℓ and σ. Display toggles stay as they are.
  $('reset').addEventListener('click', function () {
    state.obs = [];
    state.note = null;
    state.ell = START.ell; state.sigma = START.sigma;
    ellSlider.value = ellToSlider(START.ell);
    sigmaSlider.value = START.sigma;
    renderSettings();
    update(true);
  });

  // Clicking the chart: x is always where you clicked; y depends on the mode.
  function toPlot(evt) {
    var pt = svg.createSVGPoint();
    pt.x = evt.clientX; pt.y = evt.clientY;
    var q = pt.matrixTransform(svg.getScreenCTM().inverse());
    return { px: q.x, py: q.y,
      x: X_MIN + (q.x - L) / (R - L) * (X_MAX - X_MIN),
      y: Y_MIN + (B - q.y) / (B - T) * (Y_MAX - Y_MIN) };
  }

  function inPlot(p) { return p.px >= L && p.px <= R && p.py >= T && p.py <= B; }

  function yFor(p) {
    return state.mode === 'click' ? p.y : trueF(p.x) + state.sigma * gaussian(Math.random);
  }

  svg.addEventListener('click', function (e) {
    var p = toPlot(e);
    if (!inPlot(p)) return;
    add(p.x, yFor(p), state.mode === 'click' ? 'click' : 'sample');
  });

  // Mouse hover: preview where a click would add its point.
  svg.addEventListener('pointermove', function (e) {
    if (e.pointerType !== 'mouse') return;
    var p = toPlot(e);
    if (!inPlot(p)) { ghost.setAttribute('visibility', 'hidden'); return; }
    var near = state.obs.reduce(function (b, o) {
      return Math.abs(o.x - p.x) < SNAP && (!b || Math.abs(o.x - p.x) < Math.abs(b.x - p.x)) ? o : b;
    }, null);
    var x = near ? near.x : p.x, gx = sx(x);
    ghost.setAttribute('visibility', 'visible');
    ghostLine.setAttribute('x1', gx); ghostLine.setAttribute('x2', gx);
    ghostDot.setAttribute('cx', gx); ghostDot.setAttribute('cy', p.py);
    ghostDot.setAttribute('visibility', state.mode === 'click' ? 'visible' : 'hidden');
    var right = gx < (L + R) / 2;
    ghostText.setAttribute('x', right ? gx + 8 : gx - 8);
    ghostText.setAttribute('text-anchor', right ? 'start' : 'end');
    ghostText.textContent = (state.mode === 'click'
      ? 'click adds (' + f2(x) + ', ' + f2(p.y) + ')'
      : 'click adds x = ' + f2(x) + ', y = sin(x)' + (state.sigma ? ' + noise' : '')) + (near ? ' · same x' : '');
  });
  svg.addEventListener('pointerleave', function () { ghost.setAttribute('visibility', 'hidden'); });

  // Click mode
  var MODE_TEXT = {
    click: 'A click adds a point at exactly the x and the height you clicked. You choose the data, even data the hidden function would never produce.',
    truth: 'A click uses only its x. The y comes from the hidden function at that x, sin(x), plus a fresh noise draw with the current σ (no noise when noise-free). The height you click is ignored.'
  };
  Array.prototype.forEach.call(document.querySelectorAll('input[name="mode"]'), function (r) {
    r.addEventListener('change', function () {
      state.mode = r.value;
      $('modeNote').textContent = MODE_TEXT[state.mode];
    });
  });

  // Length scale: log slider 0–100, so the short end isn't cramped.
  var ellSlider = $('ellSlider');
  function ellToSlider(v) { return 100 * Math.log(v / ELL_MIN) / Math.log(ELL_MAX / ELL_MIN); }
  function sliderToEll(v) { return ELL_MIN * Math.pow(ELL_MAX / ELL_MIN, v / 100); }
  ellSlider.value = ellToSlider(START.ell);

  var sigmaSlider = $('sigmaSlider');
  sigmaSlider.max = SIGMA_MAX;
  sigmaSlider.value = START.sigma;

  var countSlider = $('countSlider');
  countSlider.max = MAX_SAMPLES;
  countSlider.value = START.samples;

  function renderSettings() {
    $('ellVal').textContent = 'ℓ = ' + f2(state.ell);
    $('ellNote').textContent = 'Inputs more than about 2ℓ = ' + (2 * state.ell).toFixed(1) +
      ' apart barely inform each other (correlation ' + rbf(0, 2 * state.ell, state.ell).toFixed(2) + ').';
    $('sigmaVal').textContent = state.sigma === 0 ? 'σ = 0 · noise-free' : 'σ = ' + f2(state.sigma);
    $('countVal').textContent = state.nSamples;
  }

  ellSlider.addEventListener('input', function () {
    state.ell = sliderToEll(parseFloat(ellSlider.value));
    state.note = null;
    renderSettings();
    update(false);
  });

  sigmaSlider.addEventListener('input', function () {
    state.sigma = Math.round(parseFloat(sigmaSlider.value) * 100) / 100;
    state.note = null;
    renderSettings();
    update(false);
  });

  countSlider.addEventListener('input', function () {
    state.nSamples = parseInt(countSlider.value, 10);
    renderSettings();
    drawCurves();
  });

  function toggle(btn, key) {
    btn.addEventListener('click', function () {
      state[key] = !state[key];
      btn.setAttribute('aria-pressed', state[key]);
      $('legendTrue').hidden = !state.showTrue;
      $('legendSamples').hidden = !state.showSamples;
      drawCurves();
    });
  }
  toggle($('trueBtn'), 'showTrue');
  toggle($('samplesBtn'), 'showSamples');

  $('redraw').addEventListener('click', function () {
    Z = Z.map(function () { return normals(Math.random, GRID_N); });
    show({ mean: current.mean, sd: current.sd, samples: drawSamples(current, Z) }, true);
  });

  $('modeNote').textContent = MODE_TEXT[state.mode];
  renderSettings();
  update(false);

})(typeof globalThis !== 'undefined' ? globalThis : this);
