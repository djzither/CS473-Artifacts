/* ============================================================================
   Which Ships Count? — SVMs in practice.
   CS 473 module 12 · Murphy, PML §17.3.5–17.4

   Rebel (y = −1) vs. Imperial (y = +1) ships on a 2D radar, then fuel use.

     score        f(x) = Σₙ αₙ yₙ K(xₙ, x) + b
     hinge loss   max(0, 1 − y f(x))
     Platt        p(Imperial | x) = σ(a f(x) + b_P), a and b_P fit on validation data
     SVR          f(x) = Σₙ θₙ K(xₙ, x) + b,  θₙ = αₙ − αₙ*  (either sign)
     ε-loss       max(0, |y − f(x)| − ε)

   Every model on the page is fitted here by one SMO solver (LIBSVM's
   maximal-violating-pair rule). The ships and fuel readings are synthetic.

     PART 1  Math      pure, DOM-free, exported for test-math.js under Node
     PART 2  Drawing   SVG helpers
     PART 3  Scenes    one setup per scene, plus navigation
   ========================================================================= */

(function (global) {
  'use strict';

  /* ==========================================================================
     PART 1 — MATH
     ====================================================================== */

  /* Seeded random numbers, so every dataset is the same on every load. */
  function mulberry32(seed) {
    var s = seed >>> 0;
    return function () {
      s = (s + 0x6D2B79F5) >>> 0;
      var t = s;
      t = Math.imul(t ^ (t >>> 15), t | 1);
      t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }
  function gauss(r) {
    var u = 0;
    while (u === 0) u = r();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * r());
  }
  function round2(v) { return Math.round(v * 100) / 100; }

  /* ----------------------------------------------------------- kernels -- */

  function dot(a, b) { return a[0] * b[0] + a[1] * b[1]; }
  function linearK(a, b) { return dot(a, b); }
  function rbfK(gamma) {
    return function (a, b) {
      var d0 = a[0] - b[0], d1 = a[1] - b[1];
      return Math.exp(-gamma * (d0 * d0 + d1 * d1));
    };
  }
  function rbf1(gamma) { return function (a, b) { return Math.exp(-gamma * (a - b) * (a - b)); }; }

  /* ---------------------------------------------------------------- SMO -- */

  /* Minimise ½ βᵀQβ + pᵀβ  s.t.  zᵀβ = 0, 0 ≤ β ≤ C,  with Q_st = z_s z_t K_st.
     Both the SVM and SVR duals have this form (as in LIBSVM).
     Kmat(s, t) returns K_st. Returns β and the offset b of the decision function. */
  function smo(Kmat, z, p, C, tol, maxIter) {
    tol = tol || 1e-7;
    maxIter = maxIter || 200000;
    var n = z.length, t, k;
    var beta = new Float64Array(n);
    var G = Float64Array.from(p);            // gradient Qβ + p
    var iters = 0, converged = false;
    while (iters < maxIter) {
      var i = -1, j = -1, gmax = -Infinity, gmin = Infinity;
      for (t = 0; t < n; t++) {
        var v = -z[t] * G[t];
        var up = z[t] > 0 ? beta[t] < C : beta[t] > 0;
        var low = z[t] > 0 ? beta[t] > 0 : beta[t] < C;
        if (up && v > gmax) { gmax = v; i = t; }
        if (low && v < gmin) { gmin = v; j = t; }
      }
      if (i < 0 || j < 0 || gmax - gmin < tol) { converged = true; break; }
      iters++;
      // Move βᵢ += zᵢ s, βⱼ −= zⱼ s (keeps zᵀβ = 0), exactly minimising along that line.
      var eta = Math.max(Kmat(i, i) + Kmat(j, j) - 2 * Kmat(i, j), 1e-12);
      var s = Math.min((gmax - gmin) / eta,
        z[i] > 0 ? C - beta[i] : beta[i],
        z[j] > 0 ? beta[j] : C - beta[j]);
      beta[i] += z[i] * s;
      beta[j] -= z[j] * s;
      if (beta[i] < 1e-13) beta[i] = 0; if (beta[i] > C) beta[i] = C;
      if (beta[j] < 1e-13) beta[j] = 0; if (beta[j] > C) beta[j] = C;
      for (k = 0; k < n; k++) G[k] += z[k] * s * (Kmat(k, i) - Kmat(k, j));
    }
    // Offset from the KKT conditions (LIBSVM's calculate_rho), b = −ρ.
    var sum = 0, cnt = 0, ub = Infinity, lb = -Infinity;
    for (t = 0; t < n; t++) {
      var yG = z[t] * G[t];
      if (beta[t] >= C * (1 - 1e-12)) { if (z[t] < 0) ub = Math.min(ub, yG); else lb = Math.max(lb, yG); }
      else if (beta[t] <= 0) { if (z[t] > 0) ub = Math.min(ub, yG); else lb = Math.max(lb, yG); }
      else { sum += yG; cnt++; }
    }
    var rho = cnt ? sum / cnt : (isFinite(ub) && isFinite(lb) ? (ub + lb) / 2 : (isFinite(ub) ? ub : lb));
    return { beta: beta, b: -rho, iters: iters, converged: converged };
  }

  function gram(X, kernel) {
    return X.map(function (a) { return X.map(function (b) { return kernel(a, b); }); });
  }

  /* Soft-margin SVM classifier. coef[n] = αₙ yₙ; support vectors have αₙ > 0. */
  function trainSVC(X, y, kernel, C) {
    var n = X.length, K = gram(X, kernel);
    var r = smo(function (s, t) { return K[s][t]; }, y, y.map(function () { return -1; }), C);
    var alpha = Array.from(r.beta), coef = alpha.map(function (a, k) { return a * y[k]; });
    var sv = [];
    for (var k = 0; k < n; k++) if (alpha[k] > 1e-8) sv.push(k);
    var b = r.b;
    function f(x) {
      var s = b;
      for (var q = 0; q < sv.length; q++) s += coef[sv[q]] * kernel(X[sv[q]], x);
      return s;
    }
    return { X: X, y: y, C: C, kernel: kernel, alpha: alpha, coef: coef, b: b, sv: sv, f: f, converged: r.converged };
  }

  /* For a linear kernel, collapse to f(x) = wᵀx + b with w = Σ αₙ yₙ xₙ. */
  function weights(m) {
    var w = [0, 0];
    m.coef.forEach(function (c, k) { w[0] += c * m.X[k][0]; w[1] += c * m.X[k][1]; });
    return w;
  }

  /* ε-insensitive support vector regression on 1D inputs.
     Variables β = (α, α*), z = (+1…, −1…), p = (ε − y, ε + y).
     Prediction coefficient θₙ = αₙ − αₙ*: positive when the reading sits above
     the tube, negative when below. */
  function trainSVR(xs, ys, kernel, C, eps) {
    var n = xs.length, K = gram(xs.map(function (v) { return [v, 0]; }), function (a, b) { return kernel(a[0], b[0]); });
    var z = [], p = [], k;
    for (k = 0; k < n; k++) { z.push(1); p.push(eps - ys[k]); }
    for (k = 0; k < n; k++) { z.push(-1); p.push(eps + ys[k]); }
    var r = smo(function (s, t) { return K[s % n][t % n]; }, z, p, C);
    var theta = [], sv = [];
    for (k = 0; k < n; k++) {
      theta.push(r.beta[k] - r.beta[k + n]);
      if (Math.abs(theta[k]) > 1e-8) sv.push(k);
    }
    var b = r.b;
    function f(x) {
      var s = b;
      for (var q = 0; q < sv.length; q++) s += theta[sv[q]] * kernel(xs[sv[q]], x);
      return s;
    }
    return { xs: xs, ys: ys, C: C, eps: eps, alpha: Array.from(r.beta.slice(0, n)), alphaStar: Array.from(r.beta.slice(n)),
      theta: theta, b: b, sv: sv, f: f, converged: r.converged };
  }

  /* ------------------------------------------------------------- losses -- */

  function hinge(m) { return Math.max(0, 1 - m); }                 // m = y f(x)
  function logLoss2(m) { return Math.log1p(Math.exp(-m)) / Math.LN2; } // log₂(1 + e^(−m))
  function epsLoss(r, eps) { return Math.max(0, Math.abs(r) - eps); }
  function sigmoid(t) { return 1 / (1 + Math.exp(-t)); }

  /* ------------------------------------------------------------- Platt -- */

  /* Fit p = σ(a f + b) by maximum likelihood on (score, label) pairs, using
     Platt's smoothed targets (N₊+1)/(N₊+2) and 1/(N₋+2). Newton with backtracking. */
  function plattFit(scores, labels) {
    var nPos = labels.filter(function (l) { return l > 0; }).length, nNeg = labels.length - nPos;
    var tPos = (nPos + 1) / (nPos + 2), tNeg = 1 / (nNeg + 2);
    var T = labels.map(function (l) { return l > 0 ? tPos : tNeg; });
    function nll(a, b) {
      var s = 0;
      for (var i = 0; i < scores.length; i++) {
        var u = a * scores[i] + b;
        // −[t log σ(u) + (1−t) log(1−σ(u))] = log(1+e^u) − t u, computed stably
        s += (u > 0 ? u + Math.log1p(Math.exp(-u)) : Math.log1p(Math.exp(u))) - T[i] * u;
      }
      return s;
    }
    var a = 0, b = Math.log((nPos + 1) / (nNeg + 1)), cur = nll(a, b);
    for (var it = 0; it < 100; it++) {
      var ga = 0, gb = 0, haa = 1e-12, hab = 0, hbb = 1e-12;
      for (var i = 0; i < scores.length; i++) {
        var pr = sigmoid(a * scores[i] + b), d = pr - T[i], w = pr * (1 - pr), f = scores[i];
        ga += d * f; gb += d; haa += w * f * f; hab += w * f; hbb += w;
      }
      if (Math.abs(ga) + Math.abs(gb) < 1e-9) break;
      var det = haa * hbb - hab * hab;
      var da = -(hbb * ga - hab * gb) / det, db = -(haa * gb - hab * ga) / det;
      var step = 1, next;
      while (step > 1e-10 && (next = nll(a + step * da, b + step * db)) > cur) step /= 2;
      a += step * da; b += step * db; cur = next;
    }
    return { a: a, b: b, nll: cur };
  }

  /* Reliability table: bin the predicted probabilities, compare with how often
     the ship really was Imperial. */
  function reliability(probs, labels, edges) {
    edges = edges || [0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0001];
    var bins = [];
    for (var k = 0; k + 1 < edges.length; k++) bins.push({ lo: edges[k], hi: Math.min(edges[k + 1], 1), n: 0, sumP: 0, pos: 0 });
    probs.forEach(function (p, i) {
      for (var k = 0; k < bins.length; k++) {
        if (p >= edges[k] && p < edges[k + 1]) { bins[k].n++; bins[k].sumP += p; if (labels[i] > 0) bins[k].pos++; break; }
      }
    });
    bins.forEach(function (bn) { bn.meanP = bn.n ? bn.sumP / bn.n : NaN; bn.frac = bn.n ? bn.pos / bn.n : NaN; });
    return bins;
  }

  function meanLogLoss(probs, labels) {
    var s = 0;
    probs.forEach(function (p, i) { p = Math.min(1 - 1e-12, Math.max(1e-12, p)); s -= labels[i] > 0 ? Math.log(p) : Math.log(1 - p); });
    return s / probs.length;
  }

  /* ------------------------------------------------------------ datasets -- */

  // Scenes 1–2: sixteen identified ships. x₁ = engine heat, x₂ = hull mass (standardized).
  var SHIPS = [
    { name: 'Red Five',       x: [-2.2, -1.0], y: -1 },
    { name: 'Gold Leader',    x: [-1.6,  0.4], y: -1 },
    { name: 'Rogue Two',      x: [-1.0, -1.8], y: -1 },
    { name: 'Tantive IV',     x: [-2.4,  1.2], y: -1 },
    { name: 'Blue Three',     x: [-0.6, -0.5], y: -1 },
    { name: 'Ghost',          x: [-0.1, -2.1], y: -1 },
    { name: 'Corellian Hauler', x: [-1.3, -0.4], y: -1 },
    { name: 'Y-wing Gray',    x: [ 0.1, -0.7], y: -1 },
    { name: 'TIE Alpha',      x: [ 1.8,  1.0], y:  1 },
    { name: 'TIE Bravo',      x: [ 0.8,  1.8], y:  1 },
    { name: 'Interceptor',    x: [ 2.2, -0.3], y:  1 },
    { name: 'Lambda Shuttle', x: [ 1.1,  0.5], y:  1 },
    { name: 'TIE Bomber',     x: [ 0.2,  2.3], y:  1 },
    { name: 'Star Destroyer', x: [ 2.4,  2.0], y:  1 },
    { name: 'Probe Carrier',  x: [ 1.6, -1.3], y:  1 },
    { name: 'Disguised TIE',  x: [-0.5,  0.6], y:  1 }
  ];
  var SHIPS_C = 1;

  // Scene 3: Rebel smugglers mask their readings near the middle; Imperial
  // patrols surround them. A few readings are noisy, so a perfect fit is a bad sign.
  function makeRing(seed, nEach) {
    var r = mulberry32(seed), out = [], i, a, rad;
    for (i = 0; i < nEach; i++) {
      a = 2 * Math.PI * r(); rad = Math.abs(gauss(r)) * 0.8;
      out.push({ x: [round2(rad * Math.cos(a)), round2(rad * Math.sin(a))], y: -1 });
    }
    for (i = 0; i < nEach; i++) {
      a = 2 * Math.PI * r(); rad = 1.85 + gauss(r) * 0.45;
      out.push({ x: [round2(rad * Math.cos(a)), round2(rad * Math.sin(a))], y: 1 });
    }
    return out.filter(function (s) { return Math.abs(s.x[0]) < 2.9 && Math.abs(s.x[1]) < 2.9; });
  }
  var RING_TRAIN = makeRing(7, 35);
  var RING_TEST = makeRing(1977, 200);
  var C_STEPS = [0.01, 0.1, 1, 10, 100, 1000];
  var GAMMA_STEPS = [0.03, 0.1, 0.3, 1, 3, 10, 30];

  // Scene 4: two overlapping fleets. Train the SVM on one set, fit Platt's a, b
  // on a second (validation), and check calibration on a third (test).
  function makeOverlap(seed, nEach) {
    var r = mulberry32(seed), out = [], i;
    for (i = 0; i < nEach; i++) out.push({ x: [-0.7 + gauss(r), -0.5 + gauss(r)], y: -1 });
    for (i = 0; i < nEach; i++) out.push({ x: [0.7 + gauss(r), 0.5 + gauss(r)], y: 1 });
    return out;
  }
  var CAL_TRAIN = makeOverlap(11, 25);
  var CAL_VAL = makeOverlap(22, 150);
  var CAL_TEST = makeOverlap(33, 400);
  var CAL_C = 0.03;

  // Scene 5: fuel use (tonnes) against cargo mass (kilotonnes). Synthetic.
  var FUEL = (function () {
    var r = mulberry32(5), xs = [], ys = [];
    for (var i = 0; i < 24; i++) {
      var x = round2(0.3 + 9.4 * i / 23 + (r() - 0.5) * 0.3);
      xs.push(x);
      ys.push(round2(2 + 0.5 * x + 1.1 * Math.sin(0.75 * x) + gauss(r) * 0.3));
    }
    return { xs: xs, ys: ys };
  })();
  var FUEL_GAMMA = 0.25, FUEL_C = 10;

  function xsOf(d) { return d.map(function (s) { return s.x; }); }
  function ysOf(d) { return d.map(function (s) { return s.y; }); }

  var MATH = {
    mulberry32: mulberry32, dot: dot, linearK: linearK, rbfK: rbfK, rbf1: rbf1, smo: smo,
    trainSVC: trainSVC, weights: weights, trainSVR: trainSVR,
    hinge: hinge, logLoss2: logLoss2, epsLoss: epsLoss, sigmoid: sigmoid,
    plattFit: plattFit, reliability: reliability, meanLogLoss: meanLogLoss,
    SHIPS: SHIPS, SHIPS_C: SHIPS_C, RING_TRAIN: RING_TRAIN, RING_TEST: RING_TEST,
    C_STEPS: C_STEPS, GAMMA_STEPS: GAMMA_STEPS,
    CAL_TRAIN: CAL_TRAIN, CAL_VAL: CAL_VAL, CAL_TEST: CAL_TEST, CAL_C: CAL_C,
    FUEL: FUEL, FUEL_GAMMA: FUEL_GAMMA, FUEL_C: FUEL_C, xsOf: xsOf, ysOf: ysOf
  };

  if (typeof module !== 'undefined' && module.exports) module.exports = MATH;
  if (typeof document === 'undefined') return;

  /* ==========================================================================
     PART 2 — DRAWING
     ====================================================================== */

  var NS = 'http://www.w3.org/2000/svg';
  var $ = function (sel) { return document.querySelector(sel); };

  function el(tag, attrs, parent) {
    var e = document.createElementNS(NS, tag);
    if (attrs) for (var k in attrs) e.setAttribute(k, attrs[k]);
    if (parent) parent.appendChild(e);
    return e;
  }
  function clear(g) { while (g.firstChild) g.removeChild(g.firstChild); }
  function fmt(v, d) {
    var s = v.toFixed(d == null ? 2 : d);
    if (/^-0\.?0*$/.test(s)) s = s.slice(1);
    return s.replace('-', '\u2212');
  }
  function paren(v, d) { return v < 0 ? '(' + fmt(v, d) + ')' : fmt(v, d); }
  function pct(v) { return Math.round(v * 100) + '%'; }

  function tex(node, src, display) {
    if (global.katex) {
      try { global.katex.render(src, node, { throwOnError: false, displayMode: !!display }); return; } catch (e) { /* fall through */ }
    }
    node.textContent = src;
    node.classList.add('tex-raw');
  }
  function renderTex(root) {
    root.querySelectorAll('[data-tex]').forEach(function (n) { tex(n, n.getAttribute('data-tex'), n.classList.contains('tex-block')); });
  }

  /* Legend items, shared by every scene. */
  var KEYS = {
    rebel: '<svg class="key-ico is-reb" viewBox="-11 -11 22 22"><use href="#ico-rebel" x="-11" y="-11" width="22" height="22"/></svg>Rebel (y = \u22121)',
    imperial: '<svg class="key-ico is-imp" viewBox="-11 -11 22 22"><use href="#ico-imperial" x="-11" y="-11" width="22" height="22"/></svg>Imperial (y = +1)',
    unknown: '<svg class="key-ico is-unk" viewBox="-11 -11 22 22"><use href="#ico-unknown" x="-11" y="-11" width="22" height="22"/></svg>unknown ship',
    boundary: '<span class="key key-boundary"></span>boundary f = 0',
    edges: '<span class="key key-edge"></span>margin edges f = \u00b11',
    edgesThin: '<span class="key key-edge"></span>f = \u00b11',
    sv: '<span class="key key-ring key-sv"></span>support vector',
    reading: '<svg class="key-ico" viewBox="-8 -8 16 16"><circle r="5" class="pt"/></svg>inside tube (\u03b8 = 0)',
    svpt: '<svg class="key-ico" viewBox="-8 -8 16 16"><circle r="5" class="pt is-sv"/></svg>on/outside tube: support vector',
    curve: '<span class="key key-boundary"></span>prediction f(x)',
    tube: '<span class="key key-corridor" style="background:rgba(127,216,232,.25)"></span>\u03b5 tube'
  };
  function fillLegends() {
    document.querySelectorAll('.legend[data-keys]').forEach(function (ul) {
      ul.innerHTML = ul.getAttribute('data-keys').split(' ').map(function (k) { return '<li>' + KEYS[k] + '</li>'; }).join('');
    });
  }

  /* Pointer position in an SVG's own coordinates. */
  function svgPoint(svg, evt) {
    var pt = svg.createSVGPoint();
    pt.x = evt.clientX; pt.y = evt.clientY;
    return pt.matrixTransform(svg.getScreenCTM().inverse());
  }

  /* A square radar showing [−3, 3]² with equal scales. */
  function Radar(svg) {
    var L = 3, PX = 420, ML = 40, MT = 10, MR = 10, MB = 40;
    svg.setAttribute('viewBox', '0 0 ' + (ML + PX + MR) + ' ' + (MT + PX + MB));
    var R = {
      svg: svg, L: L,
      sx: function (x) { return ML + (x + L) / (2 * L) * PX; },
      sy: function (y) { return MT + (L - y) / (2 * L) * PX; },
      fromEvent: function (evt) {
        var q = svgPoint(svg, evt);
        return [(q.x - ML) / PX * 2 * L - L, L - (q.y - MT) / PX * 2 * L];
      }
    };
    el('rect', { x: ML, y: MT, width: PX, height: PX, class: 'scope-bg' }, svg);
    var g = el('g', null, svg);
    for (var k = -L; k <= L; k++) {
      el('line', { x1: R.sx(k), x2: R.sx(k), y1: MT, y2: MT + PX, class: k === 0 ? 'grid-zero' : 'grid' }, g);
      el('line', { y1: R.sy(k), y2: R.sy(k), x1: ML, x2: ML + PX, class: k === 0 ? 'grid-zero' : 'grid' }, g);
      el('text', { x: R.sx(k), y: MT + PX + 16, 'text-anchor': 'middle', class: 'tick' }, g).textContent = fmt(k, 0);
      el('text', { x: ML - 7, y: R.sy(k) + 4, 'text-anchor': 'end', class: 'tick' }, g).textContent = fmt(k, 0);
    }
    el('text', { x: ML + PX / 2, y: MT + PX + 34, 'text-anchor': 'middle', class: 'axis-title' }, svg).textContent = 'engine heat x\u2081';
    el('text', { x: 12, y: MT + PX / 2, 'text-anchor': 'middle', class: 'axis-title', transform: 'rotate(-90 12 ' + (MT + PX / 2) + ')' }, svg).textContent = 'hull mass x\u2082';
    R.field = el('image', { x: ML, y: MT, width: PX, height: PX, preserveAspectRatio: 'none' }, svg);
    el('rect', { x: ML, y: MT, width: PX, height: PX, class: 'scope-frame' }, svg);
    R.lines = el('g', null, svg);
    R.marks = el('g', null, svg);
    R.top = el('g', null, svg);
    return R;
  }

  /* Tint each side of the boundary: light inside the margin |f| < 1, stronger beyond.
     Returns the grid of f values for contouring. */
  function paintField(R, f) {
    var N = 96, L = R.L, v = [], i, j;
    for (i = 0; i <= N; i++) {
      v.push(new Float64Array(N + 1));
      for (j = 0; j <= N; j++) v[i][j] = f([-L + 2 * L * i / N, -L + 2 * L * j / N]);
    }
    var cv = document.createElement('canvas');
    cv.width = cv.height = N;
    var ctx = cv.getContext('2d'), img = ctx.createImageData(N, N);
    for (i = 0; i < N; i++) for (j = 0; j < N; j++) {
      var val = (v[i][j] + v[i + 1][j] + v[i][j + 1] + v[i + 1][j + 1]) / 4;
      var k = ((N - 1 - j) * N + i) * 4;
      var c = val > 0 ? [255, 157, 66] : [78, 168, 255];
      img.data[k] = c[0]; img.data[k + 1] = c[1]; img.data[k + 2] = c[2];
      img.data[k + 3] = Math.abs(val) < 1 ? 22 : 52;
    }
    ctx.putImageData(img, 0, 0);
    R.field.setAttribute('href', cv.toDataURL());
    return { v: v, N: N };
  }

  /* Marching squares: the curve f = level, as many short segments. */
  function contour(R, grid, level) {
    var v = grid.v, N = grid.N, L = R.L, h = 2 * L / N, d = '';
    for (var i = 0; i < N; i++) for (var j = 0; j < N; j++) {
      var a = v[i][j] - level, b = v[i + 1][j] - level, c = v[i + 1][j + 1] - level, e = v[i][j + 1] - level;
      var x0 = -L + h * i, y0 = -L + h * j, pts = [];
      if ((a > 0) !== (b > 0)) pts.push([x0 + h * a / (a - b), y0]);
      if ((b > 0) !== (c > 0)) pts.push([x0 + h, y0 + h * b / (b - c)]);
      if ((e > 0) !== (c > 0)) pts.push([x0 + h * e / (e - c), y0 + h]);
      if ((a > 0) !== (e > 0)) pts.push([x0, y0 + h * a / (a - e)]);
      for (var k = 0; k + 1 < pts.length; k += 2) {
        d += 'M' + R.sx(pts[k][0]).toFixed(1) + ' ' + R.sy(pts[k][1]).toFixed(1) +
             'L' + R.sx(pts[k + 1][0]).toFixed(1) + ' ' + R.sy(pts[k + 1][1]).toFixed(1);
      }
    }
    return d;
  }

  /* The segment of {x : wᵀx + b = c} inside the radar square. */
  function lineSeg(R, w, b, c) {
    var L = R.L, pts = [], k = c - b;
    if (Math.abs(w[1]) > 1e-12) [-L, L].forEach(function (x) { var y = (k - w[0] * x) / w[1]; if (Math.abs(y) <= L + 1e-9) pts.push([x, y]); });
    if (Math.abs(w[0]) > 1e-12) [-L, L].forEach(function (y) { var x = (k - w[1] * y) / w[0]; if (Math.abs(x) <= L + 1e-9) pts.push([x, y]); });
    if (pts.length < 2) return '';
    return 'M' + R.sx(pts[0][0]) + ' ' + R.sy(pts[0][1]) + 'L' + R.sx(pts[1][0]) + ' ' + R.sy(pts[1][1]);
  }

  function drawLinear(R, w, b) {
    paintField(R, function (x) { return dot(w, x) + b; });
    clear(R.lines);
    [-1, 1].forEach(function (c) { var d = lineSeg(R, w, b, c); if (d) el('path', { d: d, class: 'edge' }, R.lines); });
    var d0 = lineSeg(R, w, b, 0); if (d0) el('path', { d: d0, class: 'boundary' }, R.lines);
  }

  function drawShip(R, g, x, y, extra) {
    var s = el('g', { class: 'ship' + (y > 0 ? ' is-imp' : '') + (extra ? ' ' + extra : ''), transform: 'translate(' + R.sx(x[0]).toFixed(1) + ',' + R.sy(x[1]).toFixed(1) + ')' }, g);
    el('use', { href: y > 0 ? '#ico-imperial' : '#ico-rebel', x: -10, y: -10, width: 20, height: 20 }, s);
    return s;
  }

  /* A plain x–y chart. */
  function Chart(svg, o) {
    var W = o.W || 520, H = o.H || 300, ML = 52, MR = 14, MT = 14, MB = 42;
    svg.setAttribute('viewBox', '0 0 ' + W + ' ' + H);
    var C = {
      svg: svg, x: o.x, y: o.y,
      sx: function (v) { return ML + (v - o.x[0]) / (o.x[1] - o.x[0]) * (W - ML - MR); },
      sy: function (v) { return MT + (o.y[1] - v) / (o.y[1] - o.y[0]) * (H - MT - MB); },
      left: ML, right: W - MR, top: MT, bottom: H - MB
    };
    el('rect', { x: ML, y: MT, width: W - ML - MR, height: H - MT - MB, class: 'scope-bg' }, svg);
    var g = el('g', null, svg);
    o.xt.forEach(function (t) {
      el('line', { x1: C.sx(t), x2: C.sx(t), y1: MT, y2: H - MB, class: 'grid' }, g);
      el('text', { x: C.sx(t), y: H - MB + 16, 'text-anchor': 'middle', class: 'tick' }, g).textContent = o.xf ? o.xf(t) : fmt(t, 0);
    });
    o.yt.forEach(function (t) {
      el('line', { y1: C.sy(t), y2: C.sy(t), x1: ML, x2: W - MR, class: 'grid' }, g);
      el('text', { x: ML - 7, y: C.sy(t) + 4, 'text-anchor': 'end', class: 'tick' }, g).textContent = o.yf ? o.yf(t) : fmt(t, 0);
    });
    el('text', { x: (ML + W - MR) / 2, y: H - 6, 'text-anchor': 'middle', class: 'axis-title' }, svg).textContent = o.xl;
    el('text', { x: 13, y: (MT + H - MB) / 2, 'text-anchor': 'middle', class: 'axis-title', transform: 'rotate(-90 13 ' + (MT + H - MB) / 2 + ')' }, svg).textContent = o.yl;
    el('rect', { x: ML, y: MT, width: W - ML - MR, height: H - MT - MB, class: 'scope-frame' }, svg);
    C.layer = el('g', null, svg);
    C.top = el('g', null, svg);
    C.path = function (fn, lo, hi, n) {
      var d = '';
      for (var i = 0; i <= n; i++) {
        var t = lo + (hi - lo) * i / n, v = Math.max(o.y[0], Math.min(o.y[1], fn(t)));
        d += (i ? 'L' : 'M') + C.sx(t).toFixed(1) + ' ' + C.sy(v).toFixed(1);
      }
      return d;
    };
    return C;
  }

  /* ==========================================================================
     PART 3 — SCENES
     ====================================================================== */

  var X1 = xsOf(SHIPS), Y1 = ysOf(SHIPS);
  var MODEL1 = trainSVC(X1, Y1, linearK, SHIPS_C);
  var W1 = weights(MODEL1), B1 = MODEL1.b, NW1 = Math.hypot(W1[0], W1[1]);
  function f1(x) { return dot(W1, x) + B1; }

  function zoneOf(f) {
    var side = f > 0 ? 'Imperial' : 'Rebel';
    if (Math.abs(f) < 0.005) return 'on the boundary (f = 0)';
    if (Math.abs(f) >= 1) return 'beyond the margin on the ' + side + ' side (|f| \u2265 1)';
    return 'inside the margin, ' + side + ' side (0 < |f| < 1): a close call';
  }

  /* ------------------------------------------------- scene 1: classify -- */
  var scene1 = (function () {
    var R, ship, u, dragging = false, START = [1.3, -2.4];
    function place() {
      ship.setAttribute('transform', 'translate(' + R.sx(u[0]).toFixed(1) + ',' + R.sy(u[1]).toFixed(1) + ')');
      var f = f1(u), imp = f > 0;
      $('#verdict1').className = 'verdict ' + (imp ? 'v-imp' : 'v-reb');
      $('#verdict1').innerHTML = '<span class="v-k">' + (imp ? '\u25ae\u2014\u25ae IMPERIAL' : '\u2715 REBEL') + '</span>' +
        '<span>score f(x) = <b class="num">' + fmt(f) + '</b> \u2192 predicted y = ' + (imp ? '+1' : '\u22121') + '</span>' +
        '<span>Position: ' + zoneOf(f) + '</span>';
      tex($('#calc1'), 'f = ' + fmt(W1[0]) + '\\cdot' + paren(u[0]) + ' + ' + fmt(W1[1]) + '\\cdot' + paren(u[1]) +
        (B1 < 0 ? ' - ' : ' + ') + fmt(Math.abs(B1)) + ' = \\mathbf{' + fmt(f).replace('\u2212', '-') + '}', true);
      $('#math1').querySelector('.live').innerHTML = 'For this ship: distance to the boundary = |f|/\u2016w\u2016 = ' +
        fmt(Math.abs(f)) + ' / ' + fmt(NW1) + ' = <b>' + fmt(Math.abs(f) / NW1) + '</b> radar units.';
    }
    function move(x) {
      u = [Math.max(-2.9, Math.min(2.9, x[0])), Math.max(-2.9, Math.min(2.9, x[1]))];
      place();
    }
    function init() {
      R = Radar($('#plot1'));
      drawLinear(R, W1, B1);
      SHIPS.forEach(function (s) { drawShip(R, R.marks, s.x, s.y); });
      ship = el('g', { class: 'ship is-unk is-draggable', tabindex: 0, 'aria-label': 'Unknown ship. Arrow keys move it.' }, R.top);
      el('circle', { r: 20, class: 'hit' }, ship);
      el('circle', { r: 16, class: 'ring ring-sel' }, ship);
      el('use', { href: '#ico-unknown', x: -12, y: -12, width: 24, height: 24 }, ship);
      var svg = R.svg;
      svg.addEventListener('pointerdown', function (e) { dragging = true; svg.setPointerCapture(e.pointerId); svg.classList.add('is-drag'); move(R.fromEvent(e)); });
      svg.addEventListener('pointermove', function (e) { if (dragging) move(R.fromEvent(e)); });
      ['pointerup', 'pointercancel'].forEach(function (t) { svg.addEventListener(t, function () { dragging = false; svg.classList.remove('is-drag'); }); });
      ship.addEventListener('keydown', function (e) {
        var d = { ArrowLeft: [-0.1, 0], ArrowRight: [0.1, 0], ArrowUp: [0, 0.1], ArrowDown: [0, -0.1] }[e.key];
        if (d) { e.preventDefault(); move([u[0] + d[0], u[1] + d[1]]); }
      });
      $('#math1').innerHTML =
        '<p>The SVM chose <span class="tex" data-tex="\\mathbf{w} = (' + fmt(W1[0]) + ',\\ ' + fmt(W1[1]) + ')"></span> and <span class="tex" data-tex="b = ' + fmt(B1) + '"></span> by minimizing</p>' +
        '<div class="tex-block" data-tex="\\tfrac12\\lVert\\mathbf{w}\\rVert^2 + C\\sum_{n=1}^{16}\\max\\big(0,\\ 1 - y_n f(\\mathbf{x}_n)\\big),\\quad C = 1."></div>' +
        '<p>The first term keeps <span class="tex" data-tex="\\lVert\\mathbf{w}\\rVert"></span> small, which keeps the margin wide; the second (scene 2) charges for ships on the wrong side of their margin edge.</p>' +
        '<p>Score and distance differ by a constant: the signed distance from the boundary is <span class="tex" data-tex="f(\\mathbf{x})/\\lVert\\mathbf{w}\\rVert"></span>, with <span class="tex" data-tex="\\lVert\\mathbf{w}\\rVert = ' + fmt(NW1) + '"></span>. So the margin edges <span class="tex" data-tex="f=\\pm1"></span> sit <span class="tex" data-tex="1/\\lVert\\mathbf{w}\\rVert = ' + fmt(1 / NW1) + '"></span> units from the line.</p>' +
        '<p class="live"></p>';
      renderTex($('#math1'));
      reset();
    }
    function reset() { move(START.slice()); }
    return { init: init, reset: reset };
  })();

  /* ---------------------------------------- scene 2: support vectors -- */
  var scene2 = (function () {
    var R, C, ships = [], sel, hidden = false, SEL0 = 7; // Y-wing Gray
    var M = X1.map(function (x, n) { return Y1[n] * f1(x); });
    var MLO = -2, MHI = 4;
    function draw() {
      ships.forEach(function (s, n) {
        s.classList.toggle('is-hidden', hidden && MODEL1.alpha[n] <= 1e-8);
        s.classList.toggle('is-sel', n === sel);
      });
      $('#hide2').setAttribute('aria-pressed', String(hidden));
      $('#hide2').textContent = hidden ? 'Show all training ships' : 'Hide non-support ships';

      // loss chart
      clear(C.layer); clear(C.top);
      el('line', { x1: C.sx(0), x2: C.sx(0), y1: C.top0, y2: C.bottom, class: 'ref-line' }, C.layer);
      el('line', { x1: C.sx(1), x2: C.sx(1), y1: C.top0, y2: C.bottom, class: 'ref-line' }, C.layer);
      el('text', { x: C.sx(-1), y: C.top0 + 14, 'text-anchor': 'middle', class: 'chart-note' }, C.layer).textContent = 'wrong side';
      el('text', { x: C.sx(0.5), y: C.top0 + 14, 'text-anchor': 'middle', class: 'chart-note' }, C.layer).textContent = 'inside';
      el('text', { x: C.sx(2.5), y: C.top0 + 14, 'text-anchor': 'middle', class: 'chart-note' }, C.layer).textContent = 'beyond margin: loss 0';
      if ($('#logit2').checked) el('path', { d: C.path(logLoss2, MLO, MHI, 120), class: 'curve c-logit' }, C.layer);
      el('path', { d: C.path(hinge, MLO, MHI, 120), class: 'curve c-hinge' }, C.layer);
      M.forEach(function (m, n) {
        if (hidden && MODEL1.alpha[n] <= 1e-8) return;
        var mm = Math.max(MLO, Math.min(MHI, m));
        var g = el('g', { class: 'ship' + (Y1[n] > 0 ? ' is-imp' : '') + (n === sel ? ' is-sel' : ''), transform: 'translate(' + C.sx(mm).toFixed(1) + ',' + C.sy(hinge(m)).toFixed(1) + ')' }, n === sel ? C.top : C.layer);
        if (n === sel) el('circle', { r: 12, class: 'ring ring-sel' }, g);
        el('use', { href: Y1[n] > 0 ? '#ico-imperial' : '#ico-rebel', x: -7, y: -7, width: 14, height: 14 }, g);
      });

      // readout
      var y = Y1[sel], f = f1(X1[sel]), m = y * f, a = MODEL1.alpha[sel], loss = hinge(m);
      var zone = m >= 1 - 1e-6 ? (m <= 1 + 1e-6 ? 'exactly on its margin edge' : 'beyond its margin edge') : m > 0 ? 'correct side, but inside the margin' : 'wrong side of the boundary (misclassified)';
      var role = a > 1e-8 ? '<b class="good">support vector</b>, \u03b1 = ' + fmt(a, 3) + (a >= SHIPS_C - 1e-6 ? ' = C' : ' (between 0 and C)') : 'not a support vector, \u03b1 = 0';
      $('#read2').innerHTML = '<div class="ro-title">' + SHIPS[sel].name + '</div>' +
        '<div class="steps">' +
        '<span>true label y</span><b>' + (y > 0 ? '+1 Imperial' : '\u22121 Rebel') + '</b>' +
        '<span>score f(x)</span><b>' + fmt(f, 3) + '</b>' +
        '<span>y \u00b7 f(x)</span><b>' + paren(y, 0) + ' \u00d7 ' + paren(f, 3) + ' = ' + fmt(m, 3) + '</b>' +
        '<span>hinge loss</span><b class="' + (loss > 1e-9 ? 'hl' : 'ok') + '">max(0, 1 \u2212 ' + paren(m, 3) + ') = ' + fmt(loss, 3) + '</b>' +
        ($('#logit2').checked ? '<span>logistic loss</span><b>log\u2082(1 + e^(\u2212' + paren(m, 3) + ')) = ' + fmt(logLoss2(m), 3) + '</b>' : '') +
        '</div><p>Position: ' + zone + '. ' + role + '.</p>';
    }
    function init() {
      R = Radar($('#plot2'));
      drawLinear(R, W1, B1);
      SHIPS.forEach(function (s, n) {
        var g = drawShip(R, R.marks, s.x, s.y);
        if (MODEL1.alpha[n] > 1e-8) {
          el('circle', { r: 15, class: 'ring ring-sv' }, g);
          var left = s.x[0] > 1 || s.name === 'Blue Three';
          el('text', { x: left ? -15 : 15, y: s.name === 'Blue Three' ? 26 : -12, 'text-anchor': left ? 'end' : 'start', class: 'ship-label' }, g).textContent = s.name;
        }
        el('circle', { r: 16, class: 'hit' }, g);
        g.setAttribute('role', 'button'); g.setAttribute('tabindex', '0'); g.setAttribute('aria-label', s.name);
        g.addEventListener('click', function () { sel = n; draw(); });
        g.addEventListener('keydown', function (e) { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); sel = n; draw(); } });
        ships.push(g);
      });
      C = Chart($('#loss2'), { W: 520, H: 250, x: [MLO, MHI], y: [0, 3.2], xt: [-2, -1, 0, 1, 2, 3, 4], yt: [0, 1, 2, 3], xl: 'y \u00b7 f(x)', yl: 'loss' });
      C.top0 = C.sy(3.2); C.bottom = C.sy(0);
      var k = MODEL1.sv.length;
      $('#tag2').textContent = k + ' of ' + SHIPS.length + ' ships are support vectors';
      $('#hide2').addEventListener('click', function () { hidden = !hidden; draw(); });
      $('#logit2').addEventListener('change', draw);
      reset();
    }
    function reset() { hidden = false; sel = SEL0; $('#logit2').checked = false; draw(); }
    return { init: init, reset: reset };
  })();

  /* -------------------------------------------------- scene 3: C, γ -- */
  var scene3 = (function () {
    var R, ci, gi, model, pending = false;
    var TX = xsOf(RING_TRAIN), TY = ysOf(RING_TRAIN), HX = xsOf(RING_TEST), HY = ysOf(RING_TEST);
    function acc(X, Y) { var k = 0; X.forEach(function (x, n) { if ((model.f(x) > 0 ? 1 : -1) === Y[n]) k++; }); return k / X.length; }
    function fit() {
      var Cv = C_STEPS[ci], g = GAMMA_STEPS[gi];
      model = trainSVC(TX, TY, rbfK(g), Cv);
      var grid = paintField(R, model.f);
      clear(R.lines);
      [-1, 1].forEach(function (c) { el('path', { d: contour(R, grid, c), class: 'edge-thin' }, R.lines); });
      el('path', { d: contour(R, grid, 0), class: 'boundary' }, R.lines);
      clear(R.marks);
      RING_TRAIN.forEach(function (s, n) {
        var sh = drawShip(R, R.marks, s.x, s.y);
        if (model.alpha[n] > 1e-8) el('circle', { r: 13, class: 'ring ring-sv', 'stroke-width': 1.4 }, sh);
      });
      var tr = acc(TX, TY), te = acc(HX, HY), gap = tr - te;
      var diag = gap > 0.08 ? '<b class="bad">Overfitting:</b> training accuracy is well above held-out accuracy.'
        : (te < 0.84 ? '<b class="bad">Underfitting:</b> the boundary is too simple; both accuracies are low.'
          : '<b class="good">Reasonable fit:</b> training and held-out accuracy are close and both high.');
      $('#c3v').textContent = 'C = ' + Cv; $('#g3v').textContent = '\u03b3 = ' + g;
      $('#read3').innerHTML = '<div class="ro-grid">' +
        '<span>training accuracy (' + TX.length + ' ships)</span><b class="num">' + pct(tr) + '</b>' +
        '<span>held-out accuracy (' + HX.length + ' new ships)</span><b class="num">' + pct(te) + '</b>' +
        '<span>support vectors</span><b class="num">' + model.sv.length + ' of ' + TX.length + '</b></div>' +
        '<p>' + diag + ' <span style="color:var(--muted)">(Rule of thumb: gap over 8 points, or held-out below 84%.)</span></p>';
      $('#math3 .live').innerHTML = 'At \u03b3 = ' + g + ', similarity falls to e\u207b\u00b9 \u2248 0.37 at distance 1/\u221a\u03b3 = <b>' + fmt(1 / Math.sqrt(g)) + '</b> radar units' +
        (1 / Math.sqrt(g) < 0.3 ? ': smaller than the gap between ships, so each one gets its own bubble.' : '.');
      document.querySelectorAll('#console [data-preset]').forEach(function (b) {
        b.setAttribute('aria-pressed', String(b.getAttribute('data-preset') === ci + ',' + gi));
      });
    }
    function schedule() { if (pending) return; pending = true; requestAnimationFrame(function () { pending = false; fit(); }); }
    function set(c, g) { ci = c; gi = g; $('#c3').value = c; $('#g3').value = g; schedule(); }
    function init() {
      R = Radar($('#plot3'));
      $('#c3').addEventListener('input', function () { ci = +this.value; schedule(); });
      $('#g3').addEventListener('input', function () { gi = +this.value; schedule(); });
      document.querySelectorAll('#console [data-preset]').forEach(function (b) {
        b.addEventListener('click', function () { var p = b.getAttribute('data-preset').split(','); set(+p[0], +p[1]); });
      });
      $('#math3').innerHTML =
        '<p>The dual problem the solver runs:</p>' +
        '<div class="tex-block" data-tex="\\max_{\\boldsymbol\\alpha}\\ \\sum_n\\alpha_n - \\tfrac12\\sum_{n,m}\\alpha_n\\alpha_m y_n y_m K(\\mathbf{x}_n,\\mathbf{x}_m)\\quad\\text{s.t. } 0\\le\\alpha_n\\le C,\\ \\sum_n\\alpha_n y_n = 0"></div>' +
        '<p><b>C</b> caps each ship&rsquo;s influence <span class="tex" data-tex="\\alpha_n"></span>. Small C: no ship can pull hard, so the boundary stays smooth and many ships sit inside the margin. Large C: the solver bends the boundary to classify every training ship.</p>' +
        '<p><b>&gamma;</b> sets the reach of each ship&rsquo;s bump <span class="tex" data-tex="K(\\mathbf{x}_n,\\cdot)"></span>. Tiny &gamma;: bumps are so wide that <span class="tex" data-tex="f"></span> is nearly linear. Huge &gamma;: <span class="tex" data-tex="f"></span> is about <span class="tex" data-tex="b"></span> everywhere except right next to a training ship.</p>' +
        '<p class="live"></p>' +
        '<p>In practice, pick C and &gamma; by cross-validation, never by training accuracy.</p>';
      renderTex($('#math3'));
      reset();
    }
    function reset() { set(2, 3); }
    return { init: init, reset: reset };
  })();

  /* ---------------------------------------------- scene 4: Platt -- */
  var scene4 = (function () {
    var S, Rl, a, b;
    var model = trainSVC(xsOf(CAL_TRAIN), ysOf(CAL_TRAIN), linearK, CAL_C);
    var VS = xsOf(CAL_VAL).map(model.f), VL = ysOf(CAL_VAL);
    var TS = xsOf(CAL_TEST).map(model.f), TL = ysOf(CAL_TEST);
    var PF = plattFit(VS, VL);
    var LIM = 3;
    function draw() {
      $('#a4').value = a; $('#b4').value = b;
      $('#a4v').textContent = fmt(a); $('#b4v').textContent = fmt(b);
      clear(S.layer);
      el('path', { d: S.path(sigmoid, -LIM, LIM, 160), class: 'curve c-ghost' }, S.layer);
      el('path', { d: S.path(function (t) { return sigmoid(a * t + b); }, -LIM, LIM, 160), class: 'curve c-sig' }, S.layer);
      el('text', { x: S.sx(-LIM) + 6, y: S.sy(1) + 13, class: 'chart-note is-ink' }, S.layer).textContent = 'Imperial validation ships (y = +1)';
      el('text', { x: S.sx(LIM) - 6, y: S.sy(0) - 7, 'text-anchor': 'end', class: 'chart-note is-ink' }, S.layer).textContent = 'Rebel validation ships (y = \u22121)';

      var probs = TS.map(function (s) { return sigmoid(a * s + b); });
      var bins = reliability(probs, TL);
      clear(Rl.layer);
      el('line', { x1: Rl.sx(0), y1: Rl.sy(0), x2: Rl.sx(1), y2: Rl.sy(1), class: 'diag' }, Rl.layer);
      el('text', { x: Rl.sx(0.3), y: Rl.sy(0.3) - 8, 'text-anchor': 'middle', class: 'chart-note', transform: 'rotate(-24 ' + Rl.sx(0.3) + ' ' + (Rl.sy(0.3) - 8) + ')' }, Rl.layer).textContent = 'perfectly calibrated';
      bins.forEach(function (bn) {
        if (bn.n < 5) return;
        el('line', { x1: Rl.sx(bn.meanP), x2: Rl.sx(bn.meanP), y1: Rl.sy(bn.meanP), y2: Rl.sy(bn.frac), class: 'rel-gap' }, Rl.layer);
        el('circle', { cx: Rl.sx(bn.meanP), cy: Rl.sy(bn.frac), r: 3 + Math.sqrt(bn.n) / 2.2, class: 'rel-dot' }, Rl.layer);
      });
      var b80 = bins[4];
      var msg = b80.n >= 5
        ? 'Test ships given 70\u201390% (average <b>' + pct(b80.meanP) + '</b>, ' + b80.n + ' ships): <b>' + pct(b80.frac) + '</b> were actually Imperial.'
        : 'Almost no test ships were given 70\u201390%, so the model never says &ldquo;about 80%.&rdquo;';
      var raw = Math.abs(a - 1) < 1e-9 && Math.abs(b) < 1e-9, fitted = Math.abs(a - PF.a) < 0.03 && Math.abs(b - PF.b) < 0.03;
      $('#read4').innerHTML = '<div class="ro-title">' + (raw ? 'Plain sigmoid \u03c3(f)' : fitted ? 'Platt-scaled \u03c3(a f + b)' : 'Your a, b') + '</div>' +
        '<p style="margin-top:0">' + msg + '</p>' +
        '<div class="ro-grid"><span>average log loss on test ships</span><b class="num">' + fmt(meanLogLoss(probs, TL), 3) + '</b>' +
        '<span>Platt\u2019s fitted values</span><b class="num">a = ' + fmt(PF.a) + ', b = ' + fmt(PF.b) + '</b></div>';
    }
    function init() {
      var hi = Math.ceil(Math.max.apply(null, VS.map(Math.abs)) * 2) / 2;
      LIM = Math.min(3, hi);
      S = Chart($('#sig4'), { W: 520, H: 300, x: [-LIM, LIM], y: [-0.08, 1.08], xt: [-3, -2, -1, 0, 1, 2, 3].filter(function (t) { return Math.abs(t) <= LIM; }), yt: [0, 0.5, 1], yf: function (t) { return fmt(t, 1); }, xl: 'SVM score f(x)', yl: 'p(Imperial | x)' });
      var rug = el('g', null, S.svg);
      S.svg.insertBefore(rug, S.layer);
      var r = mulberry32(3);
      VS.forEach(function (s, i) {
        var x = S.sx(Math.max(-LIM, Math.min(LIM, s))), imp = VL[i] > 0, y0 = imp ? 1.02 + r() * 0.05 : -0.07 + r() * 0.05;
        el('line', { x1: x, x2: x, y1: S.sy(y0), y2: S.sy(y0) + 5, class: imp ? 'rug-imp' : 'rug-reb' }, rug);
      });
      Rl = Chart($('#rel4'), { W: 520, H: 260, x: [0, 1], y: [0, 1], xt: [0, 0.2, 0.4, 0.6, 0.8, 1], yt: [0, 0.5, 1], xf: pct, yf: pct, xl: 'predicted p(Imperial), binned', yl: 'actually Imperial' });
      $('#a4').addEventListener('input', function () { a = +this.value; draw(); });
      $('#b4').addEventListener('input', function () { b = +this.value; draw(); });
      $('#raw4').addEventListener('click', function () { a = 1; b = 0; draw(); });
      $('#fit4').addEventListener('click', function () { a = PF.a; b = PF.b; draw(); });
      $('#math4').innerHTML =
        '<p>Platt scaling is a one-input logistic regression. Treat each validation score <span class="tex" data-tex="f_i"></span> as the input and its label as the target, and maximize the likelihood:</p>' +
        '<div class="tex-block" data-tex="\\min_{a,b}\\ -\\sum_i \\Big[t_i\\log\\sigma(af_i+b) + (1-t_i)\\log\\big(1-\\sigma(af_i+b)\\big)\\Big]"></div>' +
        '<p>with <span class="tex" data-tex="t_i"></span> slightly softened from 1/0 (Platt&rsquo;s <span class="tex" data-tex="\\tfrac{N_++1}{N_++2}"></span> and <span class="tex" data-tex="\\tfrac{1}{N_-+2}"></span>) to avoid overconfidence. The fit here gives <span class="tex" data-tex="a=' + fmt(PF.a) + ',\\ b=' + fmt(PF.b) + '"></span>.</p>' +
        '<p><b>Why separate data?</b> The SVM pushed its own training ships to <span class="tex" data-tex="|f|\\ge1"></span> wherever it could, so their scores look more decisive than scores on new ships. Fitting <span class="tex" data-tex="a,b"></span> on them tends to give overconfident probabilities.</p>' +
        '<p><b>Why the plain sigmoid fails here.</b> The score&rsquo;s scale depends on C and on the data, not on probability. With C = 0.03 the scores are small, so <span class="tex" data-tex="\\sigma(f)"></span> stays near 50% and is underconfident. With a different C it could happen to look fine, or be overconfident. Nothing in SVM training ties <span class="tex" data-tex="\\sigma(f)"></span> to real frequencies.</p>' +
        '<p>The calibration plot uses a third set (800 test ships), so the check is not done on the data used to fit <span class="tex" data-tex="a,b"></span>.</p>';
      renderTex($('#math4'));
      reset();
    }
    function reset() { a = 1; b = 0; draw(); }
    return { init: init, reset: reset };
  })();

  /* ------------------------------------------------- scene 5: SVR -- */
  var scene5 = (function () {
    var C, eps, model, sel, SEL0;
    var XS = FUEL.xs, YS = FUEL.ys, kern = rbf1(FUEL_GAMMA), EPS0 = 0.3;
    function draw() {
      model = trainSVR(XS, YS, kern, FUEL_C, eps);
      $('#e5').value = eps; $('#e5v').textContent = '\u03b5 = ' + fmt(eps);
      clear(C.layer); clear(C.top);
      var lo = 0, hi = 10, n = 160, up = '', dn = '';
      for (var i = 0; i <= n; i++) {
        var t = lo + (hi - lo) * i / n, f = model.f(t);
        up += (i ? 'L' : 'M') + C.sx(t).toFixed(1) + ' ' + C.sy(f + eps).toFixed(1);
        dn = 'L' + C.sx(t).toFixed(1) + ' ' + C.sy(f - eps).toFixed(1) + dn;
      }
      el('path', { d: up + dn + 'Z', class: 'tube' }, C.layer);
      el('path', { d: C.path(model.f, lo, hi, n), class: 'curve c-svr' }, C.layer);
      var nAbove = 0, nBelow = 0;
      XS.forEach(function (x, k) {
        var th = model.theta[k], isSv = Math.abs(th) > 1e-8;
        if (th > 1e-8) nAbove++; else if (th < -1e-8) nBelow++;
        if (k === sel) {
          var f = model.f(x), r = YS[k] - f, edge = f + (r > 0 ? eps : -eps);
          if (Math.abs(r) > eps) {
            el('line', { x1: C.sx(x), x2: C.sx(x), y1: C.sy(f), y2: C.sy(edge), class: 'resid-in' }, C.top);
            el('line', { x1: C.sx(x), x2: C.sx(x), y1: C.sy(edge), y2: C.sy(YS[k]), class: 'resid-out' }, C.top);
          } else el('line', { x1: C.sx(x), x2: C.sx(x), y1: C.sy(f), y2: C.sy(YS[k]), class: 'resid-in' }, C.top);
          el('circle', { cx: C.sx(x), cy: C.sy(YS[k]), r: 11, class: 'ring ring-sel' }, C.top);
        }
        var g = el('g', { tabindex: 0, role: 'button', 'aria-label': 'Reading ' + (k + 1) }, C.top);
        if (isSv) el('circle', { cx: C.sx(x), cy: C.sy(YS[k]), r: 8.5, class: 'ring ring-sv', 'stroke-width': 1.6 }, g);
        el('circle', { cx: C.sx(x), cy: C.sy(YS[k]), r: 5, class: 'pt' + (isSv ? ' is-sv' : '') }, g);
        el('circle', { cx: C.sx(x), cy: C.sy(YS[k]), r: 13, fill: 'transparent', style: 'cursor:pointer' }, g);
        g.addEventListener('click', function () { sel = k; draw(); });
        g.addEventListener('keydown', function (e) { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); sel = k; draw(); } });
      });

      var x = XS[sel], y = YS[sel], f = model.f(x), r = y - f, ab = Math.abs(r), loss = epsLoss(r, eps), th = model.theta[sel];
      var where = Math.abs(ab - eps) < 1e-4 ? 'on the tube edge' : ab < eps ? 'inside the tube' : (r > 0 ? 'above the tube' : 'below the tube');
      var thTxt = Math.abs(th) <= 1e-8 ? '\u03b8 = 0: this reading drops out of f(x)'
        : '\u03b8 = \u03b1 \u2212 \u03b1* = ' + fmt(model.alpha[sel], 3) + ' \u2212 ' + fmt(model.alphaStar[sel], 3) + ' = <b>' + fmt(th, 3) + '</b> (' + (th > 0 ? 'positive' : 'negative') + ')';
      $('#read5').innerHTML = '<div class="ro-title">Reading ' + (sel + 1) + ': cargo ' + fmt(x) + ' kt</div>' +
        '<div class="steps">' +
        '<span>actual y</span><b>' + fmt(y) + ' t</b>' +
        '<span>predicted f(x)</span><b>' + fmt(f, 3) + ' t</b>' +
        '<span>error y \u2212 f(x)</span><b>' + fmt(r, 3) + '</b>' +
        '<span>|error| \u2212 \u03b5</span><b>' + fmt(ab, 3) + ' \u2212 ' + fmt(eps) + ' = ' + fmt(ab - eps, 3) + '</b>' +
        '<span>loss</span><b class="' + (loss > 1e-6 ? 'hl' : 'ok') + '">max(0, ' + fmt(ab - eps, 3) + ') = ' + fmt(loss, 3) + '</b>' +
        '</div><p>' + where.charAt(0).toUpperCase() + where.slice(1) + '. ' + thTxt + '.</p>' +
        '<p><b>' + model.sv.length + ' of ' + XS.length + '</b> readings are support vectors (' + nAbove + ' with \u03b8 &gt; 0, ' + nBelow + ' with \u03b8 &lt; 0).</p>';
      $('#cmpSv').textContent = 'scene 5: ' + model.sv.length + ' of ' + XS.length + ' at \u03b5 = ' + fmt(eps);
    }
    function init() {
      C = Chart($('#svr5'), { W: 520, H: 320, x: [0, 10], y: [0, 9], xt: [0, 2, 4, 6, 8, 10], yt: [0, 3, 6, 9], xl: 'cargo mass x (kilotonnes)', yl: 'fuel used y (tonnes)' });
      // default selection: the reading furthest below the tube, to show a negative θ
      var m0 = trainSVR(XS, YS, kern, FUEL_C, EPS0);
      SEL0 = 0;
      XS.forEach(function (x, k) { if (YS[k] - m0.f(x) < YS[SEL0] - m0.f(XS[SEL0])) SEL0 = k; });
      $('#e5').addEventListener('input', function () { eps = +this.value; draw(); });
      reset();
    }
    function reset() { eps = EPS0; sel = SEL0; draw(); }
    return { init: init, reset: reset };
  })();

  var scene6 = { init: function () {}, reset: function () {} };

  /* ------------------------------------------------------- navigation -- */

  var SCENES = [scene1, scene2, scene3, scene4, scene5, scene6];
  var sections = Array.from(document.querySelectorAll('.scene'));
  var current = 0;

  function go(i, focus) {
    current = Math.max(0, Math.min(sections.length - 1, i));
    sections.forEach(function (s, k) { s.hidden = k !== current; });
    document.querySelectorAll('#progress button').forEach(function (b, k) {
      if (k === current) b.setAttribute('aria-current', 'step'); else b.removeAttribute('aria-current');
      b.classList.toggle('is-done', k < current);
    });
    $('#prev').disabled = current === 0;
    $('#next').disabled = current === sections.length - 1;
    $('#reset').disabled = current === sections.length - 1;
    $('#sceneCount').textContent = 'Scene ' + (current + 1) + ' / ' + sections.length;
    if (location.hash !== '#scene-' + (current + 1)) history.replaceState(null, '', '#scene-' + (current + 1));
    if (focus) $('#console').scrollIntoView({ block: 'start', behavior: 'smooth' });
  }

  function start() {
    fillLegends();
    renderTex(document);
    var ol = $('#progress');
    sections.forEach(function (s, k) {
      var li = document.createElement('li'), b = document.createElement('button');
      b.type = 'button';
      b.innerHTML = '<span class="p-num">' + (k + 1) + '</span><span class="p-lab">' + s.getAttribute('data-short') + '</span>';
      b.addEventListener('click', function () { go(k, true); });
      li.appendChild(b); ol.appendChild(li);
    });
    SCENES.forEach(function (s) { s.init(); });
    $('#prev').addEventListener('click', function () { go(current - 1, true); });
    $('#next').addEventListener('click', function () { go(current + 1, true); });
    $('#reset').addEventListener('click', function () { SCENES[current].reset(); });
    var m = /^#scene-(\d+)$/.exec(location.hash);
    go(m ? +m[1] - 1 : 0, false);
  }

  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', start);
  else start();
})(typeof window !== 'undefined' ? window : globalThis);
