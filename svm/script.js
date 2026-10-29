/* ============================================================================
   The Widest Corridor — support vector machines.
   CS 473 module 11 · Murphy, PML §17.3

   Leia classifies arriving ships as Rebel (ỹ = −1) or Imperial (ỹ = +1) from
   two standardized readings, x = (engine output, transponder strength).

     score        f(x) = wᵀx + w₀                 (linear)
                  f(x) = Σₙ αₙ ỹₙ K(x, xₙ) + w₀    (kernel, from the dual)
     hard margin  min ½‖w‖²            s.t. ỹₙ f(xₙ) ≥ 1
     soft margin  min ½‖w‖² + C Σ ξₙ   s.t. ỹₙ f(xₙ) ≥ 1 − ξₙ, ξₙ ≥ 0

   Every boundary drawn as "trained" comes from the SMO solver below, run on
   the page. Lines the reader sets by hand are labelled as such.

   Layout of this file
     PART 1  Math          pure, DOM-free, exported for the Node test suite
     PART 2  Drawing       the shared plot: axes, ships, lines, shading
     PART 3  Scenes        one setup function per scene, plus navigation

   PART 1 runs under Node (`require('./script.js')` returns the math only); the
   rest is skipped when there is no document. See test-math.js.
   ========================================================================= */

(function (global) {
  'use strict';

  /* ==========================================================================
     PART 1 — MATH
     ====================================================================== */

  var LIM = 3;                       // plot shows [−3, 3] on both axes
  var HARD_C = 1e6;                  // "infinite" C for the hard-margin problem
  var EPS = 1e-7;                    // SMO stopping tolerance (KKT gap)
  var ALPHA_TOL = 1e-6;              // αₙ above this counts as positive
  var MAX_ITER = 200000;

  // Raw sensor units. Every feature is standardized: x = (raw − mean) / sd.
  var RAW = [
    { name: 'engine output', unit: 'MW', mean: 900, sd: 300 },
    { name: 'transponder strength', unit: 'dB', mean: 18, sd: 6 }
  ];
  function toRaw(x) { return [RAW[0].mean + RAW[0].sd * x[0], RAW[1].mean + RAW[1].sd * x[1]]; }
  function standardize(raw) { return [(raw[0] - RAW[0].mean) / RAW[0].sd, (raw[1] - RAW[1].mean) / RAW[1].sd]; }

  // The identified ships (scenes 1–6). Linearly separable.
  var SHIPS = [
    { name: 'Red Five',        x: [-2.0, -1.3], y: -1 },
    { name: 'Gold Leader',     x: [-1.5, -0.3], y: -1 },
    { name: 'Rogue Two',       x: [-0.7, -1.6], y: -1 },
    { name: 'Tantive IV',      x: [-2.3,  0.6], y: -1 },
    { name: 'Blue Three',      x: [-0.3, -0.6], y: -1 },
    { name: 'Ghost',           x: [ 0.6, -2.1], y: -1 },
    { name: 'Corellian Hauler',x: [-1.2,  0.9], y: -1 },
    { name: 'TIE Alpha',       x: [ 1.7,  1.2], y:  1 },
    { name: 'TIE Bravo',       x: [ 0.5,  1.6], y:  1 },
    { name: 'Interceptor',     x: [ 2.1, -0.1], y:  1 },
    { name: 'Lambda Shuttle',  x: [ 1.0,  0.6], y:  1 },
    { name: 'TIE Bomber',      x: [ 0.0,  2.4], y:  1 },
    { name: 'Star Destroyer',  x: [ 2.5,  1.9], y:  1 },
    { name: 'Probe Carrier',   x: [ 1.8, -1.2], y:  1 }
  ];

  // Scene 8 adds two awkward ships to the same data.
  var EXTRA = [
    { name: 'Tydirium (stolen shuttle)', x: [1.3, 0.95], y: -1, special: 'shuttle' },
    { name: 'Damaged courier',           x: [0.4, -0.1], y: -1, special: 'courier' }
  ];

  // Scene 3: two hand-drawn separating lines to compare with the trained one.
  // Each is (w, w₀); they separate SHIPS but sit closer to some ship.
  var HAND_LINES = [
    { key: 'A', w: [0.97, 0.26], w0: -0.35 },
    { key: 'B', w: [0.71, 0.71], w0: -0.1 }
  ];

  // Scene 7: Rebel smugglers tune their readings to look ordinary, so they
  // sit near the middle; Imperial ships surround them. Fixed, not random.
  var RING = (function () {
    var out = [], i, a, r;
    var inner = [0.55, 0.9, 0.7, 1.05, 0.5, 0.95, 0.75, 1.0, 0.6, 0.85];
    for (i = 0; i < inner.length; i++) {
      a = 2 * Math.PI * i / inner.length + 0.3;
      r = inner[i];
      out.push({ name: 'Smuggler ' + (i + 1), x: [0.2 + r * Math.cos(a), 0.1 + r * Math.sin(a)], y: -1 });
    }
    var outer = [2.1, 2.5, 2.0, 2.4, 2.2, 2.6, 2.05, 2.35, 2.15, 2.5, 2.0, 2.3];
    for (i = 0; i < outer.length; i++) {
      a = 2 * Math.PI * i / outer.length + 0.1;
      r = outer[i];
      out.push({ name: 'Patrol ' + (i + 1), x: [0.2 + r * Math.cos(a) * 1.05, 0.1 + r * Math.sin(a) * 0.95], y: 1 });
    }
    return out.map(function (s) { s.x = [round2(s.x[0]), round2(s.x[1])]; return s; });
  })();

  function round2(v) { return Math.round(v * 100) / 100; }

  /* ---------------------------------------------------------- kernels -- */

  function dot(a, b) { return a[0] * b[0] + a[1] * b[1]; }
  function norm(w) { return Math.sqrt(dot(w, w)); }

  var KERNELS = {
    linear: function () { return function (a, b) { return dot(a, b); }; },
    quadratic: function () { return function (a, b) { var t = dot(a, b) + 1; return t * t; }; },
    rbf: function (gamma) {
      return function (a, b) { var d0 = a[0] - b[0], d1 = a[1] - b[1]; return Math.exp(-gamma * (d0 * d0 + d1 * d1)); };
    }
  };

  /* The explicit features behind the quadratic kernel:
     (xᵀz + 1)² = φ(x)ᵀφ(z) with φ(x) = (1, √2x₁, √2x₂, x₁², x₂², √2x₁x₂). */
  function phiQuadratic(x) {
    var r2 = Math.SQRT2;
    return [1, r2 * x[0], r2 * x[1], x[0] * x[0], x[1] * x[1], r2 * x[0] * x[1]];
  }

  /* ------------------------------------------------------------- SMO -- */

  /* Solve the SVM dual

       max_α  Σ αₙ − ½ Σₙ Σₘ αₙ αₘ ỹₙ ỹₘ K(xₙ, xₘ)
       s.t.   0 ≤ αₙ ≤ C,   Σ αₙ ỹₙ = 0

   by sequential minimal optimization: repeatedly pick the pair that most
   violates the optimality (KKT) conditions and solve for those two exactly.
   This is the maximal-violating-pair rule used by LIBSVM. The bias w₀ is then
   read off the KKT conditions.

   X: array of [x₁, x₂]; y: array of ±1; kernel: K(a, b); C > 0.
   Returns { alpha, w0, K, iters, converged, f(x) }. */
  function trainSVM(X, y, kernel, C) {
    var n = X.length, i, j, t;
    var K = [];
    for (i = 0; i < n; i++) {
      K.push(new Float64Array(n));
      for (j = 0; j < n; j++) K[i][j] = kernel(X[i], X[j]);
    }
    var alpha = new Float64Array(n);
    var G = new Float64Array(n);           // gradient of ½αᵀQα − Σα, Q = ỹỹᵀ∘K
    for (i = 0; i < n; i++) G[i] = -1;

    function upOK(t) { return (y[t] > 0 && alpha[t] < C) || (y[t] < 0 && alpha[t] > 0); }
    function lowOK(t) { return (y[t] > 0 && alpha[t] > 0) || (y[t] < 0 && alpha[t] < C); }

    var iters = 0, converged = false;
    while (iters < MAX_ITER) {
      var gmax = -Infinity, gmin = Infinity;
      i = -1; j = -1;
      for (t = 0; t < n; t++) {
        var v = -y[t] * G[t];
        if (upOK(t) && v > gmax) { gmax = v; i = t; }
        if (lowOK(t) && v < gmin) { gmin = v; j = t; }
      }
      if (i < 0 || j < 0 || gmax - gmin < EPS) { converged = true; break; }
      iters++;

      // Move α along the direction that keeps Σ αₙỹₙ = 0: αᵢ += ỹᵢ·s, αⱼ −= ỹⱼ·s.
      var eta = K[i][i] + K[j][j] - 2 * K[i][j];
      if (eta <= 1e-12) eta = 1e-12;
      var s = (gmax - gmin) / eta;
      // Box limits on s from 0 ≤ αᵢ, αⱼ ≤ C.
      var si = y[i] > 0 ? C - alpha[i] : alpha[i];
      var sj = y[j] > 0 ? alpha[j] : C - alpha[j];
      s = Math.min(s, si, sj);
      var dai = y[i] * s, daj = -y[j] * s;
      alpha[i] += dai; alpha[j] += daj;
      if (alpha[i] < 1e-14) alpha[i] = 0; if (alpha[i] > C - 1e-14 * C) alpha[i] = Math.min(alpha[i], C);
      if (alpha[j] < 1e-14) alpha[j] = 0; if (alpha[j] > C - 1e-14 * C) alpha[j] = Math.min(alpha[j], C);
      for (t = 0; t < n; t++) G[t] += y[t] * (y[i] * K[i][t] * dai + y[j] * K[j][t] * daj);
    }

    // w₀ from KKT. With r = −ỹₜGₜ:
    //   0 < αₜ < C  ⇒ w₀ = r (the ship sits exactly on its margin edge)
    //   αₜ = 0      ⇒ w₀ ≥ r if ỹ = +1, w₀ ≤ r if ỹ = −1
    //   αₜ = C      ⇒ the reverse
    var lo = -Infinity, hi = Infinity, sum = 0, nFree = 0;
    for (t = 0; t < n; t++) {
      var r = -y[t] * G[t];
      var free = alpha[t] > ALPHA_TOL && alpha[t] < C * (1 - 1e-9);
      if (free) { sum += r; nFree++; }
      else if ((alpha[t] <= ALPHA_TOL) === (y[t] > 0)) lo = Math.max(lo, r);
      else hi = Math.min(hi, r);
    }
    var w0 = nFree ? sum / nFree : (isFinite(lo) && isFinite(hi) ? (lo + hi) / 2 : (isFinite(lo) ? lo : hi));

    var model = {
      X: X, y: y, C: C, kernel: kernel, alpha: alpha, w0: w0, iters: iters, converged: converged,
      f: function (x) {
        var s = w0;
        for (var k = 0; k < n; k++) if (alpha[k] > 0) s += alpha[k] * y[k] * kernel(x, X[k]);
        return s;
      }
    };
    return model;
  }

  /* Linear SVM: train, then collapse w = Σ αₙỹₙxₙ. */
  function trainLinear(X, y, C) {
    var m = trainSVM(X, y, KERNELS.linear(), C);
    var w = [0, 0];
    for (var n = 0; n < X.length; n++) {
      w[0] += m.alpha[n] * y[n] * X[n][0];
      w[1] += m.alpha[n] * y[n] * X[n][1];
    }
    m.w = w;
    m.f = function (x) { return dot(w, x) + m.w0; };
    return m;
  }

  /* Hard margin: C so large it never binds. If the data can't be separated
     by a line, some ship ends up with ỹf < 1 and `separable` is false. */
  function trainHard(X, y) {
    var m = trainLinear(X, y, HARD_C);
    var worst = Infinity;
    for (var n = 0; n < X.length; n++) worst = Math.min(worst, y[n] * m.f(X[n]));
    m.separable = worst > 1 - 1e-4 && Math.max.apply(null, Array.from(m.alpha)) < HARD_C * 0.5;
    return m;
  }

  function score(w, w0, x) { return dot(w, x) + w0; }
  function signedDistance(w, w0, x) { return score(w, w0, x) / norm(w); }

  /* Geometric margin of a line on labelled data: min ỹₙ f(xₙ)/‖w‖.
     Negative if some ship is on the wrong side. */
  function geometricMargin(w, w0, X, y) {
    var m = Infinity;
    for (var n = 0; n < X.length; n++) m = Math.min(m, y[n] * signedDistance(w, w0, X[n]));
    return m;
  }

  /* ξₙ = max(0, 1 − ỹₙ f(xₙ)): how far ship n falls short of its margin edge,
     in score units. 0 = on or beyond its edge; between 0 and 1 = inside the
     corridor but still on the right side; above 1 = on the wrong side. */
  function slacks(f, X, y) {
    return X.map(function (x, n) { return Math.max(0, 1 - y[n] * f(x)); });
  }

  /* Primal soft-margin objective ½‖w‖² + C Σ ξₙ for a linear model. */
  function primalObjective(w, w0, X, y, C) {
    var xi = slacks(function (x) { return score(w, w0, x); }, X, y);
    return 0.5 * dot(w, w) + C * xi.reduce(function (a, b) { return a + b; }, 0);
  }

  /* Dual objective Σ αₙ − ½ Σ Σ αₙαₘỹₙỹₘK(xₙ, xₘ). */
  function dualObjective(model) {
    var a = model.alpha, y = model.y, X = model.X, K = model.kernel, n = a.length, s = 0, q = 0;
    for (var i = 0; i < n; i++) {
      s += a[i];
      for (var j = 0; j < n; j++) q += a[i] * a[j] * y[i] * y[j] * K(X[i], X[j]);
    }
    return s - 0.5 * q;
  }

  /* -------------------------------------------------------- geometry -- */

  /* The segment of {x : wᵀx + w₀ = c} inside the square [−lim, lim]², or null. */
  function clipLine(w, w0, c, lim) {
    lim = lim || LIM;
    var pts = [], a = w[0], b = w[1], k = c - w0, v;
    if (Math.abs(b) > 1e-12) {
      v = (k - a * -lim) / b; if (v >= -lim - 1e-9 && v <= lim + 1e-9) pts.push([-lim, v]);
      v = (k - a * lim) / b;  if (v >= -lim - 1e-9 && v <= lim + 1e-9) pts.push([lim, v]);
    }
    if (Math.abs(a) > 1e-12) {
      v = (k - b * -lim) / a; if (v >= -lim - 1e-9 && v <= lim + 1e-9) pts.push([v, -lim]);
      v = (k - b * lim) / a;  if (v >= -lim - 1e-9 && v <= lim + 1e-9) pts.push([v, lim]);
    }
    // drop near-duplicate corners
    var out = [];
    pts.forEach(function (p) {
      if (!out.some(function (q) { return Math.abs(p[0] - q[0]) < 1e-7 && Math.abs(p[1] - q[1]) < 1e-7; })) out.push(p);
    });
    return out.length >= 2 ? [out[0], out[1]] : null;
  }

  /* Clip a polygon to the half-plane sign·(wᵀx + w₀ − c) ≥ 0 (Sutherland–Hodgman). */
  function clipHalf(poly, w, w0, c, sign) {
    var out = [];
    function val(p) { return sign * (dot(w, p) + w0 - c); }
    for (var i = 0; i < poly.length; i++) {
      var p = poly[i], q = poly[(i + 1) % poly.length], vp = val(p), vq = val(q);
      if (vp >= 0) out.push(p);
      if ((vp >= 0) !== (vq >= 0)) {
        var t = vp / (vp - vq);
        out.push([p[0] + t * (q[0] - p[0]), p[1] + t * (q[1] - p[1])]);
      }
    }
    return out;
  }

  function box(lim) { lim = lim || LIM; return [[-lim, -lim], [lim, -lim], [lim, lim], [-lim, lim]]; }

  /* Marching squares: line segments where a gridded field crosses `level`.
     vals[i][j] is the field at (xs[i], ys[j]). */
  function contour(vals, xs, ys, level) {
    var segs = [], nx = xs.length, ny = ys.length;
    function lerpPt(x0, y0, v0, x1, y1, v1) {
      var t = (level - v0) / (v1 - v0);
      return [x0 + t * (x1 - x0), y0 + t * (y1 - y0)];
    }
    for (var i = 0; i < nx - 1; i++) for (var j = 0; j < ny - 1; j++) {
      var x0 = xs[i], x1 = xs[i + 1], y0 = ys[j], y1 = ys[j + 1];
      var a = vals[i][j], b = vals[i + 1][j], c = vals[i + 1][j + 1], d = vals[i][j + 1];
      var e = [];
      if ((a > level) !== (b > level)) e.push(lerpPt(x0, y0, a, x1, y0, b));
      if ((b > level) !== (c > level)) e.push(lerpPt(x1, y0, b, x1, y1, c));
      if ((c > level) !== (d > level)) e.push(lerpPt(x1, y1, c, x0, y1, d));
      if ((d > level) !== (a > level)) e.push(lerpPt(x0, y1, d, x0, y0, a));
      if (e.length === 2) segs.push(e);
      else if (e.length === 4) { segs.push([e[0], e[1]]); segs.push([e[2], e[3]]); }
    }
    return segs;
  }

  function linspace(a, b, n) {
    var out = [];
    for (var i = 0; i < n; i++) out.push(a + (b - a) * i / (n - 1));
    return out;
  }

  function xsOf(ships) { return ships.map(function (s) { return s.x.slice(); }); }
  function ysOf(ships) { return ships.map(function (s) { return s.y; }); }

  var MATH = {
    LIM: LIM, HARD_C: HARD_C, ALPHA_TOL: ALPHA_TOL, RAW: RAW, SHIPS: SHIPS, EXTRA: EXTRA, HAND_LINES: HAND_LINES, RING: RING,
    toRaw: toRaw, standardize: standardize, dot: dot, norm: norm, KERNELS: KERNELS, phiQuadratic: phiQuadratic,
    trainSVM: trainSVM, trainLinear: trainLinear, trainHard: trainHard, score: score, signedDistance: signedDistance,
    geometricMargin: geometricMargin, slacks: slacks, primalObjective: primalObjective, dualObjective: dualObjective,
    clipLine: clipLine, clipHalf: clipHalf, box: box, contour: contour, linspace: linspace, xsOf: xsOf, ysOf: ysOf
  };

  if (typeof module !== 'undefined' && module.exports) module.exports = MATH;
  if (typeof document === 'undefined') return;

  /* ==========================================================================
     PART 2 — DRAWING
     ====================================================================== */

  var NS = 'http://www.w3.org/2000/svg';
  var PL = 54, PT = 12, SIZE = 400;               // square plot area: equal scales
  var VW = PL + SIZE + 14, VH = PT + SIZE + 48;
  var reduceMotion = global.matchMedia && global.matchMedia('(prefers-reduced-motion: reduce)').matches;

  function $(id) { return document.getElementById(id); }
  function el(name, attrs, parent) {
    var e = document.createElementNS(NS, name);
    for (var k in attrs) e.setAttribute(k, attrs[k]);
    if (parent) parent.appendChild(e);
    return e;
  }
  function f2(v) { return (v < 0 ? '−' : '') + Math.abs(v).toFixed(2); }
  function fs(v) { return (v < 0 ? '− ' : '+ ') + Math.abs(v).toFixed(2); }   // signed term
  function plural(n, w) { return n + ' ' + w + (n === 1 ? '' : 's'); }
  function sideName(y) { return y > 0 ? 'Imperial' : 'Rebel'; }
  function sideTag(y) {
    return '<b class="' + (y > 0 ? 'imp' : 'reb') + '">' + sideName(y) + ' (' + (y > 0 ? '+1' : '−1') + ')</b>';
  }

  /* Render TeX with KaTeX if it loaded; otherwise show the source. */
  function tex(node, src, display) {
    if (global.katex) {
      try { global.katex.render(src, node, { throwOnError: false, displayMode: !!display }); return; } catch (e) { /* fall through */ }
    }
    node.textContent = src;
    node.classList.add('tex-raw');
  }
  function texInline(src) {
    var s = document.createElement('span');
    tex(s, src, false);
    return s.outerHTML;
  }

  /* One square scatter plot in data coordinates [−LIM, LIM]². */
  function Plot(svgId) {
    var svg = $(svgId);
    svg.setAttribute('viewBox', '0 0 ' + VW + ' ' + VH);
    function sx(x) { return PL + (x + LIM) / (2 * LIM) * SIZE; }
    function sy(y) { return PT + SIZE - (y + LIM) / (2 * LIM) * SIZE; }

    var defs = el('defs', {}, svg);
    var clipId = svgId + '-clip';
    el('rect', { x: PL, y: PT, width: SIZE, height: SIZE }, el('clipPath', { id: clipId }, defs));
    var mk = el('marker', { id: svgId + '-arrow', viewBox: '0 0 10 10', refX: 8, refY: 5, markerWidth: 7, markerHeight: 7, orient: 'auto-start-reverse' }, defs);
    el('path', { d: 'M0,0 L10,5 L0,10 Z', 'class': 'arrow-head' }, mk);

    el('rect', { 'class': 'scope-bg', x: PL, y: PT, width: SIZE, height: SIZE }, svg);
    var root = el('g', { 'clip-path': 'url(#' + clipId + ')' }, svg);
    var layers = {};
    ['shade', 'grid', 'corridor', 'lines', 'extra', 'ships', 'labels'].forEach(function (k) {
      layers[k] = el('g', { 'class': 'layer-' + k }, k === 'ships' || k === 'labels' ? svg : root);
    });
    for (var g = -LIM; g <= LIM; g++) {
      el('line', { 'class': g === 0 ? 'grid-zero' : 'grid', x1: sx(g), y1: PT, x2: sx(g), y2: PT + SIZE }, layers.grid);
      el('line', { 'class': g === 0 ? 'grid-zero' : 'grid', x1: PL, y1: sy(g), x2: PL + SIZE, y2: sy(g) }, layers.grid);
      el('text', { 'class': 'tick', x: sx(g), y: PT + SIZE + 17, 'text-anchor': 'middle' }, svg).textContent = g;
      el('text', { 'class': 'tick', x: PL - 8, y: sy(g) + 4, 'text-anchor': 'end' }, svg).textContent = g;
    }
    el('rect', { 'class': 'scope-frame', x: PL, y: PT, width: SIZE, height: SIZE }, svg);
    el('text', { 'class': 'axis-title', x: PL + SIZE / 2, y: PT + SIZE + 38, 'text-anchor': 'middle' }, svg)
      .textContent = 'x₁  engine output (standardized)';
    el('text', { 'class': 'axis-title', x: 0, y: 0, 'text-anchor': 'middle', transform: 'translate(15,' + (PT + SIZE / 2) + ') rotate(-90)' }, svg)
      .textContent = 'x₂  transponder strength (standardized)';

    function poly(pts, cls, layer) {
      if (!pts || pts.length < 3) return null;
      return el('path', { 'class': cls, d: 'M' + pts.map(function (p) { return sx(p[0]).toFixed(1) + ',' + sy(p[1]).toFixed(1); }).join('L') + 'Z' }, layers[layer || 'shade']);
    }

    var P = {
      svg: svg, sx: sx, sy: sy, layers: layers, arrowId: svgId + '-arrow',
      clear: function () { for (var k in arguments) layers[arguments[k]].textContent = ''; },

      /* The line {f(x) = c} for a linear score. */
      line: function (w, w0, c, cls, layer) {
        var seg = clipLine(w, w0, c, LIM + 0.5);
        if (!seg) return null;
        return el('line', { 'class': cls, x1: sx(seg[0][0]), y1: sy(seg[0][1]), x2: sx(seg[1][0]), y2: sy(seg[1][1]) }, layers[layer || 'lines']);
      },

      /* Tint each side of a linear boundary. */
      shadeLinear: function (w, w0) {
        layers.shade.textContent = '';
        var B = box(LIM + 0.5);
        poly(clipHalf(B, w, w0, 0, -1), 'shade-reb');
        poly(clipHalf(B, w, w0, 0, 1), 'shade-imp');
      },

      /* The corridor between f = −lo and f = +hi (score units). */
      corridor: function (w, w0, half, cls) {
        var B = box(LIM + 0.5);
        poly(clipHalf(clipHalf(B, w, w0, -half, 1), w, w0, half, -1), cls || 'corridor', 'corridor');
      },

      /* Tint by the sign of an arbitrary score field (kernel models). */
      shadeField: function (fn) {
        layers.shade.textContent = '';
        var N = 100, cv = document.createElement('canvas');
        cv.width = cv.height = N;
        var ctx = cv.getContext('2d'), img = ctx.createImageData(N, N), d = img.data;
        for (var j = 0; j < N; j++) for (var i = 0; i < N; i++) {
          var x = -LIM + (i + 0.5) / N * 2 * LIM, y = LIM - (j + 0.5) / N * 2 * LIM;
          var v = fn([x, y]), k = 4 * (j * N + i);
          if (v > 0) { d[k] = 255; d[k + 1] = 157; d[k + 2] = 66; } else { d[k] = 78; d[k + 1] = 168; d[k + 2] = 255; }
          d[k + 3] = Math.abs(v) < 1 ? 22 : 34;
        }
        ctx.putImageData(img, 0, 0);
        el('image', { href: cv.toDataURL(), x: PL, y: PT, width: SIZE, height: SIZE, preserveAspectRatio: 'none', 'class': 'field' }, layers.shade);
      },

      /* Contour of a score field at level c, via marching squares. */
      contour: function (vals, xs, ys, c, cls) {
        var segs = contour(vals, xs, ys, c), d = '';
        segs.forEach(function (s) {
          d += 'M' + sx(s[0][0]).toFixed(1) + ',' + sy(s[0][1]).toFixed(1) + 'L' + sx(s[1][0]).toFixed(1) + ',' + sy(s[1][1]).toFixed(1);
        });
        if (d) el('path', { 'class': cls, d: d }, layers.lines);
      },

      /* Draw ships. opts: selected (index), ring (fn n → class or null),
         label (fn n → text or null), onPick (fn n), dim (bool). */
      ships: function (list, opts) {
        opts = opts || {};
        layers.ships.textContent = '';
        layers.labels.textContent = '';
        var nodes = [];
        list.forEach(function (s, n) {
          var px = sx(s.x[0]), py = sy(s.x[1]);
          var g = el('g', { 'class': 'ship ' + (s.y > 0 ? 'is-imp' : 'is-reb') + (opts.selected === n ? ' is-sel' : '') + (opts.dim ? ' is-dim' : ''),
            transform: 'translate(' + px.toFixed(1) + ',' + py.toFixed(1) + ')' }, layers.ships);
          var ring = opts.ring && opts.ring(n);
          if (ring) el('circle', { 'class': 'ring ' + ring, r: 14 }, g);
          if (opts.selected === n) el('circle', { 'class': 'ring ring-sel', r: 17 }, g);
          el('circle', { 'class': 'hit', r: 15 }, g);
          el('use', { href: '#' + (s.y > 0 ? 'ico-imperial' : 'ico-rebel'), x: -10, y: -10, width: 20, height: 20 }, g);
          el('title', {}, g).textContent = s.name + ' — ' + sideName(s.y) + ' (' + (s.y > 0 ? '+1' : '−1') + ')';
          if (opts.onPick) {
            g.setAttribute('tabindex', '0');
            g.setAttribute('role', 'button');
            g.setAttribute('aria-label', s.name + ', ' + sideName(s.y));
            g.addEventListener('click', function () { opts.onPick(n); });
            g.addEventListener('keydown', function (e) { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); opts.onPick(n); } });
          }
          var lab = opts.label && opts.label(n);
          if (lab) {
            var right = px + 18 + lab.length * 7.8 < PL + SIZE;
            el('text', { 'class': 'ship-label', x: px + (right ? 16 : -16), y: py - 10, 'text-anchor': right ? 'start' : 'end' }, layers.labels).textContent = lab;
          }
          nodes.push(g);
        });
        return nodes;
      },

      /* Pointer position in data coordinates. */
      toData: function (evt) {
        var pt = svg.createSVGPoint();
        pt.x = evt.clientX; pt.y = evt.clientY;
        var q = pt.matrixTransform(svg.getScreenCTM().inverse());
        return [(q.x - PL) / SIZE * 2 * LIM - LIM, LIM - (q.y - PT) / SIZE * 2 * LIM];
      },

      /* Label the line {f = c} just inside the top (or right) edge of the plot. */
      lineLabel: function (w, w0, c, str) {
        var seg = clipLine(w, w0, c, LIM - 0.05);
        if (!seg) return;
        var e = seg[0][1] > seg[1][1] ? seg[0] : seg[1];
        P.text([Math.min(e[0] + 0.1, LIM - 0.75), e[1] - 0.22], str, 'edge-label', 'extra', 'start');
      },

      seg: function (a, b, cls, layer) {
        return el('line', { 'class': cls, x1: sx(a[0]), y1: sy(a[1]), x2: sx(b[0]), y2: sy(b[1]) }, layers[layer || 'extra']);
      },
      text: function (p, str, cls, layer, anchor) {
        var t = el('text', { 'class': cls || 'ship-label', x: sx(p[0]), y: sy(p[1]), 'text-anchor': anchor || 'middle' }, layers[layer || 'labels']);
        t.textContent = str;
        return t;
      }
    };
    return P;
  }

  /* Foot of the perpendicular from x onto {f = c}. */
  function project(w, w0, x, c) {
    var k = (c - score(w, w0, x)) / dot(w, w);
    return [x[0] + k * w[0], x[1] + k * w[1]];
  }

  var LEGEND = {
    rebel:    '<svg class="key-ico is-reb" viewBox="-11 -11 22 22"><use href="#ico-rebel" x="-11" y="-11" width="22" height="22"/></svg>Rebel, ỹ = −1',
    imperial: '<svg class="key-ico is-imp" viewBox="-11 -11 22 22"><use href="#ico-imperial" x="-11" y="-11" width="22" height="22"/></svg>Imperial, ỹ = +1',
    unknown:  '<svg class="key-ico is-unk" viewBox="-11 -11 22 22"><use href="#ico-unknown" x="-11" y="-11" width="22" height="22"/></svg>Unidentified ship',
    boundary: '<i class="key key-boundary"></i>Boundary, f(x) = 0',
    edges:    '<i class="key key-edge"></i>Margin edges, f(x) = ±1',
    corridor: '<i class="key key-corridor"></i>Corridor (margin on each side)',
    arrow:    '<i class="key key-arrow"></i>Weight vector w',
    wrong:    '<i class="key key-ring key-wrong"></i>Misclassified ship',
    nearest:  '<i class="key key-ring key-near"></i>Nearest ship(s)',
    selected: '<i class="key key-ring key-sel"></i>Selected ship',
    violated: '<i class="key key-ring key-wrong"></i>Constraint broken: ỹf(x) &lt; 1',
    sv:       '<i class="key key-ring key-sv"></i>Support vector, α &gt; 0',
    edge0:    '<i class="key key-ring key-edge0"></i>On an edge, but α = 0',
    slack:    '<i class="key key-slack"></i>Slack ξ (shortfall to its edge)'
  };

  /* ==========================================================================
     PART 3 — SCENES
     ====================================================================== */

  document.querySelectorAll('.tex').forEach(function (n) { tex(n, n.getAttribute('data-tex'), false); });
  document.querySelectorAll('.tex-block[data-tex]').forEach(function (n) { tex(n, n.getAttribute('data-tex'), true); });
  document.querySelectorAll('.legend[data-keys]').forEach(function (ul) {
    ul.innerHTML = ul.getAttribute('data-keys').split(' ').map(function (k) { return '<li>' + LEGEND[k] + '</li>'; }).join('');
  });

  var BASE_X = xsOf(SHIPS), BASE_Y = ysOf(SHIPS);
  var HARD = trainHard(BASE_X, BASE_Y);            // line C, scenes 3–5
  var TRAINED_MARGIN = 1 / norm(HARD.w);

  function rawLine(x) {
    var r = toRaw(x);
    return Math.round(r[0]) + ' ' + RAW[0].unit + ', ' + r[1].toFixed(1) + ' ' + RAW[1].unit;
  }

  /* ------------------------------------------------ 1. training data -- */
  var scene1 = (function () {
    var P = Plot('plot1'), sel = 4;
    function draw() {
      P.ships(SHIPS, { selected: sel, onPick: function (n) { sel = n; draw(); }, label: function (n) { return n === sel ? SHIPS[n].name : null; } });
      var s = SHIPS[sel];
      $('read1').innerHTML =
        '<div class="ro-title">' + s.name + ' &middot; ship <i>n</i> = ' + (sel + 1) + '</div>' +
        '<div class="ro-grid">' +
        '<span>Raw log</span><b class="num">' + rawLine(s.x) + '</b>' +
        '<span>Position</span><b>' + texInline('\\mathbf{x}_{' + (sel + 1) + '} = (' + f2(s.x[0]) + ',\\ ' + f2(s.x[1]) + ')') + '</b>' +
        '<span>Label</span><b>' + texInline('\\tilde{y}_{' + (sel + 1) + '} = ' + (s.y > 0 ? '+1' : '-1')) + ' &nbsp;' + sideTag(s.y) + '</b>' +
        '</div>';
    }
    return { draw: draw };
  })();

  /* ------------------------------------------------------- 2. scoring -- */
  var scene2 = (function () {
    var P = Plot('plot2'), START = [1, 0.3, 0.8], sel = 10;
    var s1 = $('w1'), s2 = $('w2'), s0 = $('w0');
    function set(v) { s1.value = v[0]; s2.value = v[1]; s0.value = v[2]; }
    set(START);
    function draw() {
      var w = [+s1.value, +s2.value], w0 = +s0.value;
      $('w1v').textContent = f2(w[0]); $('w2v').textContent = f2(w[1]); $('w0v').textContent = f2(w0);
      P.clear('shade', 'lines', 'extra', 'corridor');
      var zero = norm(w) < 1e-9, wrong = [];
      if (!zero) {
        P.shadeLinear(w, w0);
        P.line(w, w0, 0, 'boundary');
        // w drawn from the boundary point nearest the origin
        var foot = project(w, w0, [0, 0], 0), u = [w[0] / norm(w), w[1] / norm(w)];
        if (Math.abs(foot[0]) < LIM - 0.3 && Math.abs(foot[1]) < LIM - 0.3) {
          var tip = [foot[0] + 0.9 * u[0], foot[1] + 0.9 * u[1]];
          var a = P.seg(foot, tip, 'w-arrow');
          a.setAttribute('marker-end', 'url(#' + P.arrowId + ')');
          P.text([tip[0] + 0.18 * u[0], tip[1] + 0.18 * u[1] - 0.08], 'w', 'w-label', 'extra');
        }
      }
      SHIPS.forEach(function (s, n) { if (s.y * score(w, w0, s.x) <= 0) wrong.push(n); });
      P.ships(SHIPS, {
        selected: sel, onPick: function (n) { sel = n; draw(); },
        ring: function (n) { return wrong.indexOf(n) >= 0 ? 'ring-wrong' : null; },
        label: function (n) { return n === sel ? SHIPS[n].name : null; }
      });
      var s = SHIPS[sel], f = score(w, w0, s.x), pred = f > 0 ? 1 : f < 0 ? -1 : 0;
      $('read2').innerHTML =
        '<div class="ro-eq">' + texInline('f(\\mathbf{x}) = ' + f2(w[0]) + '\\,x_1 ' + fs(w[1]) + '\\,x_2 ' + fs(w0)) + '</div>' +
        (zero ? '<p class="warn">With w = 0 every ship gets the same score, so there is no line.</p>' : '') +
        '<div class="ro-grid">' +
        '<span>' + s.name + '</span><b>' + texInline('f = ' + f2(w[0]) + '(' + f2(s.x[0]) + ') ' + fs(w[1]) + '(' + f2(s.x[1]) + ') ' + fs(w0) + ' = ' + f2(f)) + '</b>' +
        '<span>Prediction</span><b>' + (pred ? sideTag(pred) : 'exactly on the line') + (pred && pred !== s.y ? ' <span class="bad">✕ wrong, really ' + sideName(s.y) + '</span>' : pred ? ' <span class="good">✓</span>' : '') + '</b>' +
        '<span>Mistakes</span><b class="num ' + (wrong.length ? 'bad' : 'good') + '">' + wrong.length + ' of ' + SHIPS.length + '</b>' +
        '</div>';
    }
    [s1, s2, s0].forEach(function (s) { s.addEventListener('input', draw); });
    $('reset2').addEventListener('click', function () { set(START); draw(); });
    return { draw: draw };
  })();

  /* -------------------------------------------------------- 3. margin -- */
  var scene3 = (function () {
    var P = Plot('plot3'), pick = 'C';
    var LINES = HAND_LINES.map(function (l) { return { key: l.key, w: l.w, w0: l.w0, hand: true }; })
      .concat([{ key: 'C', w: HARD.w, w0: HARD.w0, hand: false }]);
    LINES.forEach(function (l) { l.margin = geometricMargin(l.w, l.w0, BASE_X, BASE_Y); });
    var buttons = document.querySelectorAll('[data-line]');
    buttons.forEach(function (b) { b.addEventListener('click', function () { pick = b.getAttribute('data-line'); draw(); }); });
    function draw() {
      var L = LINES.filter(function (l) { return l.key === pick; })[0];
      buttons.forEach(function (b) { b.setAttribute('aria-pressed', String(b.getAttribute('data-line') === pick)); });
      P.clear('shade', 'lines', 'extra', 'corridor');
      P.shadeLinear(L.w, L.w0);
      LINES.forEach(function (o) { if (o !== L) P.line(o.w, o.w0, 0, 'line-ghost'); });
      var half = L.margin * norm(L.w);                 // margin in score units
      P.corridor(L.w, L.w0, half);
      P.line(L.w, L.w0, half, 'edge'); P.line(L.w, L.w0, -half, 'edge');
      P.line(L.w, L.w0, 0, 'boundary');
      LINES.forEach(function (o) {
        var seg = clipLine(o.w, o.w0, 0, LIM - 0.15);
        if (seg) { var e = seg[0][1] > seg[1][1] ? seg[0] : seg[1]; P.text([e[0] + 0.12, e[1] - 0.2], o.key, 'line-key' + (o === L ? ' is-on' : ''), 'extra'); }
      });
      var near = [];
      SHIPS.forEach(function (s, n) {
        if (s.y * signedDistance(L.w, L.w0, s.x) < L.margin + 1e-3) {
          near.push(n);
          P.seg(s.x, project(L.w, L.w0, s.x, 0), 'dist-seg');
        }
      });
      P.ships(SHIPS, {
        ring: function (n) { return near.indexOf(n) >= 0 ? 'ring-near' : null; },
        label: function (n) { return near.indexOf(n) >= 0 ? SHIPS[n].name : null; }
      });
      $('tag3').className = 'tag ' + (L.hand ? 'tag-hand' : 'tag-trained');
      $('tag3').textContent = L.hand ? 'Line ' + L.key + ': hand-drawn' : 'Line C: trained by the solver (scene 5)';
      $('read3').innerHTML = '<table class="ro-table"><thead><tr><th>Line</th><th>Source</th><th>Mistakes</th><th>Margin</th></tr></thead><tbody>' +
        LINES.map(function (o) {
          return '<tr' + (o === L ? ' class="is-on"' : '') + '><td>' + o.key + '</td><td>' + (o.hand ? 'hand-drawn' : 'trained') + '</td><td class="num">0</td>' +
            '<td><span class="bar" style="width:' + (o.margin / TRAINED_MARGIN * 80).toFixed(0) + 'px"></span> <span class="num">' + o.margin.toFixed(2) + '</span></td></tr>';
        }).join('') + '</tbody></table>';
    }
    return { draw: draw };
  })();

  /* ------------------------------------------------------ 4. distance -- */
  var scene4 = (function () {
    var P = Plot('plot4'), sel = 4, slider = $('s4');
    slider.value = 0;
    function draw() {
      var s = Math.exp(+slider.value), w = [s * HARD.w[0], s * HARD.w[1]], w0 = s * HARD.w0;
      $('s4v').textContent = 's = ' + s.toFixed(2);
      P.clear('shade', 'lines', 'extra', 'corridor');
      P.shadeLinear(w, w0);
      P.line(w, w0, 1, 'edge'); P.line(w, w0, -1, 'edge');
      P.lineLabel(w, w0, 1, 'f = +1'); P.lineLabel(w, w0, 0, 'f = 0'); P.lineLabel(w, w0, -1, 'f = −1');
      P.line(w, w0, 0, 'boundary');
      var x = SHIPS[sel].x, foot = project(w, w0, x, 0);
      P.seg(x, foot, 'dist-seg');
      P.ships(SHIPS, { selected: sel, onPick: function (n) { sel = n; draw(); }, label: function (n) { return n === sel ? SHIPS[n].name : null; } });
      var f = score(w, w0, x), nw = norm(w);
      $('read4').innerHTML =
        '<div class="ro-eq">' + texInline('s\\mathbf{w} = (' + f2(w[0]) + ',\\ ' + f2(w[1]) + '), \\quad s\\,w_0 = ' + f2(w0)) + '</div>' +
        '<div class="ro-grid ro-cmp">' +
        '<span>Score <i>f</i>(<b>x</b>)</span><b class="num changes">' + f2(f) + '</b><em>changes with s</em>' +
        '<span>Length ‖<b>w</b>‖</span><b class="num changes">' + nw.toFixed(2) + '</b><em>changes with s</em>' +
        '<span>Distance <i>f</i>/‖<b>w</b>‖</span><b class="num fixed">' + f2(f / nw) + '</b><em>fixed</em>' +
        '<span>f = ±1 lines sit at</span><b class="num changes">' + (1 / nw).toFixed(2) + '</b><em>from the boundary</em>' +
        '</div>';
    }
    slider.addEventListener('input', draw);
    return { draw: draw };
  })();

  /* ------------------------------------------------ 5. hard training -- */
  var scene5 = (function () {
    var P = Plot('plot5'), slider = $('t5'), labBtn = $('lab5'), showAll = false;
    slider.value = 1;
    labBtn.addEventListener('click', function () { showAll = !showAll; labBtn.setAttribute('aria-pressed', String(showAll)); draw(); });
    function draw() {
      var t = +slider.value, w = [t * HARD.w[0], t * HARD.w[1]], w0 = t * HARD.w0;
      $('t5v').textContent = 't = ' + t.toFixed(2);
      P.clear('shade', 'lines', 'extra', 'corridor');
      P.shadeLinear(w, w0);
      P.corridor(w, w0, 1);
      P.line(w, w0, 1, 'edge'); P.line(w, w0, -1, 'edge');
      P.line(w, w0, 0, 'boundary');
      P.lineLabel(w, w0, 1, 'f = +1'); P.lineLabel(w, w0, 0, 'f = 0'); P.lineLabel(w, w0, -1, 'f = −1');
      var yf = SHIPS.map(function (s) { return s.y * score(w, w0, s.x); });
      var broken = yf.filter(function (v) { return v < 1 - 1e-6; }).length;
      P.ships(SHIPS, {
        ring: function (n) { return yf[n] < 1 - 1e-6 ? 'ring-wrong' : null; },
        label: function (n) { return showAll || yf[n] < 1.05 ? yf[n].toFixed(2) : null; }
      });
      var msg;
      if (Math.abs(t - 1) < 0.005) msg = '<p class="good">This is the trained answer: the smallest ‖w‖ in this direction that keeps every ỹₙf(xₙ) ≥ 1. The nearest ships sit exactly on the edges.</p>';
      else if (broken) msg = '<p class="bad">Wider corridor, but ' + plural(broken, 'ship') + ' now score' + (broken === 1 ? 's' : '') + ' below 1. Not allowed.</p>';
      else msg = '<p>Every constraint holds, but ‖w‖ is bigger than needed, so the corridor is narrower than it could be.</p>';
      $('read5').innerHTML =
        '<div class="ro-grid">' +
        '<span>‖w‖</span><b class="num">' + norm(w).toFixed(3) + '</b>' +
        '<span>½‖w‖² (minimized)</span><b class="num">' + (0.5 * dot(w, w)).toFixed(3) + '</b>' +
        '<span>Margin 1/‖w‖</span><b class="num">' + (1 / norm(w)).toFixed(3) + '</b>' +
        '<span>Constraints broken</span><b class="num ' + (broken ? 'bad' : 'good') + '">' + broken + ' of ' + SHIPS.length + '</b>' +
        '</div>' + msg;
    }
    slider.addEventListener('input', draw);
    return { draw: draw };
  })();

  /* ---------------------------------------------- 6. support vectors -- */
  var scene6 = (function () {
    var P = Plot('plot6'), ships, model, drag = null, last = null;
    function reset() { ships = SHIPS.map(function (s) { return { name: s.name, x: s.x.slice(), y: s.y }; }); retrain(); }
    function retrain() {
      var m = trainHard(xsOf(ships), ysOf(ships));
      if (!m.separable) return false;
      model = m;
      return true;
    }
    function draw() {
      var w = model.w, w0 = model.w0;
      P.clear('shade', 'lines', 'extra', 'corridor');
      P.shadeLinear(w, w0);
      P.corridor(w, w0, 1);
      P.line(w, w0, 1, 'edge'); P.line(w, w0, -1, 'edge');
      P.line(w, w0, 0, 'boundary');
      var yf = ships.map(function (s) { return s.y * score(w, w0, s.x); });
      var nodes = P.ships(ships, {
        ring: function (n) {
          if (model.alpha[n] > ALPHA_TOL) return 'ring-sv';
          return Math.abs(yf[n] - 1) < 1e-3 ? 'ring-edge0' : null;
        },
        label: function (n) { return model.alpha[n] > ALPHA_TOL || Math.abs(yf[n] - 1) < 1e-3 ? ships[n].name : null; }
      });
      nodes.forEach(function (g, n) {
        g.classList.add('is-draggable');
        g.addEventListener('pointerdown', function (e) {
          e.preventDefault();
          drag = n; last = ships[n].x.slice();
          P.svg.setPointerCapture(e.pointerId);
        });
      });
      var maxA = Math.max.apply(null, Array.from(model.alpha));
      $('alphas6').innerHTML = '<div class="al-head"><span>Ship</span><span>α<sub>n</sub></span><span>ỹ<sub>n</sub>f(x<sub>n</sub>)</span></div>' +
        ships.map(function (s, n) {
          var a = model.alpha[n], on = a > ALPHA_TOL;
          return '<div class="al-row' + (on ? ' is-sv' : '') + '"><span class="al-name ' + (s.y > 0 ? 'imp' : 'reb') + '">' + s.name + '</span>' +
            '<span class="al-bar"><i style="width:' + (on ? Math.max(3, a / maxA * 100) : 0).toFixed(0) + '%"></i><b class="num">' + (on ? a.toFixed(3) : '0') + '</b></span>' +
            '<span class="num">' + yf[n].toFixed(2) + '</span></div>';
        }).join('');
    }
    P.svg.addEventListener('pointermove', function (e) {
      if (drag === null) return;
      var p = P.toData(e);
      var prev = ships[drag].x;
      ships[drag].x = [Math.max(-2.85, Math.min(2.85, p[0])), Math.max(-2.85, Math.min(2.85, p[1]))];
      if (retrain()) { $('warn6').hidden = true; last = ships[drag].x.slice(); }
      else {
        ships[drag].x = prev;
        $('warn6').hidden = false;
        $('warn6').textContent = 'No straight line separates the ships there, so the hard-margin problem has no solution (scene 8 fixes this). Ship kept at its last valid spot.';
      }
      draw();
    });
    function end() { drag = null; }
    P.svg.addEventListener('pointerup', end);
    P.svg.addEventListener('pointercancel', end);
    $('reset6').addEventListener('click', function () { $('warn6').hidden = true; reset(); draw(); });
    reset();
    return { draw: draw };
  })();

  /* ----------------------------------------------------- 7. kernels -- */
  var scene7 = (function () {
    var P = Plot('plot7'), kind = 'rbf', gSlider = $('g7'), sel = null, C7 = 10;
    var RX = xsOf(RING), RY = ysOf(RING);
    var GX = linspace(-LIM, LIM, 61);
    gSlider.value = Math.log(0.5);
    var buttons = document.querySelectorAll('[data-kernel]');
    buttons.forEach(function (b) { b.addEventListener('click', function () { kind = b.getAttribute('data-kernel'); draw(); }); });
    gSlider.addEventListener('input', draw);
    function draw() {
      var gamma = Math.exp(+gSlider.value);
      $('g7v').textContent = 'γ = ' + gamma.toFixed(2);
      $('gammaBox').hidden = kind !== 'rbf';
      buttons.forEach(function (b) { b.setAttribute('aria-pressed', String(b.getAttribute('data-kernel') === kind)); });
      var m = trainSVM(RX, RY, KERNELS[kind](gamma), C7);
      P.clear('shade', 'lines', 'extra', 'corridor');
      P.shadeField(m.f);
      var vals = GX.map(function (a) { return GX.map(function (b) { return m.f([a, b]); }); });
      P.contour(vals, GX, GX, 1, 'edge'); P.contour(vals, GX, GX, -1, 'edge');
      P.contour(vals, GX, GX, 0, 'boundary');
      var nSV = 0, wrong = 0;
      RING.forEach(function (s, n) { if (m.alpha[n] > ALPHA_TOL) nSV++; if (s.y * m.f(s.x) <= 0) wrong++; });
      P.ships(RING, {
        selected: sel === null ? undefined : sel,
        onPick: function (n) { sel = n; draw(); },
        ring: function (n) { return m.alpha[n] > ALPHA_TOL ? 'ring-sv' : null; }
      });
      var KTEX = { linear: 'K = \\mathbf{x}^{\\mathsf T}\\mathbf{x}_n', quadratic: 'K = (\\mathbf{x}^{\\mathsf T}\\mathbf{x}_n + 1)^2', rbf: 'K = \\exp(-' + gamma.toFixed(2) + '\\,\\lVert\\mathbf{x} - \\mathbf{x}_n\\rVert^2)' };
      var note = kind === 'linear' ? 'A straight line can&rsquo;t enclose the smugglers, so many ships end up inside the corridor or on the wrong side.'
        : kind === 'quadratic' ? 'An ellipse fits: the quadratic kernel is a straight cut in a six-reading space.'
        : gamma > 2 ? 'Very large γ: each ship&rsquo;s influence is tiny, so the boundary hugs individual ships. Fewer mistakes here, but this tends to overfit.'
        : 'The Gaussian kernel bends the boundary around the cluster.';
      $('read7').innerHTML = '<div class="ro-eq">' + texInline(KTEX[kind]) + '</div><div class="ro-grid">' +
        '<span>Training mistakes</span><b class="num ' + (wrong ? 'bad' : 'good') + '">' + wrong + ' of ' + RING.length + '</b>' +
        '<span>Support vectors</span><b class="num">' + nSV + ' of ' + RING.length + '</b></div><p>' + note + '</p>';
      if (sel !== null) {
        var a = RING[sel].x, b = RING[0].x, k = KERNELS.quadratic()(a, b), pa = phiQuadratic(a), pb = phiQuadratic(b);
        var pp = pa.reduce(function (s, v, i) { return s + v * pb[i]; }, 0);
        $('phi7').innerHTML = RING[sel].name + ' vs. ' + RING[0].name + ': ' +
          texInline('(\\mathbf{x}^{\\mathsf T}\\mathbf{z} + 1)^2 = ' + k.toFixed(4)) + ', and ' +
          texInline('\\phi(\\mathbf{x})^{\\mathsf T}\\phi(\\mathbf{z}) = ' + pp.toFixed(4)) + '. Same number, but the left side never built the six features.';
      }
    }
    return { draw: draw };
  })();

  /* -------------------------------------------------- 8. soft margin -- */
  var SOFT = { ships: SHIPS.concat(EXTRA), C: 1, model: null };
  var scene8 = (function () {
    var P = Plot('plot8'), slider = $('c8');
    var SX = xsOf(SOFT.ships), SY = ysOf(SOFT.ships);
    slider.value = 0;
    function draw() {
      var C = Math.pow(10, +slider.value);
      SOFT.C = C;
      $('c8v').textContent = 'C = ' + (C < 1 ? C.toFixed(C < 0.1 ? 3 : 2) : C.toFixed(C < 10 ? 2 : 1));
      var m = trainLinear(SX, SY, C), w = m.w, w0 = m.w0;
      SOFT.model = m;
      P.clear('shade', 'lines', 'extra', 'corridor');
      P.shadeLinear(w, w0);
      P.corridor(w, w0, 1);
      P.line(w, w0, 1, 'edge'); P.line(w, w0, -1, 'edge');
      P.line(w, w0, 0, 'boundary');
      var xi = slacks(m.f, SX, SY), inside = 0, wrongSide = 0, sum = 0;
      xi.forEach(function (v, n) {
        sum += v;
        if (v > 1e-6) {
          if (v > 1) wrongSide++; else inside++;
          var s = SOFT.ships[n];
          P.seg(s.x, project(w, w0, s.x, s.y), 'slack-seg');
        }
      });
      P.ships(SOFT.ships, {
        ring: function (n) { return m.alpha[n] > ALPHA_TOL ? 'ring-sv' : null; },
        label: function (n) {
          var s = SOFT.ships[n];
          if (s.special) return (s.special === 'shuttle' ? 'Tydirium' : 'Courier') + (xi[n] > 1e-6 ? '  ξ = ' + xi[n].toFixed(2) : '');
          return xi[n] > 1e-6 ? 'ξ = ' + xi[n].toFixed(2) : null;
        }
      });
      $('read8').innerHTML = '<div class="ro-grid">' +
        '<span>Corridor width 2/‖w‖</span><b class="num">' + (2 / norm(w)).toFixed(2) + '</b>' +
        '<span>Inside corridor, right side (0 &lt; ξ ≤ 1)</span><b class="num">' + inside + '</b>' +
        '<span>Wrong side (ξ &gt; 1)</span><b class="num ' + (wrongSide ? 'bad' : '') + '">' + wrongSide + '</b>' +
        '<span>Total slack Σξ</span><b class="num">' + sum.toFixed(2) + '</b>' +
        '<span>½‖w‖² + CΣξ</span><b class="num">' + (0.5 * dot(w, w)).toFixed(2) + ' + ' + (C * sum).toFixed(2) + ' = ' + (0.5 * dot(w, w) + C * sum).toFixed(2) + '</b>' +
        '</div>';
    }
    slider.addEventListener('input', draw);
    return { draw: draw };
  })();

  /* ------------------------------------------------ 9. final mission -- */
  var scene9 = (function () {
    var P = Plot('plot9'), probe = [-0.2, 1.5], e = $('e9'), p = $('p9');
    function syncInputs() { var r = toRaw(probe); e.value = Math.round(r[0]); p.value = r[1].toFixed(1); }
    function draw() {
      var m = SOFT.model, w = m.w, w0 = m.w0, nw = norm(w);
      $('tag9').innerHTML = 'Trained soft-margin model from scene 8, ' + texInline('C = ' + (SOFT.C < 1 ? SOFT.C.toFixed(2) : SOFT.C.toFixed(1)));
      P.clear('shade', 'lines', 'extra', 'corridor');
      P.shadeLinear(w, w0);
      P.corridor(w, w0, 1);
      P.line(w, w0, 1, 'edge'); P.line(w, w0, -1, 'edge');
      P.line(w, w0, 0, 'boundary');
      P.ships(SOFT.ships, { dim: true });
      var px = P.sx(probe[0]), py = P.sy(probe[1]);
      var g = el('g', { 'class': 'ship is-unk' + (reduceMotion ? '' : ' is-new'), transform: 'translate(' + px.toFixed(1) + ',' + py.toFixed(1) + ')' }, P.layers.ships);
      el('circle', { 'class': 'ring ring-sel', r: 17 }, g);
      el('use', { href: '#ico-unknown', x: -11, y: -11, width: 22, height: 22 }, g);
      P.seg(probe, project(w, w0, probe, 0), 'dist-seg');

      var f = score(w, w0, probe), r = f / nw, cls = f > 0 ? 1 : -1;
      var zone = Math.abs(f) >= 1 ? 'beyond the margin edge: a confident call' : 'inside the corridor: a close call';
      $('verdict9').className = 'verdict ' + (f > 0 ? 'v-imp' : 'v-reb');
      $('verdict9').innerHTML = '<span class="v-k">' + (f > 0 ? 'IMPERIAL' : 'REBEL') + '</span>' +
        '<span>score ' + texInline('f(\\mathbf{x}) = ' + f2(f)) + ' &nbsp;·&nbsp; distance ' + texInline(f2(Math.abs(r))) + ' &nbsp;·&nbsp; ' + zone + '</span>';
      tex($('calc9'),
        '\\begin{aligned}\\mathbf{x} &= (' + f2(probe[0]) + ',\\ ' + f2(probe[1]) + ')\\\\' +
        'f(\\mathbf{x}) &= ' + f2(w[0]) + '(' + f2(probe[0]) + ') ' + fs(w[1]) + '(' + f2(probe[1]) + ') ' + fs(w0) + ' = ' + f2(f) + '\\\\' +
        '\\hat{y} &= \\operatorname{sign}(' + f2(f) + ') = ' + (cls > 0 ? '+1' : '-1') + '\\\\' +
        '\\frac{f(\\mathbf{x})}{\\lVert\\mathbf{w}\\rVert} &= \\frac{' + f2(f) + '}{' + nw.toFixed(2) + '} = ' + f2(r) + '\\end{aligned}', true);
    }
    P.svg.addEventListener('click', function (evt) {
      var q = P.toData(evt);
      if (Math.abs(q[0]) > LIM || Math.abs(q[1]) > LIM) return;
      probe = [round2(q[0]), round2(q[1])];
      syncInputs(); draw();
    });
    function fromInputs() {
      var x = standardize([+e.value, +p.value]);
      if (!isFinite(x[0]) || !isFinite(x[1])) return;
      probe = [Math.max(-LIM, Math.min(LIM, x[0])), Math.max(-LIM, Math.min(LIM, x[1]))];
      draw();
    }
    e.addEventListener('input', fromInputs); p.addEventListener('input', fromInputs);
    syncInputs();
    return { draw: draw };
  })();

  /* --------------------------------------------------- navigation -- */
  var SCENES = [scene1, scene2, scene3, scene4, scene5, scene6, scene7, scene8, scene9];
  var sections = Array.from(document.querySelectorAll('.scene'));
  var current = 0;
  var prog = $('progress');
  sections.forEach(function (s, i) {
    var li = document.createElement('li'), b = document.createElement('button');
    b.type = 'button';
    b.innerHTML = '<span class="p-num">' + (i + 1) + '</span><span class="p-lab">' + s.getAttribute('data-short') + '</span>';
    b.setAttribute('aria-label', 'Scene ' + (i + 1) + ': ' + s.querySelector('h2').textContent);
    b.addEventListener('click', function () { go(i, true); });
    li.appendChild(b); prog.appendChild(li);
  });
  var progButtons = prog.querySelectorAll('button');

  function go(i, scroll) {
    current = Math.max(0, Math.min(SCENES.length - 1, i));
    sections.forEach(function (s, k) { s.hidden = k !== current; });
    progButtons.forEach(function (b, k) {
      b.classList.toggle('is-done', k < current);
      if (k === current) b.setAttribute('aria-current', 'step'); else b.removeAttribute('aria-current');
    });
    $('prev').disabled = current === 0;
    $('next').disabled = current === SCENES.length - 1;
    $('sceneCount').textContent = 'Scene ' + (current + 1) + ' of ' + SCENES.length;
    SCENES[current].draw();
    if (global.history && global.history.replaceState) global.history.replaceState(null, '', '#scene-' + (current + 1));
    if (scroll) $('console').scrollIntoView({ behavior: reduceMotion ? 'auto' : 'smooth', block: 'start' });
  }

  $('prev').addEventListener('click', function () { go(current - 1, true); });
  $('next').addEventListener('click', function () { go(current + 1, true); });
  document.addEventListener('keydown', function (e) {
    var t = e.target.tagName;
    if (t === 'INPUT' || t === 'TEXTAREA' || t === 'SELECT' || e.altKey || e.ctrlKey || e.metaKey) return;
    if (e.key === 'ArrowRight') go(current + 1, false);
    else if (e.key === 'ArrowLeft') go(current - 1, false);
  });

  var m = /scene-(\d)/.exec(global.location.hash);
  // Scene 9 reads scene 8's model, so build it once up front.
  scene8.draw();
  go(m ? +m[1] - 1 : 0, false);
})(typeof window !== 'undefined' ? window : this);
