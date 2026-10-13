/* ============================================================================
   Touch, Then Climb — EM for a Gaussian mixture, its lower bound and the KL gap.
   CS 473 module 6 · Murphy, PML §8.7.2–8.7.3

   The model: 14 fixed points on a line, two Gaussian clusters with fixed
   σ = 1 and fixed weights π = (½, ½). Cluster 2's centre μ₂ is fixed too.
   EM learns exactly ONE number, cluster 1's centre μ₁. Because μ₁ is the only
   parameter, a plot of the true score against μ₁ is the whole landscape, not a
   slice of it, so the bound picture is exactly what the optimiser sees.

   Layout of this file
     PART 1  Math          pure, DOM-free, exported for the Node test suite
     PART 2  State         the run: μ₁, q, history, animation
     PART 3  Plots         clusters, score + bound, progress
     PART 4  Narration     the side notes and phase chips
     PART 5  Controls      buttons, auto-play, sliders, dragging, presets

   PART 1 runs under Node (`require('./script.js')` returns the math only); the
   rest is skipped when there is no document. See test-math.js.
   ========================================================================= */

(function (global) {
  'use strict';

  /* ==========================================================================
     PART 1 — MATH
     Everything here is a pure function of its arguments. No DOM, no state.
     All probabilities are handled as logs and combined with log-sum-exp, so
     nothing underflows even when a centre is far from every point.
     ====================================================================== */

  var DATA = [-3.2, -2.8, -2.5, -2.2, -1.9, -1.6, -1.3, -1.0, -0.6,
              2.4, 2.9, 3.3, 3.7, 4.2];
  var SIGMA = 1;                 // both clusters, fixed
  var WEIGHTS = [0.5, 0.5];      // mixture weights π₁, π₂, fixed
  var LOG_W = WEIGHTS.map(Math.log);

  /* log N(y | μ, σ²) */
  function logNormal(y, mu, sigma) {
    var d = (y - mu) / sigma;
    return -0.5 * Math.log(2 * Math.PI) - Math.log(sigma) - 0.5 * d * d;
  }

  /* log Σ exp(v), shifted by the max so the biggest term is exp(0) = 1. */
  function logSumExp(v) {
    var m = -Infinity, i, s = 0;
    for (i = 0; i < v.length; i++) if (v[i] > m) m = v[i];
    if (m === -Infinity) return -Infinity;
    for (i = 0; i < v.length; i++) s += Math.exp(v[i] - m);
    return m + Math.log(s);
  }

  /* log p(y, z = k | θ) = log π_k + log N(y | μ_k, σ²), for k = 1, 2. */
  function logJoint(y, mu1, mu2) {
    return [LOG_W[0] + logNormal(y, mu1, SIGMA), LOG_W[1] + logNormal(y, mu2, SIGMA)];
  }

  /* The posterior p(z_n | y_n, θ) for every point, i.e. the responsibilities.
     Each row keeps the probabilities AND their logs, computed separately, so
     a responsibility of 1 − 1e-20 never turns into log(0) for the other one. */
  function posterior(mu1, mu2, data) {
    data = data || DATA;
    return data.map(function (y) {
      var a = logJoint(y, mu1, mu2), lse = logSumExp(a);
      var l1 = a[0] - lse, l2 = a[1] - lse;
      return { p: [Math.exp(l1), Math.exp(l2)], logp: [l1, l2], logEvidence: lse };
    });
  }

  /* The true score: observed-data log likelihood ℓ(θ) = Σ_n log Σ_k π_k N(y_n | μ_k). */
  function logLik(mu1, mu2, data) {
    data = data || DATA;
    var s = 0;
    for (var n = 0; n < data.length; n++) s += logSumExp(logJoint(data[n], mu1, mu2));
    return s;
  }

  /* q as a plain array of [q_n1, q_n2] rows. */
  function uniformQ(data) {
    return (data || DATA).map(function () { return [0.5, 0.5]; });
  }

  function qFromPosterior(post) {
    return post.map(function (r) { return [r.p[0], r.p[1]]; });
  }

  /* x log x with the convention 0 log 0 = 0. */
  function xlogx(x) { return x > 0 ? x * Math.log(x) : 0; }

  /* The lower bound (ELBO):
       L(θ, q) = Σ_n Σ_k q_nk [ log π_k + log N(y_n | μ_k) − log q_nk ]
     Expected complete-data log likelihood plus the entropy of q. */
  function elbo(mu1, mu2, q, data) {
    data = data || DATA;
    var s = 0;
    for (var n = 0; n < data.length; n++) {
      var a = logJoint(data[n], mu1, mu2);
      for (var k = 0; k < 2; k++) s += q[n][k] * a[k] - xlogx(q[n][k]);
    }
    return s;
  }

  /* D_KL( q_n ‖ p(z_n | y_n, θ) ) for every point, from log posteriors
     directly, so it stays finite and non-negative even at extreme μ₁. */
  function klTerms(mu1, mu2, q, data) {
    var post = posterior(mu1, mu2, data);
    return post.map(function (r, n) {
      var s = 0;
      for (var k = 0; k < 2; k++) if (q[n][k] > 0) s += q[n][k] * (Math.log(q[n][k]) - r.logp[k]);
      return Math.max(0, s);
    });
  }

  function sum(v) { var s = 0; for (var i = 0; i < v.length; i++) s += v[i]; return s; }

  /* M step for μ₁ with q held fixed. L(μ₁) is a downward parabola in μ₁:
       L(μ₁) = const − (1 / 2σ²) Σ_n q_n1 (y_n − μ₁)²
     so its single maximum is the q-weighted average of the data. If cluster 1
     owns essentially nothing, the bound is flat in μ₁ and we leave it put. */
  function mStep(q, mu1Old, data) {
    data = data || DATA;
    var w = 0, wy = 0;
    for (var n = 0; n < data.length; n++) { w += q[n][0]; wy += q[n][0] * data[n]; }
    return w > 1e-300 ? wy / w : mu1Old;
  }

  /* One whole snapshot: everything the page reports for (μ₁, μ₂, q). */
  function measure(mu1, mu2, q, data) {
    var ll = logLik(mu1, mu2, data), lb = elbo(mu1, mu2, q, data), kl = klTerms(mu1, mu2, q, data);
    return { ll: ll, elbo: lb, gap: ll - lb, klSum: sum(kl), kl: kl };
  }

  /* Run EM from a starting μ₁ and return every half-step. Used by the tests
     and by the presets' sanity checks; the page itself steps one at a time. */
  function runEM(mu1, mu2, halfSteps, data) {
    var q = uniformQ(data);
    var out = [{ kind: 'start', mu1: mu1, q: q, m: measure(mu1, mu2, q, data) }];
    for (var i = 0; i < halfSteps; i++) {
      if (i % 2 === 0) q = qFromPosterior(posterior(mu1, mu2, data));
      else mu1 = mStep(q, mu1, data);
      out.push({ kind: i % 2 === 0 ? 'E' : 'M', mu1: mu1, q: q, m: measure(mu1, mu2, q, data) });
    }
    return out;
  }

  function mulberry32(seed) {
    var a = seed >>> 0;
    return function () {
      a = (a + 0x6D2B79F5) >>> 0;
      var t = a;
      t = Math.imul(t ^ (t >>> 15), 1 | t);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }

  var MATH = {
    DATA: DATA, SIGMA: SIGMA, WEIGHTS: WEIGHTS,
    logNormal: logNormal, logSumExp: logSumExp, logJoint: logJoint,
    posterior: posterior, logLik: logLik, uniformQ: uniformQ, qFromPosterior: qFromPosterior,
    elbo: elbo, klTerms: klTerms, mStep: mStep, measure: measure, runEM: runEM,
    mulberry32: mulberry32
  };

  if (typeof module !== 'undefined' && module.exports) module.exports = MATH;
  if (typeof document === 'undefined') return;


  /* ==========================================================================
     PART 2 — STATE
     ====================================================================== */

  var X_MIN = -6, X_MAX = 7;          // shared μ / y axis for plots 1 and 2
  var SLIDER_MIN = -5.5, SLIDER_MAX = 6.5;
  var SETTLE_TOL = 5e-4;              // |Δμ₁| below this after an M step = settled
  var MAX_HALF_STEPS = 60;
  var DUR = 750;                      // ms per animated half-step
  var PAUSE = 650;                    // ms between half-steps on auto-play

  var reduceMotion = window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches;

  var S = {
    mu1Start: -0.5,
    mu2: -2,
    mu1: -0.5,
    q: uniformQ(),
    next: 'E',          // which half-step is allowed next
    last: null,         // 'E1' first E step, 'M', 'E' a later E step
    history: [],        // { kind, mu1, ll, elbo, gap }
    prev: null,         // snapshot before the last half-step, for narration
    settled: false,
    playing: false,
    anim: null,         // { kind, t0, from, to, done }
    view: null,         // what is drawn this frame: { mu1, q }
    yRange: null        // plot 2's y range, fixed for a run
  };

  function $(id) { return document.getElementById(id); }

  function fmt(x, d) {
    var s = x.toFixed(d === undefined ? 3 : d);
    return /^-0\.?0*$/.test(s) ? s.slice(1) : s;
  }

  /* Signed number with a real minus sign, for prose. */
  function sn(x, d) { return fmt(x, d).replace('-', '−'); }

  function snapshot() {
    var m = measure(S.mu1, S.mu2, S.q);
    return { mu1: S.mu1, ll: m.ll, elbo: m.elbo, gap: m.gap, klSum: m.klSum };
  }

  /* The global and local peaks of ℓ(μ₁) for the current μ₂, from a fine grid. */
  function scoreLandscape() {
    var xs = [], ys = [], peaks = [], i, n = 1300;
    for (i = 0; i <= n; i++) {
      var x = X_MIN + (X_MAX - X_MIN) * i / n;
      xs.push(x); ys.push(logLik(x, S.mu2));
    }
    for (i = 1; i < n; i++) if (ys[i] > ys[i - 1] && ys[i] >= ys[i + 1]) peaks.push({ x: xs[i], y: ys[i] });
    var lo = Math.min.apply(null, ys), hi = Math.max.apply(null, ys);
    return { lo: lo, hi: hi, peaks: peaks };
  }

  function resetRun() {
    stopPlay();
    S.anim = null;
    S.mu1 = S.mu1Start;
    S.q = uniformQ();
    S.next = 'E';
    S.last = null;
    S.prev = null;
    S.settled = false;
    var land = scoreLandscape();
    S.land = land;
    var span = land.hi - land.lo;
    S.yRange = [land.lo - 0.1 * span, land.hi + 0.1 * span];
    S.history = [];
    S.history.push(Object.assign({ kind: 'start' }, snapshot()));
    S.view = { mu1: S.mu1, q: S.q };
    render();
  }


  /* ----------------------------------------------------------- half steps -- */

  function ease(t) { return t < 0.5 ? 2 * t * t : 1 - Math.pow(-2 * t + 2, 2) / 2; }

  function finishAnim() {
    if (!S.anim) return;
    var a = S.anim;
    S.anim = null;
    a.done();
  }

  function animate(kind, from, to, done) {
    S.anim = { kind: kind, from: from, to: to, t0: performance.now(), done: done };
    if (reduceMotion) { finishAnim(); return; }
    var tick = function (now) {
      if (!S.anim || S.anim.kind !== kind || S.anim.from !== from) return;
      var t = Math.min(1, (now - S.anim.t0) / DUR), e = ease(t);
      if (kind === 'E') {
        // Every intermediate q is a genuine distribution, so the bound drawn
        // mid-animation is a genuine lower bound, not a cartoon.
        S.view = { mu1: S.mu1, q: from.map(function (r, n) {
          return [r[0] + (to[n][0] - r[0]) * e, r[1] + (to[n][1] - r[1]) * e];
        }) };
      } else {
        S.view = { mu1: from + (to - from) * e, q: S.q };
      }
      render(true);
      if (t < 1) requestAnimationFrame(tick); else finishAnim();
    };
    requestAnimationFrame(tick);
  }

  function eStep() {
    finishAnim();
    if (S.next !== 'E' || S.history.length > MAX_HALF_STEPS) return;
    S.prev = snapshot();
    var qOld = S.q, qNew = qFromPosterior(posterior(S.mu1, S.mu2));
    S.next = 'M';
    S.last = S.history.length === 1 ? 'E1' : 'E';
    animate('E', qOld, qNew, function () {
      S.q = qNew;
      S.view = { mu1: S.mu1, q: S.q };
      S.history.push(Object.assign({ kind: 'E' }, snapshot()));
      render();
    });
    render(true);
  }

  function mStepAction() {
    finishAnim();
    if (S.next !== 'M') return;
    S.prev = snapshot();
    var muOld = S.mu1, muNew = mStep(S.q, S.mu1);
    S.next = 'E';
    S.last = 'M';
    animate('M', muOld, muNew, function () {
      S.mu1 = muNew;
      S.view = { mu1: S.mu1, q: S.q };
      S.history.push(Object.assign({ kind: 'M' }, snapshot()));
      if (Math.abs(muNew - muOld) < SETTLE_TOL) S.settled = true;
      render();
    });
    render(true);
  }


  /* ==========================================================================
     PART 3 — PLOTS
     All three are inline SVG rebuilt from strings each frame. Plots 1 and 2
     share an x axis, so a centre in plot 1 sits directly above the same μ₁ in
     plot 2.
     ====================================================================== */

  var W = 760, ML = 66, MR = 18;

  function sx(x) { return ML + (x - X_MIN) / (X_MAX - X_MIN) * (W - ML - MR); }
  function invSx(px) { return X_MIN + (px - ML) / (W - ML - MR) * (X_MAX - X_MIN); }

  var COLORS = {};
  function readColors() {
    var cs = getComputedStyle(document.documentElement);
    ['--c1', '--c2', '--panel'].forEach(function (k) { COLORS[k] = cs.getPropertyValue(k).trim(); });
  }

  function hexToRgb(h) {
    h = h.replace('#', '');
    if (h.length === 3) h = h.replace(/(.)/g, '$1$1');
    var v = parseInt(h, 16);
    return [(v >> 16) & 255, (v >> 8) & 255, v & 255];
  }

  /* Colour for "probability p of cluster 1": blue at 1, red at 0, mixed between. */
  function mixColor(p1) {
    var a = hexToRgb(COLORS['--c1'] || '#2f6be0'), b = hexToRgb(COLORS['--c2'] || '#d6394e');
    var c = [0, 1, 2].map(function (i) { return Math.round(b[i] + (a[i] - b[i]) * p1); });
    return 'rgb(' + c.join(',') + ')';
  }

  function niceTicks(lo, hi, count) {
    var step = Math.pow(10, Math.floor(Math.log10((hi - lo) / count)));
    var err = (hi - lo) / count / step;
    if (err >= 7.5) step *= 10; else if (err >= 3.5) step *= 5; else if (err >= 1.5) step *= 2;
    var out = [];
    for (var v = Math.ceil(lo / step) * step; v <= hi + 1e-9; v += step) out.push(+v.toFixed(10));
    return out;
  }

  function pathFrom(pts) {
    return pts.map(function (p, i) { return (i ? 'L' : 'M') + p[0].toFixed(1) + ' ' + p[1].toFixed(1); }).join('');
  }

  function xAxis(y0, label) {
    var s = '<line class="ax" x1="' + ML + '" x2="' + (W - MR) + '" y1="' + y0 + '" y2="' + y0 + '"/>';
    for (var x = X_MIN; x <= X_MAX; x++) {
      s += '<line class="tick" x1="' + sx(x) + '" x2="' + sx(x) + '" y1="' + y0 + '" y2="' + (y0 + 5) + '"/>';
      s += '<text class="tick-l" x="' + sx(x) + '" y="' + (y0 + 20) + '" text-anchor="middle">' + sn(x, 0) + '</text>';
    }
    if (label) s += '<text class="ax-l" x="' + (W - MR) + '" y="' + (y0 + 40) + '" text-anchor="end">' + label + '</text>';
    return s;
  }


  /* ------------------------------------------------- plot 1: the clusters -- */

  var P1 = { H: 330, top: 58, base: 196, rowQ: 232, rowP: 268 };

  function densityY(d) {            // weighted density → svg y
    return P1.base - d / 0.3 * (P1.base - P1.top);
  }

  function renderClusters() {
    var v = S.view, mu1 = v.mu1, mu2 = S.mu2, s = '';
    var post = posterior(mu1, mu2);
    var n, i, steps = 260;

    // the two weighted components and their sum
    var c1 = [], c2 = [], mix = [];
    for (i = 0; i <= steps; i++) {
      var x = X_MIN + (X_MAX - X_MIN) * i / steps;
      var d1 = WEIGHTS[0] * Math.exp(logNormal(x, mu1, SIGMA));
      var d2 = WEIGHTS[1] * Math.exp(logNormal(x, mu2, SIGMA));
      c1.push([sx(x), densityY(d1)]); c2.push([sx(x), densityY(d2)]); mix.push([sx(x), densityY(d1 + d2)]);
    }
    var close = 'L' + sx(X_MAX) + ' ' + P1.base + 'L' + sx(X_MIN) + ' ' + P1.base + 'Z';
    s += '<path class="comp-fill c2-fill" d="' + pathFrom(c2) + close + '"/>';
    s += '<path class="comp-fill c1-fill" d="' + pathFrom(c1) + close + '"/>';
    s += '<clipPath id="clip1"><rect x="' + ML + '" y="' + (P1.top - 6) + '" width="' + (W - ML - MR) + '" height="' + (P1.base - P1.top + 30) + '"/></clipPath>';
    s += '<path class="mix" clip-path="url(#clip1)" d="' + pathFrom(mix) + '"/>';
    s += '<path class="comp c2-line" d="' + pathFrom(c2) + '"/>';
    s += '<path class="comp c1-line" d="' + pathFrom(c1) + '"/>';
    s += '<line class="ax" x1="' + ML + '" x2="' + (W - MR) + '" y1="' + P1.base + '" y2="' + P1.base + '"/>';

    // start ghost for μ₁
    if (Math.abs(mu1 - S.mu1Start) > 0.02) {
      s += '<line class="ghost" x1="' + sx(S.mu1Start) + '" x2="' + sx(S.mu1Start) + '" y1="' + (P1.base - 30) + '" y2="' + P1.base + '"/>';
      s += '<text class="ghost-l" x="' + sx(S.mu1Start) + '" y="' + (P1.base - 34) + '" text-anchor="middle">start</text>';
    }

    // centre handles at the top of each bump
    var peakY = densityY(WEIGHTS[0] * Math.exp(logNormal(0, 0, SIGMA)));
    var stack = Math.abs(sx(mu1) - sx(mu2)) < 190;
    [[mu2, 'c2', '2', 'fixed'], [mu1, 'c1', '1', S.history.length > 1 ? 'learning' : 'drag me']].forEach(function (h) {
      var lift = stack && h[2] === '2' ? 20 : 0;
      var hx = sx(h[0]);
      s += '<line class="ctr-line ' + h[1] + '-line" x1="' + hx + '" x2="' + hx + '" y1="' + peakY + '" y2="' + P1.base + '"/>';
      s += '<g class="handle" data-h="' + h[2] + '"><circle class="hit" cx="' + hx + '" cy="' + (peakY - 2) + '" r="24"/>';
      s += '<circle class="ctr ' + h[1] + '-ctr" cx="' + hx + '" cy="' + (peakY - 2) + '" r="14"/>';
      s += '<text class="ctr-t" x="' + hx + '" y="' + (peakY + 3) + '" text-anchor="middle">' + h[2] + '</text></g>';
      s += '<text class="ctr-sub ' + h[1] + '-t" x="' + hx + '" y="' + (peakY - 24 - lift) + '" text-anchor="middle">μ' +
           (h[2] === '1' ? '₁' : '₂') + ' = ' + sn(h[0], 2) + ' · ' + h[3] + '</text>';
    });

    // rows: q in use, and the posterior under the centres as drawn now
    s += '<text class="row-l" x="' + ML + '" y="' + (P1.rowQ + 5) + '">q (in use)</text>';
    s += '<text class="row-l" x="' + ML + '" y="' + (P1.rowP + 5) + '">posterior</text>';
    s += '<line class="row-rule" x1="' + ML + '" x2="' + (W - MR) + '" y1="' + P1.rowQ + '" y2="' + P1.rowQ + '"/>';
    s += '<line class="row-rule" x1="' + ML + '" x2="' + (W - MR) + '" y1="' + P1.rowP + '" y2="' + P1.rowP + '"/>';
    for (n = 0; n < DATA.length; n++) {
      var px = sx(DATA[n]);
      var stale = Math.abs(v.q[n][0] - post[n].p[0]) > 0.02;
      s += '<line class="drop" x1="' + px + '" x2="' + px + '" y1="' + P1.base + '" y2="' + P1.rowQ + '"/>';
      s += '<circle class="pt" cx="' + px + '" cy="' + P1.base + '" r="4.5"/>';
      s += '<circle class="dot" cx="' + px + '" cy="' + P1.rowQ + '" r="8" fill="' + mixColor(v.q[n][0]) + '"/>';
      s += '<circle class="dot' + (stale ? ' dot-stale' : '') + '" cx="' + px + '" cy="' + P1.rowP + '" r="8" fill="' + mixColor(post[n].p[0]) + '"/>';
    }
    s += xAxis(P1.H - 30, '');

    $('plotClusters').innerHTML = s;
  }


  /* ---------------------------------------- plot 2: true score and bound -- */

  var P2 = { H: 340, top: 18, bot: 290 };

  function sy2(v) {
    var r = S.yRange;
    return P2.bot - (v - r[0]) / (r[1] - r[0]) * (P2.bot - P2.top);
  }

  function renderScore() {
    var v = S.view, mu2 = S.mu2, s = '', i, steps = 320;
    var r = S.yRange;

    s += '<defs><clipPath id="clip2"><rect x="' + ML + '" y="' + P2.top + '" width="' + (W - ML - MR) + '" height="' + (P2.bot - P2.top) + '"/></clipPath></defs>';

    niceTicks(r[0], r[1], 5).forEach(function (t) {
      s += '<line class="grid" x1="' + ML + '" x2="' + (W - MR) + '" y1="' + sy2(t) + '" y2="' + sy2(t) + '"/>';
      s += '<text class="tick-l" x="' + (ML - 8) + '" y="' + (sy2(t) + 4) + '" text-anchor="end">' + sn(t, 0) + '</text>';
    });

    var ll = [], lb = [];
    for (i = 0; i <= steps; i++) {
      var x = X_MIN + (X_MAX - X_MIN) * i / steps;
      ll.push([sx(x), sy2(logLik(x, mu2))]);
      lb.push([sx(x), sy2(elbo(x, mu2, v.q))]);
    }
    // the gap region: between the two curves at every μ₁, it is Σ KL
    var band = pathFrom(ll) + lb.slice().reverse().map(function (p) { return 'L' + p[0].toFixed(1) + ' ' + p[1].toFixed(1); }).join('') + 'Z';
    s += '<g clip-path="url(#clip2)"><path class="gap-band" d="' + band + '"/>';
    s += '<path class="bound" d="' + pathFrom(lb) + '"/>';
    s += '<path class="score" d="' + pathFrom(ll) + '"/></g>';

    // the trail of earlier μ₁ values
    var seen = {};
    S.history.forEach(function (h) {
      var key = h.mu1.toFixed(3);
      if (seen[key]) return; seen[key] = 1;
      s += '<circle class="trail" cx="' + sx(h.mu1) + '" cy="' + sy2(h.ll) + '" r="3.5"/>';
    });

    // μ₂, fixed
    s += '<line class="mu2-tick" x1="' + sx(mu2) + '" x2="' + sx(mu2) + '" y1="' + (P2.bot - 8) + '" y2="' + P2.bot + '"/>';

    // where the bound peaks: the M step's destination
    var peakMu = mStep(v.q, v.mu1), peakVal = elbo(peakMu, mu2, v.q);
    var showPeak = S.next === 'M' || (S.anim && S.anim.kind === 'M');
    if (S.anim && S.anim.kind === 'M') { peakMu = S.anim.to; peakVal = elbo(peakMu, mu2, v.q); }
    if (showPeak && peakVal > r[0]) {
      var px = sx(peakMu), py = sy2(peakVal);
      s += '<path class="peak-mark" d="M' + px + ' ' + (py - 4) + 'l-7 -11h14z"/>';
      s += '<text class="peak-l" x="' + (px - 10) + '" y="' + (py - 10) + '" text-anchor="end">bound’s peak</text>';
    }

    // the current μ₁, its two values, and the gap between them
    var cx = sx(v.mu1), lv = logLik(v.mu1, mu2), bv = elbo(v.mu1, mu2, v.q), gap = lv - bv;
    s += '<line class="cur" x1="' + cx + '" x2="' + cx + '" y1="' + P2.top + '" y2="' + P2.bot + '"/>';
    var yl = sy2(lv), yb = Math.min(P2.bot, sy2(bv));
    if (gap > 5e-4) {
      s += '<line class="gap-bracket" x1="' + (cx + 9) + '" x2="' + (cx + 9) + '" y1="' + yl + '" y2="' + yb + '"/>';
      s += '<line class="gap-bracket" x1="' + (cx + 5) + '" x2="' + (cx + 13) + '" y1="' + yl + '" y2="' + yl + '"/>';
      s += '<line class="gap-bracket" x1="' + (cx + 5) + '" x2="' + (cx + 13) + '" y1="' + yb + '" y2="' + yb + '"/>';
      var right = cx < W - 190;
      s += '<text class="gap-l" x="' + (right ? cx + 18 : cx - 10) + '" y="' + ((yl + yb) / 2 + 5) + '" text-anchor="' + (right ? 'start' : 'end') + '">KL gap ' + fmt(gap) + '</text>';
    } else {
      s += '<text class="touch-l" x="' + (cx + 14) + '" y="' + (yl + 22) + '">touching · gap 0</text>';
    }
    if (bv >= r[0]) s += '<circle class="bound-dot" cx="' + cx + '" cy="' + sy2(bv) + '" r="6"/>';
    s += '<circle class="score-dot" cx="' + cx + '" cy="' + yl + '" r="6"/>';

    // a settled run that missed a higher peak gets the higher peak pointed out
    var best = S.land.peaks.reduce(function (a, b) { return b.y > a.y ? b : a; }, { y: -Infinity });
    if (S.settled && best.y - lv > 0.05) {
      s += '<circle class="best" cx="' + sx(best.x) + '" cy="' + sy2(best.y) + '" r="7"/>';
      s += '<text class="best-l" x="' + sx(best.x) + '" y="' + (sy2(best.y) - 14) + '" text-anchor="middle">higher peak EM never visited</text>';
    }

    s += xAxis(P2.bot, 'learned centre μ₁');
    s += '<text class="ax-l" x="12" y="' + (P2.top + 4) + '" transform="rotate(-90 12 ' + (P2.top + 4) + ')" text-anchor="end">score (nats)</text>';

    $('plotScore').innerHTML = s;
  }


  /* -------------------------------------------- plot 3: progress and gap -- */

  var P3 = { W: 760, H: 380, ml: 66, mr: 18, top: 18, bot: 196, gTop: 256, gBot: 336 };

  function renderProgress() {
    var h = S.history, s = '', i;
    var n = Math.max(h.length - 1, 8);
    var px = function (i) { return P3.ml + i / n * (P3.W - P3.ml - P3.mr); };

    var vals = [];
    h.forEach(function (e) { vals.push(e.ll, e.elbo); });
    var lo = Math.min.apply(null, vals), hi = Math.max.apply(null, vals);
    if (hi - lo < 1) { lo -= 0.5; hi += 0.5; }
    var pad = 0.08 * (hi - lo); lo -= pad; hi += pad;
    var py = function (v) { return P3.bot - (v - lo) / (hi - lo) * (P3.bot - P3.top); };

    niceTicks(lo, hi, 4).forEach(function (t) {
      s += '<line class="grid" x1="' + P3.ml + '" x2="' + (P3.W - P3.mr) + '" y1="' + py(t) + '" y2="' + py(t) + '"/>';
      s += '<text class="tick-l" x="' + (P3.ml - 8) + '" y="' + (py(t) + 4) + '" text-anchor="end">' + sn(t, 0) + '</text>';
    });

    // the gap between the two lines at each recorded half-step
    var top = h.map(function (e, i) { return [px(i), py(e.ll)]; });
    var bot = h.map(function (e, i) { return [px(i), py(e.elbo)]; });
    if (h.length > 1) {
      s += '<path class="gap-band" d="' + pathFrom(top) + bot.slice().reverse().map(function (p) { return 'L' + p[0].toFixed(1) + ' ' + p[1].toFixed(1); }).join('') + 'Z"/>';
    }
    s += '<path class="bound" d="' + pathFrom(bot) + '"/>';
    s += '<path class="score" d="' + pathFrom(top) + '"/>';
    h.forEach(function (e, i) {
      s += '<circle class="bound-dot sm" cx="' + px(i) + '" cy="' + py(e.elbo) + '" r="4"/>';
      s += '<circle class="score-dot sm" cx="' + px(i) + '" cy="' + py(e.ll) + '" r="4"/>';
    });
    s += '<text class="ax-l" x="12" y="' + (P3.top + 4) + '" transform="rotate(-90 12 ' + (P3.top + 4) + ')" text-anchor="end">score (nats)</text>';

    // the gap itself, on its own scale
    var gmax = Math.max.apply(null, h.map(function (e) { return e.gap; }));
    gmax = Math.max(gmax, 1e-9);
    var bw = Math.min(26, (P3.W - P3.ml - P3.mr) / (n + 1) * 0.6);
    s += '<text class="sub-t" x="' + P3.ml + '" y="' + (P3.gTop - 26) + '">KL gap after each half-step · own scale, tallest = ' + fmt(gmax) + '</text>';
    s += '<line class="ax" x1="' + P3.ml + '" x2="' + (P3.W - P3.mr) + '" y1="' + P3.gBot + '" y2="' + P3.gBot + '"/>';
    h.forEach(function (e, i) {
      var hgt = e.gap / gmax * (P3.gBot - P3.gTop);
      var x = px(i);
      if (e.gap > 5e-4) s += '<rect class="gap-bar" x="' + (x - bw / 2) + '" y="' + (P3.gBot - hgt) + '" width="' + bw + '" height="' + Math.max(hgt, 1.5) + '"/>';
      else s += '<text class="zero-l" x="' + x + '" y="' + (P3.gBot - 6) + '" text-anchor="middle">0</text>';
      if (e.gap > 5e-4 && n <= 16) s += '<text class="bar-v" x="' + x + '" y="' + (P3.gBot - hgt - 5) + '" text-anchor="middle">' + fmt(e.gap, e.gap < 0.1 ? 3 : 2) + '</text>';
      var lab = e.kind === 'start' ? 'start' : e.kind;
      if (n <= 24 || e.kind === 'start' || i === h.length - 1) {
        s += '<text class="step-l step-' + e.kind + '" x="' + x + '" y="' + (P3.gBot + 20) + '" text-anchor="middle">' + lab + '</text>';
      }
    });
    s += '<text class="ax-l" x="' + (P3.W - P3.mr) + '" y="' + (P3.gBot + 40) + '" text-anchor="end">half-steps →</text>';

    $('plotProgress').innerHTML = s;
  }


  /* ==========================================================================
     PART 4 — NARRATION
     Every sentence that quotes a number quotes the number just computed.
     ====================================================================== */

  function b(cls, txt) { return '<b class="' + cls + '">' + txt + '</b>'; }

  function renderNotes() {
    // Mid-animation the half-step isn't committed yet; keep the old notes up.
    if (S.anim) {
      $('phaseHead').textContent = S.anim.kind === 'E' ? 'E step in progress…' : 'M step in progress…';
      return;
    }
    var cur = snapshot(), p = S.prev, last = S.last;
    var n1, n2, n3, head;

    if (!last) {
      head = 'Start: nothing has happened yet';
      n1 = 'The top row shows <b>q</b>, our current belief about which cluster made each point. ' +
           'We haven’t looked at the data yet, so every point is <b>50/50</b> (purple). The bottom row is ' +
           'the model’s actual posterior under these centres. Drag either centre to choose where EM starts.';
      n2 = 'The ' + b('t-bound', 'green bound') + ' is built from the 50/50 beliefs. It sits under the ' +
           b('t-score', 'true score') + ' everywhere, as every bound must, but it doesn’t touch it. ' +
           'The amber band between them is the KL gap: ' + b('t-gap', fmt(cur.gap)) + '.';
      n3 = 'One point so far: the starting score ' + b('t-score', sn(cur.ll)) + ' and bound ' + b('t-bound', sn(cur.elbo)) +
           '. Press <b>E step</b> to begin.';
    } else if (last === 'E1' || last === 'E') {
      head = last === 'E1' ? 'E step: update our beliefs about membership' : 'E step: rebuild the bound at the new model';
      n1 = (last === 'E1'
            ? '<b>Update our beliefs about membership.</b> Each point’s colour is now its <b>responsibility</b>: '
            : '<b>Rebuild the bound at the new model.</b> The beliefs catch up with the centre that just moved. Each colour is again the <b>responsibility</b>: ') +
           'the probability, under the current centres, that cluster 1 (blue) or cluster 2 (red) made that point. ' +
           'The two rows now agree. ' + b('t-c1', 'Neither centre moved.');
      n2 = 'Same μ₁, same true score: ' + b('t-score', sn(p.ll) + ' → ' + sn(cur.ll)) + '. ' +
           'Only the ' + b('t-bound', 'bound') + ' changed. Rebuilt from the new q, it rose from ' + b('t-bound', sn(p.elbo)) +
           ' to ' + b('t-bound', sn(cur.elbo)) + ' and now <b>touches</b> the true score at the current μ₁. ' +
           'The triangle marks the bound’s peak, which is where the M step will go.';
      n3 = 'The KL gap went from ' + b('t-gap', fmt(p.gap)) + ' to ' + b('t-gap', '0') +
           ': q is now exactly the posterior, and a distribution has zero KL divergence from itself. The true score stayed flat.';
    } else {
      var dMu = cur.mu1 - p.mu1;
      head = 'M step: improve the model with those beliefs fixed';
      n1 = '<b>Improve the model with those beliefs fixed.</b> Centre 1 moved ' + b('t-c1', sn(p.mu1, 3) + ' → ' + sn(cur.mu1, 3)) +
           ': the average of all the points, each weighted by how much it belongs to cluster 1. ' +
           'The top row (q) did not change, so it is now a step stale. ' +
           (cur.gap > 5e-4 ? 'The bottom row shows how the posterior shifted when the centre moved; the ringed points changed most.'
                           : 'The centre barely moved, so the posterior barely shifted.');
      n2 = 'The ' + b('t-bound', 'bound') + ' stayed frozen and μ₁ slid to its peak, so the bound rose ' +
           b('t-bound', sn(p.elbo) + ' → ' + sn(cur.elbo)) + '. The true score sits on or above the bound, so it rose at least that much: ' +
           b('t-score', sn(p.ll) + ' → ' + sn(cur.ll)) + '. ' +
           (cur.gap > 5e-4 ? 'At the new μ₁ the two no longer touch.'
                           : 'μ₁ hardly moved, so they still (almost) touch.');
      n3 = 'The KL gap reopened to ' + b('t-gap', fmt(cur.gap)) + '. That is not a mistake: q was built for the old μ₁, ' +
           'and the posterior moved. How big the gap gets depends on how far the posterior moved, so it can be ' +
           '<b>larger</b> than an earlier gap. The next E step closes it.';
      if (S.settled) {
        var best = S.land.peaks.reduce(function (a, c) { return c.y > a.y ? c : a; }, { y: -Infinity });
        var missed = best.y - cur.ll > 0.05;
        head = 'Settled';
        n3 = 'The last M step moved μ₁ by only ' + fmt(Math.abs(dMu), 5) + ', so EM has <b>settled</b> at a point where ' +
             'the bound’s peak is where it already stands, a flat spot of the true score. ' +
             (missed ? 'It is <b>not</b> the best one: the true score is higher, ' + b('t-score', sn(best.y)) +
                       ', near μ₁ = ' + sn(best.x, 2) + '. EM only promises the score never goes down, not that it finds the top.'
                     : 'Here it happens to be the highest point of the curve, but EM only promised that the score never goes down.');
      }
    }

    $('note1').innerHTML = n1;
    $('note2').innerHTML = n2;
    $('note3').innerHTML = n3;
    $('phaseHead').textContent = head;

    // phase chips
    var on = last === 'E1' ? 'e1' : last === 'M' ? 'm' : last === 'E' ? 'e2' : '';
    var nextChip = S.next === 'M' ? 'm' : (S.history.length <= 1 ? 'e1' : 'e2');
    ['e1', 'm', 'e2'].forEach(function (k) {
      var c = $('chip-' + k);
      c.classList.toggle('is-on', k === on);
      c.classList.toggle('is-next', k === nextChip && k !== on);
    });

    // the identity, with live numbers
    $('idLL').textContent = sn(cur.ll);
    $('idLB').textContent = sn(cur.elbo);
    $('idGap').textContent = fmt(cur.gap);
    $('idKL').textContent = fmt(cur.klSum);
  }

  function renderControls() {
    var busy = S.playing;
    $('btnE').disabled = busy || S.next !== 'E' || S.history.length > MAX_HALF_STEPS;
    $('btnM').disabled = busy || S.next !== 'M';
    $('btnE').classList.toggle('is-next', S.next === 'E');
    $('btnM').classList.toggle('is-next', S.next === 'M');
    $('btnPlay').textContent = S.playing ? 'Pause' : 'Auto-play';
    $('btnPlay').setAttribute('aria-pressed', S.playing ? 'true' : 'false');
    var steps = S.history.length - 1;
    $('stepCount').textContent = steps === 0 ? 'No steps yet'
      : steps + ' half-step' + (steps === 1 ? '' : 's') + ' · ' + Math.floor(steps / 2) + ' full iteration' + (Math.floor(steps / 2) === 1 ? '' : 's');
    $('nextUp').textContent = S.settled ? 'Settled. Further steps change almost nothing.' : 'Next: ' + (S.next === 'E' ? 'E step' : 'M step');
    $('sMu1').value = S.mu1Start; $('oMu1').textContent = sn(S.mu1Start, 2);
    $('sMu2').value = S.mu2; $('oMu2').textContent = sn(S.mu2, 2);
    Array.prototype.forEach.call(document.querySelectorAll('[data-mu1]'), function (btn) {
      var hit = Math.abs(+btn.getAttribute('data-mu1') - S.mu1Start) < 1e-9 && Math.abs(+btn.getAttribute('data-mu2') - S.mu2) < 1e-9;
      btn.classList.toggle('is-on', hit);
      btn.setAttribute('aria-pressed', hit ? 'true' : 'false');
    });
  }

  function render(frameOnly) {
    renderClusters();
    renderScore();
    if (!frameOnly || !S.anim) renderProgress();
    renderNotes();
    renderControls();
  }


  /* ==========================================================================
     PART 5 — CONTROLS
     ====================================================================== */

  var playTimer = null;

  function stopPlay() {
    S.playing = false;
    if (playTimer) { clearTimeout(playTimer); playTimer = null; }
  }

  function playStep() {
    if (!S.playing) return;
    if (S.settled || S.history.length > MAX_HALF_STEPS) { stopPlay(); render(); return; }
    if (S.next === 'E') eStep(); else mStepAction();
    playTimer = setTimeout(playStep, (reduceMotion ? 0 : DUR) + PAUSE);
  }

  function togglePlay() {
    if (S.playing) { stopPlay(); render(); return; }
    if (S.settled) resetRun();
    S.playing = true;
    renderControls();
    playStep();
  }

  function setStart(mu1, mu2) {
    S.mu1Start = Math.max(SLIDER_MIN, Math.min(SLIDER_MAX, Math.round(mu1 * 20) / 20));
    S.mu2 = Math.max(SLIDER_MIN, Math.min(SLIDER_MAX, Math.round(mu2 * 20) / 20));
    resetRun();
  }

  /* Dragging a centre in plot 1. Moving either one restarts the run, because
     a run is defined by where it starts. */
  function wireDrag() {
    var svg = $('plotClusters'), dragging = null;

    function toX(evt) {
      var pt = svg.createSVGPoint();
      pt.x = evt.clientX; pt.y = evt.clientY;
      return invSx(pt.matrixTransform(svg.getScreenCTM().inverse()).x);
    }

    svg.addEventListener('pointerdown', function (evt) {
      var g = evt.target.closest && evt.target.closest('.handle');
      if (!g) return;
      dragging = g.getAttribute('data-h');
      svg.setPointerCapture(evt.pointerId);
      svg.classList.add('is-dragging');
      evt.preventDefault();
    });
    svg.addEventListener('pointermove', function (evt) {
      if (!dragging) return;
      var x = toX(evt);
      if (dragging === '1') setStart(x, S.mu2); else setStart(S.mu1Start, x);
    });
    function end() { dragging = null; svg.classList.remove('is-dragging'); }
    svg.addEventListener('pointerup', end);
    svg.addEventListener('pointercancel', end);
  }

  function boot() {
    readColors();
    if (window.matchMedia) {
      var mq = window.matchMedia('(prefers-color-scheme: dark)');
      var onTheme = function () { readColors(); render(); };
      if (mq.addEventListener) mq.addEventListener('change', onTheme); else if (mq.addListener) mq.addListener(onTheme);
    }

    $('btnE').addEventListener('click', eStep);
    $('btnM').addEventListener('click', mStepAction);
    $('btnReset').addEventListener('click', resetRun);
    $('btnPlay').addEventListener('click', togglePlay);

    $('sMu1').addEventListener('input', function () { setStart(+$('sMu1').value, S.mu2); });
    $('sMu2').addEventListener('input', function () { setStart(S.mu1Start, +$('sMu2').value); });

    Array.prototype.forEach.call(document.querySelectorAll('[data-mu1]'), function (btn) {
      btn.addEventListener('click', function () {
        setStart(+btn.getAttribute('data-mu1'), +btn.getAttribute('data-mu2'));
        document.querySelector('.bench').scrollIntoView({ behavior: reduceMotion ? 'auto' : 'smooth', block: 'start' });
      });
    });

    wireDrag();
    resetRun();
  }

  boot();

})(typeof globalThis !== 'undefined' ? globalThis : this);
