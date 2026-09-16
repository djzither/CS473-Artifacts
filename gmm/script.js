/* ============================================================================
   Who Made This Point? — an interactive 2-component Gaussian mixture model.
   CS 473 module 2 · Murphy, PML §3.5 (Eq. 3.94–3.101)

   Layout of this file
     PART 1  Math          pure, DOM-free, exported for the Node test suite
     PART 2  State         parameters, dataset, presets
     PART 3  Canvas        sizing, world<->screen, palette
     PART 4  Scalar field  responsibility + density grid, marching squares
     PART 5  Plot          the 2D instrument
     PART 6  1D chart      adding densities, not samples
     PART 7  Inspector     per-point readout and the plain-language verdict
     PART 8  Controls      sliders, buttons, walkthrough, tooltips
     PART 9  Interaction   drag, hover, click, keyboard

   PART 1 runs under Node (`require('./script.js')` returns the math only); the
   rest is skipped when there is no document. See test-math.js.
   ========================================================================= */

(function (global) {
  'use strict';

  /* ==========================================================================
     PART 1 — MATH
     Everything here is a pure function of its arguments. No DOM, no state.
     ====================================================================== */

  var TAU = Math.PI * 2;
  var LOG_TAU = Math.log(TAU);

  /* A covariance matrix built from two spreads and a rotation:
        Σ = R(θ) · diag(σx², σy²) · R(θ)ᵀ
     Constructing Σ this way rather than letting the user type entries means it
     is positive definite by construction for any σx, σy > 0 and any θ — the
     model can never be handed an invalid covariance. Its determinant is
     exactly σx²σy² (the rotation contributes nothing), which the test suite
     checks. Stored as {a, b, d} for [[a, b], [b, d]]. */
  function covFrom(sx, sy, theta) {
    var c = Math.cos(theta), s = Math.sin(theta);
    var vx = sx * sx, vy = sy * sy;
    return {
      a: vx * c * c + vy * s * s,
      b: (vx - vy) * c * s,
      d: vx * s * s + vy * c * c
    };
  }

  function covDet(S) { return S.a * S.d - S.b * S.b; }

  /* Squared Mahalanobis distance (y−μ)ᵀ Σ⁻¹ (y−μ).
     For a 2×2, Σ⁻¹ = (1/det)·[[d, −b], [−b, a]]. */
  function maha2(S, dx, dy) {
    return (S.d * dx * dx - 2 * S.b * dx * dy + S.a * dy * dy) / covDet(S);
  }

  /* log N(y | μ, Σ) in 2D. Kept in logs throughout: densities under a tight
     covariance overflow a float fast, and responsibilities only ever need
     differences of logs. */
  function logGauss2(x, y, c) {
    return -0.5 * maha2(c.S, x - c.mx, y - c.my)
           - LOG_TAU
           - 0.5 * Math.log(covDet(c.S));
  }

  function logGauss1(x, mu, sd) {
    var z = (x - mu) / sd;
    return -0.5 * z * z - 0.5 * LOG_TAU - Math.log(sd);
  }

  function gauss1(x, mu, sd) { return Math.exp(logGauss1(x, mu, sd)); }

  /* Overflow-safe logistic. Responsibilities in a 2-component mixture are
     exactly a logistic of the log-odds, which is why the whole plot can be
     driven by one scalar field (see PART 4). */
  function sigmoid(z) {
    if (z >= 0) return 1 / (1 + Math.exp(-z));
    var e = Math.exp(z);
    return e / (1 + e);
  }

  /* log-odds of component 1:  log π₁p₁(y) − log π₂p₂(y). */
  function logOdds(x, y, model) {
    return (Math.log(model.pi[0]) + logGauss2(x, y, model.comps[0]))
         - (Math.log(model.pi[1]) + logGauss2(x, y, model.comps[1]));
  }

  /* r₁ = π₁p₁ / (π₁p₁ + π₂p₂), r₂ = 1 − r₁.
     Computing r₁ as sigmoid(log-odds) makes "sums to 1" hold identically
     rather than approximately — there is no separate normaliser to drift. */
  function responsibilities(x, y, model) {
    var r1 = sigmoid(logOdds(x, y, model));
    return [r1, 1 - r1];
  }

  /* log p(y) = log Σ πₖ pₖ(y), via log-sum-exp. */
  function logMixture(x, y, model) {
    var l1 = Math.log(model.pi[0]) + logGauss2(x, y, model.comps[0]);
    var l2 = Math.log(model.pi[1]) + logGauss2(x, y, model.comps[1]);
    var m = Math.max(l1, l2);
    if (!isFinite(m)) return -Infinity;
    return m + Math.log(Math.exp(l1 - m) + Math.exp(l2 - m));
  }

  /* Eigendecomposition of a 2×2 symmetric matrix.
     The drawn ellipse is derived from Σ through THIS function rather than from
     the sliders, so the picture cannot silently disagree with the matrix: the
     contour at Mahalanobis distance t has semi-axes t√λ₁ and t√λ₂ along the
     eigenvectors. The test suite walks the drawn ellipse and confirms every
     point on it really is at Mahalanobis distance t. */
  function eig2(S) {
    var tr = S.a + S.d, det = covDet(S);
    var disc = Math.sqrt(Math.max(0, tr * tr / 4 - det));
    var l1 = tr / 2 + disc, l2 = tr / 2 - disc;
    var ang;
    if (Math.abs(S.b) > 1e-12) ang = Math.atan2(l1 - S.a, S.b);
    else ang = (S.a >= S.d) ? 0 : Math.PI / 2;
    return { l1: Math.max(l1, 1e-12), l2: Math.max(l2, 1e-12), angle: ang };
  }

  /* --- the GMM → K-means blend -------------------------------------------
     t = 0 gives the model exactly as configured. As t → 1 three things happen
     together, matching the three assumptions K-means makes:

       1. weights flatten toward 0.5 / 0.5      (no prior)
       2. Σ rotates out its off-diagonal and    (no shape, no correlation)
          both matrices converge on the same
          spherical v̄·I
       3. that shared matrix shrinks            (responsibilities harden to 0/1)

     At t = 1, log πₖ + log N(y|μₖ, cI) = const − ‖y−μₖ‖²/(2c) for both k, so
     argmax_k rₖ = argmin_k ‖y−μₖ‖² — Murphy Eq. 3.101, the nearest-centroid
     rule. Step 3 is what turns soft assignment into hard assignment; it does
     not move the boundary, because equal spherical covariances already put the
     boundary on the perpendicular bisector.

     Every intermediate Σ is a convex combination of two positive definite
     matrices scaled by a positive number, so it stays positive definite. */
  function modelAt(p, t) {
    var c0 = p.comps[0], c1 = p.comps[1];
    var S0 = covFrom(c0.sx, c0.sy, c0.rot);
    var S1 = covFrom(c1.sx, c1.sy, c1.rot);

    if (t <= 0) {
      return {
        pi: [p.pi1, 1 - p.pi1],
        comps: [{ mx: c0.mx, my: c0.my, S: S0 }, { mx: c1.mx, my: c1.my, S: S1 }]
      };
    }

    var vbar = (c0.sx * c0.sx + c0.sy * c0.sy + c1.sx * c1.sx + c1.sy * c1.sy) / 4;
    // Hold the ellipses readable through the first half of the slider, then
    // collapse hard. A gentler decay (1/(1+60t²)) leaves the shared variance
    // large enough that two centres sitting close together still produce soft
    // responsibilities at t = 1 — which would contradict the copy promising a
    // hard nearest-centre assignment. t⁶ keeps the middle of the slider legible
    // and still reaches ~1e-6 at the end. Verified in test-math.js §7.
    var shrink = Math.exp(-14 * Math.pow(t, 6));
    var morph = function (S) {
      return {
        a: ((1 - t) * S.a + t * vbar) * shrink,
        b: ((1 - t) * S.b) * shrink,
        d: ((1 - t) * S.d + t * vbar) * shrink
      };
    };
    var p1 = p.pi1 + (0.5 - p.pi1) * t;
    return {
      pi: [p1, 1 - p1],
      comps: [{ mx: c0.mx, my: c0.my, S: morph(S0) },
              { mx: c1.mx, my: c1.my, S: morph(S1) }]
    };
  }

  /* Draw y = μ + R·diag(σx, σy)·z with z ~ N(0, I).
     The covariance of the result is R·diag(σx², σy²)·Rᵀ, i.e. exactly covFrom.
     Because z is held fixed per point (PART 2), moving a slider deforms the
     existing cloud instead of resampling it. */
  function samplePoint(c, z1, z2) {
    var co = Math.cos(c.rot), si = Math.sin(c.rot);
    var u = c.sx * z1, v = c.sy * z2;
    return { x: c.mx + co * u - si * v, y: c.my + si * u + co * v };
  }

  /* Reproducible PRNG so "generate new data" is a seed change, not entropy. */
  function mulberry32(seed) {
    var a = seed >>> 0;
    return function () {
      a = (a + 0x6D2B79F5) >>> 0;
      var t = a;
      t = Math.imul(t ^ (t >>> 15), t | 1);
      t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }

  function gaussPair(rnd) {
    var u = Math.max(rnd(), 1e-12), v = rnd();
    var r = Math.sqrt(-2 * Math.log(u));
    return [r * Math.cos(TAU * v), r * Math.sin(TAU * v)];
  }

  var MATH = {
    covFrom: covFrom, covDet: covDet, maha2: maha2, eig2: eig2,
    logGauss2: logGauss2, logGauss1: logGauss1, gauss1: gauss1,
    sigmoid: sigmoid, logOdds: logOdds, responsibilities: responsibilities,
    logMixture: logMixture, modelAt: modelAt, samplePoint: samplePoint,
    mulberry32: mulberry32, gaussPair: gaussPair
  };

  global.GMMMath = MATH;
  if (typeof module !== 'undefined' && module.exports) module.exports = MATH;

  /* Node stops here — everything below needs a browser. */
  if (typeof document === 'undefined') return;


  /* ==========================================================================
     PART 2 — STATE
     ====================================================================== */

  var MAXN = 600;
  var XSPAN = 12.4;             // world units across the plot
  var $ = function (id) { return document.getElementById(id); };

  var DEFAULTS = {
    comps: [
      { mx: -2.15, my: -0.85, sx: 1.35, sy: 0.65, rot: 25 * Math.PI / 180 },
      { mx: 2.05, my: 0.95, sx: 0.90, sy: 1.20, rot: -15 * Math.PI / 180 }
    ],
    pi1: 0.5,
    n: 260,
    km: 0,
    mu1d: [-1.6, 1.4]
  };

  var S = clone(DEFAULTS);
  var seed = 7;
  var draws = [];               // fixed z-pairs + uniform, one per potential point
  var pts = [];                 // current positions, recomputed each render
  var selected = -1;            // index into pts, or -1
  var demo = null;              // the "watch one point get made" point
  var activeComp = 0;           // which mean the arrow keys move
  var dragging = -1;

  var SD1D = [1.0, 1.45];       // fixed spreads for the 1D panel

  function clone(o) { return JSON.parse(JSON.stringify(o)); }

  /* Fixed standard-normal draws + a uniform per point. The uniform is compared
     against π₁ to decide which component made the point, so moving the π₁
     slider genuinely re-rolls the hidden coin — points migrate between clouds,
     which is what a prior over z actually means. */
  function reseed() {
    var rnd = mulberry32(seed);
    draws = [];
    for (var i = 0; i < MAXN; i++) {
      var g = gaussPair(rnd);
      draws.push({ z1: g[0], z2: g[1], u: rnd() });
    }
  }

  function rebuildPoints() {
    pts = [];
    for (var i = 0; i < S.n; i++) {
      var d = draws[i];
      var k = d.u < S.pi1 ? 0 : 1;
      var p = samplePoint(S.comps[k], d.z1, d.z2);
      p.k = k;
      pts.push(p);
    }
  }


  /* ==========================================================================
     PART 3 — CANVAS
     ====================================================================== */

  var plot = $('plot'), oned = $('oned');
  var pctx = plot.getContext('2d'), octx = oned.getContext('2d');
  var PW = 0, PH = 0, OW = 0, OH = 0, scale = 1;
  var hm = document.createElement('canvas');   // offscreen heatmap
  var hctx = hm.getContext('2d');

  var pal = {};
  function readPalette() {
    var cs = getComputedStyle(document.documentElement);
    var get = function (n) { return cs.getPropertyValue(n).trim(); };
    pal = {
      blue: hex(get("--blue")), red: hex(get("--red")), violet: hex(get("--violet")),
      mutedRGB: hex(get("--muted")),
      ink: get('--ink'), ink2: get('--ink-2'), muted: get('--muted'),
      grid: get('--grid'), axis: get('--axis'), line: get('--line'),
      panel: get('--panel'), sunken: get('--sunken')
    };
  }

  function hex(h) {
    h = h.replace('#', '');
    if (h.length === 3) h = h[0] + h[0] + h[1] + h[1] + h[2] + h[2];
    return [parseInt(h.slice(0, 2), 16), parseInt(h.slice(2, 4), 16), parseInt(h.slice(4, 6), 16)];
  }

  function rgba(c, a) { return 'rgba(' + c[0] + ',' + c[1] + ',' + c[2] + ',' + a + ')'; }

  /* Responsibility → colour. Three stops: red at r=0, a deliberately
     low-chroma violet at r=0.5, blue at r=1. Because the midpoint is the least
     saturated colour on the ramp, uncertainty reads as "washed out" as well as
     "purple" — certainty is carried by chroma and hue at once. */
  function mixColor(r) {
    var a, b, t;
    if (r < 0.5) { a = pal.red; b = pal.violet; t = r * 2; }
    else { a = pal.violet; b = pal.blue; t = (r - 0.5) * 2; }
    return [
      Math.round(a[0] + (b[0] - a[0]) * t),
      Math.round(a[1] + (b[1] - a[1]) * t),
      Math.round(a[2] + (b[2] - a[2]) * t)
    ];
  }

  function fit() {
    var dpr = Math.min(window.devicePixelRatio || 1, 2);

    PW = plot.clientWidth || plot.parentElement.clientWidth;
    PH = Math.round(Math.min(560, Math.max(300, PW * 0.68)));
    plot.style.height = PH + 'px';
    plot.width = Math.round(PW * dpr); plot.height = Math.round(PH * dpr);
    pctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    scale = PW / XSPAN;

    OW = oned.clientWidth || oned.parentElement.clientWidth;
    OH = Math.round(Math.min(230, Math.max(150, OW * 0.42)));
    oned.style.height = OH + 'px';
    oned.width = Math.round(OW * dpr); oned.height = Math.round(OH * dpr);
    octx.setTransform(dpr, 0, 0, dpr, 0, 0);
  }

  // world <-> screen
  function sx(x) { return PW / 2 + x * scale; }
  function sy(y) { return PH / 2 - y * scale; }
  function wx(px) { return (px - PW / 2) / scale; }
  function wy(py) { return (PH / 2 - py) / scale; }
  function yHalf() { return PH / (2 * scale); }


  /* ==========================================================================
     PART 4 — SCALAR FIELD + CONTOUR
     ====================================================================== */

  var GRID_PX = 5;
  var field = { nx: 0, ny: 0, odds: null, dens: null, maxDens: 0 };

  /* One pass fills both layers the plot needs:
       odds[i] = log π₁p₁ − log π₂p₂   → r₁ = sigmoid(odds), boundary = {odds = 0}
       dens[i] = log p(y)              → the mixture-density background
     Sampling the log-odds once and reusing it for the tint, the contour and the
     decision boundary guarantees all three agree with each other. */
  function computeField(model) {
    var nx = Math.ceil(PW / GRID_PX), ny = Math.ceil(PH / GRID_PX);
    var n = (nx + 1) * (ny + 1);
    if (field.nx !== nx || field.ny !== ny) {
      field.nx = nx; field.ny = ny;
      field.odds = new Float64Array(n);
      field.dens = new Float64Array(n);
    }
    var maxD = -Infinity;
    for (var j = 0; j <= ny; j++) {
      var py = j * GRID_PX, wyj = wy(py);
      for (var i = 0; i <= nx; i++) {
        var wxi = wx(i * GRID_PX), idx = j * (nx + 1) + i;
        field.odds[idx] = logOdds(wxi, wyj, model);
        var d = logMixture(wxi, wyj, model);
        field.dens[idx] = d;
        if (d > maxD) maxD = d;
      }
    }
    field.maxDens = maxD;
  }

  function drawHeatmap(mode) {
    if (mode === 'off') return;
    var nx = field.nx, ny = field.ny, w = nx + 1, h = ny + 1;
    if (hm.width !== w || hm.height !== h) { hm.width = w; hm.height = h; }
    var img = hctx.createImageData(w, h);
    var px = img.data;

    for (var j = 0; j < h; j++) {
      for (var i = 0; i < w; i++) {
        var idx = j * w + i, o = idx * 4;
        var r1 = sigmoid(field.odds[idx]);
        var c = mixColor(r1);
        var a;
        if (mode === 'dens') {
          // exp(log p − log p_max) keeps the ratio finite even when the
          // covariance has collapsed and the raw density is astronomical.
          var rel = Math.exp(field.dens[idx] - field.maxDens);
          a = 0.62 * Math.pow(rel, 0.55);
        } else {
          // Tint by owner, with strength following confidence: the boundary
          // region stays pale, which is the honest picture.
          a = 0.08 + 0.17 * Math.abs(2 * r1 - 1);
        }
        px[o] = c[0]; px[o + 1] = c[1]; px[o + 2] = c[2];
        px[o + 3] = Math.round(Math.max(0, Math.min(1, a)) * 255);
      }
    }
    hctx.putImageData(img, 0, 0);

    pctx.save();
    pctx.imageSmoothingEnabled = true;
    pctx.imageSmoothingQuality = 'high';
    pctx.drawImage(hm, 0, 0, w, h, 0, 0, nx * GRID_PX, ny * GRID_PX);
    pctx.restore();
  }

  /* Marching squares on the log-odds grid at level 0.
     The r₁ = r₂ boundary of a two-Gaussian mixture is a conic — a line when the
     covariances match, otherwise an ellipse, parabola or hyperbola. Marching
     squares draws whichever one it is without special-casing, and handles the
     disconnected branches of a hyperbola for free. Corner bits: 1 = bottom-left,
     2 = bottom-right, 4 = top-right, 8 = top-left. */
  function contourSegments(level) {
    var nx = field.nx, ny = field.ny, f = field.odds, w = nx + 1;
    var segs = [];

    for (var j = 0; j < ny; j++) {
      for (var i = 0; i < nx; i++) {
        var v00 = f[j * w + i], v10 = f[j * w + i + 1];
        var v11 = f[(j + 1) * w + i + 1], v01 = f[(j + 1) * w + i];
        var idx = 0;
        if (v00 > level) idx |= 1;
        if (v10 > level) idx |= 2;
        if (v11 > level) idx |= 4;
        if (v01 > level) idx |= 8;
        if (idx === 0 || idx === 15) continue;

        var X = i * GRID_PX, Y = j * GRID_PX, g = GRID_PX;
        var ip = function (va, vb) {
          var den = vb - va;
          return Math.abs(den) < 1e-15 ? 0.5 : (level - va) / den;
        };
        var B = [X + g * ip(v00, v10), Y];
        var R = [X + g, Y + g * ip(v10, v11)];
        var T = [X + g * ip(v01, v11), Y + g];
        var L = [X, Y + g * ip(v00, v01)];

        switch (idx) {
          case 1: case 14: segs.push([L, B]); break;
          case 2: case 13: segs.push([B, R]); break;
          case 3: case 12: segs.push([L, R]); break;
          case 4: case 11: segs.push([R, T]); break;
          case 6: case 9:  segs.push([B, T]); break;
          case 7: case 8:  segs.push([L, T]); break;
          // Saddles: the four corners alone are ambiguous, so break the tie
          // with the cell-centre average.
          case 5:
            if ((v00 + v10 + v11 + v01) / 4 > level) { segs.push([L, T]); segs.push([B, R]); }
            else { segs.push([L, B]); segs.push([R, T]); }
            break;
          case 10:
            if ((v00 + v10 + v11 + v01) / 4 > level) { segs.push([L, B]); segs.push([R, T]); }
            else { segs.push([L, T]); segs.push([B, R]); }
            break;
        }
      }
    }
    return segs;
  }


  /* ==========================================================================
     PART 5 — THE 2D PLOT
     ====================================================================== */

  function drawGrid() {
    pctx.save();
    pctx.strokeStyle = pal.grid;
    pctx.lineWidth = 1;
    var yh = yHalf();
    for (var x = -6; x <= 6; x++) {
      pctx.beginPath(); pctx.moveTo(sx(x), 0); pctx.lineTo(sx(x), PH); pctx.stroke();
    }
    for (var y = -Math.ceil(yh); y <= Math.ceil(yh); y++) {
      pctx.beginPath(); pctx.moveTo(0, sy(y)); pctx.lineTo(PW, sy(y)); pctx.stroke();
    }
    pctx.strokeStyle = pal.axis;
    pctx.globalAlpha = 0.55;
    pctx.beginPath(); pctx.moveTo(0, sy(0)); pctx.lineTo(PW, sy(0)); pctx.stroke();
    pctx.beginPath(); pctx.moveTo(sx(0), 0); pctx.lineTo(sx(0), PH); pctx.stroke();
    pctx.restore();
  }

  /* Ellipse geometry comes from the eigendecomposition of Σ, never from the
     sliders — so what is drawn is provably the Mahalanobis contour of the
     matrix the model is actually using. The screen rotation is negated because
     the canvas y-axis points down. */
  function drawEllipse(c, t, col, lw, dash, alpha) {
    var e = eig2(c.S);
    pctx.save();
    pctx.setLineDash(dash || []);
    pctx.strokeStyle = rgba(col, alpha);
    pctx.lineWidth = lw;
    pctx.beginPath();
    pctx.ellipse(sx(c.mx), sy(c.my),
                 t * Math.sqrt(e.l1) * scale, t * Math.sqrt(e.l2) * scale,
                 -e.angle, 0, TAU);
    pctx.stroke();
    pctx.restore();
  }

  function drawBoundary(model) {
    var segs = contourSegments(0);
    if (!segs.length) return;

    pctx.save();
    // Halo first so the line reads over both the heatmap and the points.
    pctx.strokeStyle = pal.panel;
    pctx.globalAlpha = 0.8;
    pctx.lineWidth = 5;
    pctx.lineCap = 'round';
    drawSegs(segs);
    pctx.globalAlpha = 0.85;
    pctx.strokeStyle = pal.ink;
    pctx.lineWidth = 1.8;
    // Solid, not dashed: marching squares emits many short independent
    // segments, and a dash pattern restarts on each one, so it would render as
    // a solid line anyway. Saying so beats a setting that quietly does nothing.
    drawSegs(segs);
    pctx.restore();

    // Label the boundary at whichever piece of it runs nearest the midpoint
    // between the two centres.
    var mxp = sx((model.comps[0].mx + model.comps[1].mx) / 2);
    var myp = sy((model.comps[0].my + model.comps[1].my) / 2);
    var best = null, bd = Infinity;
    for (var i = 0; i < segs.length; i++) {
      var p = segs[i][0];
      var dd = (p[0] - mxp) * (p[0] - mxp) + (p[1] - myp) * (p[1] - myp);
      if (dd < bd) { bd = dd; best = p; }
    }
    if (best && best[0] > 30 && best[0] < PW - 30 && best[1] > 18 && best[1] < PH - 10) {
      pctx.save();
      pctx.font = '500 11px "IBM Plex Mono", ui-monospace, monospace';
      // ASCII only: Unicode subscripts fall back to a different face at 11px
      // on canvas and render as mush.
      var txt = 'r1 = r2';
      var tw = pctx.measureText(txt).width;
      pctx.fillStyle = pal.panel;
      pctx.globalAlpha = 0.86;
      pctx.fillRect(best[0] - tw / 2 - 5, best[1] - 17, tw + 10, 15);
      pctx.globalAlpha = 1;
      pctx.fillStyle = pal.ink2;
      pctx.textAlign = 'center';
      pctx.fillText(txt, best[0], best[1] - 6);
      pctx.restore();
    }
  }

  function drawSegs(segs) {
    pctx.beginPath();
    for (var i = 0; i < segs.length; i++) {
      pctx.moveTo(segs[i][0][0], segs[i][0][1]);
      pctx.lineTo(segs[i][1][0], segs[i][1][1]);
    }
    pctx.stroke();
  }

  function drawMean(c, col, label, isActive) {
    var X = sx(c.mx), Y = sy(c.my);
    pctx.save();
    if (isActive) {
      pctx.beginPath(); pctx.arc(X, Y, 15, 0, TAU);
      pctx.fillStyle = rgba(col, 0.13); pctx.fill();
    }
    pctx.beginPath(); pctx.arc(X, Y, 9, 0, TAU);
    pctx.fillStyle = rgba(col, 1); pctx.fill();
    pctx.lineWidth = 2.5; pctx.strokeStyle = pal.panel; pctx.stroke();
    pctx.beginPath(); pctx.arc(X, Y, 11.4, 0, TAU);
    pctx.lineWidth = 1.4; pctx.strokeStyle = rgba(col, 0.75); pctx.stroke();

    pctx.font = '600 11px "IBM Plex Sans", system-ui, sans-serif';
    pctx.fillStyle = pal.panel;
    pctx.textAlign = 'center'; pctx.textBaseline = 'middle';
    pctx.fillText(label, X, Y + 0.5);
    pctx.restore();
  }

  function render() {
    var base = modelAt(S, 0);
    var model = modelAt(S, S.km);

    rebuildPoints();
    computeField(model);

    pctx.clearRect(0, 0, PW, PH);
    pctx.fillStyle = pal.sunken;
    pctx.fillRect(0, 0, PW, PH);

    drawHeatmap($('heatMode').value);
    drawGrid();

    // Ellipses at Mahalanobis 1 (solid) and 2 (dashed).
    for (var k = 0; k < 2; k++) {
      var col = k === 0 ? pal.blue : pal.red;
      drawEllipse(model.comps[k], 2, col, 1.2, [5, 4], 0.45);
      drawEllipse(model.comps[k], 1, col, 2, null, 0.95);
    }

    drawBoundary(model);

    // Points, coloured by responsibility under the current (possibly blended)
    // model. Ambiguous points additionally get a ring, so the "unsure" state
    // survives greyscale printing and colour-vision deficiency.
    var flips = 0;
    for (var i = 0; i < pts.length; i++) {
      var p = pts[i];
      var r = responsibilities(p.x, p.y, model)[0];
      var rb = responsibilities(p.x, p.y, base)[0];
      if ((r >= 0.5) !== (rb >= 0.5)) flips++;

      var X = sx(p.x), Y = sy(p.y);
      var conf = Math.abs(2 * r - 1);
      var c = mixColor(r);

      if (conf < 0.36) {
        pctx.beginPath(); pctx.arc(X, Y, 5.4, 0, TAU);
        pctx.strokeStyle = rgba(pal.violet, 0.55);
        pctx.lineWidth = 1.2; pctx.stroke();
      }
      pctx.beginPath(); pctx.arc(X, Y, 3.2, 0, TAU);
      pctx.fillStyle = rgba(c, 0.9); pctx.fill();
      if (i === selected) {
        pctx.beginPath(); pctx.arc(X, Y, 7.5, 0, TAU);
        pctx.strokeStyle = pal.ink; pctx.lineWidth = 2; pctx.stroke();
      }
    }
    $('kmFlips').textContent = flips;

    // The demo point from "watch one point get made".
    if (demo) drawDemo(model);

    // Distance lines from the selected point to both centres — the comparison
    // K-means makes, drawn next to the comparison a GMM makes.
    var sel = selected >= 0 ? pts[selected] : (demo && demo.phase >= 3 ? demo : null);
    if (sel) drawSpokes(sel, model);

    for (var m = 0; m < 2; m++) {
      drawMean(model.comps[m], m === 0 ? pal.blue : pal.red, String(m + 1), activeComp === m);
    }

    drawOneD();
    syncReadouts(model);
  }

  function drawSpokes(p, model) {
    pctx.save();
    pctx.setLineDash([4, 4]);
    pctx.lineWidth = 1.5;
    for (var k = 0; k < 2; k++) {
      var c = model.comps[k], col = k === 0 ? pal.blue : pal.red;
      pctx.strokeStyle = rgba(col, 0.8);
      pctx.beginPath();
      pctx.moveTo(sx(p.x), sy(p.y));
      pctx.lineTo(sx(c.mx), sy(c.my));
      pctx.stroke();

      var dist = Math.hypot(p.x - c.mx, p.y - c.my);
      var mx = (sx(p.x) + sx(c.mx)) / 2, my = (sy(p.y) + sy(c.my)) / 2;
      pctx.setLineDash([]);
      pctx.font = '500 10.5px "IBM Plex Mono", ui-monospace, monospace';
      var t = dist.toFixed(2);
      var tw = pctx.measureText(t).width;
      pctx.fillStyle = pal.panel; pctx.globalAlpha = 0.88;
      pctx.fillRect(mx - tw / 2 - 4, my - 8, tw + 8, 14);
      pctx.globalAlpha = 1;
      pctx.fillStyle = rgba(col, 1);
      pctx.textAlign = 'center'; pctx.textBaseline = 'middle';
      pctx.fillText(t, mx, my - 0.5);
      pctx.setLineDash([4, 4]);
    }
    pctx.restore();
  }

  function drawDemo(model) {
    var X = sx(demo.x), Y = sy(demo.y);
    var col;
    if (demo.phase === 1) col = demo.k === 0 ? pal.blue : pal.red;   // true label shown
    else if (demo.phase === 2) col = pal.mutedRGB;            // label thrown away
    else col = mixColor(responsibilities(demo.x, demo.y, model)[0]);

    pctx.save();
    pctx.beginPath(); pctx.arc(X, Y, 13, 0, TAU);
    pctx.strokeStyle = rgba(col, 0.35); pctx.lineWidth = 2; pctx.stroke();
    pctx.beginPath(); pctx.arc(X, Y, 6, 0, TAU);
    pctx.fillStyle = rgba(col, 1); pctx.fill();
    pctx.strokeStyle = pal.panel; pctx.lineWidth = 2; pctx.stroke();
    pctx.restore();
  }


  /* ==========================================================================
     PART 6 — THE 1D PANEL
     "We add the curves, not the samples."
     ====================================================================== */

  var hover1d = null;
  var X1 = -6, X2 = 6;

  function drawOneD() {
    var pad = { l: 8, r: 8, t: 12, b: 20 };
    var w = OW - pad.l - pad.r, h = OH - pad.t - pad.b;
    var mu = S.mu1d, pi = [S.pi1, 1 - S.pi1];

    // Scale to whichever is taller: an unweighted component or the mixture.
    var yMax = 0, i, x;
    for (i = 0; i <= 240; i++) {
      x = X1 + (X2 - X1) * i / 240;
      var a = gauss1(x, mu[0], SD1D[0]), b = gauss1(x, mu[1], SD1D[1]);
      yMax = Math.max(yMax, a, b, pi[0] * a + pi[1] * b);
    }
    yMax *= 1.12;

    var px = function (v) { return pad.l + (v - X1) / (X2 - X1) * w; };
    var py = function (v) { return pad.t + h - (v / yMax) * h; };

    octx.clearRect(0, 0, OW, OH);
    octx.fillStyle = pal.sunken;
    octx.fillRect(0, 0, OW, OH);

    // baseline + ticks
    octx.save();
    octx.strokeStyle = pal.grid; octx.lineWidth = 1;
    for (i = -6; i <= 6; i += 2) {
      octx.beginPath(); octx.moveTo(px(i), pad.t); octx.lineTo(px(i), pad.t + h); octx.stroke();
    }
    octx.strokeStyle = pal.axis; octx.globalAlpha = 0.6;
    octx.beginPath(); octx.moveTo(pad.l, py(0)); octx.lineTo(pad.l + w, py(0)); octx.stroke();
    octx.globalAlpha = 1;
    octx.font = '400 10px "IBM Plex Mono", ui-monospace, monospace';
    octx.fillStyle = pal.muted; octx.textAlign = 'center';
    for (i = -6; i <= 6; i += 2) octx.fillText(String(i), px(i), OH - 6);
    octx.restore();

    var cols = [pal.blue, pal.red];

    // 1. weighted contributions πₖ·pₖ, filled
    for (var k = 0; k < 2; k++) {
      octx.beginPath();
      octx.moveTo(px(X1), py(0));
      for (i = 0; i <= 240; i++) {
        x = X1 + (X2 - X1) * i / 240;
        octx.lineTo(px(x), py(pi[k] * gauss1(x, mu[k], SD1D[k])));
      }
      octx.lineTo(px(X2), py(0));
      octx.closePath();
      octx.fillStyle = rgba(cols[k], 0.20); octx.fill();
      octx.strokeStyle = rgba(cols[k], 0.95); octx.lineWidth = 2; octx.stroke();
    }

    // 2. the raw, unweighted bell curves, dashed — what each component would
    //    look like on its own, before the mixture weight scales it down
    octx.save();
    octx.setLineDash([3, 3]); octx.lineWidth = 1.1;
    for (k = 0; k < 2; k++) {
      octx.beginPath();
      for (i = 0; i <= 240; i++) {
        x = X1 + (X2 - X1) * i / 240;
        var v = py(gauss1(x, mu[k], SD1D[k]));
        if (i === 0) octx.moveTo(px(x), v); else octx.lineTo(px(x), v);
      }
      octx.strokeStyle = rgba(cols[k], 0.5); octx.stroke();
    }
    octx.restore();

    // 3. the mixture — the sum of the two filled curves
    octx.beginPath();
    for (i = 0; i <= 240; i++) {
      x = X1 + (X2 - X1) * i / 240;
      var s = pi[0] * gauss1(x, mu[0], SD1D[0]) + pi[1] * gauss1(x, mu[1], SD1D[1]);
      if (i === 0) octx.moveTo(px(x), py(s)); else octx.lineTo(px(x), py(s));
    }
    octx.strokeStyle = pal.ink; octx.lineWidth = 2.4; octx.stroke();

    // 4. hover: stack the two contributions to show they add to the black curve
    if (hover1d !== null) {
      var hx = Math.max(X1, Math.min(X2, hover1d));
      var c1 = pi[0] * gauss1(hx, mu[0], SD1D[0]);
      var c2 = pi[1] * gauss1(hx, mu[1], SD1D[1]);
      var X = px(hx);

      octx.save();
      octx.strokeStyle = pal.axis; octx.setLineDash([2, 3]); octx.lineWidth = 1;
      octx.beginPath(); octx.moveTo(X, pad.t); octx.lineTo(X, py(0)); octx.stroke();
      octx.setLineDash([]);

      // stacked bar, 2px gap between segments
      octx.lineWidth = 7; octx.lineCap = 'butt';
      octx.strokeStyle = rgba(pal.blue, 1);
      octx.beginPath(); octx.moveTo(X, py(0)); octx.lineTo(X, py(c1)); octx.stroke();
      octx.strokeStyle = rgba(pal.red, 1);
      octx.beginPath(); octx.moveTo(X, py(c1) - 2); octx.lineTo(X, py(c1 + c2)); octx.stroke();

      octx.beginPath(); octx.arc(X, py(c1 + c2), 4, 0, TAU);
      octx.fillStyle = pal.ink; octx.fill();
      octx.strokeStyle = pal.panel; octx.lineWidth = 2; octx.stroke();
      octx.restore();

      $('sumLine').innerHTML =
        'p(<b>' + hx.toFixed(2) + '</b>) =<br>' +
        '<span class="t1">' + pi[0].toFixed(2) + ' × ' + gauss1(hx, mu[0], SD1D[0]).toFixed(4) + '</span> + ' +
        '<span class="t2">' + pi[1].toFixed(2) + ' × ' + gauss1(hx, mu[1], SD1D[1]).toFixed(4) + '</span><br>' +
        '= <b>' + (c1 + c2).toFixed(4) + '</b>';
    }
  }


  /* ==========================================================================
     PART 7 — INSPECTOR
     ====================================================================== */

  function fmtDensity(v) {
    if (v >= 0.0001 || v === 0) return v.toFixed(4);
    return v.toExponential(1);
  }

  function showPoint(p, model) {
    var base = modelAt(S, S.km);
    var c1 = base.comps[0], c2 = base.comps[1];
    var d1 = Math.exp(logGauss2(p.x, p.y, c1));
    var d2 = Math.exp(logGauss2(p.x, p.y, c2));
    var r = responsibilities(p.x, p.y, base);
    var e1 = Math.hypot(p.x - c1.mx, p.y - c1.my);
    var e2 = Math.hypot(p.x - c2.mx, p.y - c2.my);
    var m1 = Math.sqrt(maha2(c1.S, p.x - c1.mx, p.y - c1.my));
    var m2 = Math.sqrt(maha2(c2.S, p.x - c2.mx, p.y - c2.my));

    $('inspEmpty').hidden = true;
    $('inspBody').hidden = false;
    $('pCoord').textContent = 'y = (' + p.x.toFixed(2) + ', ' + p.y.toFixed(2) + ')';
    $('pd1').textContent = fmtDensity(d1);
    $('pd2').textContent = fmtDensity(d2);
    $('pr1').textContent = r[0].toFixed(3);
    $('pr2').textContent = r[1].toFixed(3);
    $('pe1').textContent = e1.toFixed(2);
    $('pe2').textContent = e2.toFixed(2);
    $('pm1').textContent = m1.toFixed(2) + 'σ';
    $('pm2').textContent = m2.toFixed(2) + 'σ';
    $('bar1').style.width = (r[0] * 100).toFixed(1) + '%';
    $('bar2').style.width = (r[1] * 100).toFixed(1) + '%';

    $('verdict').className = 'verdict ' + verdictClass(r[0]);
    $('verdict').innerHTML = explain(p, base, r, d1, d2, e1, e2, m1, m2);
  }

  function verdictClass(r1) {
    if (r1 > 0.65) return 'c1';
    if (r1 < 0.35) return 'c2';
    return 'tie';
  }

  /* The plain-language panel. Rather than restating the numbers, it names the
     reason the numbers came out that way — and flags the two cases that are
     genuinely surprising:
       · the prior overturned the likelihood
       · the nearer centre lost anyway, because ellipse shape outranks raw
         distance (which is exactly what K-means cannot do) */
  function explain(p, model, r, d1, d2, e1, e2, m1, m2) {
    var win = r[0] >= r[1] ? 0 : 1;
    var lead = Math.max(r[0], r[1]);
    var name = ['Component 1', 'Component 2'][win];
    var other = ['Component 2', 'Component 1'][win];
    var mWin = win === 0 ? m1 : m2, mLose = win === 0 ? m2 : m1;
    var html = '';

    if (lead < 0.62) {
      html = '<b>Neither component owns this point.</b> It sits ' + mWin.toFixed(1) +
             'σ into ' + name + ' and ' + mLose.toFixed(1) + 'σ into ' + other +
             ' — close enough to the same that the model splits the difference ' +
             r[0].toFixed(2) + ' / ' + r[1].toFixed(2) + '. This is the honest answer, ' +
             'not a failure: the point really could have come from either.';
    } else {
      html = '<b>' + name + ' explains this point better.</b> ';
      if (mWin < mLose * 0.96) {
        html += 'Measured in each cloud’s own units it lies just ' + mWin.toFixed(1) +
                'σ from ' + name + '’s centre, against ' + mLose.toFixed(1) +
                'σ from ' + other + '’s — well inside the first ellipse, ' +
                'out in the thin tail of the second.';
      } else {
        html += 'Both centres are about equally far away, but ' + name + '’s ellipse is ' +
                'tighter here, so the same point lands in a much denser part of it.';
      }
      html += ' That makes the density ' + fmtDensity(win === 0 ? d1 : d2) + ' against ' +
              fmtDensity(win === 0 ? d2 : d1) + ', and responsibility ' + lead.toFixed(2) + '.';
    }

    // Did the mixture weight decide it? Re-run with equal weights and compare.
    var flat = { pi: [0.5, 0.5], comps: model.comps };
    var rFlat = responsibilities(p.x, p.y, flat);
    var winFlat = rFlat[0] >= rFlat[1] ? 0 : 1;
    if (winFlat !== win) {
      html += '<span class="flag">↑ The prior decided this one. With equal mixture ' +
              'weights ' + other + ' would win (r would be ' + rFlat[win === 0 ? 0 : 1].toFixed(2) +
              '); π₁ = ' + S.pi1.toFixed(2) + ' tipped it the other way.</span>';
    } else if (Math.abs(rFlat[0] - r[0]) > 0.06) {
      html += '<span class="flag">With equal weights r₁ would be ' + rFlat[0].toFixed(2) +
              ' instead of ' + r[0].toFixed(2) + ' — the prior is nudging, not deciding.</span>';
    }

    // Did raw distance disagree with Mahalanobis distance?
    var nearest = e1 <= e2 ? 0 : 1;
    if (nearest !== win && lead > 0.62) {
      html += '<span class="flag">↑ Note: this point is physically <em>closer</em> to ' +
              other + '’s centre (' + Math.min(e1, e2).toFixed(2) + ' vs ' +
              Math.max(e1, e2).toFixed(2) + '), and the GMM assigned it the other way anyway. ' +
              'Shape beats raw distance — K-means would get this one wrong.</span>';
    }
    return html;
  }

  function refreshInspector() {
    var model = modelAt(S, S.km);
    if (selected >= 0 && selected < pts.length) showPoint(pts[selected], model);
    else if (demo && demo.phase >= 3) showPoint(demo, model);
  }


  /* ==========================================================================
     PART 8 — CONTROLS & READOUTS
     ====================================================================== */

  function fmtSigma(S2) {
    return '[ ' + S2.a.toFixed(2) + '  ' + S2.b.toFixed(2) + ' ; ' +
           S2.b.toFixed(2) + '  ' + S2.d.toFixed(2) + ' ]';
  }

  function syncReadouts(model) {
    $('sig1').textContent = fmtSigma(model.comps[0].S);
    $('sig2').textContent = fmtSigma(model.comps[1].S);
    $('pi2v').textContent = (1 - S.pi1).toFixed(2);
    $('stPi1').textContent = S.pi1.toFixed(2);

    var t = S.km;
    $('kmPi').textContent = model.pi[0].toFixed(2) + ' / ' + model.pi[1].toFixed(2);
    $('kmPiNote').textContent = t > 0.98 ? 'forced equal — the prior is gone'
                              : t > 0.02 ? 'flattening toward 50/50' : 'as you set them';
    $('kmShape').textContent = t > 0.98
      ? 'one small circle, shared — tilt and width discarded'
      : t > 0.02 ? 'rounding off; tilt draining away' : 'tilted, unequal — each cloud keeps its own Σ';
    $('kmBoundary').textContent = t > 0.98
      ? 'a straight line — the perpendicular bisector of the two centres'
      : t > 0.02 ? 'straightening out' : 'curved — bends around the narrower cloud';
    $('kmLabel').textContent = t < 0.02 ? 'Full GMM' : t > 0.98 ? 'K-means' : 'blend ' + t.toFixed(2);
  }

  /* Bind a range input to a path in S, with an optional value formatter. */
  function bind(id, get, set, fmt) {
    var el = $(id), out = $(id + 'v');
    var apply = function () {
      set(parseFloat(el.value));
      if (out) out.textContent = fmt ? fmt(parseFloat(el.value)) : el.value;
      render();
      refreshInspector();
    };
    el.addEventListener('input', apply);
    return { el: el, sync: function () {
      el.value = get();
      if (out) out.textContent = fmt ? fmt(parseFloat(el.value)) : el.value;
    } };
  }

  var deg = function (v) { return v.toFixed(0) + '°'; };
  var two = function (v) { return v.toFixed(2); };

  var binds = [
    bind('sx1', function () { return S.comps[0].sx; }, function (v) { S.comps[0].sx = v; }, two),
    bind('sy1', function () { return S.comps[0].sy; }, function (v) { S.comps[0].sy = v; }, two),
    bind('rot1', function () { return S.comps[0].rot * 180 / Math.PI; }, function (v) { S.comps[0].rot = v * Math.PI / 180; }, deg),
    bind('sx2', function () { return S.comps[1].sx; }, function (v) { S.comps[1].sx = v; }, two),
    bind('sy2', function () { return S.comps[1].sy; }, function (v) { S.comps[1].sy = v; }, two),
    bind('rot2', function () { return S.comps[1].rot * 180 / Math.PI; }, function (v) { S.comps[1].rot = v * Math.PI / 180; }, deg),
    bind('pi1', function () { return S.pi1; }, function (v) { S.pi1 = v; }, two),
    bind('npts', function () { return S.n; }, function (v) { S.n = Math.round(v); selected = -1; clearInspector(); }),
    bind('kmeans', function () { return S.km; }, function (v) { S.km = v; }, two),
    bind('mu1d1', function () { return S.mu1d[0]; }, function (v) { S.mu1d[0] = v; }, two),
    bind('mu1d2', function () { return S.mu1d[1]; }, function (v) { S.mu1d[1] = v; }, two)
  ];

  function syncAll() { for (var i = 0; i < binds.length; i++) binds[i].sync(); }

  function clearInspector() {
    $('inspEmpty').hidden = false;
    $('inspBody').hidden = true;
  }

  $('heatMode').addEventListener('change', render);

  $('gen').addEventListener('click', function () {
    seed = (seed * 1664525 + 1013904223) >>> 0;
    reseed();
    selected = -1; demo = null;
    clearInspector();
    render();
    say('New sample drawn.');
  });

  $('reset').addEventListener('click', function () {
    S = clone(DEFAULTS);
    seed = 7; reseed();
    selected = -1; demo = null; activeComp = 0;
    $('heatMode').value = 'resp';
    setStepButton(-1);
    clearInspector();
    syncAll(); render();
    say('Everything reset.');
  });

  /* --- "watch one point get made" -----------------------------------------
     The whole page in four seconds: roll the hidden coin, draw from the cloud
     it picked, throw the label away, then infer it back. */
  var demoTimers = [];
  $('draw1').addEventListener('click', function () {
    demoTimers.forEach(clearTimeout); demoTimers = [];
    var rnd = mulberry32((Date.now() & 0xffff) >>> 0);
    var k = rnd() < S.pi1 ? 0 : 1;
    var g = gaussPair(rnd);
    var p = samplePoint(S.comps[k], g[0], g[1]);
    demo = { x: p.x, y: p.y, k: k, phase: 0 };
    selected = -1;
    clearInspector();

    var reduced = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
    var step = reduced ? 1 : 1100;

    lightStep(1);
    demoTimers.push(setTimeout(function () {
      demo.phase = 1; lightStep(2); render();
      say('Component ' + (k + 1) + ' was chosen, and produced a point.');
    }, step * 0.55));
    demoTimers.push(setTimeout(function () {
      demo.phase = 2; lightStep(3); render();
      say('The component label is now hidden.');
    }, step * 1.7));
    demoTimers.push(setTimeout(function () {
      demo.phase = 3; lightStep(4); render(); refreshInspector();
      say('Responsibilities computed.');
    }, step * 2.8));
    render();
  });

  function lightStep(n) {
    for (var i = 1; i <= 4; i++) $('st' + i).classList.toggle('lit', i === n);
  }

  /* --- guided walkthrough --------------------------------------------------
     Each step is a target parameter set; we tween into it so the change is
     legible rather than a jump cut. */
  var STEPS = [
    {
      note: '<b>Far apart, and the model is sure.</b> Almost every point is solid blue or solid red — ' +
            'responsibility is pinned at 1 and 0. The ellipses barely touch, so there is nothing to argue about.',
      set: { c: [{ mx: -2.9, my: -1.0, sx: 1.1, sy: 0.8, rot: 20 }, { mx: 2.9, my: 1.0, sx: 1.0, sy: 0.9, rot: -10 }], pi1: 0.5, km: 0 }
    },
    {
      note: '<b>Now they overlap.</b> The purple, ringed points in the middle are the ones the model genuinely ' +
            'cannot attribute — both clouds explain them about equally well. Click one and read the numbers.',
      set: { c: [{ mx: -0.85, my: -0.15, sx: 1.5, sy: 1.0, rot: 20 }, { mx: 0.85, my: 0.2, sx: 1.4, sy: 1.1, rot: -25 }], pi1: 0.5, km: 0 }
    },
    {
      note: '<b>Same geometry, different prior.</b> Only π₁ changed. Component 1 is now picked 4 times ' +
            'out of 5, so every contested point drifts blue — and the boundary slides toward red’s side. ' +
            'Where the evidence is weak, the prior decides.',
      set: { c: [{ mx: -0.85, my: -0.15, sx: 1.5, sy: 1.0, rot: 20 }, { mx: 0.85, my: 0.2, sx: 1.4, sy: 1.1, rot: -25 }], pi1: 0.8, km: 0 }
    },
    {
      note: '<b>Watch what K-means throws away.</b> The blue cloud is long and tilted; the red one is round. ' +
            'The GMM boundary curves to respect that. Drag the slider to 1 and the shapes collapse, the weights ' +
            'flatten, the boundary snaps straight — and the flip counter shows how many points change hands.',
      set: { c: [{ mx: -1.5, my: -0.4, sx: 2.3, sy: 0.5, rot: 32 }, { mx: 1.7, my: 0.7, sx: 0.95, sy: 0.95, rot: 0 }], pi1: 0.5, km: 0 }
    }
  ];

  var tweenRAF = null;
  function tweenTo(target, ms) {
    if (tweenRAF) cancelAnimationFrame(tweenRAF);
    var from = clone(S);
    var reduced = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
    if (reduced) ms = 0;
    var t0 = performance.now();

    var lerp = function (a, b, u) { return a + (b - a) * u; };
    var frame = function (now) {
      var u = ms <= 0 ? 1 : Math.min(1, (now - t0) / ms);
      var e = u < 0.5 ? 4 * u * u * u : 1 - Math.pow(-2 * u + 2, 3) / 2;
      for (var k = 0; k < 2; k++) {
        var f = from.comps[k], g = target.c[k];
        S.comps[k].mx = lerp(f.mx, g.mx, e);
        S.comps[k].my = lerp(f.my, g.my, e);
        S.comps[k].sx = lerp(f.sx, g.sx, e);
        S.comps[k].sy = lerp(f.sy, g.sy, e);
        S.comps[k].rot = lerp(f.rot, g.rot * Math.PI / 180, e);
      }
      S.pi1 = lerp(from.pi1, target.pi1, e);
      S.km = lerp(from.km, target.km, e);
      syncAll(); render(); refreshInspector();
      if (u < 1) tweenRAF = requestAnimationFrame(frame); else tweenRAF = null;
    };
    tweenRAF = requestAnimationFrame(frame);
  }

  function setStepButton(i) {
    var bs = document.querySelectorAll('.steps button');
    for (var j = 0; j < bs.length; j++) bs[j].classList.toggle('on', j === i);
  }

  document.querySelectorAll('.steps button').forEach(function (b) {
    b.addEventListener('click', function () {
      var i = parseInt(b.dataset.step, 10);
      setStepButton(i);
      $('stepNote').innerHTML = STEPS[i].note;
      tweenTo(STEPS[i].set, 700);
      say('Step ' + (i + 1) + ' set.');
    });
  });

  /* --- tooltips ------------------------------------------------------------ */
  var tip = $('tip'), tipOwner = null;

  function showTip(btn) {
    tip.textContent = btn.dataset.tip;
    tip.hidden = false;
    var r = btn.getBoundingClientRect();
    var tr = tip.getBoundingClientRect();
    var left = Math.min(Math.max(8, r.left + r.width / 2 - tr.width / 2), window.innerWidth - tr.width - 8);
    var top = r.top - tr.height - 8;
    if (top < 8) top = r.bottom + 8;
    tip.style.left = left + 'px';
    tip.style.top = top + 'px';
    tipOwner = btn;
  }
  function hideTip() { tip.hidden = true; tipOwner = null; }

  document.querySelectorAll('.q').forEach(function (b) {
    b.setAttribute('aria-label', 'Explain: ' + b.dataset.tip.slice(0, 40));
    b.addEventListener('mouseenter', function () { showTip(b); });
    b.addEventListener('mouseleave', function () { if (document.activeElement !== b) hideTip(); });
    b.addEventListener('focus', function () { showTip(b); });
    b.addEventListener('blur', hideTip);
    b.addEventListener('click', function (e) {
      e.preventDefault();
      if (tipOwner === b && !tip.hidden) hideTip(); else showTip(b);
    });
  });
  document.addEventListener('keydown', function (e) { if (e.key === 'Escape') hideTip(); });

  function say(msg) { $('live').textContent = msg; }


  /* ==========================================================================
     PART 9 — INTERACTION
     ====================================================================== */

  function localPos(ev) {
    var r = plot.getBoundingClientRect();
    return { x: ev.clientX - r.left, y: ev.clientY - r.top };
  }

  function meanHit(pos) {
    var model = modelAt(S, S.km);
    for (var k = 0; k < 2; k++) {
      var c = model.comps[k];
      if (Math.hypot(pos.x - sx(c.mx), pos.y - sy(c.my)) <= 17) return k;
    }
    return -1;
  }

  function pointHit(pos) {
    var best = -1, bd = 13 * 13;
    for (var i = 0; i < pts.length; i++) {
      var dx = pos.x - sx(pts[i].x), dy = pos.y - sy(pts[i].y);
      var d = dx * dx + dy * dy;
      if (d < bd) { bd = d; best = i; }
    }
    return best;
  }

  // Keep a centre inside the visible window so it can never be dragged away.
  function clampMean(k) {
    var xh = XSPAN / 2 - 0.3, yh = yHalf() - 0.3;
    S.comps[k].mx = Math.max(-xh, Math.min(xh, S.comps[k].mx));
    S.comps[k].my = Math.max(-yh, Math.min(yh, S.comps[k].my));
  }

  plot.addEventListener('pointerdown', function (ev) {
    var pos = localPos(ev);
    var k = meanHit(pos);
    if (k >= 0) {
      dragging = k; activeComp = k;
      plot.setPointerCapture(ev.pointerId);
      plot.classList.add('grabbing');
      render();
      return;
    }
    var i = pointHit(pos);
    if (i >= 0) {
      selected = i; demo = null;
      render(); refreshInspector();
    }
  });

  plot.addEventListener('pointermove', function (ev) {
    var pos = localPos(ev);
    if (dragging >= 0) {
      S.comps[dragging].mx = wx(pos.x);
      S.comps[dragging].my = wy(pos.y);
      clampMean(dragging);
      render(); refreshInspector();
      return;
    }
    // Hover preview: show a point's numbers without committing a selection.
    if (selected < 0) {
      var i = pointHit(pos);
      if (i >= 0) showPoint(pts[i], modelAt(S, S.km));
      else if (!demo) clearInspector();
    }
    plot.style.cursor = meanHit(pos) >= 0 ? 'grab' : (pointHit(pos) >= 0 ? 'pointer' : 'crosshair');
  });

  function endDrag(ev) {
    if (dragging >= 0) {
      dragging = -1;
      plot.classList.remove('grabbing');
      try { plot.releasePointerCapture(ev.pointerId); } catch (e) { /* already gone */ }
    }
  }
  plot.addEventListener('pointerup', endDrag);
  plot.addEventListener('pointercancel', endDrag);
  plot.addEventListener('pointerleave', function () {
    if (dragging < 0 && selected < 0 && !demo) clearInspector();
  });

  // Keyboard equivalent for dragging the means.
  plot.addEventListener('keydown', function (e) {
    var step = e.shiftKey ? 0.4 : 0.12, moved = false;
    if (e.key === '1') { activeComp = 0; moved = true; }
    else if (e.key === '2') { activeComp = 1; moved = true; }
    else if (e.key === 'ArrowLeft') { S.comps[activeComp].mx -= step; moved = true; }
    else if (e.key === 'ArrowRight') { S.comps[activeComp].mx += step; moved = true; }
    else if (e.key === 'ArrowUp') { S.comps[activeComp].my += step; moved = true; }
    else if (e.key === 'ArrowDown') { S.comps[activeComp].my -= step; moved = true; }
    if (!moved) return;
    e.preventDefault();
    clampMean(activeComp);
    render(); refreshInspector();
    say('Component ' + (activeComp + 1) + ' centre at ' +
        S.comps[activeComp].mx.toFixed(1) + ', ' + S.comps[activeComp].my.toFixed(1));
  });

  // 1D hover
  function onedX(ev) {
    var r = oned.getBoundingClientRect();
    var pad = 8, w = OW - 16;
    return X1 + (X2 - X1) * ((ev.clientX - r.left - pad) / w);
  }
  oned.addEventListener('pointermove', function (ev) { hover1d = onedX(ev); drawOneD(); });
  oned.addEventListener('pointerdown', function (ev) { hover1d = onedX(ev); drawOneD(); });
  oned.addEventListener('pointerleave', function () {
    hover1d = null;
    $('sumLine').textContent = 'Hover the curve to see the sum.';
    drawOneD();
  });


  /* ==========================================================================
     BOOT
     ====================================================================== */

  function boot() {
    readPalette();
    fit();
    reseed();
    syncAll();
    render();
  }

  var ro = new ResizeObserver(function () { fit(); render(); });
  ro.observe(plot.parentElement);
  ro.observe(oned.parentElement);

  var mq = window.matchMedia('(prefers-color-scheme: dark)');
  (mq.addEventListener ? mq.addEventListener.bind(mq, 'change') : mq.addListener.bind(mq))(function () {
    readPalette(); render();
  });

  if (document.fonts && document.fonts.ready) document.fonts.ready.then(render);
  boot();

})(typeof globalThis !== 'undefined' ? globalThis : this);
