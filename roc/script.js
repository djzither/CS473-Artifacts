/* ============================================================================
   Where Do You Draw the Line? — an interactive ROC curve.
   CS 473 module 3 · Murphy, PML §5.1.2–5.1.3 (Eq. 5.9–5.19, Tables 5.3–5.5)

   Layout of this file
     PART 1  Math          pure, DOM-free, exported for the Node test suite
     PART 2  State         parameters, the fixed draws, presets
     PART 3  Canvas        sizing, palette, hatch pattern
     PART 4  Score chart   the two distributions and the draggable threshold
     PART 5  ROC chart     the curve, its parts, the operating point
     PART 6  PR chart      the same model seen through precision
     PART 7  Readouts      confusion matrix, rates, cost panel
     PART 8  Controls      sliders, checkboxes, walkthrough, tooltips
     PART 9  Interaction   drag the line, drag the dot, keyboard

   PART 1 runs under Node (`require('./script.js')` returns the math only); the
   rest is skipped when there is no document. See test-math.js.
   ========================================================================= */

(function (global) {
  'use strict';

  /* ==========================================================================
     PART 1 — MATH
     Everything here is a pure function of its arguments. No DOM, no state.
     ====================================================================== */

  var SQRT2 = Math.SQRT2;
  var INV_SQRT_TAU = 1 / Math.sqrt(2 * Math.PI);

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

  /* Box–Muller. Returns two independent standard normals per call. */
  function gaussPair(rnd) {
    var u1 = 1 - rnd(), u2 = rnd();
    var r = Math.sqrt(-2 * Math.log(u1)), th = 2 * Math.PI * u2;
    return [r * Math.cos(th), r * Math.sin(th)];
  }

  function sigmoid(z) {
    if (z >= 0) return 1 / (1 + Math.exp(-z));
    var e = Math.exp(z);
    return e / (1 + e);
  }

  function logit(p) { return Math.log(p / (1 - p)); }

  /* Abramowitz & Stegun 7.1.26 — plenty for drawing, and the test suite only
     leans on it to ~1e-7, which it comfortably holds. */
  function erf(x) {
    var s = x < 0 ? -1 : 1;
    x = Math.abs(x);
    var t = 1 / (1 + 0.3275911 * x);
    var y = 1 - ((((1.061405429 * t - 1.453152027) * t + 1.421413741) * t
                - 0.284496736) * t + 0.254829592) * t * Math.exp(-x * x);
    return s * y;
  }

  function normCdf(x) { return 0.5 * (1 + erf(x / SQRT2)); }
  function normPdf(x) { return INV_SQRT_TAU * Math.exp(-0.5 * x * x); }

  /* --------------------------------------------------------------- the data */

  /* Each sample carries two draws that never change: a uniform `u` deciding
     which class it belongs to, and a standard normal `z` that becomes its
     score. Keeping them fixed means moving the "positives" slider genuinely
     migrates samples between classes, and moving "separation" slides existing
     scores rather than resampling — the picture deforms instead of flickering. */
  function makeDraws(seed, n) {
    var rnd = mulberry32(seed);
    var u = new Float64Array(n), z = new Float64Array(n);
    for (var i = 0; i < n; i++) {
      u[i] = rnd();
      var g = gaussPair(rnd);
      z[i] = g[0];
    }
    return { u: u, z: z };
  }

  /* Negatives ~ N(−sep/2, 1), positives ~ N(+sep/2, 1): the binormal model.
     Equal unit variances are not laziness — they are what makes the posterior
     an exact logistic of the score (see `posterior`), which in turn lets the
     cost-optimal threshold be solved in closed form instead of searched for. */
  function dataset(draws, n, prev, sep) {
    n = Math.min(n, draws.u.length);
    var s = new Float64Array(n), y = new Uint8Array(n);
    var nPos = 0;
    for (var i = 0; i < n; i++) {
      var pos = draws.u[i] < prev ? 1 : 0;
      y[i] = pos;
      if (pos) nPos++;
      s[i] = (pos ? sep / 2 : -sep / 2) + draws.z[i];
    }
    return { s: s, y: y, n: n, nPos: nPos, nNeg: n - nPos };
  }

  /* Murphy Eq. 5.16–5.17, counted directly: ŷ = I(score ≥ τ). */
  function confusion(d, t) {
    var tn = 0, fp = 0, fn = 0, tp = 0;
    for (var i = 0; i < d.n; i++) {
      var yh = d.s[i] >= t ? 1 : 0;
      if (d.y[i]) { if (yh) tp++; else fn++; }
      else { if (yh) fp++; else tn++; }
    }
    return { tn: tn, fp: fp, fn: fn, tp: tp, n: d.n };
  }

  /* Tables 5.4 and 5.5 in one object. Row-normalised quantities (tpr, fpr,
     tnr, fnr) divide inside a single true class; column-normalised ones
     (prec, npv) divide inside a single predicted class. That difference is
     the entire class-imbalance story, so both live here side by side. */
  function rates(cm) {
    var P = cm.tp + cm.fn, N = cm.fp + cm.tn;
    var PP = cm.tp + cm.fp, PN = cm.tn + cm.fn;
    return {
      tpr: P ? cm.tp / P : NaN,
      fnr: P ? cm.fn / P : NaN,
      fpr: N ? cm.fp / N : NaN,
      tnr: N ? cm.tn / N : NaN,
      prec: PP ? cm.tp / PP : NaN,
      fdr: PP ? cm.fp / PP : NaN,
      npv: PN ? cm.tn / PN : NaN,
      forate: PN ? cm.fn / PN : NaN,
      acc: cm.n ? (cm.tp + cm.tn) / cm.n : NaN
    };
  }

  /* The empirical ROC: sweep the threshold from +∞ down past every observed
     score. Equal scores are consumed as one group, which is what keeps the
     curve correct (and the AUC consistent with the rank formula) when ties
     exist. Vertex k carries the threshold that produces it, so dragging the
     dot on the curve can hand a real threshold back to the rest of the page. */
  function rocCurve(d) {
    if (!d.nPos || !d.nNeg) return [{ t: Infinity, fpr: 0, tpr: 0 }];
    var ord = new Int32Array(d.n);
    for (var i = 0; i < d.n; i++) ord[i] = i;
    var s = d.s;
    ord.sort(function (a, b) { return s[b] - s[a]; });

    var pts = [{ t: Infinity, fpr: 0, tpr: 0 }];
    var tp = 0, fp = 0, j = 0;
    while (j < d.n) {
      var v = s[ord[j]];
      while (j < d.n && s[ord[j]] === v) {
        if (d.y[ord[j]]) tp++; else fp++;
        j++;
      }
      pts.push({ t: v, fpr: fp / d.nNeg, tpr: tp / d.nPos });
    }
    return pts;
  }

  /* AUC by the rank (Mann–Whitney U) identity: the probability that a random
     positive outscores a random negative, counting a tie as half. This is the
     same number as the trapezoidal area under `rocCurve`, which the test suite
     checks — computing it independently means the headline number and the
     drawn shape can never quietly disagree. */
  function auc(d) {
    if (!d.nPos || !d.nNeg) return NaN;
    var ord = new Int32Array(d.n);
    for (var i = 0; i < d.n; i++) ord[i] = i;
    var s = d.s;
    ord.sort(function (a, b) { return s[a] - s[b]; });

    var rankSum = 0, j = 0;
    while (j < d.n) {
      var k = j;
      while (k < d.n && s[ord[k]] === s[ord[j]]) k++;
      var avg = (j + k + 1) / 2;            // average 1-based rank of the tie group
      for (var m = j; m < k; m++) if (d.y[ord[m]]) rankSum += avg;
      j = k;
    }
    return (rankSum - d.nPos * (d.nPos + 1) / 2) / (d.nPos * d.nNeg);
  }

  function aucTrapezoid(pts) {
    var a = 0;
    for (var i = 1; i < pts.length; i++) {
      a += (pts[i].fpr - pts[i - 1].fpr) * (pts[i].tpr + pts[i - 1].tpr) / 2;
    }
    return a;
  }

  /* Equal error rate: where FPR = FNR = 1 − TPR, i.e. where the curve crosses
     the anti-diagonal. Linear interpolation inside the straddling segment. */
  function eer(pts) {
    for (var i = 1; i < pts.length; i++) {
      var g0 = pts[i - 1].tpr + pts[i - 1].fpr - 1;
      var g1 = pts[i].tpr + pts[i].fpr - 1;
      if (g0 <= 0 && g1 >= 0) {
        var u = g1 === g0 ? 0 : -g0 / (g1 - g0);
        var fpr = pts[i - 1].fpr + u * (pts[i].fpr - pts[i - 1].fpr);
        var tpr = pts[i - 1].tpr + u * (pts[i].tpr - pts[i - 1].tpr);
        return { fpr: fpr, tpr: tpr, rate: (fpr + (1 - tpr)) / 2 };
      }
    }
    return null;
  }

  /* Precision–recall, from the same sweep. Skipped at k = 0, where nothing is
     predicted positive and precision is 0/0 rather than any particular value. */
  function prCurve(d) {
    if (!d.nPos || !d.nNeg) return [];
    var ord = new Int32Array(d.n);
    for (var i = 0; i < d.n; i++) ord[i] = i;
    var s = d.s;
    ord.sort(function (a, b) { return s[b] - s[a]; });

    var out = [], tp = 0, fp = 0, j = 0;
    while (j < d.n) {
      var v = s[ord[j]];
      while (j < d.n && s[ord[j]] === v) {
        if (d.y[ord[j]]) tp++; else fp++;
        j++;
      }
      out.push({ t: v, recall: tp / d.nPos, prec: tp / (tp + fp) });
    }
    return out;
  }

  /* ---------------------------------------------------- scores to probability */

  /* With equal unit variances the posterior is exactly logistic in the score:
        p(y=1 | s) = σ( sep·s + logit(π) )
     so this page can talk about a score threshold and a probability threshold
     as the same object, and Murphy's 1/(1+c) rule lands on the score axis
     without any approximation. */
  function posterior(s, sep, prev) {
    return sigmoid(sep * s + logit(prev));
  }

  function scoreForProb(p, sep, prev) {
    if (sep <= 0) return p > prev ? Infinity : (p < prev ? -Infinity : 0);
    return (logit(p) - logit(prev)) / sep;
  }

  function probThreshold(c) { return 1 / (1 + c); }

  /* The cost-optimal score threshold, in closed form.
       cost(t) = c·π·FNR(t) + (1−π)·FPR(t),  FNR = Φ(t − sep/2), FPR = 1 − Φ(t + sep/2)
       d/dt = 0  ⇒  c·π·φ(t − sep/2) = (1−π)·φ(t + sep/2)  ⇒  exp(t·sep) = (1−π)/(cπ)
     so t* = (−ln c − logit π) / sep, which is exactly scoreForProb(1/(1+c)).
     The test suite checks both that identity and that a brute-force grid search
     over the population cost lands on the same place. */
  function optScore(c, sep, prev) {
    return scoreForProb(probThreshold(c), sep, prev);
  }

  function popTPR(t, sep) { return 1 - normCdf(t - sep / 2); }
  function popFPR(t, sep) { return 1 - normCdf(t + sep / 2); }

  function popCost(t, sep, prev, c) {
    return c * prev * normCdf(t - sep / 2) + (1 - prev) * (1 - normCdf(t + sep / 2));
  }

  /* Empirical cost per sample, with a false positive costing 1 unit and a
     false negative costing c (Murphy sets ℓ10 = c·ℓ01 and ℓ00 = ℓ11 = 0). */
  function empCost(cm, c) {
    return cm.n ? (c * cm.fn + cm.fp) / cm.n : NaN;
  }

  /* The cheapest threshold actually available on this sample, found by walking
     every ROC vertex. Each vertex already carries its own counts implicitly
     (tp = TPR·P, fp = FPR·N), so this costs one pass over the curve rather
     than a fresh scan of the data per vertex -- it runs on every drag frame. */
  function bestEmpCost(d, pts, c) {
    var best = Infinity, bt = Infinity;
    if (!d.n) return { cost: NaN, t: Infinity };
    for (var i = 0; i < pts.length; i++) {
      var tp = Math.round(pts[i].tpr * d.nPos);
      var fp = Math.round(pts[i].fpr * d.nNeg);
      var v = (c * (d.nPos - tp) + fp) / d.n;
      if (v < best) { best = v; bt = pts[i].t; }
    }
    return { cost: best, t: bt };
  }

  /* Slope of an iso-cost line in (FPR, TPR) coordinates: holding
        cost = c·P·(1 − TPR) + N·FPR
     constant gives dTPR/dFPR = N / (c·P). The ROC curve's own slope at
     threshold t is exp(t·sep), so the two are equal exactly at t* — which is
     what "slide the ruler until it touches" means, and what the suite proves. */
  function isoCostSlope(c, prev) {
    return (1 - prev) / (c * prev);
  }

  var MATH = {
    mulberry32: mulberry32, gaussPair: gaussPair,
    sigmoid: sigmoid, logit: logit, erf: erf, normCdf: normCdf, normPdf: normPdf,
    makeDraws: makeDraws, dataset: dataset,
    confusion: confusion, rates: rates,
    rocCurve: rocCurve, auc: auc, aucTrapezoid: aucTrapezoid, eer: eer, prCurve: prCurve,
    posterior: posterior, scoreForProb: scoreForProb, probThreshold: probThreshold,
    optScore: optScore, popTPR: popTPR, popFPR: popFPR, popCost: popCost,
    empCost: empCost, bestEmpCost: bestEmpCost, isoCostSlope: isoCostSlope
  };

  if (typeof module !== 'undefined' && module.exports) module.exports = MATH;

  /* Node stops here — everything below needs a browser. */
  if (typeof document === 'undefined') return;


  /* ==========================================================================
     PART 2 — STATE
     ====================================================================== */

  var MAXN = 2000;
  var X1 = -5.2, X2 = 5.2;             // the score axis
  var $ = function (id) { return document.getElementById(id); };

  var DEFAULTS = { sep: 2, prev: 0.5, n: 600, c: 1, t: 0 };

  var S = {
    sep: DEFAULTS.sep, prev: DEFAULTS.prev, n: DEFAULTS.n,
    c: DEFAULTS.c, t: DEFAULTS.t,
    seed: 20250930,
    showPts: true, showAUC: true, showEER: true, showCost: true, showLbl: true,
    cmMode: 'count',
    hi: null                            // which anatomy part is lit, if any
  };

  var draws = makeDraws(S.seed, MAXN);
  var D = null, ROC = null, PR = null, AUC = NaN, EER = null;

  function rebuild() {
    D = dataset(draws, S.n, S.prev, S.sep);
    ROC = rocCurve(D);
    PR = prCurve(D);
    AUC = auc(D);
    EER = eer(ROC);
  }

  /* The ROC vertex nearest a given threshold — the dot always sits ON the
     drawn curve rather than floating near it. */
  function vertexFor(t) {
    var best = ROC[0], bd = Infinity;
    for (var i = 0; i < ROC.length; i++) {
      var dt = Math.abs((ROC[i].t === Infinity ? X2 + 1 : ROC[i].t) - t);
      if (dt < bd) { bd = dt; best = ROC[i]; }
    }
    return best;
  }


  /* ==========================================================================
     PART 3 — CANVAS
     ====================================================================== */

  var dist = $('dist'), dctx = dist.getContext('2d');
  var roc = $('roc'), rctx = roc.getContext('2d');
  var prc = $('pr'), pctx = prc.getContext('2d');

  var DW = 0, DH = 0, RW = 0, RH = 0, PW = 0, PH = 0;

  var pal = {};
  function readPalette() {
    var cs = getComputedStyle(document.documentElement);
    var g = function (n) { return cs.getPropertyValue(n).trim(); };
    pal = {
      ink: g('--ink'), ink2: g('--ink-2'), muted: g('--muted'),
      grid: g('--grid'), axis: g('--axis'), line: g('--line'), line2: g('--line-2'),
      panel: g('--panel'), sunken: g('--sunken'), bg: g('--bg'),
      blue: g('--blue'), red: g('--red'), violet: g('--violet'), gold: g('--gold'),
      blueRGB: hex(g('--blue')), redRGB: hex(g('--red')),
      violetRGB: hex(g('--violet')), goldRGB: hex(g('--gold')), inkRGB: hex(g('--ink'))
    };
    hatchBlue = makeHatch(pal.blue);
    hatchRed = makeHatch(pal.red);
  }

  function hex(h) {
    h = h.replace('#', '');
    if (h.length === 3) h = h[0] + h[0] + h[1] + h[1] + h[2] + h[2];
    return [parseInt(h.slice(0, 2), 16), parseInt(h.slice(2, 4), 16), parseInt(h.slice(4, 6), 16)];
  }

  function rgba(c, a) { return 'rgba(' + c[0] + ',' + c[1] + ',' + c[2] + ',' + a + ')'; }

  /* Mistakes are hatched, not just differently coloured. Hue says which class;
     texture says right or wrong. Neither reading depends on the other. */
  var hatchBlue = null, hatchRed = null;
  function makeHatch(color) {
    var c = document.createElement('canvas');
    c.width = c.height = 8;
    var x = c.getContext('2d');
    x.strokeStyle = color;
    x.globalAlpha = 0.55;
    x.lineWidth = 1.4;
    x.beginPath();
    x.moveTo(-2, 10); x.lineTo(10, -2);
    x.moveTo(-2, 18); x.lineTo(18, -2);
    x.moveTo(-10, 10); x.lineTo(10, -10);
    x.stroke();
    return dctx.createPattern(c, 'repeat');
  }

  function fit() {
    var dpr = Math.min(window.devicePixelRatio || 1, 2);

    DW = dist.clientWidth || dist.parentElement.clientWidth;
    DH = Math.round(Math.min(360, Math.max(230, DW * 0.52)));
    dist.style.height = DH + 'px';
    dist.width = Math.round(DW * dpr); dist.height = Math.round(DH * dpr);
    dctx.setTransform(dpr, 0, 0, dpr, 0, 0);

    RW = roc.clientWidth || roc.parentElement.clientWidth;
    RH = Math.round(Math.min(470, Math.max(290, RW * 0.86)));
    roc.style.height = RH + 'px';
    roc.width = Math.round(RW * dpr); roc.height = Math.round(RH * dpr);
    rctx.setTransform(dpr, 0, 0, dpr, 0, 0);

    PW = prc.clientWidth || prc.parentElement.clientWidth;
    PH = Math.round(Math.min(300, Math.max(190, PW * 0.76)));
    prc.style.height = PH + 'px';
    prc.width = Math.round(PW * dpr); prc.height = Math.round(PH * dpr);
    pctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  }


  /* ==========================================================================
     PART 4 — THE SCORE CHART
     Two distributions, one line through them, four regions.
     ====================================================================== */

  var DPAD = { l: 12, r: 12, t: 18, b: 46 };
  var RUG = 30;                          // height of the two sample lanes

  function dx(s) { return DPAD.l + (s - X1) / (X2 - X1) * (DW - DPAD.l - DPAD.r); }
  function dsx(px) { return X1 + (px - DPAD.l) / (DW - DPAD.l - DPAD.r) * (X2 - X1); }

  function drawDist() {
    var ctx = dctx;
    ctx.clearRect(0, 0, DW, DH);

    var base = DH - DPAD.b;              // the score axis
    var top = DPAD.t;
    var peak = normPdf(0);
    // Both curves share a scale so their AREAS stay proportional to their counts:
    // the positive hump really does shrink when positives get rare.
    var big = Math.max(D.nNeg, D.nPos, 1) * peak;
    var hgt = base - top;

    function curveY(s, mu, count) {
      return base - (count * normPdf(s - mu) / big) * hgt;
    }

    var tx = Math.max(DPAD.l, Math.min(DW - DPAD.r, dx(S.t)));

    // grid
    ctx.strokeStyle = pal.grid; ctx.lineWidth = 1;
    for (var g = -5; g <= 5; g++) {
      var gx = dx(g);
      ctx.beginPath(); ctx.moveTo(gx, top - 4); ctx.lineTo(gx, base); ctx.stroke();
    }

    var muN = -S.sep / 2, muP = S.sep / 2;

    // ---- filled regions, drawn correct-first so hatching sits on top
    region(muN, D.nNeg, X1, S.t, rgba(pal.blueRGB, 0.17), null);   // TN
    region(muP, D.nPos, S.t, X2, rgba(pal.redRGB, 0.17), null);    // TP
    region(muN, D.nNeg, S.t, X2, rgba(pal.blueRGB, 0.05), hatchBlue); // FP
    region(muP, D.nPos, X1, S.t, rgba(pal.redRGB, 0.05), hatchRed);   // FN

    function region(mu, count, a, b, fill, pattern) {
      a = Math.max(X1, Math.min(X2, a));
      b = Math.max(X1, Math.min(X2, b));
      if (b <= a) return;
      ctx.beginPath();
      ctx.moveTo(dx(a), base);
      for (var s = a; s <= b + 1e-9; s += (X2 - X1) / 420) {
        ctx.lineTo(dx(s), curveY(Math.min(s, b), mu, count));
      }
      ctx.lineTo(dx(b), base);
      ctx.closePath();
      ctx.fillStyle = fill; ctx.fill();
      if (pattern) { ctx.fillStyle = pattern; ctx.fill(); }
    }

    // ---- outlines
    outline(muN, D.nNeg, pal.blue);
    outline(muP, D.nPos, pal.red);

    function outline(mu, count, color) {
      ctx.beginPath();
      for (var s = X1, first = true; s <= X2 + 1e-9; s += (X2 - X1) / 420) {
        var yy = curveY(s, mu, count);
        if (first) { ctx.moveTo(dx(s), yy); first = false; } else ctx.lineTo(dx(s), yy);
      }
      ctx.strokeStyle = color; ctx.lineWidth = 1.8; ctx.stroke();
    }

    // ---- the samples themselves, in two lanes under the axis
    if (S.showPts) {
      var laneN = base + 11, laneP = base + 24;
      for (var i = 0; i < D.n; i++) {
        var pos = D.y[i], correct = (D.s[i] >= S.t) === (pos === 1);
        var px = dx(Math.max(X1, Math.min(X2, D.s[i])));
        var py = (pos ? laneP : laneN) + ((i % 5) - 2) * 0.9;
        ctx.beginPath();
        ctx.arc(px, py, correct ? 1.5 : 2.1, 0, Math.PI * 2);
        ctx.fillStyle = rgba(pos ? pal.redRGB : pal.blueRGB, correct ? 0.26 : 0.92);
        ctx.fill();
      }
      ctx.font = '10px "IBM Plex Mono", monospace';
      ctx.fillStyle = pal.muted; ctx.textAlign = 'left'; ctx.textBaseline = 'middle';
      ctx.fillText('neg', 3, laneN);
      ctx.fillText('pos', 3, laneP);
    }

    // ---- axis
    ctx.strokeStyle = pal.axis; ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(DPAD.l, base); ctx.lineTo(DW - DPAD.r, base); ctx.stroke();

    ctx.font = '10.5px "IBM Plex Mono", monospace';
    ctx.fillStyle = pal.muted; ctx.textAlign = 'center'; ctx.textBaseline = 'top';
    for (var k = -4; k <= 4; k += 2) ctx.fillText(String(k), dx(k), DH - 15);
    ctx.textAlign = 'right';
    ctx.fillText('score →', DW - DPAD.r, DH - 15);

    // ---- region labels, laid out as the same 2x2 as the matrix card: the
    //      negative row above the positive row, "called 0" left of the line and
    //      "called 1" right of it. Carrying the counts here is what makes the
    //      "shaded area = number of cases" reading concrete.
    ctx.font = '500 11px "IBM Plex Mono", monospace';
    ctx.textBaseline = 'alphabetic';
    var leftS = (X1 + S.t) / 2, rightS = (S.t + X2) / 2;
    label('TN', leftS, 0, pal.blue, cmNow.tn);
    label('FP', rightS, 0, pal.blue, cmNow.fp);
    label('FN', leftS, 1, pal.red, cmNow.fn);
    label('TP', rightS, 1, pal.red, cmNow.tp);

    function label(txt, atS, row, color, count) {
      var lx = dx(Math.max(X1 + 0.3, Math.min(X2 - 0.3, atS)));
      if (Math.abs(lx - tx) < 20) return;          // never crowd the threshold
      if (lx < DPAD.l + 14 || lx > DW - DPAD.r - 14) return;
      ctx.fillStyle = color; ctx.textAlign = 'center';
      ctx.globalAlpha = count > 0 ? 0.95 : 0.3;
      ctx.fillText(txt + '  ' + count, lx, top + 13 + row * 14);
      ctx.globalAlpha = 1;
    }

    // ---- the cost-optimal threshold, as a quiet gold tick
    if (S.showCost) {
      var ts = optScore(S.c, S.sep, S.prev);
      if (isFinite(ts)) {
        var ox = dx(Math.max(X1, Math.min(X2, ts)));
        ctx.save();
        ctx.setLineDash([3, 3]);
        ctx.strokeStyle = pal.gold; ctx.lineWidth = 1.5;
        ctx.beginPath(); ctx.moveTo(ox, top); ctx.lineTo(ox, base + 4); ctx.stroke();
        ctx.restore();
        // Parked down at the axis, clear of the region labels up top.
        ctx.fillStyle = pal.gold; ctx.font = '10px "IBM Plex Mono", monospace';
        ctx.textAlign = ox > DW - 60 ? 'right' : 'left';
        ctx.textBaseline = 'top';
        ctx.fillText(ox > DW - 60 ? 'cost ' : ' cost', ox, base + 1);
      }
    }

    // ---- the threshold itself: the one thing on this page you move
    ctx.strokeStyle = pal.violet; ctx.lineWidth = 2.4;
    ctx.beginPath(); ctx.moveTo(tx, top - 6); ctx.lineTo(tx, base + RUG + 4); ctx.stroke();

    ctx.fillStyle = pal.violet;
    roundRect(ctx, tx - 15, top - 15, 30, 13, 3.5);
    ctx.fill();
    ctx.beginPath();
    ctx.moveTo(tx - 4, top - 2); ctx.lineTo(tx + 4, top - 2); ctx.lineTo(tx, top + 3);
    ctx.closePath(); ctx.fill();

    ctx.fillStyle = pal.panel;
    ctx.font = '500 9.5px "IBM Plex Mono", monospace';
    ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
    ctx.fillText('↔', tx, top - 8.5);
  }

  function roundRect(ctx, x, y, w, h, r) {
    ctx.beginPath();
    ctx.moveTo(x + r, y);
    ctx.arcTo(x + w, y, x + w, y + h, r);
    ctx.arcTo(x + w, y + h, x, y + h, r);
    ctx.arcTo(x, y + h, x, y, r);
    ctx.arcTo(x, y, x + w, y, r);
    ctx.closePath();
  }


  /* ==========================================================================
     PART 5 — THE ROC CHART
     ====================================================================== */

  var RPAD = { l: 44, r: 16, t: 16, b: 40 };

  function rx(f) { return RPAD.l + f * (RW - RPAD.l - RPAD.r); }
  function ry(t) { return RH - RPAD.b - t * (RH - RPAD.t - RPAD.b); }
  function rfx(px) { return (px - RPAD.l) / (RW - RPAD.l - RPAD.r); }
  function rty(py) { return (RH - RPAD.b - py) / (RH - RPAD.t - RPAD.b); }

  /* When a part is lit from the anatomy list, everything else steps back.
     Dimming rather than hiding keeps the shape of the whole chart readable. */
  function em(part) {
    if (!S.hi) return 1;
    return S.hi === part ? 1 : 0.17;
  }

  function drawRoc() {
    var ctx = rctx;
    ctx.clearRect(0, 0, RW, RH);

    var x0 = rx(0), x1 = rx(1), y0 = ry(0), y1 = ry(1);

    // ---- grid
    ctx.globalAlpha = em('axes');
    ctx.strokeStyle = pal.grid; ctx.lineWidth = 1;
    for (var i = 0; i <= 4; i++) {
      var v = i / 4;
      ctx.beginPath(); ctx.moveTo(x0, ry(v)); ctx.lineTo(x1, ry(v)); ctx.stroke();
      ctx.beginPath(); ctx.moveTo(rx(v), y0); ctx.lineTo(rx(v), y1); ctx.stroke();
    }

    // ---- axes + ticks
    ctx.strokeStyle = pal.axis; ctx.lineWidth = 1.2;
    ctx.beginPath(); ctx.moveTo(x0, y1); ctx.lineTo(x0, y0); ctx.lineTo(x1, y0); ctx.stroke();

    ctx.font = '10.5px "IBM Plex Mono", monospace';
    ctx.fillStyle = pal.muted;
    ctx.textAlign = 'center'; ctx.textBaseline = 'top';
    for (i = 0; i <= 4; i++) ctx.fillText((i / 4).toFixed(2), rx(i / 4), y0 + 6);
    ctx.textAlign = 'right'; ctx.textBaseline = 'middle';
    for (i = 0; i <= 4; i++) ctx.fillText((i / 4).toFixed(2), x0 - 7, ry(i / 4));

    ctx.font = '500 11px "IBM Plex Sans", sans-serif';
    ctx.fillStyle = S.hi === 'axes' ? pal.ink : pal.ink2;
    ctx.textAlign = 'center'; ctx.textBaseline = 'alphabetic';
    ctx.fillText('FPR  —  false alarm rate  →', (x0 + x1) / 2, RH - 8);
    ctx.save();
    ctx.translate(13, (y0 + y1) / 2); ctx.rotate(-Math.PI / 2);
    ctx.fillText('TPR  —  positives caught  →', 0, 0);
    ctx.restore();
    ctx.globalAlpha = 1;

    // ---- chance diagonal
    ctx.globalAlpha = em('chance');
    ctx.save();
    ctx.setLineDash([5, 4]);
    ctx.strokeStyle = S.hi === 'chance' ? pal.ink : pal.axis;
    ctx.lineWidth = S.hi === 'chance' ? 2 : 1.3;
    ctx.beginPath(); ctx.moveTo(x0, y0); ctx.lineTo(x1, y1); ctx.stroke();
    ctx.restore();
    if (S.hi === 'chance') {
      ctx.save();
      ctx.translate((x0 + x1) / 2 + 26, (y0 + y1) / 2 + 26);
      ctx.rotate(-Math.atan2(y0 - y1, x1 - x0));
      ctx.fillStyle = pal.ink2; ctx.font = '10.5px "IBM Plex Mono", monospace';
      ctx.textAlign = 'center';
      ctx.fillText('coin flipping', 0, 0);
      ctx.restore();
    }
    ctx.globalAlpha = 1;

    // ---- the EER diagonal, shown only while that part is lit
    if (S.showEER && EER) {
      ctx.globalAlpha = em('eer');
      if (S.hi === 'eer') {
        ctx.save();
        ctx.setLineDash([3, 4]);
        ctx.strokeStyle = pal.muted; ctx.lineWidth = 1.2;
        ctx.beginPath(); ctx.moveTo(x0, y1); ctx.lineTo(x1, y0); ctx.stroke();
        ctx.restore();
      }
      ctx.globalAlpha = 1;
    }

    // ---- area under the curve
    if (S.showAUC) {
      ctx.globalAlpha = em('auc');
      ctx.beginPath();
      ctx.moveTo(rx(0), ry(0));
      for (i = 0; i < ROC.length; i++) ctx.lineTo(rx(ROC[i].fpr), ry(ROC[i].tpr));
      ctx.lineTo(rx(1), ry(0));
      ctx.closePath();
      ctx.fillStyle = rgba(pal.violetRGB, S.hi === 'auc' ? 0.26 : 0.13);
      ctx.fill();
      ctx.globalAlpha = 1;
    }

    // ---- the curve
    ctx.globalAlpha = em('curve');
    ctx.beginPath();
    for (i = 0; i < ROC.length; i++) {
      var px = rx(ROC[i].fpr), py = ry(ROC[i].tpr);
      if (i === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py);
    }
    ctx.strokeStyle = pal.ink;
    ctx.lineWidth = S.hi === 'curve' ? 3.2 : 2;
    ctx.lineJoin = 'round';
    ctx.stroke();
    ctx.globalAlpha = 1;

    // ---- the cost ruler and where it touches
    if (S.showCost) {
      ctx.globalAlpha = em('cost');
      var m = isoCostSlope(S.c, S.prev);
      var ts = optScore(S.c, S.sep, S.prev);
      var ov = vertexFor(isFinite(ts) ? ts : (ts > 0 ? X2 + 1 : X1 - 1));

      // the line through the tangent point, clipped to the unit box
      var bx = ov.fpr, by = ov.tpr;
      var seg = clipLine(bx, by, m);
      if (seg) {
        ctx.save();
        ctx.setLineDash([6, 4]);
        ctx.strokeStyle = pal.gold;
        ctx.lineWidth = S.hi === 'cost' ? 2.4 : 1.6;
        ctx.beginPath();
        ctx.moveTo(rx(seg[0]), ry(seg[1]));
        ctx.lineTo(rx(seg[2]), ry(seg[3]));
        ctx.stroke();
        ctx.restore();
      }

      ctx.beginPath();
      var gx = rx(ov.fpr), gy = ry(ov.tpr), hr = 5.5;
      ctx.moveTo(gx, gy - hr); ctx.lineTo(gx + hr, gy);
      ctx.lineTo(gx, gy + hr); ctx.lineTo(gx - hr, gy);
      ctx.closePath();
      ctx.fillStyle = pal.panel; ctx.fill();
      ctx.strokeStyle = pal.gold; ctx.lineWidth = 2; ctx.stroke();
      ctx.globalAlpha = 1;
    }

    // ---- equal error rate
    if (S.showEER && EER) {
      ctx.globalAlpha = em('eer');
      ctx.beginPath();
      ctx.arc(rx(EER.fpr), ry(EER.tpr), S.hi === 'eer' ? 6 : 4.5, 0, Math.PI * 2);
      ctx.fillStyle = pal.panel; ctx.fill();
      ctx.strokeStyle = pal.muted; ctx.lineWidth = 2; ctx.stroke();
      ctx.globalAlpha = 1;
    }

    // ---- corner marks
    if (S.showLbl) {
      corner('origin', 0, 0, 'threshold at its highest', 'left', 'bottom');
      corner('corner', 1, 1, 'threshold at its lowest', 'right', 'top');
      corner('perfect', 0, 1, 'perfect', 'left', 'top');
    }

    function corner(part, f, t, text, ha, va) {
      var lit = S.hi === part;
      ctx.globalAlpha = em(part);
      var cx = rx(f), cy = ry(t);
      ctx.beginPath();
      ctx.arc(cx, cy, lit ? 6.5 : 3.6, 0, Math.PI * 2);
      ctx.fillStyle = lit ? pal.ink : pal.axis;
      ctx.fill();
      if (lit) {
        ctx.font = '500 11px "IBM Plex Mono", monospace';
        ctx.fillStyle = pal.ink;
        ctx.textAlign = ha === 'left' ? 'left' : 'right';
        ctx.textBaseline = va === 'top' ? 'top' : 'bottom';
        var ox = (ha === 'left' ? 11 : -11), oy = (va === 'top' ? 11 : -11);
        ctx.fillText('(' + f + ', ' + t + ')  ' + text, cx + ox, cy + oy);
      }
      ctx.globalAlpha = 1;
    }

    // ---- the operating point: always last, always on top
    var v = vertexFor(S.t);
    ctx.globalAlpha = em('op');
    var ax = rx(v.fpr), ay = ry(v.tpr);
    ctx.beginPath(); ctx.arc(ax, ay, 11, 0, Math.PI * 2);
    ctx.fillStyle = rgba(pal.violetRGB, 0.22); ctx.fill();
    ctx.beginPath(); ctx.arc(ax, ay, 6.2, 0, Math.PI * 2);
    ctx.fillStyle = pal.violet; ctx.fill();
    ctx.strokeStyle = pal.panel; ctx.lineWidth = 2; ctx.stroke();
    ctx.globalAlpha = 1;
  }

  /* Clip the line through (bx,by) of slope m to the unit square, so the gold
     ruler spans the plot instead of stopping at the tangent point. */
  function clipLine(bx, by, m) {
    if (!isFinite(m)) return [bx, 0, bx, 1];
    var pts = [];
    var yAt0 = by + m * (0 - bx), yAt1 = by + m * (1 - bx);
    if (yAt0 >= 0 && yAt0 <= 1) pts.push([0, yAt0]);
    if (yAt1 >= 0 && yAt1 <= 1) pts.push([1, yAt1]);
    if (m !== 0) {
      var xAt0 = bx + (0 - by) / m, xAt1 = bx + (1 - by) / m;
      if (xAt0 > 0 && xAt0 < 1) pts.push([xAt0, 0]);
      if (xAt1 > 0 && xAt1 < 1) pts.push([xAt1, 1]);
    }
    if (pts.length < 2) return null;
    return [pts[0][0], pts[0][1], pts[1][0], pts[1][1]];
  }


  /* ==========================================================================
     PART 6 — THE PR CHART
     ====================================================================== */

  var PPAD = { l: 40, r: 14, t: 14, b: 34 };

  function px_(r) { return PPAD.l + r * (PW - PPAD.l - PPAD.r); }
  function py_(p) { return PH - PPAD.b - p * (PH - PPAD.t - PPAD.b); }

  function drawPr() {
    var ctx = pctx;
    ctx.clearRect(0, 0, PW, PH);

    var x0 = px_(0), x1 = px_(1), y0 = py_(0), y1 = py_(1);

    ctx.strokeStyle = pal.grid; ctx.lineWidth = 1;
    for (var i = 0; i <= 4; i++) {
      ctx.beginPath(); ctx.moveTo(x0, py_(i / 4)); ctx.lineTo(x1, py_(i / 4)); ctx.stroke();
      ctx.beginPath(); ctx.moveTo(px_(i / 4), y0); ctx.lineTo(px_(i / 4), y1); ctx.stroke();
    }

    ctx.strokeStyle = pal.axis; ctx.lineWidth = 1.2;
    ctx.beginPath(); ctx.moveTo(x0, y1); ctx.lineTo(x0, y0); ctx.lineTo(x1, y0); ctx.stroke();

    ctx.font = '10.5px "IBM Plex Mono", monospace';
    ctx.fillStyle = pal.muted;
    ctx.textAlign = 'center'; ctx.textBaseline = 'top';
    for (i = 0; i <= 4; i += 2) ctx.fillText((i / 4).toFixed(1), px_(i / 4), y0 + 5);
    ctx.textAlign = 'right'; ctx.textBaseline = 'middle';
    for (i = 0; i <= 4; i += 2) ctx.fillText((i / 4).toFixed(1), x0 - 6, py_(i / 4));

    ctx.font = '500 11px "IBM Plex Sans", sans-serif';
    ctx.fillStyle = pal.ink2;
    ctx.textAlign = 'center'; ctx.textBaseline = 'alphabetic';
    ctx.fillText('recall →', (x0 + x1) / 2, PH - 7);
    ctx.save();
    ctx.translate(12, (y0 + y1) / 2); ctx.rotate(-Math.PI / 2);
    ctx.fillText('precision →', 0, 0);
    ctx.restore();

    // the floor a coin flip would reach: precision = base rate
    var base = D.n ? D.nPos / D.n : 0;
    ctx.save();
    ctx.setLineDash([5, 4]);
    ctx.strokeStyle = pal.axis; ctx.lineWidth = 1.3;
    ctx.beginPath(); ctx.moveTo(x0, py_(base)); ctx.lineTo(x1, py_(base)); ctx.stroke();
    ctx.restore();
    ctx.font = '10px "IBM Plex Mono", monospace';
    ctx.fillStyle = pal.muted; ctx.textAlign = 'left'; ctx.textBaseline = 'bottom';
    ctx.fillText('chance = base rate', x0 + 5, py_(base) - 3);

    if (!PR.length) return;

    ctx.beginPath();
    for (i = 0; i < PR.length; i++) {
      var qx = px_(PR[i].recall), qy = py_(PR[i].prec);
      if (i === 0) ctx.moveTo(qx, qy); else ctx.lineTo(qx, qy);
    }
    ctx.strokeStyle = pal.ink; ctx.lineWidth = 2; ctx.lineJoin = 'round';
    ctx.stroke();

    // the same operating point as everywhere else
    var r = rates(cmNow);
    if (!isNaN(r.prec) && !isNaN(r.tpr)) {
      var ax = px_(r.tpr), ay = py_(r.prec);
      ctx.beginPath(); ctx.arc(ax, ay, 9, 0, Math.PI * 2);
      ctx.fillStyle = rgba(pal.violetRGB, 0.22); ctx.fill();
      ctx.beginPath(); ctx.arc(ax, ay, 5, 0, Math.PI * 2);
      ctx.fillStyle = pal.violet; ctx.fill();
      ctx.strokeStyle = pal.panel; ctx.lineWidth = 2; ctx.stroke();
    }
  }


  /* ==========================================================================
     PART 7 — READOUTS
     ====================================================================== */

  var cmNow = { tn: 0, fp: 0, fn: 0, tp: 0, n: 0 };

  function f2(v) { return isNaN(v) ? '—' : v.toFixed(2); }
  function f3(v) { return isNaN(v) ? '—' : v.toFixed(3); }
  function pct(v) { return isNaN(v) ? '—' : (v * 100).toFixed(1) + '%'; }

  var CM_NOTE = {
    count: 'Counts, straight from the samples — Murphy’s Table 5.3.',
    row: 'Each row divided by its own total: p(ŷ | y), Table 5.4. The top row gives TNR and FPR, the bottom row FNR and TPR. Nothing here depends on how common each class is.',
    col: 'Each column divided by its own total: p(y | ŷ), Table 5.5. The right column gives FDR and precision. These DO depend on how common each class is.'
  };

  function cellText(which, cm) {
    var P = cm.tp + cm.fn, N = cm.fp + cm.tn;
    var PP = cm.tp + cm.fp, PN = cm.tn + cm.fn;
    if (S.cmMode === 'count') return String(cm[which]);
    if (S.cmMode === 'row') {
      var rd = (which === 'tn' || which === 'fp') ? N : P;
      return rd ? (cm[which] / rd).toFixed(2) : '—';
    }
    var cd = (which === 'tn' || which === 'fn') ? PN : PP;
    return cd ? (cm[which] / cd).toFixed(2) : '—';
  }

  var CELL_SUB = {
    count: { tn: 'correctly cleared', fp: 'false alarms', fn: 'missed', tp: 'caught' },
    row:   { tn: 'TNR · specificity', fp: 'FPR · type I', fn: 'FNR · miss rate', tp: 'TPR · recall' },
    col:   { tn: 'NPV', fp: 'FDR', fn: 'FOR', tp: 'precision · PPV' }
  };

  function syncReadouts() {
    cmNow = confusion(D, S.t);
    var r = rates(cmNow);

    ['tn', 'fp', 'fn', 'tp'].forEach(function (k) {
      $('c' + k.toUpperCase()).textContent = cellText(k, cmNow);
      $('s' + k.toUpperCase()).textContent = CELL_SUB[S.cmMode][k];
    });
    $('cmNote').textContent = CM_NOTE[S.cmMode];

    $('rTPR').textContent = f2(r.tpr);
    $('rFPR').textContent = f2(r.fpr);
    $('rPREC').textContent = f2(r.prec);
    $('rACC').textContent = f2(r.acc);
    $('rAUC').textContent = f3(AUC);
    $('rEER').textContent = EER ? f2(EER.rate) : '—';

    // threshold strip
    var p = posterior(S.t, S.sep, S.prev);
    $('thrRead').textContent = 's ≥ ' + S.t.toFixed(2);
    $('thrProb').textContent = S.sep > 0.01
      ? 'p(y=1 | x) ≥ ' + p.toFixed(3)
      : 'nothing at all — with no separation the score carries no information';

    $('opVerdict').innerHTML = verdictFor(r);

    // cost panel
    var pOpt = probThreshold(S.c);
    var tOpt = optScore(S.c, S.sep, S.prev);
    var costNow = empCost(cmNow, S.c);
    var best = bestEmpCost(D, ROC, S.c);

    $('costC').textContent = S.c.toFixed(S.c < 1 ? 2 : 1);
    $('costThrP').textContent = pOpt.toFixed(3);
    $('fThrP').textContent = S.sep > 0.01 ? p.toFixed(3) : '—';
    $('fOptP').textContent = pOpt.toFixed(3);
    $('fCostNow').textContent = f3(costNow);
    $('fCostBest').textContent = f3(best.cost);
    $('fCostCmp').textContent = costNow <= best.cost + 1e-12
      ? 'you are at the cheapest point'
      : (((costNow / Math.max(best.cost, 1e-9)) - 1) * 100).toFixed(0) + '% above the best';
    $('fThr').classList.toggle('lit', Math.abs(S.t - tOpt) < 0.06);
    $('slopeTxt').textContent = isoCostSlope(S.c, S.prev).toFixed(2);
    $('costVerdict').innerHTML = costVerdict(tOpt, pOpt);

    // imbalance panel
    $('iPrev').textContent = pct(D.n ? D.nPos / D.n : NaN);
    $('iCounts').textContent = D.nPos + ' positive of ' + D.n;
    $('iAUC').textContent = f3(AUC);
    $('iPrec').textContent = f2(r.prec);
    $('iBase').textContent = f2(D.n ? D.nPos / D.n : NaN);

    $('live').textContent =
      'Threshold ' + S.t.toFixed(2) + '. TPR ' + f2(r.tpr) + ', FPR ' + f2(r.fpr) +
      ', precision ' + f2(r.prec) + '. AUC ' + f3(AUC) + '.';
  }

  /* A sentence about where the dot is, in the language a person would use. */
  function verdictFor(r) {
    if (isNaN(r.tpr) || isNaN(r.fpr)) {
      return 'Not enough of one class in this sample to say anything. Add samples, or even out the base rate.';
    }
    if (r.tpr < 0.03) {
      return '<b>You are refusing to call anything positive.</b> No false alarms, but you caught '
        + cmNow.tp + ' of ' + (cmNow.tp + cmNow.fn) + '. This is the bottom-left corner.';
    }
    if (r.fpr > 0.97) {
      return '<b>You are calling almost everything positive.</b> You caught nearly all of them, '
        + 'and falsely flagged ' + cmNow.fp + ' innocent cases to do it. This is the top-right corner.';
    }
    var miss = 1 - r.tpr;
    return 'You catch <b>' + pct(r.tpr) + '</b> of the real positives, miss <b>' + pct(miss)
      + '</b> of them, and falsely flag <b>' + pct(r.fpr) + '</b> of the negatives. '
      + 'Of everything you flagged, <b>' + pct(r.prec) + '</b> was right.';
  }

  function costVerdict(tOpt, pOpt) {
    if (S.sep <= 0.01) {
      return '<b>With no separation there is nothing to threshold.</b> The score says nothing, so the '
        + 'cheapest policy is to call every case the same way &mdash; whichever way costs less.';
    }
    var dir = S.c > 1 ? 'lower' : (S.c < 1 ? 'higher' : 'exactly 0.5');
    var txt = S.c === 1
      ? '<b>Both mistakes cost the same</b>, so the optimal cut-off is 0.5 and the MAP rule &mdash; '
        + 'pick the most probable class &mdash; is the right one.'
      : '<b>A miss costs ' + S.c.toFixed(S.c < 1 ? 2 : 1) + '× a false alarm</b>, so the optimal '
        + 'cut-off drops to ' + pOpt.toFixed(3) + ' &mdash; ' + dir + ' than 0.5. You should declare '
        + (S.c > 1 ? 'positive on weaker evidence than MAP would.' : 'positive only on stronger evidence than MAP would.');
    return txt + ' On the score axis that is <span class="mono">s = '
      + (isFinite(tOpt) ? tOpt.toFixed(2) : (tOpt > 0 ? '+∞' : '−∞')) + '</span>.';
  }


  /* ==========================================================================
     PART 8 — CONTROLS
     ====================================================================== */

  function render() {
    syncReadouts();     // cmNow is used by both charts, so it goes first
    drawDist();
    drawRoc();
    drawPr();
  }

  function recompute() { rebuild(); render(); }

  function setThreshold(t, fromSlider) {
    S.t = Math.max(X1, Math.min(X2, t));
    if (!fromSlider) $('thr').value = String(S.t);
    render();
  }

  // ---- cost slider is log-scaled, so 1:1 sits dead centre
  function costLabel(c) {
    if (Math.abs(c - 1) < 0.005) return '1 : 1';
    return c > 1 ? c.toFixed(c < 10 ? 1 : 0) + ' : 1' : '1 : ' + (1 / c).toFixed(1 / c < 10 ? 1 : 0);
  }

  function bindRange(id, outId, fmt, apply) {
    var el = $(id), out = $(outId);
    el.addEventListener('input', function () {
      apply(parseFloat(el.value));
      out.textContent = fmt();
    });
    out.textContent = fmt();
  }

  bindRange('sep', 'sepv', function () { return S.sep.toFixed(2); }, function (v) {
    S.sep = v; recompute();
  });
  bindRange('prev', 'prevv', function () { return Math.round(S.prev * 100) + '%'; }, function (v) {
    S.prev = v; recompute();
  });
  bindRange('npts', 'nptsv', function () { return String(S.n); }, function (v) {
    S.n = Math.round(v); recompute();
  });
  bindRange('cost', 'costv', function () { return costLabel(S.c); }, function (v) {
    S.c = Math.pow(10, v); render();
  });

  $('thr').addEventListener('input', function () {
    setThreshold(parseFloat(this.value), true);
  });

  $('cmMode').addEventListener('change', function () { S.cmMode = this.value; render(); });

  [['showPts', 'showPts'], ['showAUC', 'showAUC'], ['showEER', 'showEER'],
   ['showCost', 'showCost'], ['showLbl', 'showLbl']].forEach(function (p) {
    $(p[0]).addEventListener('change', function () { S[p[1]] = this.checked; render(); });
  });

  $('goOpt').addEventListener('click', function () {
    var t = optScore(S.c, S.sep, S.prev);
    setThreshold(isFinite(t) ? t : (t > 0 ? X2 : X1));
  });

  $('gen').addEventListener('click', function () {
    S.seed = (Date.now() & 0x7fffffff) >>> 0;
    draws = makeDraws(S.seed, MAXN);
    recompute();
  });

  $('reset').addEventListener('click', function () {
    S.sep = DEFAULTS.sep; S.prev = DEFAULTS.prev; S.n = DEFAULTS.n;
    S.c = DEFAULTS.c; S.t = DEFAULTS.t;
    S.cmMode = 'count';
    S.showPts = S.showAUC = S.showEER = S.showCost = S.showLbl = true;
    syncControls();
    recompute();
  });

  function syncControls() {
    $('sep').value = String(S.sep); $('sepv').textContent = S.sep.toFixed(2);
    $('prev').value = String(S.prev); $('prevv').textContent = Math.round(S.prev * 100) + '%';
    $('npts').value = String(S.n); $('nptsv').textContent = String(S.n);
    $('cost').value = String(Math.log(S.c) / Math.LN10); $('costv').textContent = costLabel(S.c);
    $('thr').value = String(S.t);
    $('cmMode').value = S.cmMode;
    $('showPts').checked = S.showPts; $('showAUC').checked = S.showAUC;
    $('showEER').checked = S.showEER; $('showCost').checked = S.showCost;
    $('showLbl').checked = S.showLbl;
  }

  // ---- anatomy list: hover or focus lights the matching part of the chart
  Array.prototype.forEach.call($('parts').querySelectorAll('button'), function (b) {
    var part = b.getAttribute('data-part');
    function on() { S.hi = part; markParts(); drawRoc(); }
    function off() { S.hi = null; markParts(); drawRoc(); }
    b.addEventListener('pointerenter', on);
    b.addEventListener('pointerleave', off);
    b.addEventListener('focus', on);
    b.addEventListener('blur', off);
    b.addEventListener('click', function () {
      S.hi = (S.hi === part) ? null : part; markParts(); drawRoc();
    });
  });

  function markParts() {
    Array.prototype.forEach.call($('parts').querySelectorAll('button'), function (b) {
      b.classList.toggle('on', S.hi === b.getAttribute('data-part'));
    });
  }

  // ---- walkthrough
  var STEPS = [
    { note: '<b>Push the threshold all the way right.</b> Nothing clears the bar, so nothing gets '
          + 'called positive: no false alarms, no catches. The dot is pinned at (0,&nbsp;0) &mdash; and '
          + 'notice the accuracy is still high, which is exactly why accuracy is a poor thing to trust.',
      set: function () { S.t = X2; } },
    { note: '<b>Now all the way left.</b> Everything clears the bar, so everything is called positive. '
          + 'You catch every single positive &mdash; and flag every negative too. The dot is at '
          + '(1,&nbsp;1). Both corners are free; the whole skill is in between.',
      set: function () { S.t = X1; } },
    { note: '<b>Separation down to 0.3.</b> The two humps now sit almost on top of each other, so no '
          + 'threshold separates them and the curve flattens onto the diagonal. AUC falls to about 0.5. '
          + 'This is what a worthless classifier looks like &mdash; and no threshold can rescue it.',
      set: function () { S.sep = 0.3; S.t = 0; } },
    { note: '<b>A miss now costs 10&times; a false alarm.</b> The gold cut-off slides left: you should '
          + 'declare positive on much weaker evidence. Hit &ldquo;snap the line to the optimum&rdquo; and '
          + 'watch the dot slide along a curve that never moved. The model didn’t change &mdash; your '
          + 'loss function did.',
      set: function () { S.sep = 2; S.c = 10; S.prev = 0.5; } },
    { note: '<b>Only 3&#37; of cases are positive now.</b> Look hard at the ROC curve: it barely '
          + 'flinched, because TPR and FPR each divide inside one true class. Now look at precision and '
          + 'the PR curve below. <b>Same model, same threshold, and most of what you flag is wrong.</b>',
      set: function () { S.sep = 2; S.c = 1; S.prev = 0.03; S.n = 2000; S.t = 0; } }
  ];

  Array.prototype.forEach.call(document.querySelectorAll('.steps button'), function (b) {
    b.addEventListener('click', function () {
      var i = +b.getAttribute('data-step');
      STEPS[i].set();
      syncControls();
      recompute();
      $('stepNote').innerHTML = STEPS[i].note;
      Array.prototype.forEach.call(document.querySelectorAll('.steps button'), function (o) {
        o.classList.toggle('on', o === b);
      });
    });
  });

  // ---- tooltips
  var tip = $('tip');
  function showTip(btn) {
    tip.textContent = btn.getAttribute('data-tip');
    tip.hidden = false;
    var r = btn.getBoundingClientRect();
    var tr = tip.getBoundingClientRect();
    var left = Math.min(window.innerWidth - tr.width - 10, Math.max(10, r.left - tr.width / 2 + r.width / 2));
    var top = r.top - tr.height - 9;
    if (top < 8) top = r.bottom + 9;
    tip.style.left = left + 'px';
    tip.style.top = top + 'px';
  }
  function hideTip() { tip.hidden = true; }

  Array.prototype.forEach.call(document.querySelectorAll('.q'), function (b) {
    b.addEventListener('pointerenter', function () { showTip(b); });
    b.addEventListener('pointerleave', hideTip);
    b.addEventListener('focus', function () { showTip(b); });
    b.addEventListener('blur', hideTip);
    b.addEventListener('click', function (e) { e.preventDefault(); });
  });


  /* ==========================================================================
     PART 9 — INTERACTION
     ====================================================================== */

  // ---- drag the threshold line
  var dragDist = false;
  dist.addEventListener('pointerdown', function (ev) {
    dragDist = true;
    dist.setPointerCapture(ev.pointerId);
    dist.classList.add('grabbing');
    setThreshold(dsx(ev.clientX - dist.getBoundingClientRect().left));
  });
  dist.addEventListener('pointermove', function (ev) {
    if (!dragDist) return;
    setThreshold(dsx(ev.clientX - dist.getBoundingClientRect().left));
  });
  function endDist(ev) {
    if (!dragDist) return;
    dragDist = false;
    dist.classList.remove('grabbing');
    if (ev && ev.pointerId !== undefined && dist.hasPointerCapture(ev.pointerId)) {
      dist.releasePointerCapture(ev.pointerId);
    }
  }
  dist.addEventListener('pointerup', endDist);
  dist.addEventListener('pointercancel', endDist);

  dist.addEventListener('keydown', function (ev) {
    var step = ev.shiftKey ? 0.4 : 0.08;
    if (ev.key === 'ArrowLeft') { setThreshold(S.t - step); ev.preventDefault(); }
    if (ev.key === 'ArrowRight') { setThreshold(S.t + step); ev.preventDefault(); }
  });

  // ---- drag the operating point: snap to the nearest vertex of the curve,
  //      then hand that vertex's threshold back to everything else
  var dragRoc = false;
  function rocPick(ev) {
    var r = roc.getBoundingClientRect();
    var f = rfx(ev.clientX - r.left), t = rty(ev.clientY - r.top);
    var best = null, bd = Infinity;
    for (var i = 0; i < ROC.length; i++) {
      var dd = (ROC[i].fpr - f) * (ROC[i].fpr - f) + (ROC[i].tpr - t) * (ROC[i].tpr - t);
      if (dd < bd) { bd = dd; best = ROC[i]; }
    }
    if (best) setThreshold(best.t === Infinity ? X2 : best.t);
  }
  roc.addEventListener('pointerdown', function (ev) {
    dragRoc = true;
    roc.setPointerCapture(ev.pointerId);
    roc.classList.add('grabbing');
    rocPick(ev);
  });
  roc.addEventListener('pointermove', function (ev) { if (dragRoc) rocPick(ev); });
  function endRoc(ev) {
    if (!dragRoc) return;
    dragRoc = false;
    roc.classList.remove('grabbing');
    if (ev && ev.pointerId !== undefined && roc.hasPointerCapture(ev.pointerId)) {
      roc.releasePointerCapture(ev.pointerId);
    }
  }
  roc.addEventListener('pointerup', endRoc);
  roc.addEventListener('pointercancel', endRoc);

  roc.addEventListener('keydown', function (ev) {
    var step = ev.shiftKey ? 0.4 : 0.08;
    if (ev.key === 'ArrowLeft' || ev.key === 'ArrowDown') { setThreshold(S.t + step); ev.preventDefault(); }
    if (ev.key === 'ArrowRight' || ev.key === 'ArrowUp') { setThreshold(S.t - step); ev.preventDefault(); }
  });


  /* ==========================================================================
     BOOT
     ====================================================================== */

  function boot() {
    readPalette();
    fit();
    syncControls();
    recompute();
  }

  var ro = new ResizeObserver(function () { fit(); render(); });
  ro.observe(dist.parentElement);
  ro.observe(roc.parentElement);
  ro.observe(prc.parentElement);

  var mq = window.matchMedia('(prefers-color-scheme: dark)');
  (mq.addEventListener ? mq.addEventListener.bind(mq, 'change') : mq.addListener.bind(mq))(function () {
    readPalette(); render();
  });

  if (document.fonts && document.fonts.ready) document.fonts.ready.then(render);
  boot();

})(typeof globalThis !== 'undefined' ? globalThis : this);
