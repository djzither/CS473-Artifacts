/* ============================================================================
   Math checks for the ROC module.   Run:  node test-math.js

   This exercises the SHIPPED script.js, not a copy of its formulas — the file
   exports its pure-math half under Node and skips everything DOM-bound, so a
   green run here is a statement about the code the browser actually executes.
   ========================================================================= */

var M = require('./script.js');

var pass = 0, fail = 0;

function ok(name, cond, detail) {
  if (cond) { pass++; console.log('  PASS  ' + name); }
  else { fail++; console.log('  FAIL  ' + name + (detail ? '   -> ' + detail : '')); }
}

function near(a, b, tol) { return Math.abs(a - b) <= tol; }

function section(t) { console.log('\n' + t); }

var rnd = M.mulberry32(473473);
function u(lo, hi) { return lo + (hi - lo) * rnd(); }

/* A standard sample to hammer invariants with. */
function sample(n, prev, sep, seed) {
  return M.dataset(M.makeDraws(seed || 7777, n), n, prev, sep);
}

function randomCase() {
  var n = Math.round(u(120, 700));
  return { n: n, prev: u(0.12, 0.88), sep: u(0.2, 3.6), seed: Math.round(u(1, 1e9)) };
}


/* -------------------------------------------------------- 1. the matrix ---- */
section('1. Confusion matrix (Table 5.3)');
(function () {
  var bad = 0, badEnds = 0;
  for (var i = 0; i < 300; i++) {
    var p = randomCase();
    var d = sample(p.n, p.prev, p.sep, p.seed);
    for (var k = 0; k < 6; k++) {
      var cm = M.confusion(d, u(-5, 5));
      if (cm.tn + cm.fp + cm.fn + cm.tp !== d.n) bad++;
      if (cm.tp + cm.fn !== d.nPos || cm.fp + cm.tn !== d.nNeg) bad++;
    }
    var lo = M.confusion(d, -Infinity), hi = M.confusion(d, Infinity);
    if (lo.tp !== d.nPos || lo.fp !== d.nNeg || lo.tn !== 0 || lo.fn !== 0) badEnds++;
    if (hi.tn !== d.nNeg || hi.fn !== d.nPos || hi.tp !== 0 || hi.fp !== 0) badEnds++;
  }
  ok('the four cells always sum to n, and the row totals are the class counts', bad === 0,
     bad + ' violations');
  ok('threshold at -inf calls everything positive; at +inf, everything negative', badEnds === 0,
     badEnds + ' violations');
})();


/* -------------------------------------------------------- 2. the rates ----- */
section('2. Rate identities (Tables 5.4 and 5.5)');
(function () {
  var worst = 0, undef = 0, mismatched = 0;

  /* Precision and NPV are genuinely undefined when nothing is predicted into
     that column -- 0/0, not a bug -- so the identity is asserted where it is
     defined, and the undefined cases are checked to be exactly those. */
  function pair(a, b, cm, columnIsEmpty) {
    if (isNaN(a) || isNaN(b)) {
      undef++;
      if (!columnIsEmpty) mismatched++;
      return;
    }
    if (columnIsEmpty) mismatched++;
    worst = Math.max(worst, Math.abs(a - (1 - b)));
  }

  for (var i = 0; i < 400; i++) {
    var p = randomCase();
    var d = sample(p.n, p.prev, p.sep, p.seed);
    var cm = M.confusion(d, u(-4, 4));
    var r = M.rates(cm);
    worst = Math.max(worst,
      Math.abs(r.fnr - (1 - r.tpr)),
      Math.abs(r.fpr - (1 - r.tnr)));
    pair(r.fdr, r.prec, cm, cm.tp + cm.fp === 0);
    pair(r.forate, r.npv, cm, cm.tn + cm.fn === 0);
  }
  ok('FNR=1-TPR, FPR=1-TNR, FDR=1-precision, FOR=1-NPV', worst < 1e-12,
     'worst deviation ' + worst.toExponential(2));
  ok('a rate is NaN exactly when its column is empty (' + undef + ' such cases)',
     mismatched === 0, mismatched + ' mismatches');

  var bad = 0;
  for (i = 0; i < 300; i++) {
    var q = randomCase();
    var dd = sample(q.n, q.prev, q.sep, q.seed);
    var rr = M.rates(M.confusion(dd, u(-4, 4)));
    ['tpr', 'fpr', 'tnr', 'fnr', 'prec', 'npv', 'acc'].forEach(function (k) {
      if (!isNaN(rr[k]) && (rr[k] < 0 || rr[k] > 1)) bad++;
    });
  }
  ok('every rate stays inside [0, 1]', bad === 0, bad + ' violations');
})();


/* ----------------------------------------------------- 3. the ROC curve ---- */
section('3. The ROC curve');
(function () {
  var badMono = 0, badEnds = 0;
  for (var i = 0; i < 200; i++) {
    var p = randomCase();
    var d = sample(p.n, p.prev, p.sep, p.seed);
    var pts = M.rocCurve(d);
    for (var j = 1; j < pts.length; j++) {
      if (pts[j].fpr < pts[j - 1].fpr - 1e-12 || pts[j].tpr < pts[j - 1].tpr - 1e-12) badMono++;
    }
    var a = pts[0], z = pts[pts.length - 1];
    if (a.fpr !== 0 || a.tpr !== 0) badEnds++;
    if (!near(z.fpr, 1, 1e-12) || !near(z.tpr, 1, 1e-12)) badEnds++;
  }
  ok('the curve never moves left or down, over 200 datasets', badMono === 0,
     badMono + ' backward steps');
  ok('it starts exactly at (0,0) and ends exactly at (1,1)', badEnds === 0,
     badEnds + ' bad endpoints');
})();

/* Each vertex claims a threshold. Re-deriving the matrix from that threshold
   has to reproduce the vertex, or the dot the user drags would be lying about
   which confusion matrix it stands for. */
(function () {
  var worst = 0, checked = 0;
  for (var i = 0; i < 25; i++) {
    var p = randomCase(); p.n = 400;
    var d = sample(p.n, p.prev, p.sep, p.seed);
    var pts = M.rocCurve(d);
    for (var j = 0; j < pts.length; j++) {
      var cm = M.confusion(d, pts[j].t);
      var r = M.rates(cm);
      worst = Math.max(worst, Math.abs(r.tpr - pts[j].tpr), Math.abs(r.fpr - pts[j].fpr));
      checked++;
    }
  }
  ok('every vertex reproduces its own confusion matrix (' + checked + ' vertices)',
     worst < 1e-12, 'worst deviation ' + worst.toExponential(2));
})();


/* ------------------------------------------------------------- 4. AUC ----- */
section('4. AUC');
(function () {
  var worst = 0;
  for (var i = 0; i < 250; i++) {
    var p = randomCase();
    var d = sample(p.n, p.prev, p.sep, p.seed);
    var byRank = M.auc(d);
    var byArea = M.aucTrapezoid(M.rocCurve(d));
    worst = Math.max(worst, Math.abs(byRank - byArea));
  }
  ok('the rank formula and the area under the drawn curve agree', worst < 1e-12,
     'worst deviation ' + worst.toExponential(2));
})();

/* The binormal truth: with negatives ~ N(-d/2,1) and positives ~ N(+d/2,1),
   P(positive outscores negative) = Phi(d / sqrt(2)). */
(function () {
  var worst = 0, rows = [];
  [0.5, 1, 2, 3].forEach(function (sep) {
    var d = sample(200000, 0.5, sep, 20250930);
    var got = M.auc(d), want = M.normCdf(sep / Math.SQRT2);
    worst = Math.max(worst, Math.abs(got - want));
    rows.push('sep ' + sep + ': ' + got.toFixed(4) + ' vs ' + want.toFixed(4));
  });
  ok('empirical AUC matches Phi(sep/sqrt2) at n=200000  [' + rows.join('; ') + ']',
     worst < 5e-3, 'worst deviation ' + worst.toExponential(2));
})();

/* A classifier with no signal has to sit on the diagonal. */
(function () {
  var d = sample(200000, 0.5, 0, 31337);
  ok('separation 0 gives AUC 0.5', near(M.auc(d), 0.5, 5e-3), 'got ' + M.auc(d).toFixed(4));
})();


/* ------------------------------------------ 5. the class-imbalance claim --- */
section('5. The ROC curve really is blind to class imbalance');

/* The page's central claim, checked exactly rather than statistically:
   tripling the negatives changes the base rate and leaves TPR/FPR — and so the
   whole curve and its AUC — bit-for-bit identical, while precision collapses. */
(function () {
  var base = sample(500, 0.5, 1.8, 5150);

  var s = [], y = [];
  for (var i = 0; i < base.n; i++) {
    var reps = base.y[i] ? 1 : 3;            // keep every positive, triple every negative
    for (var k = 0; k < reps; k++) { s.push(base.s[i]); y.push(base.y[i]); }
  }
  var blown = {
    s: Float64Array.from(s), y: Uint8Array.from(y), n: s.length,
    nPos: base.nPos, nNeg: base.nNeg * 3
  };

  var p1 = M.rocCurve(base), p2 = M.rocCurve(blown);
  var sameLen = p1.length === p2.length;
  var worst = 0;
  if (sameLen) {
    for (i = 0; i < p1.length; i++) {
      worst = Math.max(worst, Math.abs(p1[i].fpr - p2[i].fpr), Math.abs(p1[i].tpr - p2[i].tpr));
    }
  }
  ok('tripling the negatives leaves every ROC vertex identical', sameLen && worst === 0,
     sameLen ? 'worst deviation ' + worst : 'vertex counts differ: ' + p1.length + ' vs ' + p2.length);
  ok('...and leaves AUC identical to the last bit', M.auc(base) === M.auc(blown),
     M.auc(base) + ' vs ' + M.auc(blown));

  var r1 = M.rates(M.confusion(base, 0.4));
  var r2 = M.rates(M.confusion(blown, 0.4));
  ok('...while TPR and FPR are unchanged at a fixed threshold',
     r1.tpr === r2.tpr && r1.fpr === r2.fpr,
     'tpr ' + r1.tpr + '/' + r2.tpr + ', fpr ' + r1.fpr + '/' + r2.fpr);
  ok('...and precision drops, which is the whole point  ('
     + r1.prec.toFixed(3) + ' -> ' + r2.prec.toFixed(3) + ')',
     r2.prec < r1.prec - 0.1, 'prec ' + r1.prec + ' -> ' + r2.prec);
})();


/* -------------------------------------------- 6. scores and probabilities -- */
section('6. Scores and probabilities are two views of one threshold');
(function () {
  var worst = 0;
  for (var i = 0; i < 2000; i++) {
    var sep = u(0.2, 4), prev = u(0.05, 0.95), p = u(0.001, 0.999);
    var s = M.scoreForProb(p, sep, prev);
    worst = Math.max(worst, Math.abs(M.posterior(s, sep, prev) - p));
  }
  ok('posterior and scoreForProb are exact inverses', worst < 1e-12,
     'worst deviation ' + worst.toExponential(2));
})();

/* The posterior claim is only worth anything if it is actually calibrated —
   that among samples scoring near s, the fraction that really are positive is
   the number the page prints. Checked by Monte Carlo on the shipped sampler. */
(function () {
  var sep = 1.8, prev = 0.35, n = 400000;
  var d = M.dataset(M.makeDraws(99221, n), n, prev, sep);
  var W = 0.1, LO = -3, HI = 3;
  var nb = Math.round((HI - LO) / W);
  var cnt = new Float64Array(nb), pos = new Float64Array(nb);
  for (var i = 0; i < d.n; i++) {
    var b = Math.floor((d.s[i] - LO) / W);
    if (b >= 0 && b < nb) { cnt[b]++; if (d.y[i]) pos[b]++; }
  }
  var worst = 0, used = 0;
  for (b = 0; b < nb; b++) {
    if (cnt[b] < 500) continue;
    var mid = LO + (b + 0.5) * W;
    worst = Math.max(worst, Math.abs(pos[b] / cnt[b] - M.posterior(mid, sep, prev)));
    used++;
  }
  ok('the posterior is calibrated against the data (' + used + ' bins, n=400000)',
     worst < 0.03, 'worst bin off by ' + worst.toFixed(4));
})();


/* ------------------------------------------------- 7. cost-sensitive rule -- */
section('7. The cost-sensitive threshold (Murphy §5.1.2.2)');
(function () {
  var worst = 0;
  for (var i = 0; i < 2000; i++) {
    var c = Math.pow(10, u(-1.3, 1.3)), sep = u(0.3, 4), prev = u(0.05, 0.95);
    var t = M.optScore(c, sep, prev);
    worst = Math.max(worst, Math.abs(M.posterior(t, sep, prev) - 1 / (1 + c)));
  }
  ok('the optimal score threshold sits exactly at p = 1/(1+c)', worst < 1e-12,
     'worst deviation ' + worst.toExponential(2));
})();

/* Murphy's worked example: a false negative costing twice a false positive
   moves the decision threshold to 1/3. */
(function () {
  ok('c = 2 gives a probability threshold of 1/3', near(M.probThreshold(2), 1 / 3, 1e-15));
  ok('c = 1 gives 0.5 — zero-one loss, i.e. the MAP rule', M.probThreshold(1) === 0.5);
})();

/* The closed form has to agree with just searching for the cheapest threshold. */
(function () {
  var worst = 0, rows = 0;
  for (var i = 0; i < 40; i++) {
    var c = Math.pow(10, u(-1.2, 1.2)), sep = u(0.8, 3.5), prev = u(0.15, 0.85);
    var bestT = 0, bestC = Infinity;
    for (var t = -8; t <= 8; t += 0.002) {
      var v = M.popCost(t, sep, prev, c);
      if (v < bestC) { bestC = v; bestT = t; }
    }
    worst = Math.max(worst, Math.abs(bestT - M.optScore(c, sep, prev)));
    rows++;
  }
  ok('brute-force search over the population cost finds the same threshold ('
     + rows + ' cases)', worst < 6e-3, 'worst gap ' + worst.toExponential(2));
})();

/* "Slide the ruler until it touches" is a real tangency claim: the slope of
   the ROC curve at t* equals the slope of the iso-cost line. */
(function () {
  var worstRel = 0;
  for (var i = 0; i < 300; i++) {
    var c = Math.pow(10, u(-1, 1)), sep = u(0.8, 3.5), prev = u(0.15, 0.85);
    var t = M.optScore(c, sep, prev);
    var h = 1e-3;
    var dT = (M.popTPR(t + h, sep) - M.popTPR(t - h, sep));
    var dF = (M.popFPR(t + h, sep) - M.popFPR(t - h, sep));
    var slopeCurve = dT / dF;
    var slopeIso = M.isoCostSlope(c, prev);
    worstRel = Math.max(worstRel, Math.abs(slopeCurve - slopeIso) / slopeIso);
  }
  ok('the ROC curve is tangent to the iso-cost line exactly at the optimum',
     worstRel < 2e-3, 'worst relative gap ' + worstRel.toExponential(2));
})();

/* The fast per-vertex cost scan the page runs every frame has to agree with
   the slow, obviously-correct version that rescans the data at each vertex. */
(function () {
  var bad = 0, worst = 0;
  for (var i = 0; i < 30; i++) {
    var p = randomCase(); p.n = 400;
    var d = sample(p.n, p.prev, p.sep, p.seed);
    var pts = M.rocCurve(d);
    var c = Math.pow(10, u(-1.2, 1.2));

    var slow = Infinity;
    for (var j = 0; j < pts.length; j++) {
      slow = Math.min(slow, M.empCost(M.confusion(d, pts[j].t), c));
    }
    var fast = M.bestEmpCost(d, pts, c).cost;
    worst = Math.max(worst, Math.abs(slow - fast));
    if (Math.abs(slow - fast) > 1e-12) bad++;
  }
  ok('the per-frame cost scan matches a full rescan at every vertex', bad === 0,
     'worst deviation ' + worst.toExponential(2));
})();

/* At c = 1 the cost per sample is just the error rate. */
(function () {
  var worst = 0;
  for (var i = 0; i < 300; i++) {
    var p = randomCase();
    var d = sample(p.n, p.prev, p.sep, p.seed);
    var cm = M.confusion(d, u(-4, 4));
    worst = Math.max(worst, Math.abs(M.empCost(cm, 1) - (1 - M.rates(cm).acc)));
  }
  ok('at c = 1, cost per sample equals 1 - accuracy', worst < 1e-12,
     'worst deviation ' + worst.toExponential(2));
})();


/* ------------------------------------------------------------- 8. EER ----- */
section('8. Equal error rate');
(function () {
  var worst = 0, missing = 0;
  for (var i = 0; i < 200; i++) {
    var p = randomCase();
    var d = sample(p.n, p.prev, p.sep, p.seed);
    var e = M.eer(M.rocCurve(d));
    if (!e) { missing++; continue; }
    worst = Math.max(worst, Math.abs(e.fpr - (1 - e.tpr)));
  }
  ok('at the EER point, the false alarm rate equals the miss rate', worst < 1e-12,
     'worst deviation ' + worst.toExponential(2));
  ok('an EER point exists for every curve', missing === 0, missing + ' missing');
})();

(function () {
  var d = sample(200000, 0.5, 2, 4242);
  var e = M.eer(M.rocCurve(d));
  // For the binormal model the EER is Phi(-sep/2).
  var want = M.normCdf(-1);
  ok('EER matches Phi(-sep/2) for the binormal model  (' + e.rate.toFixed(4)
     + ' vs ' + want.toFixed(4) + ')', near(e.rate, want, 5e-3));
})();


/* --------------------------------------------- 9. population vs. sample --- */
section('9. The closed-form rates match the sampled ones');
(function () {
  var worst = 0;
  var d = M.dataset(M.makeDraws(616161, 300000), 300000, 0.5, 2.2);
  for (var t = -3; t <= 3; t += 0.25) {
    var r = M.rates(M.confusion(d, t));
    worst = Math.max(worst, Math.abs(r.tpr - M.popTPR(t, 2.2)), Math.abs(r.fpr - M.popFPR(t, 2.2)));
  }
  ok('popTPR and popFPR track the sampled TPR and FPR at n=300000', worst < 6e-3,
     'worst deviation ' + worst.toExponential(2));
})();

(function () {
  var worst = 0;
  [-3, -1.5, -0.4, 0, 0.7, 2, 3.5].forEach(function (x) {
    var want = 0.5 * (1 + erfRef(x / Math.SQRT2));
    worst = Math.max(worst, Math.abs(M.normCdf(x) - want));
  });
  function erfRef(z) {                  // high-order series, independent of the shipped erf
    var s = z, term = z;
    for (var k = 1; k < 80; k++) {
      term *= -z * z / k;
      s += term / (2 * k + 1);
    }
    return 2 / Math.sqrt(Math.PI) * s;
  }
  ok('normCdf agrees with an independent series expansion', worst < 2e-7,
     'worst deviation ' + worst.toExponential(2));
  ok('normCdf(0) = 0.5 and the tails are symmetric',
     near(M.normCdf(0), 0.5, 1e-9) && near(M.normCdf(1.3) + M.normCdf(-1.3), 1, 1e-9));
})();


/* ------------------------------------------------------------------ done -- */
console.log('\n' + pass + ' passed, ' + fail + ' failed.');
process.exit(fail ? 1 : 0);
