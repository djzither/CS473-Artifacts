/* ============================================================================
   How Tree Ensembles Learn: the models.
   CS 473 module 14 · Murphy, PML §18.4–18.5.4

   No DOM here. Every tree, weight, residual and vote on the page comes from this
   file applied to the passengers in data.js. Under Node it exports itself for
   test-math.js; in the browser it defines window.ENS.
   ========================================================================= */

(function (global) {
  'use strict';

  var ROWS = (typeof TITANIC !== 'undefined') ? TITANIC : (typeof require !== 'undefined' ? require('./data.js') : []);

  var SEED = 473;          // split seed and base seed for every random forest tree
  var N_TRAIN = 500;       // the other 214 passengers are the test set
  var M = 50;              // rounds (trees) per run
  var DEPTH = 2;           // every tree is at most depth 2: up to 4 leaves
  var MIN_LEAF = 5;        // a split must leave at least 5 distinct passengers on each side
  var MTRY = 2;            // random forest: features drawn per split (of 4)
  var Z_MAX = 4;           // LogitBoost working-response clip (Friedman, Hastie & Tibshirani)

  var FEATS = [
    { key: 'sex', name: 'Sex' },
    { key: 'pclass', name: 'Class' },
    { key: 'age', name: 'Age' },
    { key: 'fare', name: 'Fare' }
  ];

  /* ---- seeded RNG (mulberry32) ---- */
  function rng(seed) {
    var a = seed >>> 0;
    return function () {
      a = (a + 0x6D2B79F5) >>> 0;
      var t = a;
      t = Math.imul(t ^ (t >>> 15), t | 1);
      t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }
  /* Tree m of the forest gets its own seed, so it never depends on trees 1..m-1. */
  function treeSeed(m) { return (Math.imul(SEED, 1000003) + Math.imul(m, 7919) + 17) >>> 0; }

  /* ---- passengers and the fixed split ---- */
  var ALL = ROWS.map(function (r, k) {
    return { k: k, id: r[0], y: r[1], pclass: r[2], female: r[3], age: r[4], fare: r[5], name: r[6],
      x: [r[3] ? 0 : 1, r[2], r[4], r[5]] };   // x[0] = 1 for male, so "x0 > 0.5" reads "male"
  });
  (function split() {
    var order = ALL.map(function (_, i) { return i; }), r = rng(SEED);
    for (var i = order.length - 1; i > 0; i--) {
      var j = Math.floor(r() * (i + 1)), t = order[i]; order[i] = order[j]; order[j] = t;
    }
    order.forEach(function (k, pos) { ALL[k].train = pos < N_TRAIN; });
  })();
  var TRAIN = ALL.filter(function (p) { return p.train; });
  var TEST = ALL.filter(function (p) { return !p.train; });
  TRAIN.forEach(function (p, i) { p.ti = i; });
  TEST.forEach(function (p) { p.ti = -1; });
  var NT = TRAIN.length;

  /* ---- CART for depth-limited trees with case weights ----
     opt.kind  'gini' (classification, y in {0,1}) or 'mse' (regression on targets t)
     opt.t     per-training-row target: label 0/1 or real value
     opt.w     per-training-row weight (bootstrap counts, AdaBoost or LogitBoost weights)
     opt.pick  optional function returning the features a node may use (random forest)
     A node splits the rows with x[f] <= thr to the LEFT (the "yes" branch). */
  function nodeStats(kind, rows, t, w) {
    var W = 0, S = 0, Q = 0;
    for (var i = 0; i < rows.length; i++) { var r = rows[i], wi = w[r]; W += wi; S += wi * t[r]; Q += wi * t[r] * t[r]; }
    var mean = W > 0 ? S / W : 0;
    var imp = kind === 'gini' ? 1 - mean * mean - (1 - mean) * (1 - mean) : Math.max(0, Q / W - mean * mean);
    return { n: rows.length, W: W, mean: mean, imp: W > 0 ? imp : 0 };
  }
  function impOf(kind, W, S, Q) {
    if (W <= 0) return 0;
    var m = S / W;
    return kind === 'gini' ? 1 - m * m - (1 - m) * (1 - m) : Math.max(0, Q / W - m * m);
  }
  function bestSplitOn(kind, rows, f, t, w) {
    var X = TRAIN;
    var sorted = rows.slice().sort(function (a, b) { return (X[a].x[f] - X[b].x[f]) || (a - b); });
    var Wt = 0, St = 0, Qt = 0, i;
    for (i = 0; i < sorted.length; i++) { var r0 = sorted[i]; Wt += w[r0]; St += w[r0] * t[r0]; Qt += w[r0] * t[r0] * t[r0]; }
    var WL = 0, SL = 0, QL = 0, best = null;
    for (i = 0; i < sorted.length - 1; i++) {
      var r = sorted[i];
      WL += w[r]; SL += w[r] * t[r]; QL += w[r] * t[r] * t[r];
      var a = X[r].x[f], b = X[sorted[i + 1]].x[f];
      if (a === b) continue;
      var nL = i + 1, nR = sorted.length - nL;
      if (nL < MIN_LEAF || nR < MIN_LEAF) continue;
      var WR = Wt - WL;
      if (WL <= 0 || WR <= 0) continue;
      var score = (WL * impOf(kind, WL, SL, QL) + WR * impOf(kind, WR, St - SL, Qt - QL)) / Wt;
      if (!best || score < best.score - 1e-12) best = { f: f, thr: (a + b) / 2, score: score };
    }
    return best;
  }
  function growTree(opt) {
    var kind = opt.kind, t = opt.t, w = opt.w;
    var rows = [];
    for (var i = 0; i < NT; i++) if (w[i] > 0) rows.push(i);
    function grow(rows, depth, id) {
      var st = nodeStats(kind, rows, t, w);
      var node = { id: id, depth: depth, n: st.n, W: st.W, mean: st.mean, imp: st.imp, leaf: true };
      if (depth >= DEPTH || st.imp <= 1e-12 || rows.length < 2 * MIN_LEAF) return node;
      var feats = opt.pick ? opt.pick() : [0, 1, 2, 3];
      var cands = feats.map(function (f) { return bestSplitOn(kind, rows, f, t, w) || { f: f, thr: null, score: null }; });
      var best = null;
      cands.forEach(function (c) { if (c.score !== null && (!best || c.score < best.score - 1e-12)) best = c; });
      node.cands = cands;
      if (!best || best.score >= st.imp - 1e-12) return node;
      var L = [], R = [];
      rows.forEach(function (r) { (TRAIN[r].x[best.f] <= best.thr ? L : R).push(r); });
      node.leaf = false; node.f = best.f; node.thr = best.thr; node.score = best.score;
      node.left = grow(L, depth + 1, id + 'L');
      node.right = grow(R, depth + 1, id + 'R');
      return node;
    }
    var root = grow(rows, 0, 'r');
    eachLeaf(root, function (lf) { lf.value = opt.leafValue(lf); });
    return root;
  }
  function eachLeaf(node, fn) { if (node.leaf) fn(node); else { eachLeaf(node.left, fn); eachLeaf(node.right, fn); } }
  function eachNode(node, fn) { fn(node); if (!node.leaf) { eachNode(node.left, fn); eachNode(node.right, fn); } }
  function route(node, x) {
    var path = [node];
    while (!node.leaf) { node = x[node.f] <= node.thr ? node.left : node.right; path.push(node); }
    return path;
  }
  function leafOf(node, x) { while (!node.leaf) node = x[node.f] <= node.thr ? node.left : node.right; return node; }
  function applyTree(tree) {
    var out = new Float64Array(ALL.length);
    for (var k = 0; k < ALL.length; k++) out[k] = leafOf(tree, ALL[k].x).value;
    return out;
  }

  /* ---- helpers over all passengers (index k) ---- */
  function accuracy(pred, set) {
    var ok = 0; set.forEach(function (p) { if (pred(p.k) === p.y) ok++; });
    return ok / set.length;
  }
  function sigma2(f) { return 1 / (1 + Math.exp(-2 * f)); }

  var LOSSES = {
    squared: function (y, f) { return (y - f) * (y - f); },
    exponential: function (y, f) { return Math.exp(-(2 * y - 1) * f); },
    logistic: function (y, f) { var m = -2 * (2 * y - 1) * f; return m > 30 ? m : Math.log1p(Math.exp(m)); }
  };
  function trainLoss(loss, f) {
    var s = 0; TRAIN.forEach(function (p) { s += LOSSES[loss](p.y, f[p.k]); });
    return s / NT;
  }
  /* Line search over beta for the round's tree: L(beta) = mean loss of f_{m-1} + beta F_m */
  function lineLoss(loss, fPrev, F, beta) {
    var s = 0; TRAIN.forEach(function (p) { s += LOSSES[loss](p.y, fPrev[p.k] + beta * F[p.k]); });
    return s / NT;
  }
  function lineSearch(loss, fPrev, F) {   // golden section on [-1, 4] after a coarse grid
    var lo = -1, hi = 4, best = 0, bestV = Infinity, i;
    for (i = 0; i <= 100; i++) { var b = lo + (hi - lo) * i / 100, v = lineLoss(loss, fPrev, F, b); if (v < bestV) { bestV = v; best = b; } }
    var a = best - 0.05, c = best + 0.05, g = (Math.sqrt(5) - 1) / 2;
    for (i = 0; i < 80; i++) {
      var x1 = c - g * (c - a), x2 = a + g * (c - a);
      if (lineLoss(loss, fPrev, F, x1) < lineLoss(loss, fPrev, F, x2)) c = x2; else a = x1;
    }
    return (a + c) / 2;
  }

  /* ======================================================== RANDOM FOREST == */
  function rfTree(m) {
    var r = rng(treeSeed(m));
    var counts = new Float64Array(NT), i;
    for (i = 0; i < NT; i++) counts[Math.floor(r() * NT)]++;
    var y = TRAIN.map(function (p) { return p.y; });
    function pick() {
      var f = [0, 1, 2, 3];
      for (var j = 0; j < MTRY; j++) { var s = j + Math.floor(r() * (4 - j)), tmp = f[j]; f[j] = f[s]; f[s] = tmp; }
      return f.slice(0, MTRY).sort();
    }
    var tree = growTree({ kind: 'gini', t: y, w: counts, pick: pick,
      leafValue: function (lf) { return lf.mean > 0.5 ? 1 : 0; } });
    return { tree: tree, counts: counts };
  }
  function runRF() {
    var votes = new Float64Array(ALL.length), rounds = [];
    for (var m = 1; m <= M; m++) {
      var b = rfTree(m), F = applyTree(b.tree);
      var before = Float64Array.from(votes);
      for (var k = 0; k < ALL.length; k++) votes[k] += F[k];
      var after = Float64Array.from(votes);
      var distinct = 0; for (var i = 0; i < NT; i++) if (b.counts[i] > 0) distinct++;
      rounds.push({ m: m, tree: b.tree, counts: b.counts, F: F, votesBefore: before, votesAfter: after, distinct: distinct,
        fBefore: scale(before, m - 1), fAfter: scale(after, m) });
    }
    return finish('rf', rounds, function (f) { return function (k) { return f[k] > 0.5 ? 1 : 0; }; }, null);
  }
  function scale(v, m) { var o = new Float64Array(v.length); for (var k = 0; k < v.length; k++) o[k] = m ? v[k] / m : 0.5; return o; }

  /* ============================================== LEAST SQUARES BOOSTING == */
  function runLS() {
    var f0 = 0; TRAIN.forEach(function (p) { f0 += p.y; }); f0 /= NT;
    var f = new Float64Array(ALL.length).fill(f0), rounds = [];
    for (var m = 1; m <= M; m++) {
      var r = TRAIN.map(function (p) { return p.y - f[p.k]; });
      var w = new Float64Array(NT).fill(1);
      var tree = growTree({ kind: 'mse', t: r, w: w, leafValue: function (lf) { return lf.mean; } });
      var F = applyTree(tree);
      var num = 0, den = 0;
      TRAIN.forEach(function (p, i) { num += r[i] * F[p.k]; den += F[p.k] * F[p.k]; });
      var beta = den > 0 ? num / den : 0;          // least-squares line search; equals 1 for leaf means
      var before = Float64Array.from(f);
      for (var k = 0; k < ALL.length; k++) f[k] += beta * F[k];
      rounds.push({ m: m, tree: tree, F: F, beta: beta, resid: r, fBefore: before, fAfter: Float64Array.from(f) });
    }
    var run = finish('ls', rounds, function (f) { return function (k) { return f[k] > 0.5 ? 1 : 0; }; }, 'squared');
    run.f0 = f0;
    return run;
  }

  /* ============================================================ ADABOOST == */
  function runAda() {
    var f = new Float64Array(ALL.length), rounds = [];
    var y = TRAIN.map(function (p) { return p.y; });
    for (var m = 1; m <= M; m++) {
      var raw = TRAIN.map(function (p) { return Math.exp(-(2 * p.y - 1) * f[p.k]); });
      var Z = raw.reduce(function (a, b) { return a + b; }, 0);
      var w = Float64Array.from(raw, function (v) { return v / Z; });
      var tree = growTree({ kind: 'gini', t: y, w: w, leafValue: function (lf) { return lf.mean > 0.5 ? 1 : -1; } });
      var F = applyTree(tree);
      var err = 0;
      TRAIN.forEach(function (p, i) { if (F[p.k] !== 2 * p.y - 1) err += w[i]; });
      var e = Math.min(Math.max(err, 1e-10), 1 - 1e-10);
      var beta = 0.5 * Math.log((1 - e) / e);
      var before = Float64Array.from(f);
      for (var k = 0; k < ALL.length; k++) f[k] += beta * F[k];
      var raw2 = TRAIN.map(function (p, i) { return w[i] * Math.exp(-beta * (2 * p.y - 1) * F[p.k]); });
      var Z2 = raw2.reduce(function (a, b) { return a + b; }, 0);
      rounds.push({ m: m, tree: tree, F: F, beta: beta, err: err, w: w,
        wNext: Float64Array.from(raw2, function (v) { return v / Z2; }), fBefore: before, fAfter: Float64Array.from(f) });
    }
    return finish('ada', rounds, function (f) { return function (k) { return f[k] > 0 ? 1 : 0; }; }, 'exponential');
  }

  /* ========================================================== LOGITBOOST == */
  function runLogit() {
    var f = new Float64Array(ALL.length), rounds = [];
    for (var m = 1; m <= M; m++) {
      var pi = TRAIN.map(function (p) { return sigma2(f[p.k]); });
      var z = TRAIN.map(function (p, i) {
        var v = (p.y - pi[i]) / (pi[i] * (1 - pi[i]));
        return Math.max(-Z_MAX, Math.min(Z_MAX, v));
      });
      var w = Float64Array.from(pi, function (q) { return Math.max(q * (1 - q), 1e-12); });
      var tree = growTree({ kind: 'mse', t: z, w: w, leafValue: function (lf) { return lf.mean; } });
      var F = applyTree(tree);
      var before = Float64Array.from(f);
      for (var k = 0; k < ALL.length; k++) f[k] += 0.5 * F[k];
      rounds.push({ m: m, tree: tree, F: F, beta: 0.5, pi: pi, z: z, w: w, fBefore: before, fAfter: Float64Array.from(f) });
    }
    return finish('logit', rounds, function (f) { return function (k) { return f[k] > 0 ? 1 : 0; }; }, 'logistic');
  }

  /* Accuracy, loss and line-search numbers for every round. */
  function finish(key, rounds, predOf, loss) {
    rounds.forEach(function (rd) {
      var pb = predOf(rd.fBefore), pa = predOf(rd.fAfter);
      rd.predBefore = function (k) { return rd.m === 1 && key === 'rf' ? null : pb(k); };
      rd.predAfter = pa;
      rd.accTrainBefore = accuracy(pb, TRAIN); rd.accTrainAfter = accuracy(pa, TRAIN);
      rd.accTestBefore = accuracy(pb, TEST); rd.accTestAfter = accuracy(pa, TEST);
      var flips = 0; TRAIN.forEach(function (p) { if (pb(p.k) !== pa(p.k)) flips++; });
      rd.flips = flips;
      if (loss) {
        rd.lossBefore = trainLoss(loss, rd.fBefore); rd.lossAfter = trainLoss(loss, rd.fAfter);
        rd.betaStar = lineSearch(loss, rd.fBefore, rd.F);
      }
    });
    return { key: key, rounds: rounds, loss: loss, pred: predOf };
  }

  /* ---- readable split rules ---- */
  function rule(f, thr) {
    if (f === 0) return 'Male?';
    if (f === 1) return thr < 2 ? 'Class = 1st?' : 'Class 1st or 2nd?';
    if (f === 2) return 'Age ≤ ' + fmtNum(thr) + '?';
    return 'Fare ≤ £' + thr.toFixed(2) + '?';
  }
  function fmtNum(v) { return Math.abs(v - Math.round(v)) < 1e-9 ? String(Math.round(v)) : String(+v.toFixed(2)); }

  var ENS = {
    SEED: SEED, N_TRAIN: N_TRAIN, M: M, DEPTH: DEPTH, MIN_LEAF: MIN_LEAF, MTRY: MTRY, Z_MAX: Z_MAX,
    FEATS: FEATS, ALL: ALL, TRAIN: TRAIN, TEST: TEST, rng: rng, treeSeed: treeSeed,
    growTree: growTree, bestSplitOn: bestSplitOn, nodeStats: nodeStats, route: route, leafOf: leafOf,
    eachLeaf: eachLeaf, eachNode: eachNode, applyTree: applyTree, rfTree: rfTree,
    runRF: runRF, runLS: runLS, runAda: runAda, runLogit: runLogit,
    LOSSES: LOSSES, lineLoss: lineLoss, lineSearch: lineSearch, trainLoss: trainLoss, sigma2: sigma2,
    rule: rule, fmtNum: fmtNum
  };
  if (typeof module !== 'undefined' && module.exports) module.exports = ENS;
  else global.ENS = ENS;
})(typeof window !== 'undefined' ? window : globalThis);
