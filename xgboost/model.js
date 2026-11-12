/* ============================================================================
   Boosting Through the Highstorm: the models.
   CS 473 module 15 · gradient boosting and XGBoost (Murphy, PML §18.5)

   No DOM. Plain gradient boosting and XGBoost on the same invented bridge data,
   both with squared-error loss l(y, F) = 1/2 (y - F)^2, so
     gradient  g = F - y       (minus the residual)
     hessian   h = 1           (the loss curves the same amount everywhere)
   Exported under Node for test-math.js; window.GBX in the browser.
   ========================================================================= */

(function (global) {
  'use strict';

  /* INVENTED FOR TEACHING. Not canon. No bridge, number or name here comes from the books.
     age      = months since the bridge was built
     strength = quality of timber and lashings, 1 (green softwood) to 10 (seasoned, bronze-banded)
     exposure = how squarely it faces the storm's leading edge, 1 (sheltered) to 10 (open to the east)
     y        = highstorms it survived before it had to be rebuilt */
  var BRIDGES = [
    ['Sunrise Span', 2, 6, 4, 24],
    ['Chasm Nine Crossing', 35, 5, 6, 9],
    ['Windward Causeway', 20, 4, 3, 19],
    ['Old Kholin Bridge', 7, 5, 6, 21],
    ['Plateau Sixteen Link', 17, 3, 3, 19],
    ['Crem Hollow Span', 25, 6, 5, 15],
    ['Shard Road Span', 14, 8, 1, 29],
    ['Lighteyes Walk', 24, 8, 1, 26],
    ['Greatshell Ford', 36, 6, 2, 18],
    ['Stormward Arch', 38, 5, 9, 2],
    ['Market Crossing', 29, 8, 10, 9],
    ['Eastgate Span', 8, 8, 10, 18]
  ].map(function (r, i) { return { i: i, name: r[0], x: [r[1], r[2], r[3]], y: r[4] }; });

  var FEATS = [
    { key: 'age', name: 'Age', unit: 'months', short: 'age' },
    { key: 'strength', name: 'Strength', unit: '/10', short: 'strength' },
    { key: 'exposure', name: 'Exposure', unit: '/10', short: 'exposure' }
  ];
  var N = BRIDGES.length;
  var M = 20;            // rounds
  var MIN_LEAF = 2;      // every leaf keeps at least two bridges

  function sum(a) { var s = 0; for (var i = 0; i < a.length; i++) s += a[i]; return s; }

  /* Score of a group of bridges with gradient sum G and hessian sum H.
     XGBoost: G^2 / (H + lambda). With lambda = 0 and h = 1 this is n * (mean residual)^2,
     so the gain below is exactly the drop in summed squared residuals, halved. */
  function sim(G, H, lambda) { return G * G / (H + lambda); }

  /* All candidate splits of a node, best first (ties: earlier feature, smaller threshold). */
  function candidates(rows, g, h, lambda, gamma) {
    var out = [];
    var G = 0, H = 0;
    rows.forEach(function (r) { G += g[r]; H += h[r]; });
    for (var f = 0; f < FEATS.length; f++) {
      var vals = rows.map(function (r) { return BRIDGES[r].x[f]; }).filter(function (v, k, a) { return a.indexOf(v) === k; }).sort(function (a, b) { return a - b; });
      for (var k = 0; k < vals.length - 1; k++) {
        var thr = (vals[k] + vals[k + 1]) / 2, GL = 0, HL = 0, nL = 0;
        rows.forEach(function (r) { if (BRIDGES[r].x[f] <= thr) { GL += g[r]; HL += h[r]; nL++; } });
        var nR = rows.length - nL;
        if (nL < MIN_LEAF || nR < MIN_LEAF) continue;
        var GR = G - GL, HR = H - HL;
        var raw = 0.5 * (sim(GL, HL, lambda) + sim(GR, HR, lambda) - sim(G, H, lambda));
        out.push({ f: f, thr: thr, nL: nL, nR: nR, GL: GL, HL: HL, GR: GR, HR: HR, raw: raw, gain: raw - gamma });
      }
    }
    out.sort(function (a, b) { return (b.gain - a.gain) || (a.f - b.f) || (a.thr - b.thr); });
    return out;
  }

  /* Grow one tree of the given depth on gradients g and hessians h.
     opt.lambda (L2 on leaf weights), opt.gamma (cost per split), opt.depth.
     Growth takes the best split whenever it lowers the loss at all (raw gain > 0).
     Then, as XGBoost does, prune from the bottom up: a split whose children are both
     leaves and whose gain after paying gamma is negative is removed. */
  function growTree(g, h, opt) {
    var lambda = opt.lambda, gamma = opt.gamma;
    function grow(rows, depth, id) {
      var G = 0, H = 0;
      rows.forEach(function (r) { G += g[r]; H += h[r]; });
      var node = { id: id, depth: depth, rows: rows, G: G, H: H, w: -G / (H + lambda), leaf: true, cands: [] };
      if (depth >= opt.depth) return node;
      node.cands = candidates(rows, g, h, lambda, gamma);
      var best = node.cands[0];
      if (!best || best.raw <= 1e-12) return node;
      node.leaf = false; node.f = best.f; node.thr = best.thr; node.best = best;
      node.left = grow(rows.filter(function (r) { return BRIDGES[r].x[best.f] <= best.thr; }), depth + 1, id + 'L');
      node.right = grow(rows.filter(function (r) { return BRIDGES[r].x[best.f] > best.thr; }), depth + 1, id + 'R');
      return node;
    }
    function prune(n) {
      if (n.leaf) return;
      prune(n.left); prune(n.right);
      if (n.left.leaf && n.right.leaf && n.best.gain < 0) {
        n.pruned = { f: n.f, thr: n.thr, best: n.best, left: n.left, right: n.right };
        n.leaf = true; delete n.left; delete n.right;
      }
    }
    var all = []; for (var i = 0; i < N; i++) all.push(i);
    var root = grow(all, 0, 'r');
    prune(root);
    return root;
  }
  function leafOf(n, x) { while (!n.leaf) n = x[n.f] <= n.thr ? n.left : n.right; return n; }
  function route(n, x) { var p = [n]; while (!n.leaf) { n = x[n.f] <= n.thr ? n.left : n.right; p.push(n); } return p; }
  function eachNode(n, fn) { fn(n); if (!n.leaf) { eachNode(n.left, fn); eachNode(n.right, fn); } }
  function leaves(n) { var a = []; eachNode(n, function (q) { if (q.leaf) a.push(q); }); return a; }

  /* One full run. mode 'gb': plain gradient boosting (lambda = gamma = 0).
     mode 'xgb': XGBoost with the given lambda and gamma. Both use F0 = mean(y). */
  function run(mode, p) {
    var lambda = mode === 'gb' ? 0 : p.lambda, gamma = mode === 'gb' ? 0 : p.gamma;
    var y = BRIDGES.map(function (b) { return b.y; });
    var F0 = sum(y) / N;
    var F = y.map(function () { return F0; });
    var rounds = [];
    for (var m = 1; m <= M; m++) {
      var g = F.map(function (f, i) { return f - y[i]; });
      var h = F.map(function () { return 1; });
      var tree = growTree(g, h, { lambda: lambda, gamma: gamma, depth: p.depth });
      var step = BRIDGES.map(function (b) { return leafOf(tree, b.x).w; });
      var before = F.slice();
      F = F.map(function (f, i) { return f + p.eta * step[i]; });
      var T = leaves(tree);
      var omega = gamma * T.length + 0.5 * lambda * sum(T.map(function (l) { return l.w * l.w; }));
      rounds.push({ m: m, g: g, h: h, tree: tree, step: step, before: before, after: F.slice(),
        lossBefore: loss(y, before), lossAfter: loss(y, F), omega: omega, leaves: T.length });
    }
    return { mode: mode, lambda: lambda, gamma: gamma, eta: p.eta, depth: p.depth, F0: F0, y: y, rounds: rounds };
  }
  /* Total loss: sum of 1/2 (y - F)^2 */
  function loss(y, F) { var s = 0; for (var i = 0; i < y.length; i++) s += 0.5 * (y[i] - F[i]) * (y[i] - F[i]); return s; }
  function rmse(y, F) { return Math.sqrt(2 * loss(y, F) / y.length); }

  function rule(f, thr) {
    var t = Math.abs(thr - Math.round(thr)) < 1e-9 ? String(Math.round(thr)) : String(thr);
    return FEATS[f].name + ' ≤ ' + t;
  }

  var GBX = { BRIDGES: BRIDGES, FEATS: FEATS, N: N, M: M, MIN_LEAF: MIN_LEAF,
    sim: sim, candidates: candidates, growTree: growTree, leafOf: leafOf, route: route, eachNode: eachNode, leaves: leaves,
    run: run, loss: loss, rmse: rmse, rule: rule };
  if (typeof module !== 'undefined' && module.exports) module.exports = GBX;
  else global.GBX = GBX;
})(typeof window !== 'undefined' ? window : globalThis);
