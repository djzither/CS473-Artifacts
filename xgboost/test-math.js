/* ============================================================================
   Math checks for "Boosting Through the Highstorm".   Run:  node test-math.js
   Exercises the SHIPPED model.js and checks every claim the page makes in words.
   ========================================================================= */

var G = require('./model.js');
var pass = 0, fail = 0;
function ok(name, cond, detail) {
  if (cond) { pass++; console.log('  PASS  ' + name); }
  else { fail++; console.log('  FAIL  ' + name + (detail !== undefined ? '   -> ' + detail : '')); }
}
function section(t) { console.log('\n' + t); }
function close(a, b, t) { return Math.abs(a - b) <= (t || 1e-9); }
var B = G.BRIDGES, y = B.map(function (b) { return b.y; });

section('Data');
ok('12 invented bridges with 3 features each', B.length === 12 && B.every(function (b) { return b.x.length === 3; }));
ok('every target lies on the chart (0 to 32 storms)', y.every(function (v) { return v >= 0 && v <= 32; }));

var gb = G.run('gb', { eta: 0.3, depth: 2 });
var xg = G.run('xgb', { eta: 0.3, depth: 2, lambda: 1, gamma: 2 });

section('Starting forecast');
var mean = y.reduce(function (a, b) { return a + b; }, 0) / 12;
ok('F0 is the average target (' + mean.toFixed(2) + ') in both modes', close(gb.F0, mean) && close(xg.F0, mean));
var Lmean = G.loss(y, y.map(function () { return mean; }));
ok('the average beats any other constant under squared error', [mean - 1, mean + 1, mean - 0.01, mean + 0.01].every(function (c) {
  return G.loss(y, y.map(function () { return c; })) > Lmean; }));

section('Gradient boosting (mode 1)');
ok('round-1 slope g = F0 - y, and the miss r = -g', gb.rounds[0].g.every(function (g, i) { return close(g, mean - y[i]); }));
ok('every hessian is 1', gb.rounds.every(function (r) { return r.h.every(function (h) { return h === 1; }); }));
ok('each leaf value is the average miss of its bridges', gb.rounds.every(function (r) {
  return G.leaves(r.tree).every(function (l) {
    var s = 0; l.rows.forEach(function (i) { s += -r.g[i]; }); return close(l.w, s / l.rows.length, 1e-9);
  });
}));
ok('F_m = F_{m-1} + eta * leaf for every bridge and round', gb.rounds.every(function (r) {
  return B.every(function (b, i) { return close(r.after[i], r.before[i] + 0.3 * G.leafOf(r.tree, b.x).w, 1e-12); });
}));
ok('training loss falls every round', gb.rounds.every(function (r) { return r.lossAfter < r.lossBefore; }));
ok('the drop column (2 x raw gain) equals the fall in summed squared misses', (function () {
  var r = gb.rounds[0], t = r.tree, b = t.best;
  var miss = r.g.map(function (g) { return -g; });
  function sse(rows) { var m = rows.reduce(function (a, i) { return a + miss[i]; }, 0) / rows.length; return rows.reduce(function (a, i) { return a + (miss[i] - m) * (miss[i] - m); }, 0); }
  return close(2 * b.raw, sse(t.rows) - sse(t.left.rows) - sse(t.right.rows), 1e-9);
})());
ok('each split took the highest-scoring candidate', gb.rounds.every(function (r) {
  var good = true; G.eachNode(r.tree, function (n) { if (!n.leaf && n.cands.some(function (c) { return c.gain > n.best.gain + 1e-12; })) good = false; }); return good;
}));
ok('every leaf keeps at least 2 bridges', [gb, xg].every(function (R) { return R.rounds.every(function (r) { return G.leaves(r.tree).every(function (l) { return l.rows.length >= 2; }); }); }));
ok('round 1 root question is "Age <= 24.5"', G.rule(gb.rounds[0].tree.f, gb.rounds[0].tree.thr) === 'Age ≤ 24.5');

section('XGBoost (mode 2)');
ok('leaf weight w = -G / (H + lambda)', xg.rounds.every(function (r) { return G.leaves(r.tree).every(function (l) { return close(l.w, -l.G / (l.H + 1), 1e-12); }); }));
ok('H counts the bridges in a node (h = 1)', xg.rounds.every(function (r) { var g = true; G.eachNode(r.tree, function (n) { if (n.H !== n.rows.length) g = false; }); return g; }));
ok('lambda pulls every leaf toward zero vs. the plain average', xg.rounds.every(function (r) {
  return G.leaves(r.tree).every(function (l) { var avg = -l.G / l.H; return Math.abs(l.w) <= Math.abs(avg) + 1e-12 && l.w * avg >= 0; });
}));
ok('w = -G/(H+lambda) is the bottom of the leaf parabola sum(g w + h w^2 / 2) + lambda w^2 / 2', (function () {
  var l = G.leaves(xg.rounds[0].tree)[0];
  function q(w) { return l.G * w + 0.5 * (l.H + 1) * w * w; }
  return q(l.w) < q(l.w + 0.01) && q(l.w) < q(l.w - 0.01);
})());
ok('gain formula: 1/2 [GL^2/(HL+l) + GR^2/(HR+l) - G^2/(H+l)] - gamma', (function () {
  var c = xg.rounds[0].tree.best, l = 1;
  var Gt = c.GL + c.GR, Ht = c.HL + c.HR;
  return close(c.gain, 0.5 * (c.GL * c.GL / (c.HL + l) + c.GR * c.GR / (c.HR + l) - Gt * Gt / (Ht + l)) - 2, 1e-9);
})());
ok('no kept split has negative gain where both children are leaves', xg.rounds.every(function (r) {
  var g = true; G.eachNode(r.tree, function (n) { if (!n.leaf && n.left.leaf && n.right.leaf && n.best.gain < 0) g = false; }); return g;
}));
ok('pruned splits all had negative gain', xg.rounds.every(function (r) {
  var g = true; G.eachNode(r.tree, function (n) { if (n.pruned && n.pruned.best.gain >= 0) g = false; }); return g;
}));
ok('with gamma = 2, some rounds prune a split', xg.rounds.some(function (r) { var p = false; G.eachNode(r.tree, function (n) { if (n.pruned) p = true; }); return p; }));
ok('Omega = gamma T + lambda/2 sum w^2', xg.rounds.every(function (r) {
  var T = G.leaves(r.tree); return close(r.omega, 2 * T.length + 0.5 * T.reduce(function (a, l) { return a + l.w * l.w; }, 0), 1e-9);
}));
ok('XGBoost training loss also falls (or holds) every round', xg.rounds.every(function (r) { return r.lossAfter <= r.lossBefore + 1e-12; }));
ok('after 20 rounds XGBoost fits the training bridges less tightly than mode 1', G.rmse(y, xg.rounds[19].after) > G.rmse(y, gb.rounds[19].after));

section('The two modes are connected');
[[0.3, 2], [0.3, 1], [1, 2], [0.1, 2]].forEach(function (p) {
  var a = G.run('gb', { eta: p[0], depth: p[1] }), b = G.run('xgb', { eta: p[0], depth: p[1], lambda: 0, gamma: 0 });
  ok('lambda = gamma = 0 reproduces mode 1 exactly (eta ' + p[0] + ', depth ' + p[1] + ')', JSON.stringify(a.rounds.map(function (r) { return r.after; })) === JSON.stringify(b.rounds.map(function (r) { return r.after; })));
});
ok('a huge gamma stops every split: each tree is one leaf', G.run('xgb', { eta: 0.3, depth: 2, lambda: 1, gamma: 1e6 }).rounds.every(function (r) { return r.tree.leaf; }));

console.log('\n' + pass + ' passed, ' + fail + ' failed');
process.exit(fail ? 1 : 0);
