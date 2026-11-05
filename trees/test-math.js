/* ============================================================================
   Math checks for "Build a Decision Tree with LEGO".   Run:  node test-math.js
   Exercises the SHIPPED script.js (it exports its math half under Node) and
   checks every number the page states in words.
   ========================================================================= */

var M = require('./script.js');
var pass = 0, fail = 0;
function ok(name, cond, detail) {
  if (cond) { pass++; console.log('  PASS  ' + name); }
  else { fail++; console.log('  FAIL  ' + name + (detail !== undefined ? '   -> ' + detail : '')); }
}
function section(t) { console.log('\n' + t); }
function fr(f) { return f.d === 1 ? String(f.n) : f.n + '/' + f.d; }
function is(f, s) { return fr(f) === s; }

/* ------------------------------------------------------------ scene 1 -- */
section('Scene 1: classification tree on six bricks');
ok('labels are blue, blue, red, red, blue, blue', M.CLS.map(function (r) { return r.c; }).join('') === '001100');
ok('thresholds 1-5 are the candidates at the root', M.candidates(M.CLS, 0).join(',') === '1,2,3,4,5');
var cmp = M.compareSplits('cls', M.CLS, 0);
ok('compare splits: 2/5, 1/3, 4/9, 1/3, 2/5', cmp.all.map(function (s) { return fr(s.score); }).join(', ') === '2/5, 1/3, 4/9, 1/3, 2/5',
  cmp.all.map(function (s) { return fr(s.score); }).join(', '));
ok('t = 2 and t = 4 tie for best', cmp.best.map(function (k) { return cmp.all[k].t; }).join(',') === '2,4');
var s2 = M.scoreSplit('cls', M.CLS, 0, 2);
ok('t = 2: left tray is 2 blue, right tray is 2 blue + 2 red',
  M.classCounts(s2.L).join(',') === '2,0' && M.classCounts(s2.R).join(',') === '2,2');
ok('t = 2: G_L = 0, G_R = 1/2', is(s2.GL, '0') && is(s2.GR, '1/2'));
ok('t = 2: (2/6)(0) + (4/6)(1/2) = 1/3', s2.nL === 2 && s2.nR === 4 && is(s2.score, '1/3'));
ok('root impurity 1 - (4/6)^2 - (2/6)^2 = 4/9', is(M.gini(M.CLS), '4/9'));

var right = s2.R;
ok('right branch has thresholds 3, 4, 5', M.candidates(right, 0).join(',') === '3,4,5');
var cR = M.compareSplits('cls', right, 0);
ok('right branch: t = 4 is the unique best with score 0',
  cR.best.length === 1 && cR.all[cR.best[0]].t === 4 && is(cR.all[cR.best[0]].score, '0'));
var r4 = M.scoreSplit('cls', right, 0, 4);
ok('right branch at t = 4 gives two pure leaves (red | blue)',
  M.classCounts(r4.L).join(',') === '0,2' && M.classCounts(r4.R).join(',') === '2,0');

var s4 = M.scoreSplit('cls', M.CLS, 0, 4), left = s4.L;
ok('mirrored path: t = 4 first leaves a pure blue right tray', M.classCounts(s4.R).join(',') === '2,0' && is(s4.score, '1/3'));
var cL = M.compareSplits('cls', left, 0);
ok('mirrored path: left branch best is t = 2, score 0',
  cL.best.length === 1 && cL.all[cL.best[0]].t === 2 && is(cL.all[cL.best[0]].score, '0'));

/* route 3.5 through both finished trees */
function hand(splits) {   // build the tree the user builds: list of [path, t]
  var rootN = { rows: M.CLS, leaf: true, path: '' }, by = { '': rootN };
  splits.forEach(function (op) {
    var n = by[op[0]], s = M.splitRows(n.rows, 0, op[1]);
    n.leaf = false; n.f = 0; n.t = op[1];
    n.left = { rows: s.L, leaf: true, path: op[0] + 'L' }; n.right = { rows: s.R, leaf: true, path: op[0] + 'R' };
    by[n.left.path] = n.left; by[n.right.path] = n.right;
  });
  Object.keys(by).forEach(function (k) { if (by[k].leaf) by[k].value = M.leafValue('cls', by[k].rows); });
  return rootN;
}
[[['', 2], ['R', 4]], [['', 4], ['L', 2]]].forEach(function (sp) {
  var tr = hand(sp), path = M.route(tr, [3.5]), leaf = path[path.length - 1];
  ok('x = 3.5 reaches a pure red leaf (first split t = ' + sp[0][1] + ')',
    leaf.value === 1 && M.gini(leaf.rows).n === 0 && path.length === 3);
});
/* every possible hand-built path ends pure; 3.5 always lands red */
(function () {
  var allPure = true, allRed = true;
  function explore(splits, open) {
    if (!open.length) {
      var tr = hand(splits), p = M.route(tr, [3.5]);
      if (p[p.length - 1].value !== 1) allRed = false;
      return;
    }
    var path = open[0], tr0 = hand(splits), n = tr0;
    for (var i = 0; i < path.length; i++) n = path[i] === 'L' ? n.left : n.right;
    M.candidates(n.rows, 0).forEach(function (t) {
      var s = M.splitRows(n.rows, 0, t), nxt = open.slice(1);
      if (M.gini(s.L).n) nxt.push(path + 'L');
      if (M.gini(s.R).n) nxt.push(path + 'R');
      explore(splits.concat([[path, t]]), nxt);
    });
  }
  explore([], ['']);
  ok('whatever thresholds you commit, x = 3.5 ends in a red leaf', allRed && allPure);
})();

/* ------------------------------------------------------------ scene 2 -- */
section('Scene 2: regression tree');
ok('targets 2, 4, 6, 10, 12, 14', M.REG.map(function (r) { return r.v; }).join(',') === '2,4,6,10,12,14');
var cg = M.compareSplits('reg', M.REG, 0);
ok('t = 3 is the unique best regression split', cg.best.length === 1 && cg.all[cg.best[0]].t === 3);
var g3 = M.scoreSplit('reg', M.REG, 0, 3);
ok('leaf with targets 2, 4, 6 has mean 4', g3.L.map(function (r) { return r.v; }).join(',') === '2,4,6' && is(M.mean(g3.L), '4'));
ok('its MSE is ((2-4)^2 + 0 + (6-4)^2)/3 = 8/3', is(M.mse(g3.L), '8/3'));
ok('right leaf mean 12, MSE 8/3', is(M.mean(g3.R), '12') && is(M.mse(g3.R), '8/3'));
ok('split score (3/6)(8/3) + (3/6)(8/3) = 8/3', is(g3.score, '8/3'));
ok('root mean 8, MSE 56/3', is(M.mean(M.REG), '8') && is(M.mse(M.REG), '56/3'));
ok('every input routed to the left leaf gets 4',
  [0.5, 1, 1.7, 2.2, 3].every(function (x) { return x <= 3; }) && M.leafValue('reg', g3.L).n === 4);
ok('MSE formula equals the brute-force mean square',
  M.REG.every(function (r, i) {
    var rows = M.REG.slice(0, i + 1), m = rows.reduce(function (a, r) { return a + r.v; }, 0) / rows.length;
    var b = rows.reduce(function (a, r) { return a + (r.v - m) * (r.v - m); }, 0) / rows.length;
    return Math.abs(M.fval(M.mse(rows)) - b) < 1e-12;
  }));

/* ------------------------------------------------------------ scene 3 -- */
section('Scene 3: ensemble');
function preds(mode, rf, seed) {
  return M.PRESETS[mode].map(function (s, k) {
    var p = M.route(M.ensembleTree(s, mode, rf, seed * 31 + k + 1), M.ENS_TEST);
    return p[p.length - 1];
  });
}
ok('presets are bootstrap samples (10 draws from 10 bricks)',
  ['cls', 'reg'].every(function (m) { return M.PRESETS[m].every(function (s) { return s.length === 10 && s.every(function (i) { return i >= 0 && i < 10; }); }); }));
ok('every preset sample repeats some bricks and leaves some out',
  ['cls', 'reg'].every(function (m) { return M.PRESETS[m].every(function (s) { var c = M.counts(s, 10); return c.indexOf(0) >= 0 && c.some(function (v) { return v > 1; }); }); }));
var pc = preds('cls', false, 7);
ok('classification preset votes 1, 0, 1', pc.map(function (l) { return l.value; }).join(',') === '1,0,1', pc.map(function (l) { return l.value; }).join(','));
ok('... with no tied leaves', pc.every(function (l) { var k = M.classCounts(l.rows); return k[0] !== k[1]; }));
ok('... majority is 1', M.majority(pc.map(function (l) { return l.value; })).winner === 1);
var pr = preds('reg', false, 7);
ok('regression preset predicts 4, 6, 8', pr.map(function (l) { return fr(l.value); }).join(',') === '4,6,8', pr.map(function (l) { return fr(l.value); }).join(','));
ok('... average (4 + 6 + 8)/3 = 6', is(M.average(pr.map(function (l) { return l.value; })), '6'));
ok('the three classification trees differ',
  (function () { var d = M.PRESETS.cls.map(function (s, k) { var t = M.ensembleTree(s, 'cls', false, 1); return t.f + ':' + t.t; }); return d[0] !== d[1] || d[1] !== d[2]; })());

/* bagging vs random forest eligibility */
(function () {
  var okB = true, okR = true, okD = true;
  for (var seed = 1; seed < 60; seed++) {
    var r = M.rng(seed), s = M.bootstrap(r, 10);
    ['cls', 'reg'].forEach(function (mode) {
      (function walk(n) { if (n.eligible.join(',') !== '0,1') okB = false; if (!n.leaf) { walk(n.left); walk(n.right); } })(M.ensembleTree(s, mode, false, seed));
      (function walk(n) {
        if (n.eligible.length !== 1) okR = false;
        if (!n.leaf && n.eligible.indexOf(n.f) < 0) okR = false;
        if (n.depth > M.ENS_DEPTH) okD = false;
        if (!n.leaf) { walk(n.left); walk(n.right); }
      })(M.ensembleTree(s, mode, true, seed));
    });
  }
  ok('bagging: every node may use x1 and x2', okB);
  ok('random forest: every node sees exactly 1 feature and splits only on it', okR);
  ok('trees never exceed depth 2', okD);
})();

/* CART picks the true minimum over every eligible (feature, threshold) */
(function () {
  var good = true;
  for (var seed = 100; seed < 160; seed++) {
    var s = M.bootstrap(M.rng(seed), 10);
    (function walk(n) {
      if (n.leaf) return;
      n.eligible.forEach(function (f) {
        M.candidates(n.rows, f).forEach(function (t) {
          if (M.fcmp(M.scoreSplit('cls', n.rows, f, t).score, n.split.score) < 0) good = false;
        });
      });
      walk(n.left); walk(n.right);
    })(M.ensembleTree(s, 'cls', false, 1));
  }
  ok('each grown node uses the lowest-scoring eligible split', good);
})();

var e3 = M.majorityError(0.3, 3);
ok('3 independent trees at 30% error: majority wrong 21.6%', Math.abs(e3 - 0.216) < 1e-12, e3);
ok('25 trees: under 3%', M.majorityError(0.3, 25) < 0.03, M.majorityError(0.3, 25));

console.log('\n' + pass + ' passed, ' + fail + ' failed');
process.exitCode = fail ? 1 : 0;
