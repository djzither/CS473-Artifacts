/* ============================================================================
   Math checks for "How Tree Ensembles Learn".   Run:  node test-math.js
   Exercises the SHIPPED model.js and data.js and checks every claim the page
   makes in words.
   ========================================================================= */

var E = require('./model.js');
var pass = 0, fail = 0;
function ok(name, cond, detail) {
  if (cond) { pass++; console.log('  PASS  ' + name); }
  else { fail++; console.log('  FAIL  ' + name + (detail !== undefined ? '   -> ' + detail : '')); }
}
function section(t) { console.log('\n' + t); }
function close(a, b, tol) { return Math.abs(a - b) <= (tol || 1e-9); }

var RF = E.runRF(), LS = E.runLS(), ADA = E.runAda(), LOGIT = E.runLogit();
var BOOST = [LS, ADA, LOGIT];

/* --------------------------------------------------------------- data -- */
section('Data and split');
ok('714 passengers with a recorded age', E.ALL.length === 714);
ok('500 training / 214 test', E.TRAIN.length === 500 && E.TEST.length === 214);
ok('no passenger is in both sets', E.ALL.every(function (p) { return p.train === (p.ti >= 0); }));
ok('every age and fare is a number', E.ALL.every(function (p) { return isFinite(p.age) && isFinite(p.fare); }));
var again = (function () {
  var order = E.ALL.map(function (_, i) { return i; }), r = E.rng(E.SEED);
  for (var i = order.length - 1; i > 0; i--) { var j = Math.floor(r() * (i + 1)), t = order[i]; order[i] = order[j]; order[j] = t; }
  return order.slice(0, 500).sort(function (a, b) { return a - b; }).join(',');
})();
ok('split is reproducible from seed 473', again === E.TRAIN.map(function (p) { return p.k; }).join(','));
var heik = E.ALL.filter(function (p) { return p.id === 3; })[0];
ok('default passenger Heikkinen, Miss. Laina is a training passenger who survived', heik.train && heik.y === 1 && heik.pclass === 3);
ok('x[0] = 1 means male', E.ALL.every(function (p) { return p.x[0] === (p.female ? 0 : 1); }));

/* -------------------------------------------------------------- trees -- */
section('Trees');
function allTrees() {
  var t = [];
  [RF, LS, ADA, LOGIT].forEach(function (R) { R.rounds.forEach(function (r) { t.push({ key: R.key, tree: r.tree }); }); });
  return t;
}
var depthOK = true, leafOK = true;
allTrees().forEach(function (o) {
  E.eachNode(o.tree, function (n) { if (n.depth > E.DEPTH) depthOK = false; if (n.leaf && n.n < E.MIN_LEAF) leafOK = false; });
});
ok('every tree has depth <= 2', depthOK);
ok('every leaf holds at least 5 distinct passengers', leafOK);

section('Random forest');
var candOK = true, bestOK = true;
RF.rounds.forEach(function (r) {
  E.eachNode(r.tree, function (n) {
    if (!n.cands) return;
    if (n.cands.length !== E.MTRY) candOK = false;
    if (!n.leaf) {
      var valid = n.cands.filter(function (c) { return c.score !== null; });
      var min = Math.min.apply(null, valid.map(function (c) { return c.score; }));
      if (!close(n.score, min, 1e-12) || n.cands.every(function (c) { return c.f !== n.f; })) bestOK = false;
    }
  });
});
ok('every split node saw exactly 2 of the 4 features', candOK);
ok('every split took the best-scoring candidate', bestOK);
var indep = [7, 23, 50].every(function (m) {
  return JSON.stringify(E.rfTree(m).tree) === JSON.stringify(RF.rounds[m - 1].tree);
});
ok('trees 7, 23, 50 grown alone equal the same trees grown in sequence (order is only playback)', indep);
var reversed = [];
for (var m = E.M; m >= 1; m--) reversed[m - 1] = JSON.stringify(E.rfTree(m).tree);
ok('growing all 50 trees in reverse order gives the same forest', reversed.every(function (s, i) { return s === JSON.stringify(RF.rounds[i].tree); }));
var bsum = RF.rounds.every(function (r) { var s = 0; r.counts.forEach(function (c) { s += c; }); return s === 500; });
ok('each bootstrap sample has 500 draws', bsum);
var meanDistinct = RF.rounds.reduce(function (a, r) { return a + r.distinct; }, 0) / E.M / 500;
ok('about 63% of passengers appear in each sample (got ' + (100 * meanDistinct).toFixed(1) + '%)', meanDistinct > 0.6 && meanDistinct < 0.66);
var roots = {}; RF.rounds.forEach(function (r) { if (!r.tree.leaf) roots[r.tree.f] = 1; });
ok('sex does not win every root (roots use ' + Object.keys(roots).length + ' different features)', Object.keys(roots).length >= 2);
ok('running vote equals the count of trees voting survived', RF.rounds.every(function (r) {
  return E.ALL.every(function (p) { var v = 0; for (var j = 0; j < r.m; j++) v += RF.rounds[j].F[p.k]; return v === r.votesAfter[p.k]; });
}));

section('Least squares boosting');
var ybar = E.TRAIN.reduce(function (a, p) { return a + p.y; }, 0) / 500;
ok('f0 is the training survival rate ' + ybar.toFixed(3), close(LS.f0, ybar));
ok('round-1 residuals sum to zero', Math.abs(LS.rounds[0].resid.reduce(function (a, b) { return a + b; }, 0)) < 1e-9);
ok('line search gives beta = 1 every round', LS.rounds.every(function (r) { return close(r.beta, 1, 1e-9); }));
ok('numeric line search agrees (beta* = 1 to 3 decimals)', LS.rounds.every(function (r) { return Math.abs(r.betaStar - 1) < 1e-3; }));
ok('each leaf predicts the mean residual of its passengers', LS.rounds.every(function (r) {
  var ok2 = true;
  E.eachLeaf(r.tree, function (lf) {
    var s = 0, n = 0;
    E.TRAIN.forEach(function (p, i) { if (E.leafOf(r.tree, p.x) === lf) { s += r.resid[i]; n++; } });
    if (n !== lf.n || !close(s / n, lf.value, 1e-9)) ok2 = false;
  });
  return ok2;
}));

section('AdaBoost');
ok('round 1 starts with equal weights', ADA.rounds[0].w.every(function (w) { return close(w, 1 / 500); }));
ok('weights are e^(-y f) normalized', ADA.rounds.every(function (r) {
  var raw = E.TRAIN.map(function (p) { return Math.exp(-(2 * p.y - 1) * r.fBefore[p.k]); }), Z = raw.reduce(function (a, b) { return a + b; }, 0);
  return raw.every(function (v, i) { return close(v / Z, r.w[i], 1e-12); });
}));
ok('beta = 1/2 log((1 - err) / err)', ADA.rounds.every(function (r) { return close(r.beta, 0.5 * Math.log((1 - r.err) / r.err), 1e-12); }));
ok('weighted error is below 1/2 every round, so beta > 0', ADA.rounds.every(function (r) { return r.err < 0.5 && r.beta > 0; }));
ok('closed-form beta equals the numeric line search', ADA.rounds.every(function (r) { return Math.abs(r.betaStar - r.beta) < 1e-3; }));
ok('after reweighting, the passengers tree m got wrong hold exactly half the weight', ADA.rounds.every(function (r) {
  var s = 0; E.TRAIN.forEach(function (p, i) { if (r.F[p.k] !== 2 * p.y - 1) s += r.wNext[i]; });
  return close(s, 0.5, 1e-9);
}));
ok('wNext of round m equals w of round m + 1', ADA.rounds.slice(0, -1).every(function (r, j) {
  var w2 = ADA.rounds[j + 1].w; return r.wNext.every(function (v, i) { return close(v, w2[i], 1e-12); });
}));
ok('leaves output +1 or -1', ADA.rounds.every(function (r) { var g = true; E.eachLeaf(r.tree, function (l) { if (l.value !== 1 && l.value !== -1) g = false; }); return g; }));
ok('round-1 tree is the first tree and beta1 = 0.693', close(ADA.rounds[0].beta, 0.5 * Math.log(0.8 / 0.2), 0.01), ADA.rounds[0].beta);

section('LogitBoost');
ok('round 1: every pi = 0.5 and weight = 0.25', LOGIT.rounds[0].pi.every(function (q) { return q === 0.5; }) && LOGIT.rounds[0].w.every(function (w) { return w === 0.25; }));
ok('z is (y - pi) / (pi (1 - pi)) clipped to [-4, 4]', LOGIT.rounds.every(function (r) {
  return E.TRAIN.every(function (p, i) { var v = (p.y - r.pi[i]) / (r.pi[i] * (1 - r.pi[i])); return close(Math.max(-4, Math.min(4, v)), r.z[i], 1e-9); });
}));
ok('w * z = y - pi whenever z is not clipped', LOGIT.rounds.every(function (r) {
  return E.TRAIN.every(function (p, i) { return Math.abs(r.z[i]) >= 4 - 1e-12 || close(r.w[i] * r.z[i], p.y - r.pi[i], 1e-9); });
}));
ok('f_m = f_{m-1} + F_m / 2', LOGIT.rounds.every(function (r) { return E.ALL.every(function (p) { return close(r.fAfter[p.k], r.fBefore[p.k] + 0.5 * r.F[p.k], 1e-12); }); }));
ok('the 1/2 step differs from the exact line search in at least one round', LOGIT.rounds.some(function (r) { return Math.abs(r.betaStar - 0.5) > 0.05; }));

section('Forward stagewise (all three boosting runs)');
BOOST.forEach(function (R) {
  ok(R.key + ': f_m = f_{m-1} + beta_m F_m for every passenger and round', R.rounds.every(function (r) {
    return E.ALL.every(function (p) { return close(r.fAfter[p.k], r.fBefore[p.k] + r.beta * r.F[p.k], 1e-12); });
  }));
  ok(R.key + ': rounds chain (f_before of round m+1 = f_after of round m)', R.rounds.slice(0, -1).every(function (r, j) {
    var n = R.rounds[j + 1]; return E.ALL.every(function (p) { return n.fBefore[p.k] === r.fAfter[p.k]; });
  }));
});
[LS, ADA].forEach(function (R) {
  ok(R.key + ': training loss never goes up (exact line search)', R.rounds.every(function (r) { return r.lossAfter <= r.lossBefore + 1e-12; }));
});
ok('logit: training log loss lower after 50 rounds than at the start', LOGIT.rounds[49].lossAfter < LOGIT.rounds[0].lossBefore);

section('Accuracy figures');
var baseline = E.TEST.filter(function (p) { return !p.y; }).length / E.TEST.length;
[RF, LS, ADA, LOGIT].forEach(function (R) {
  var r = R.rounds[R.rounds.length - 1];
  ok(R.key + ': after 50 rounds, test accuracy ' + (100 * r.accTestAfter).toFixed(1) + '% beats the ' + (100 * baseline).toFixed(1) + '% all-died baseline',
    r.accTestAfter > baseline + 0.1);
});

console.log('\n' + pass + ' passed, ' + fail + ' failed');
process.exit(fail ? 1 : 0);
