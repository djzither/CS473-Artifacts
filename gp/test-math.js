/* ============================================================================
   Math checks for the Gaussian process module.
   Run:  node test-math.js

   This exercises the SHIPPED script.js, not a copy of its formulas — the file
   exports its pure-math half under Node and skips everything DOM-bound.

   Besides the numbers, it checks the claims the page makes in words: the prior
   is flat with a constant band, noise-free means pass through the data with
   the band pinched shut, noisy means don't have to, the band shows f and not
   y, repeated inputs don't break anything, and samples are joint draws.
   ========================================================================= */

var M = require('./script.js');

var pass = 0, fail = 0;

function ok(name, cond, detail) {
  if (cond) { pass++; console.log('  PASS  ' + name); }
  else { fail++; console.log('  FAIL  ' + name + (detail ? '   -> ' + detail : '')); }
}

function section(t) { console.log('\n' + t); }

function maxAbs(a) { var m = 0; for (var i = 0; i < a.length; i++) m = Math.max(m, Math.abs(a[i])); return m; }
function allFinite(a) { for (var i = 0; i < a.length; i++) if (!isFinite(a[i])) return false; return true; }

function preset(n, sigma) {
  var obs = [];
  for (var i = 0; i < n; i++) obs.push({ x: M.PRESET_X[i], y: M.presetY(i, sigma), src: 'preset' });
  return obs;
}

/* Dense reference: solve (K + (σ² + jitter)I) with Gauss–Jordan, no Cholesky. */
function referencePosterior(obs, ell, sigma, xs, jitter) {
  var n = obs.length, i, j, k;
  var A = obs.map(function (p, r) {
    return obs.map(function (q, c) { return M.rbf(p.x, q.x, ell) + (r === c ? sigma * sigma + jitter : 0); });
  });
  var inv = obs.map(function (_, r) { return obs.map(function (_, c) { return r === c ? 1 : 0; }); });
  for (i = 0; i < n; i++) {
    var piv = A[i][i];
    for (j = 0; j < n; j++) { A[i][j] /= piv; inv[i][j] /= piv; }
    for (k = 0; k < n; k++) if (k !== i) {
      var f = A[k][i];
      for (j = 0; j < n; j++) { A[k][j] -= f * A[i][j]; inv[k][j] -= f * inv[i][j]; }
    }
  }
  return xs.map(function (x) {
    var ks = obs.map(function (p) { return M.rbf(p.x, x, ell); });
    var mean = 0, q = 0;
    for (i = 0; i < n; i++) for (j = 0; j < n; j++) {
      mean += ks[i] * inv[i][j] * obs[j].y;
      q += ks[i] * inv[i][j] * ks[j];
    }
    return { mean: mean, v: 1 - q };
  });
}


/* -------------------------------------------------------- 1. the kernel -- */
section('1. The RBF kernel');

ok('K(x, x) = 1', M.rbf(1.3, 1.3, 0.7) === 1);
ok('K(0, ℓ) = e^(−1/2)', Math.abs(M.rbf(0, 2, 2) - Math.exp(-0.5)) < 1e-15);
ok('symmetric', M.rbf(-1, 2.5, 0.8) === M.rbf(2.5, -1, 0.8));
ok('larger ℓ gives higher correlation at a fixed distance', M.rbf(0, 1, 2) > M.rbf(0, 1, 0.5));


/* ------------------------------------------------------- 2. zero points -- */
section('2. The prior (zero observations)');

var prior = M.posterior([], 1, 0);
ok('prior mean is 0 everywhere', maxAbs(prior.mean) === 0);
ok('prior sd is 1 everywhere', prior.sd.every(function (s) { return Math.abs(s - 1) < 1e-15; }));
ok('prior covariance is the kernel', Math.abs(prior.cov[10][30] - M.rbf(M.GRID[10], M.GRID[30], 1)) < 1e-15);
var priorNoisy = M.posterior([], 1, 0.8);
ok('noise does not widen the band for f (sd still 1 with σ = 0.8)',
  priorNoisy.sd.every(function (s) { return Math.abs(s - 1) < 1e-15; }));


/* --------------------------------------------------------- 3. one point -- */
section('3. One observation');

var one = [{ x: 0.5, y: 1.2 }], ell = 1.3, sg = 0.4;
var post1 = M.posterior(one, ell, sg);
var bad1 = 0;
M.GRID.forEach(function (x, a) {
  var k = M.rbf(0.5, x, ell), d = 1 + sg * sg + M.JITTER;
  if (Math.abs(post1.mean[a] - k * 1.2 / d) > 1e-12) bad1++;
  if (Math.abs(post1.cov[a][a] - (1 - k * k / d)) > 1e-12) bad1++;
});
ok('matches the closed form m = k·y/(1+σ²), v = 1 − k²/(1+σ²) at all ' + M.GRID_N + ' grid points', bad1 === 0);
var a0 = 110; // x = 0.5 on the grid
ok('grid index 110 is x = 0.5', Math.abs(M.GRID[a0] - 0.5) < 1e-12);
ok('sd is smallest at the observed input', post1.sd.every(function (s) { return s >= post1.sd[a0] - 1e-15; }));
ok('far from the point the posterior is back to the prior (|x − 0.5| = 5.5)',
  Math.abs(post1.sd[0] - 1) < 1e-3 && Math.abs(post1.mean[0]) < 1e-3);


/* ---------------------------------------------- 4. matches a reference -- */
section('4. Cholesky posterior matches a direct inverse');

[[3, 0, 1], [6, 0.3, 0.6], [10, 0.1, 2], [10, 0, 0.4]].forEach(function (c) {
  var obs = preset(c[0], c[1]), xs = [-4.1, -0.3, 0.77, 2.2, 4.9];
  var p = M.posterior(obs, c[2], c[1], xs), ref = referencePosterior(obs, c[2], c[1], xs, p.jitter);
  var err = 0;
  ref.forEach(function (r, i) { err = Math.max(err, Math.abs(r.mean - p.mean[i]), Math.abs(r.v - p.cov[i][i])); });
  ok(c[0] + ' points, σ = ' + c[1] + ', ℓ = ' + c[2] + ': mean and variance agree to 1e-6', err < 1e-6, 'max diff ' + err);
});


/* -------------------------------------------- 5. noise-free interpolation -- */
section('5. Noise-free interpolation');

[0.3, 1, 3].forEach(function (L) {
  var obs = preset(10, 0), xs = obs.map(function (p) { return p.x; });
  var p = M.posterior(obs, L, 0, xs);
  var err = 0, sd = 0;
  obs.forEach(function (o, i) { err = Math.max(err, Math.abs(p.mean[i] - o.y)); sd = Math.max(sd, p.sd[i]); });
  ok('ℓ = ' + L + ': mean passes through all 10 points (max miss ' + err.toExponential(1) + ')', err < 1e-3);
  ok('ℓ = ' + L + ': sd at the points is essentially zero (max ' + sd.toExponential(1) + ')', sd < 1e-2);
});


/* ------------------------------------------------------------- 6. noise -- */
section('6. Noisy observations');

var noisyObs = preset(6, 0.3), nx = noisyObs.map(function (p) { return p.x; });
var pn = M.posterior(noisyObs, 1, 0.3, nx);
var missed = noisyObs.filter(function (o, i) { return Math.abs(pn.mean[i] - o.y) > 0.01; }).length;
ok('σ = 0.3: the mean misses at least one point by more than 0.01 (' + missed + ' of 6)', missed > 0);
ok('σ = 0.3: sd for f stays above 0.1 at every observed input',
  Array.prototype.every.call(pn.sd, function (s) { return s > 0.1; }));
var pq = M.posterior(noisyObs, 1, 0.3), pf = M.posterior(noisyObs, 1, 0);
ok('the noisy band is wider than the noise-free one at every grid point',
  Array.prototype.every.call(pq.sd, function (s, a) { return s >= pf.sd[a] - 1e-9; }));
var obsAt = M.posterior([{ x: 1, y: 0 }], 1, 0.5, [1]);
ok('one point, σ = 0.5: Var f(x₁) = 1 − 1/1.25 = 0.2, not 0.2 + σ²', Math.abs(obsAt.cov[0][0] - 0.2) < 1e-7);


/* ---------------------------------------------------- 7. more data -- */
section('7. More data, fixed settings');

[0, 0.3].forEach(function (s) {
  var never = true;
  for (var n = 1; n <= 10; n++) {
    var a = M.posterior(preset(n - 1, s), 1, s), b = M.posterior(preset(n, s), 1, s);
    for (var i = 0; i < M.GRID_N; i++) if (b.sd[i] > a.sd[i] + 1e-9) never = false;
  }
  ok('σ = ' + s + ': adding each preset point never widens the band anywhere', never);
});
var anyWorse = false;
for (var n = 1; n <= 10; n++) {
  if (M.worseFraction(M.posterior(preset(n - 1, 0), 1, 0).mean, M.posterior(preset(n, 0), 1, 0).mean) > 0) anyWorse = true;
}
ok('...but at least one step makes the mean less accurate somewhere (the page’s caveat is real)', anyWorse);


/* ------------------------------------------------- 8. repeated inputs -- */
section('8. Repeated and nearly repeated inputs');

var rep = [{ x: 1, y: 0.5 }, { x: 1, y: -0.5 }, { x: -2, y: 0.3 }];
var pr = M.posterior(rep, 1, 0, [1, 0, -2]);
ok('exact repeat, noise-free: finite', allFinite(pr.mean) && allFinite(pr.sd));
ok('exact repeat, conflicting y: the mean at that x is their average', Math.abs(pr.mean[0]) < 1e-4);
ok('exact repeat: other points still interpolated', Math.abs(pr.mean[2] - 0.3) < 1e-4);
var same = M.posterior([{ x: 1, y: 0.7 }, { x: 1, y: 0.7 }], 1, 0, [1, 2]);
ok('exact repeat, same y: same posterior as a single point',
  Math.abs(same.mean[0] - 0.7) < 1e-6 && Math.abs(same.mean[1] - 0.7 * Math.exp(-0.5)) < 1e-6);
var tiny = M.posterior([{ x: 1, y: 0.2 }, { x: 1 + 1e-12, y: 0.2 }, { x: 1 + 2e-12, y: 0.2 }], 3, 0);
ok('three inputs 1e-12 apart with ℓ = 3: Cholesky still succeeds and the result is finite',
  allFinite(tiny.mean) && allFinite(tiny.sd));
var rpn = M.posterior(rep, 1, 0.3, [1]);
ok('exact repeat with noise: finite and between the two y values',
  isFinite(rpn.mean[0]) && rpn.mean[0] > -0.5 && rpn.mean[0] < 0.5);
var snap = M.addObservation([{ x: 1.2, y: 0 }], 1.25, 1);
ok('a click within ' + M.SNAP + ' of an input reuses that x exactly', snap.repeat && snap.obs[1].x === 1.2);
var noSnap = M.addObservation([{ x: 1.2, y: 0 }], 1.4, 1);
ok('a click farther away keeps its own x', !noSnap.repeat && noSnap.obs[1].x === 1.4);
ok('addObservation does not change the array it was given', snap.obs.length === 2 && M.addObservation([], 0, 0).obs.length === 1);


/* -------------------------------------------------------- 9. samples -- */
section('9. Posterior samples');

var ps = M.posterior(preset(3, 0), 1, 0);
var big = [];
var r = M.mulberry32(99);
for (var s = 0; s < 4000; s++) big.push(M.normals(r, M.GRID_N));
var draws = M.drawSamples(ps, big);
function empCov(i, j) {
  var mi = 0, mj = 0, c = 0, N = draws.length;
  draws.forEach(function (d) { mi += d[i]; mj += d[j]; });
  mi /= N; mj /= N;
  draws.forEach(function (d) { c += (d[i] - mi) * (d[j] - mj); });
  return { mi: mi, c: c / (N - 1) };
}
var pairs = [[40, 40], [40, 45], [150, 160], [20, 120]], covOk = true, detail = '';
pairs.forEach(function (pq2) {
  var e = empCov(pq2[0], pq2[1]), want = ps.cov[pq2[0]][pq2[1]];
  if (Math.abs(e.c - want) > 0.06 || Math.abs(e.mi - ps.mean[pq2[0]]) > 0.06) { covOk = false; detail += ' (' + pq2 + ': ' + e.c.toFixed(3) + ' vs ' + want.toFixed(3) + ')'; }
});
ok('4000 draws reproduce the posterior mean and covariance, including between different x', covOk, detail);
var e45 = empCov(40, 45);
ok('neighboring points in a sample are strongly correlated (joint draw, not independent): ' + (e45.c / Math.sqrt(empCov(40, 40).c * empCov(45, 45).c)).toFixed(2),
  e45.c / Math.sqrt(empCov(40, 40).c * empCov(45, 45).c) > 0.9);
var noisefree = M.posterior(preset(5, 0), 1, 0), d5 = M.drawSamples(noisefree, big.slice(0, 10));
var gx = M.PRESET_X.slice(0, 5).map(function (x) { return Math.round((x - M.X_MIN) / (M.X_MAX - M.X_MIN) * (M.GRID_N - 1)); });
var through = d5.every(function (f) { return gx.every(function (a, i) { return Math.abs(f[a] - M.presetY(i, 0)) < 0.01; }); });
ok('noise-free: every sample passes through the observed points', through);
[[0.2, 0], [3, 0], [3, 0.5]].forEach(function (c) {
  var dd = M.drawSamples(M.posterior(c[1] ? preset(10, c[1]) : [], c[0], c[1]), big.slice(0, 3));
  ok('drawSamples is finite for ℓ = ' + c[0] + ', σ = ' + c[1] + (c[1] ? ', 10 points' : ', prior'),
    dd.every(allFinite));
});


/* ------------------------------------------------------- 10. the reset -- */
section('10. Reset and the preset sequence');

ok('preset inputs are distinct and on the axis', M.PRESET_X.every(function (x, i) {
  return x >= M.X_MIN && x <= M.X_MAX && M.PRESET_X.indexOf(x) === i;
}));
ok('noise-free preset y is exactly sin(x)', M.presetY(2, 0) === Math.sin(M.PRESET_X[2]));
var again = M.posterior([], M.START.ell, M.START.sigma);
ok('after reset (no data, starting ℓ and σ) the posterior is the prior again',
  maxAbs(again.mean) === 0 && again.sd.every(function (v) { return v === 1; }));


console.log('\n' + pass + ' passed, ' + fail + ' failed');
if (fail) process.exit(1);
