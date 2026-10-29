/* ============================================================================
   Math checks for the support vector machine module.
   Run:  node test-math.js

   This exercises the SHIPPED script.js, not a copy of its formulas — the file
   exports its pure-math half under Node and skips everything DOM-bound.

   Besides the numbers, it checks the claims the page makes in words: line C
   really is the widest corridor, scaling w and w₀ never moves a ship's
   distance, support vectors sit on the margin edges (and one on-edge ship has
   α = 0), the captured shuttle makes the data inseparable, larger C narrows
   the corridor, and the quadratic kernel is a dot product of six features.
   ========================================================================= */

var M = require('./script.js');

var pass = 0, fail = 0;

function ok(name, cond, detail) {
  if (cond) { pass++; console.log('  PASS  ' + name); }
  else { fail++; console.log('  FAIL  ' + name + (detail ? '   -> ' + detail : '')); }
}
function section(t) { console.log('\n' + t); }
function close(a, b, tol) { return Math.abs(a - b) <= (tol || 1e-6); }
function sum(a) { return a.reduce(function (s, v) { return s + v; }, 0); }

var X = M.xsOf(M.SHIPS), Y = M.ysOf(M.SHIPS);
var SX = X.concat(M.xsOf(M.EXTRA)), SY = Y.concat(M.ysOf(M.EXTRA));
var RX = M.xsOf(M.RING), RY = M.ysOf(M.RING);

/* KKT conditions for a trained model, with tolerance. */
function kkt(m, tol) {
  tol = tol || 1e-4;
  var bad = [];
  m.X.forEach(function (x, n) {
    var a = m.alpha[n], yf = m.y[n] * m.f(x);
    if (a < -1e-12 || a > m.C * (1 + 1e-12)) bad.push(n + ': α out of box');
    else if (a <= M.ALPHA_TOL && yf < 1 - tol) bad.push(n + ': α = 0 but ỹf = ' + yf.toFixed(5));
    else if (a > M.ALPHA_TOL && a < m.C * (1 - 1e-9) && !close(yf, 1, tol)) bad.push(n + ': free α but ỹf = ' + yf.toFixed(5));
    else if (a >= m.C * (1 - 1e-9) && yf > 1 + tol) bad.push(n + ': α = C but ỹf = ' + yf.toFixed(5));
  });
  var bal = 0;
  m.X.forEach(function (_, n) { bal += m.alpha[n] * m.y[n]; });
  if (Math.abs(bal) > 1e-8) bad.push('Σαỹ = ' + bal);
  return bad;
}

/* ------------------------------------------------------ 1. the data -- */
section('1. Data and standardization');

ok('14 identified ships, 7 Rebel and 7 Imperial', X.length === 14 && Y.filter(function (y) { return y < 0; }).length === 7);
ok('every ship is inside the plotted square', SX.concat(RX).every(function (x) { return Math.abs(x[0]) < M.LIM && Math.abs(x[1]) < M.LIM; }));
var rt = M.standardize(M.toRaw([0.37, -1.4]));
ok('standardize undoes toRaw', close(rt[0], 0.37, 1e-12) && close(rt[1], -1.4, 1e-12));
var r5 = M.toRaw(M.SHIPS[4].x);
ok('Blue Three logs 810 MW and 14.4 dB (as the page shows)', close(r5[0], 810, 1e-9) && close(r5[1], 14.4, 1e-9));

/* ------------------------------------------------ 2. hard margin -- */
section('2. Hard-margin SVM (line C)');

var H = M.trainHard(X, Y);
ok('solver converged', H.converged, 'iters ' + H.iters);
ok('data are separable', H.separable);
ok('every constraint ỹₙf(xₙ) ≥ 1 holds', X.every(function (x, n) { return Y[n] * H.f(x) >= 1 - 1e-6; }));
ok('KKT conditions hold', kkt(H).length === 0, kkt(H).join('; '));
var wSum = [0, 0];
X.forEach(function (x, n) { wSum[0] += H.alpha[n] * Y[n] * x[0]; wSum[1] += H.alpha[n] * Y[n] * x[1]; });
ok('w = Σ αₙ ỹₙ xₙ', close(wSum[0], H.w[0], 1e-12) && close(wSum[1], H.w[1], 1e-12));
var margin = 1 / M.norm(H.w);
ok('geometric margin 1/‖w‖ equals min ỹₙf(xₙ)/‖w‖', close(margin, M.geometricMargin(H.w, H.w0, X, Y), 1e-6));
ok('margin rounds to 0.75 (scene 3 and 5 text): ' + margin.toFixed(4), margin.toFixed(2) === '0.75');
ok('strong duality: ½‖w‖² equals the dual objective',
  close(0.5 * M.dot(H.w, H.w), M.dualObjective(H), 1e-5), (0.5 * M.dot(H.w, H.w)) + ' vs ' + M.dualObjective(H));

// Brute-force widest corridor: for each direction u, the best gap is
// (min over Imperial uᵀx − max over Rebel uᵀx) / 2.
var best = -Infinity;
for (var k = 0; k < 200000; k++) {
  var t = 2 * Math.PI * k / 200000, u = [Math.cos(t), Math.sin(t)], lo = Infinity, hi = -Infinity;
  X.forEach(function (x, n) { var p = M.dot(u, x); if (Y[n] > 0) lo = Math.min(lo, p); else hi = Math.max(hi, p); });
  best = Math.max(best, (lo - hi) / 2);
}
ok('no direction gives a wider corridor (brute force over 200,000 angles): ' + best.toFixed(5), best <= margin + 1e-6 && best > margin - 1e-4);

var sv = [], edge = [];
X.forEach(function (x, n) {
  if (H.alpha[n] > M.ALPHA_TOL) sv.push(M.SHIPS[n].name);
  if (close(Y[n] * H.f(x), 1, 1e-4)) edge.push(M.SHIPS[n].name);
});
ok('support vectors: ' + sv.join(', '), sv.length === 3);
ok('every support vector sits on a margin edge, not on the boundary',
  X.every(function (x, n) { return H.alpha[n] <= M.ALPHA_TOL || (close(Y[n] * H.f(x), 1, 1e-4) && Math.abs(H.f(x)) > 0.5); }));
ok('Blue Three is on an edge but has α = 0 (scene 6 text)',
  edge.indexOf('Blue Three') >= 0 && H.alpha[4] <= M.ALPHA_TOL, 'α = ' + H.alpha[4]);
ok('complementary slackness αₙ[ỹₙf(xₙ) − 1] = 0 for every ship',
  X.every(function (x, n) { return Math.abs(H.alpha[n] * (Y[n] * H.f(x) - 1)) < 1e-5; }));

/* ------------------------------------------- 3. hand-drawn lines -- */
section('3. Hand-drawn lines A and B');

M.HAND_LINES.forEach(function (l) {
  var g = M.geometricMargin(l.w, l.w0, X, Y);
  ok('line ' + l.key + ' separates all 14 ships, margin ' + g.toFixed(3) + ' < trained ' + margin.toFixed(3), g > 0 && g < margin);
});
ok('line A margin rounds to 0.27 (scene 3 text)', M.geometricMargin(M.HAND_LINES[0].w, M.HAND_LINES[0].w0, X, Y).toFixed(2) === '0.27');

/* ------------------------------------------------- 4. scaling -- */
section('4. Scaling w and w₀');

[0.25, 0.5, 2, 4].forEach(function (s) {
  var w = [s * H.w[0], s * H.w[1]], w0 = s * H.w0;
  var sameDist = X.every(function (x) { return close(M.signedDistance(w, w0, x), M.signedDistance(H.w, H.w0, x), 1e-12); });
  var scaled = X.every(function (x) { return close(M.score(w, w0, x), s * H.f(x), 1e-12); });
  ok('s = ' + s + ': scores scale by s, distances unchanged', sameDist && scaled);
});
ok('the f = ±1 lines sit at 1/(s‖w‖) from the boundary', close(M.signedDistance([2 * H.w[0], 2 * H.w[1]], 2 * H.w0, M.clipLine([2 * H.w[0], 2 * H.w[1]], 2 * H.w0, 1)[0]), margin / 2, 1e-9));

/* -------------------------------------- 5. the t slider (scene 5) -- */
section('5. Rescaling the trained solution');

ok('t < 1 breaks at least one constraint', X.some(function (x, n) { return Y[n] * 0.95 * H.f(x) < 1; }));
ok('t > 1 keeps every constraint but narrows the margin',
  X.every(function (x, n) { return Y[n] * 1.05 * H.f(x) >= 1; }) && 1 / (1.05 * M.norm(H.w)) < margin);

/* ---------------------------------------- 6. dragging (scene 6) -- */
section('6. Moving ships');

var moved = X.map(function (x) { return x.slice(); });
moved[0] = [-2.6, -2.4];                       // Red Five: far from the line, α = 0
var H2 = M.trainHard(moved, Y);
ok('moving a non-support ship leaves w and w₀ unchanged',
  close(H2.w[0], H.w[0], 1e-6) && close(H2.w[1], H.w[1], 1e-6) && close(H2.w0, H.w0, 1e-6));
var moved2 = X.map(function (x) { return x.slice(); });
moved2[5] = [0.9, -2.3];                        // Ghost: a support vector
var H3 = M.trainHard(moved2, Y);
ok('moving a support vector changes the line', !close(H3.w[0], H.w[0], 1e-3) || !close(H3.w0, H.w0, 1e-3));
var bad = X.map(function (x) { return x.slice(); });
bad[0] = [2.2, 1.5];                            // a Rebel deep among Imperials
ok('trainHard reports an inseparable layout', !M.trainHard(bad, Y).separable);

/* --------------------------------------------- 7. soft margin -- */
section('7. Soft margin with the shuttle and courier');

ok('with the extra ships, no line separates the data (hard margin fails)', !M.trainHard(SX, SY).separable);
var widths = [], slackSums = [];
[0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100].forEach(function (C) {
  var m = M.trainLinear(SX, SY, C);
  var xi = M.slacks(m.f, SX, SY), w = m.w;
  widths.push(2 / M.norm(w)); slackSums.push(sum(xi));
  var k = kkt(m, 1e-4);
  ok('C = ' + C + ': converged, KKT holds, width ' + (2 / M.norm(w)).toFixed(2) + ', Σξ ' + sum(xi).toFixed(2),
    m.converged && k.length === 0, k.join('; '));
  ok('C = ' + C + ': strong duality, primal = dual',
    close(M.primalObjective(w, m.w0, SX, SY, C), M.dualObjective(m), 1e-4 * Math.max(1, C)),
    M.primalObjective(w, m.w0, SX, SY, C) + ' vs ' + M.dualObjective(m));
  ok('C = ' + C + ': ξₙ matches max(0, 1 − ỹₙf) and every ξ > 0 ship has α = C',
    xi.every(function (v, n) { return v === Math.max(0, 1 - SY[n] * m.f(SX[n])) && (v < 1e-4 || m.alpha[n] >= C * (1 - 1e-9)); }));
});
ok('larger C never widens the corridor', widths.every(function (v, i) { return i === 0 || v <= widths[i - 1] + 1e-6; }), widths.map(function (v) { return v.toFixed(3); }).join(' '));
ok('larger C never increases total slack', slackSums.every(function (v, i) { return i === 0 || v <= slackSums[i - 1] + 1e-6; }), slackSums.map(function (v) { return v.toFixed(3); }).join(' '));
var m1 = M.trainLinear(SX, SY, 1), xi1 = M.slacks(m1.f, SX, SY);
ok('at C = 1 some ship is inside the corridor yet correctly classified (slack ≠ mistake)',
  xi1.some(function (v) { return v > 1e-6 && v <= 1; }));
ok('at C = 1 the Tydirium is on the wrong side (ξ > 1)', xi1[14] > 1, 'ξ = ' + xi1[14]);

/* ----------------------------------------------------- 8. kernels -- */
section('8. Kernels on the ring data');

var rng = M.RING;
ok('ring data: 10 Rebel smugglers inside 12 Imperial patrols', RY.filter(function (y) { return y < 0; }).length === 10 && RY.length === 22);
var lin = M.trainSVM(RX, RY, M.KERNELS.linear(), 10);
var linErr = RX.filter(function (x, n) { return RY[n] * lin.f(x) <= 0; }).length;
ok('a straight line makes training mistakes (' + linErr + ')', linErr > 0);
[['quadratic', M.KERNELS.quadratic()], ['RBF γ = 0.5', M.KERNELS.rbf(0.5)], ['RBF γ = 0.1', M.KERNELS.rbf(0.1)], ['RBF γ = 5', M.KERNELS.rbf(5)]].forEach(function (c) {
  var m = M.trainSVM(RX, RY, c[1], 10);
  var err = RX.filter(function (x, n) { return RY[n] * m.f(x) <= 0; }).length;
  ok(c[0] + ': converged, KKT holds, ' + err + ' training mistakes', m.converged && kkt(m).length === 0, kkt(m).join('; '));
});
var q = M.trainSVM(RX, RY, M.KERNELS.quadratic(), 10);
ok('quadratic kernel separates the ring', RX.every(function (x, n) { return RY[n] * q.f(x) > 0; }));
var phiOk = true;
for (var a = 0; a < RX.length; a++) for (var b = 0; b < RX.length; b++) {
  var pa = M.phiQuadratic(RX[a]), pb = M.phiQuadratic(RX[b]), d = 0;
  for (var i = 0; i < 6; i++) d += pa[i] * pb[i];
  if (!close(d, M.KERNELS.quadratic()(RX[a], RX[b]), 1e-10)) phiOk = false;
}
ok('(xᵀz + 1)² = φ(x)ᵀφ(z) for every pair of ring ships', phiOk);

// A linear SVM through trainSVM equals the explicit-w version.
var viaK = M.trainSVM(SX, SY, M.KERNELS.linear(), 1);
ok('kernel form with K = xᵀz gives the same scores as wᵀx + w₀',
  SX.every(function (x) { return close(viaK.f(x), m1.f(x), 1e-6); }));

/* --------------------------------------------------- 9. geometry -- */
section('9. Geometry helpers');

var seg = M.clipLine([1, 0], -0.5, 0);
ok('clipLine: x₁ = 0.5 spans the square vertically', seg && close(seg[0][0], 0.5) && close(seg[1][0], 0.5) && close(Math.abs(seg[0][1] - seg[1][1]), 6));
ok('clipLine: a line outside the square gives null', M.clipLine([1, 0], -5, 0) === null);
var half = M.clipHalf(M.box(), [1, 1], 0, 0, 1);
var area = 0;
half.forEach(function (p, i) { var r = half[(i + 1) % half.length]; area += p[0] * r[1] - r[0] * p[1]; });
ok('clipHalf: the diagonal cuts the square in half (area 18)', close(Math.abs(area) / 2, 18, 1e-9));
var gx = M.linspace(-3, 3, 61);
var vals = gx.map(function (u) { return gx.map(function (v) { return u * u + v * v; }); });
var segs = M.contour(vals, gx, gx, 4);
ok('contour: the circle of radius 2 is traced at the right radius',
  segs.length > 20 && segs.every(function (s) { return s.every(function (p) { return Math.abs(Math.hypot(p[0], p[1]) - 2) < 0.02; }); }));

console.log('\n' + pass + ' passed, ' + fail + ' failed');
if (fail) process.exit(1);
