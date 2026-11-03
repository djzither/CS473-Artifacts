/* ============================================================================
   Math checks for "Which Ships Count?".   Run:  node test-math.js
   Exercises the SHIPPED script.js (it exports its math half under Node) and
   checks the claims the page makes in words.
   ========================================================================= */

var M = require('./script.js');
var pass = 0, fail = 0;
function ok(name, cond, detail) {
  if (cond) { pass++; console.log('  PASS  ' + name); }
  else { fail++; console.log('  FAIL  ' + name + (detail !== undefined ? '   -> ' + detail : '')); }
}
function section(t) { console.log('\n' + t); }
function close(a, b, tol) { return Math.abs(a - b) <= (tol || 1e-6); }

/* KKT conditions for a trained classifier. */
function kktSVC(m, tol) {
  tol = tol || 1e-4;
  var bad = [], bal = 0;
  m.X.forEach(function (x, n) {
    var a = m.alpha[n], yf = m.y[n] * m.f(x);
    bal += a * m.y[n];
    if (a < -1e-12 || a > m.C + 1e-9) bad.push(n + ' box');
    else if (a <= 1e-8 && yf < 1 - tol) bad.push(n + ' a=0 yf=' + yf);
    else if (a > 1e-8 && a < m.C - 1e-8 && !close(yf, 1, tol)) bad.push(n + ' free yf=' + yf);
    else if (a >= m.C - 1e-8 && yf > 1 + tol) bad.push(n + ' a=C yf=' + yf);
  });
  if (Math.abs(bal) > 1e-8) bad.push('sum ay = ' + bal);
  return bad;
}

/* ------------------------------------------------- scenes 1-2 -- */
section('Scenes 1-2: linear SVM and hinge loss');
var X = M.xsOf(M.SHIPS), Y = M.ysOf(M.SHIPS);
var m = M.trainSVC(X, Y, M.linearK, M.SHIPS_C), w = M.weights(m);
ok('solver converged', m.converged);
ok('KKT conditions hold', kktSVC(m).length === 0, kktSVC(m).join('; '));
ok('w.x + b equals the kernel expansion everywhere',
  [[0, 0], [1.3, -2.4], [-2, 2], [2.5, 2.5]].every(function (x) { return close(M.dot(w, x) + m.b, m.f(x), 1e-9); }));
var primal = 0.5 * M.dot(w, w), dual = 0;
X.forEach(function (x, n) { primal += M.SHIPS_C * M.hinge(Y[n] * m.f(x)); dual += m.alpha[n]; });
X.forEach(function (x, n) { X.forEach(function (z, k) { dual -= 0.5 * m.alpha[n] * m.alpha[k] * Y[n] * Y[k] * M.dot(x, z); }); });
ok('strong duality (primal = dual)', close(primal, dual, 1e-5), primal + ' vs ' + dual);
ok('4 support vectors', m.sv.length === 4, m.sv.length);
var byName = {};
M.SHIPS.forEach(function (s, n) { byName[s.name] = n; });
var yg = byName['Y-wing Gray'], ygm = Y[yg] * m.f(X[yg]);
ok('Y-wing Gray is correctly classified but inside the margin (0 < yf < 1)', ygm > 0 && ygm < 1, ygm);
ok('... and still has positive hinge loss and alpha = C', M.hinge(ygm) > 0 && close(m.alpha[yg], M.SHIPS_C, 1e-6));
var dt = byName['Disguised TIE'];
ok('Disguised TIE is misclassified, hinge loss > 1', Y[dt] * m.f(X[dt]) < 0 && M.hinge(Y[dt] * m.f(X[dt])) > 1);
['Blue Three', 'Probe Carrier'].forEach(function (nm) {
  var k = byName[nm];
  ok(nm + ' sits on its margin edge: loss 0, yet a support vector', close(Y[k] * m.f(X[k]), 1, 1e-4) && m.alpha[k] > 1e-6);
});
var sx = m.sv.map(function (k) { return X[k]; }), sc = m.sv.map(function (k) { return m.coef[k]; });
function fsv(x) { var s = m.b; sx.forEach(function (z, i) { s += sc[i] * M.dot(z, x); }); return s; }
ok('prediction from support vectors alone is identical', [[0.3, 0.1], [-2, 1], [2, -2]].every(function (x) { return close(fsv(x), m.f(x), 1e-12); }));
var m2 = M.trainSVC(sx, m.sv.map(function (k) { return Y[k]; }), M.linearK, M.SHIPS_C), w2 = M.weights(m2);
ok('retraining on support vectors alone gives the same line', close(w2[0], w[0], 1e-4) && close(w2[1], w[1], 1e-4) && close(m2.b, m.b, 1e-4));
ok('logistic loss is never 0; log2 form passes through (0, 1)', M.logLoss2(10) > 0 && close(M.logLoss2(0), 1, 1e-12));

/* ---------------------------------------------------- scene 3 -- */
section('Scene 3: C and gamma');
var RX = M.xsOf(M.RING_TRAIN), RY = M.ysOf(M.RING_TRAIN), TX = M.xsOf(M.RING_TEST), TY = M.ysOf(M.RING_TEST);
function acc(mod, A, B) { return A.filter(function (x, i) { return (mod.f(x) > 0 ? 1 : -1) === B[i]; }).length / A.length; }
var allKKT = true, allConv = true;
M.C_STEPS.forEach(function (C) {
  M.GAMMA_STEPS.forEach(function (g) {
    var r = M.trainSVC(RX, RY, M.rbfK(g), C);
    if (!r.converged) allConv = false;
    if (kktSVC(r, 1e-3).length) allKKT = false;
  });
});
ok('every C x gamma setting converges', allConv);
ok('every C x gamma setting satisfies KKT', allKKT);
// presets in index.html: data-preset="cIndex,gammaIndex"
var under = M.trainSVC(RX, RY, M.rbfK(M.GAMMA_STEPS[0]), M.C_STEPS[1]);
var bal = M.trainSVC(RX, RY, M.rbfK(M.GAMMA_STEPS[2]), M.C_STEPS[2]);
var over = M.trainSVC(RX, RY, M.rbfK(M.GAMMA_STEPS[6]), M.C_STEPS[5]);
ok('Underfit preset: both accuracies below 84%', acc(under, RX, RY) < 0.84 && acc(under, TX, TY) < 0.84);
ok('Balanced preset: held-out >= 85% and within 8 points of training', acc(bal, TX, TY) >= 0.85 && acc(bal, RX, RY) - acc(bal, TX, TY) < 0.08);
ok('Overfit preset: 100% training, held-out at least 15 points lower', acc(over, RX, RY) === 1 && acc(over, RX, RY) - acc(over, TX, TY) > 0.15);
ok('Balanced beats both on held-out data', acc(bal, TX, TY) > acc(under, TX, TY) && acc(bal, TX, TY) > acc(over, TX, TY));

/* ---------------------------------------------------- scene 4 -- */
section('Scene 4: Platt scaling');
var cm = M.trainSVC(M.xsOf(M.CAL_TRAIN), M.ysOf(M.CAL_TRAIN), M.linearK, M.CAL_C);
var VS = M.xsOf(M.CAL_VAL).map(cm.f), VL = M.ysOf(M.CAL_VAL), TS = M.xsOf(M.CAL_TEST).map(cm.f), TL = M.ysOf(M.CAL_TEST);
var pf = M.plattFit(VS, VL);
function nllAt(a, b) {
  var nP = VL.filter(function (l) { return l > 0; }).length, nN = VL.length - nP, s = 0;
  VS.forEach(function (f, i) {
    var t = VL[i] > 0 ? (nP + 1) / (nP + 2) : 1 / (nN + 2), p = M.sigmoid(a * f + b);
    s -= t * Math.log(p) + (1 - t) * Math.log(1 - p);
  });
  return s;
}
var d = 1e-4;
ok('Platt fit is a minimum (nudging a or b only increases the loss)',
  [[d, 0], [-d, 0], [0, d], [0, -d]].every(function (e) { return nllAt(pf.a + e[0], pf.b + e[1]) >= nllAt(pf.a, pf.b) - 1e-9; }));
var raw = TS.map(function (s) { return M.sigmoid(s); }), plt = TS.map(function (s) { return M.sigmoid(pf.a * s + pf.b); });
var rb = M.reliability(raw, TL)[4], pb = M.reliability(plt, TL)[4];
ok('plain sigmoid(f) is underconfident in the 70-90% bin (by > 10 points)', rb.frac - rb.meanP > 0.1, rb.meanP + ' vs ' + rb.frac);
ok('Platt-scaled 70-90% bin is within 5 points', Math.abs(pb.frac - pb.meanP) < 0.05, pb.meanP + ' vs ' + pb.frac);
ok('Platt lowers test log loss', M.meanLogLoss(plt, TL) < M.meanLogLoss(raw, TL));
ok('validation and test sets are 300 and 800 ships', VS.length === 300 && TS.length === 800);

/* ---------------------------------------------------- scene 5 -- */
section('Scene 5: SVR');
var xs = M.FUEL.xs, ys = M.FUEL.ys, k1 = M.rbf1(M.FUEL_GAMMA);
var prev = Infinity, mono = true, signs = true, tubeOK = true, conv = true, balOK = true;
[0, 0.1, 0.2, 0.3, 0.5, 0.8, 1.2, 1.5].forEach(function (e) {
  var r = M.trainSVR(xs, ys, k1, M.FUEL_C, e);
  if (!r.converged) conv = false;
  if (r.sv.length > prev) mono = false;
  prev = r.sv.length;
  var s = 0;
  xs.forEach(function (x, n) {
    var res = ys[n] - r.f(x), th = r.theta[n];
    s += th;
    if (Math.abs(res) < e - 1e-4 && Math.abs(th) > 1e-6) tubeOK = false;                 // strictly inside => theta = 0
    if (Math.abs(res) > e + 1e-4 && !close(Math.abs(th), M.FUEL_C, 1e-6)) tubeOK = false; // outside => |theta| = C
    if (e > 0 && r.alpha[n] > 1e-8 && r.alphaStar[n] > 1e-8) tubeOK = false;            // never both positive (eps > 0)
    if (th > 1e-6 && res < e - 1e-4) signs = false;                                     // theta > 0 => on/above top edge
    if (th < -1e-6 && res > -e + 1e-4) signs = false;                                   // theta < 0 => on/below bottom edge
  });
  if (Math.abs(s) > 1e-8) balOK = false;
});
ok('SVR converges for every epsilon', conv);
ok('inside tube => theta = 0; outside => |theta| = C; alpha, alpha* never both positive when eps > 0', tubeOK);
ok('theta > 0 only on/above the tube, theta < 0 only on/below', signs);
ok('sum of theta = 0', balOK);
ok('support vectors never increase as epsilon grows', mono);
var r3 = M.trainSVR(xs, ys, k1, M.FUEL_C, 0.3);
ok('coefficients take both signs at epsilon = 0.3', r3.theta.some(function (t) { return t > 1e-6; }) && r3.theta.some(function (t) { return t < -1e-6; }));
ok('epsilon = 0 uses every reading; epsilon = 1.2 uses at most 6',
  M.trainSVR(xs, ys, k1, M.FUEL_C, 0).sv.length === xs.length && M.trainSVR(xs, ys, k1, M.FUEL_C, 1.2).sv.length <= 6);
ok('epsilon-loss is max(0, |r| - eps)', close(M.epsLoss(-0.529, 0.3), 0.229, 1e-12) && M.epsLoss(0.2, 0.3) === 0);

console.log('\n' + pass + ' passed, ' + fail + ' failed');
process.exit(fail ? 1 : 0);
