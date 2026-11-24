/* Math checks for the PCA Interactive Learning Lab.   Run:  node test-math.js */

var P = require('./model.js');
var Q = require('./practice.js');
var pass = 0, fail = 0;
function ok(name, cond, detail) {
  if (cond) { pass++; console.log('  PASS  ' + name); }
  else { fail++; console.log('  FAIL  ' + name + (detail !== undefined ? '   -> ' + detail : '')); }
}
function close(a, b, tol) { return Math.abs(a - b) <= (tol || 1e-9); }
function maxAbs(A) { var m = 0; A.forEach(function (r) { r.forEach(function (v) { m = Math.max(m, Math.abs(v)); }); }); return m; }

console.log('\nEigendecomposition (Jacobi)');
var r = P.rng(7), A = P.zeros(6, 6);
for (var i = 0; i < 6; i++) for (var j = 0; j <= i; j++) { A[i][j] = A[j][i] = P.gauss(r); }
var e = P.eigSym(A), V = e.vectors;
ok('A V = V Λ', maxAbs(P.sub(P.mul(A, V), P.mul(V, P.diag(e.values)))) < 1e-10);
ok('Vᵀ V = I', maxAbs(P.sub(P.mul(P.T(V), V), P.eye(6))) < 1e-10);
ok('eigenvalues sorted descending', e.values.every(function (v, k) { return k === 0 || e.values[k - 1] >= v; }));
var e2 = P.eigSym([[2, 1], [1, 2]]);
ok('[[2,1],[1,2]] has eigenvalues 3, 1', close(e2.values[0], 3) && close(e2.values[1], 1));
ok('…with first eigenvector (1,1)/√2', close(e2.vectors[0][0], Math.SQRT1_2) && close(e2.vectors[1][0], Math.SQRT1_2));
ok('diagonal matrix needs no rotation', close(P.eigSym([[1, 0], [0, 5]]).values[0], 5));

console.log('\nSVD');
var X = [[2, 0, 1], [1, 3, 2], [4, 1, 0], [0, 2, 5], [3, 3, 3]], Xc = P.center(X).Xc;
var s = P.svd(Xc);
var Xr = P.mul(P.mul(s.U, P.diag(s.s)), P.T(s.V));
ok('U S Vᵀ reproduces Xc', maxAbs(P.sub(Xr, Xc)) < 1e-9);
ok('Uᵀ U = I', maxAbs(P.sub(P.mul(P.T(s.U), s.U), P.eye(3))) < 1e-9);
var lamCov = P.eigSym(P.cov(X)).values;
ok('λ_j = s_j² / N', s.s.every(function (sv, k) { return close(sv * sv / X.length, lamCov[k], 1e-9); }));
var wide = P.center([[1, 2, 0, 4], [3, 1, 2, 2], [0, 0, 5, 1]]).Xc, sw = P.svd(wide);
ok('wide centered matrix (N < D): rank ≤ N − 1, last s = 0', sw.s[2] === 0);
ok('wide: U still orthonormal after completion', maxAbs(P.sub(P.mul(P.T(sw.U), sw.U), P.eye(3))) < 1e-9);
ok('wide: U S Vᵀ reproduces Xc', maxAbs(P.sub(P.mul(P.mul(sw.U, P.diag(sw.s)), P.T(sw.V)), wide)) < 1e-9);

console.log('\nPCA = truncated SVD');
var fit = P.pcaFit(X);
for (var L = 0; L <= 3; L++) {
  var Xh = P.reconstruct(fit, X, L);
  var err = P.reconError(X, Xh);
  ok('L = ' + L + ': training error = Σ discarded λ (Eq. 20.26)', close(err, P.residualFromEigs(fit.lambdas, L), 1e-9), err);
  var UL = P.cols(s.U, L), SL = P.diag(s.s.slice(0, L)), VL = P.cols(s.V, L);
  if (L > 0) {
    var Xsvd = P.mul(P.mul(UL, SL), P.T(VL)).map(function (row) { return row.map(function (v, j) { return v + fit.mean[j]; }); });
    ok('L = ' + L + ': PCA reconstruction = U_L S_L V_Lᵀ + x̄', maxAbs(P.sub(Xsvd, Xh)) < 1e-9);
  }
}
ok('Z = U_L S_L equals Xc W_L', maxAbs(P.sub(P.mul(P.cols(s.U, 2), P.diag(s.s.slice(0, 2))), P.mul(Xc, P.cols(s.V, 2)))) < 1e-9);

console.log('\nCovariance vs correlation');
var hw = P.gaussian2D(3, 200, [170, 70], 9, 12, 0.6);
var R1 = P.corr(hw), hwmm = hw.map(function (p) { return [p[0] * 10, p[1]]; });
ok('correlation matrix has a unit diagonal', close(R1[0][0], 1) && close(R1[1][1], 1));
ok('rescaling a feature leaves the correlation matrix unchanged', maxAbs(P.sub(R1, P.corr(hwmm))) < 1e-12);
var c1 = P.eigSym(P.cov(hw)).vectors, c2 = P.eigSym(P.cov(hwmm)).vectors;
ok('…but changes the covariance PC1 (cm → mm)', Math.abs(c1[0][0] - c2[0][0]) > 0.05, c1[0][0] + ' vs ' + c2[0][0]);
var eR = P.eigSym(R1);
ok('2×2 correlation eigenvalues are 1 ± |ρ|', close(eR.values[0], 1 + Math.abs(R1[0][1])) && close(eR.values[1], 1 - Math.abs(R1[0][1])));
ok('PCA on standardized data = eigen of the correlation matrix', maxAbs(P.sub(P.cov(P.standardize(hw).Z), R1)) < 1e-12);

console.log('\nGram matrix trick (Eq. 20.22)');
var Xg = P.center([[1, 2, 0, 4, 1], [3, 1, 2, 2, 0], [0, 0, 5, 1, 2], [2, 2, 2, 0, 4]]).Xc;
var g = P.gramTrick(Xg);
var big = P.eigSym(P.mul(P.T(Xg), Xg));
ok('XXᵀ and XᵀX share their nonzero eigenvalues', g.keep.every(function (jj) { return close(g.values[jj], big.values[jj], 1e-9); }));
ok('||Xᵀu_j||² = λ_j (why we divide by √λ)', g.keep.every(function (jj) { return close(P.norm(P.col(g.XtU, jj)) ** 2, g.values[jj], 1e-9); }));
ok('V = XᵀUΛ^{-1/2} has unit columns', g.keep.every(function (_, c) { return close(P.norm(P.col(g.V, c)), 1, 1e-9); }));
var XtX = P.mul(P.T(Xg), Xg);
ok('each recovered v satisfies (XᵀX) v = λ v', g.keep.every(function (jj, c) {
  var v = P.col(g.V, c), Av = XtX.map(function (row) { return P.dot(row, v); });
  return Av.every(function (val, k) { return close(val, g.values[jj] * v[k], 1e-8); });
}));
ok('centered N = 4 data has at most 3 usable eigenvalues', g.keep.length === 3);

console.log('\nSpectrum: Eqs. 20.26–20.27');
var lams = [5, 3, 1, 0.5, 0.5];
ok('F(2) = 8 / 10 = 0.8', close(P.explained(lams, 2), 0.8));
ok('L_2 residual = 1 + 0.5 + 0.5 = 2', close(P.residualFromEigs(lams, 2), 2));
ok('min L for 90% = 3 (cumulative 0.9 exactly)', P.minComponents(lams, 0.9) === 3);
ok('min L for 95% = 4', P.minComponents(lams, 0.95) === 4);
ok('min L for 100% = all', P.minComponents(lams, 1) === 5);

console.log('\nTrain/test reconstruction error');
var tt = P.trainTest([9, 6, 3, 1, 0.6, 0.4, 0.3, 0.2], 60, 200, 11);
ok('training error never increases with L', tt.train.every(function (v, k) { return k === 0 || v <= tt.train[k - 1] + 1e-12; }));
ok('test error never increases with L (no U shape)', tt.test.every(function (v, k) { return k === 0 || v <= tt.test[k - 1] + 1e-12; }));
ok('both reach 0 at L = D', close(tt.train[8], 0, 1e-9) && close(tt.test[8], 0, 1e-9));
ok('training error = Σ discarded sample eigenvalues', tt.train.every(function (v, k) { return close(v, P.residualFromEigs(tt.sampleLams, k), 1e-8); }));

console.log('\nProfile likelihood (Eqs. 20.28–20.31)');
var pl = [10, 9, 8, 2, 1.5, 1, 0.8, 0.5];
var pr = P.profile(pl, 3);
ok('μ1 = mean of first L', close(pr.mu1, 9));
ok('μ2 = mean of the rest', close(pr.mu2, (2 + 1.5 + 1 + 0.8 + 0.5) / 5));
ok('σ² = pooled SS / Lmax', close(pr.sigma2, (pr.ss1 + pr.ss2) / 8));
ok('ℓ(L) = −(Lmax/2)·log(2πσ²) − Lmax/2', close(pr.ll, -4 * Math.log(2 * Math.PI * pr.sigma2) - 4, 1e-9));
ok('clear gap after 3 → argmax L = 3', P.profileAll(pl).best.L === 3);
ok('valid splits are L = 1 … Lmax − 1', P.profileAll(pl).all.length === 7);
ok('elbow heuristic also picks 3 here', P.elbow(pl) === 3, P.elbow(pl));

console.log('\nPractice generators (each answer matches its own worked solution)');
var bad = [];
Q.TOPICS.forEach(function (t) {
  for (var k = 0; k < 40; k++) {
    var q = Q.make(t.key, P.rng(1000 + k * 17 + t.key.length));
    if (!q || !q.prompt || !q.steps || !q.steps.length) { bad.push(t.key + ': missing fields'); break; }
    if (q.kind === 'num' && !(isFinite(q.answer))) { bad.push(t.key + ': non-finite answer'); break; }
    if (q.kind === 'num' && !Q.grade(q, String(q.answer))) { bad.push(t.key + ': own answer rejected'); break; }
    if (q.kind === 'mc' && (q.correct < 0 || q.correct >= q.options.length)) { bad.push(t.key + ': bad option index'); break; }
    if (q.kind === 'dim' && !Q.grade(q, q.answer[0] + ' x ' + q.answer[1])) { bad.push(t.key + ': own dims rejected'); break; }
  }
});
ok('all ' + Q.TOPICS.length + ' topics × 40 random questions are self-consistent', bad.length === 0, bad.join('; '));
var qe = Q.make('explained', P.rng(5));
ok('explained-variance answers accept a percent too', Q.grade(qe, String(qe.answer * 100) + '%'));

console.log('\n' + pass + ' passed, ' + fail + ' failed');
process.exitCode = fail ? 1 : 0;
