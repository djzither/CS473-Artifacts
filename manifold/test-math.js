/* Math checks for "Manifold Learning Playground".   Run:  node test-math.js
   Every claim the page makes (success / failure presets, quiz numbers) is
   checked here against the same model.js the page loads. */

var M = require('./model.js');
var pass = 0, fail = 0;
function ok(name, cond, detail) {
  if (cond) { pass++; console.log('  PASS  ' + name + (detail !== undefined ? '   (' + detail + ')' : '')); }
  else { fail++; console.log('  FAIL  ' + name + (detail !== undefined ? '   -> ' + detail : '')); }
}
function close(a, b, tol) { return Math.abs(a - b) <= (tol || 1e-9); }
function f3(x) { return (+x).toFixed(3); }

console.log('\nEigensolver');
(function () {
  var r = M.rng(3), n = 60, A = new Float64Array(n * n);
  for (var i = 0; i < n; i++) for (var j = i; j < n; j++) { var v = M.randn(r); A[i * n + j] = A[j * n + i] = v; }
  var E = M.eigSym(A, n), res = 0, orth = 0, sorted = true;
  for (var l = 0; l < n; l++) {
    var x = E.vectors[l];
    for (i = 0; i < n; i++) { var s = 0; for (j = 0; j < n; j++) s += A[i * n + j] * x[j]; res = Math.max(res, Math.abs(s - E.values[l] * x[i])); }
    for (var m = 0; m < n; m++) { var d = M.dot(Array.from(x), Array.from(E.vectors[m])); orth = Math.max(orth, Math.abs(d - (l === m ? 1 : 0))); }
    if (l && E.values[l] < E.values[l - 1]) sorted = false;
  }
  ok('A v = λ v for a random 60 × 60 symmetric matrix', res < 1e-10, res.toExponential(1));
  ok('eigenvectors are orthonormal', orth < 1e-10, orth.toExponential(1));
  ok('eigenvalues come out ascending', sorted);
  var D = M.eigSym(new Float64Array([2, 0, 0, 0, 5, 0, 0, 0, -1]), 3);
  ok('diagonal matrix: eigenvalues -1, 2, 5', close(D.values[0], -1) && close(D.values[1], 2) && close(D.values[2], 5));
})();

console.log('\nSwiss roll geometry');
(function () {
  var worst = 0;
  [5, 7.5, 10, 13].forEach(function (t) { worst = Math.max(worst, Math.abs(M.spiralArcInv(M.spiralArc(t)) - t)); });
  ok('arc length s(t) inverts', worst < 1e-10, worst.toExponential(1));
  var h = 1e-6, t = 9, sp = (M.spiralArc(t + h) - M.spiralArc(t - h)) / (2 * h);
  ok('ds/dt = √(1 + t²), the speed along the spiral', close(sp, Math.sqrt(1 + t * t), 1e-5));
  var ds = M.makeData('swiss', 500, 0, 1), n = 500, D2 = M.sqdist(ds.X), nb = M.knn(D2, n, 1), r = 0;
  for (var i = 0; i < n; i++) { var j = nb[i][0], e = Math.sqrt(D2[i * n + j]); r = Math.max(r, Math.abs(ds.TD[i * n + j] / e - 1)); }
  ok('(s, h) are isometric: nearest-neighbour chords match true distances within 3%', r < 0.03, f3(r));
})();

console.log('\nSurfaces and curvature (section 1)');
(function () {
  var S = M.SURFACES;
  ok('sphere of radius 3: K = 1/9', close(M.surfGeom(S.sphere, 1, 1.2).K, 1 / 9, 1e-4), f3(M.surfGeom(S.sphere, 1, 1.2).K));
  ok('Swiss roll: K = 0 (intrinsically flat)', Math.abs(M.surfGeom(S.swiss, 3 * Math.PI, 0.5).K) < 1e-4, M.surfGeom(S.swiss, 3 * Math.PI, 0.5).K.toExponential(1));
  ok('saddle at origin: K = −(2/3.5)²', close(M.surfGeom(S.saddle, 0, 0).K, -Math.pow(2 / 3.5, 2), 1e-4));
  ok('torus outer equator: K = 1/(r(R+r)) > 0', close(M.surfGeom(S.torus, 0.5, 0).K, 1 / 3.6, 1e-4));
  ok('torus inner equator: K < 0', M.surfGeom(S.torus, 0.5, Math.PI).K < 0);
  var small = M.patch(S.sphere, 1, 1.2, 0.2), big = M.patch(S.sphere, 1, 1.2, 2);
  ok('locally Euclidean: δ/r shrinks with r (sphere)', small.delta / 0.2 < 0.05 && big.delta / 2 > 0.3, f3(small.delta / 0.2) + ' vs ' + f3(big.delta / 2));
  ok('δ ≈ r²/(2R) on the sphere for small r', close(small.delta, 0.04 / 6, 4e-4), f3(small.delta));
  var tang = M.surfGeom(S.sphere, 1, 1.2);
  ok('tangent basis ⟂ normal', Math.abs(M.dot(tang.e1, tang.n)) < 1e-8 && Math.abs(M.dot(tang.e2, tang.n)) < 1e-8);
})();

console.log('\nPCA, MDS, kernel PCA');
(function () {
  var lin = M.makeData('linear', 300, 0, 2), p = M.pca(lin.X);
  ok('linear plane: third PCA eigenvalue ≈ 0', p.spectrum[2] < 1e-12, p.spectrum[2].toExponential(1));
  ok('linear plane: PCA keeps every neighbourhood', M.neighborKeep(lin.TD, p.Y, 300, 10) > 0.99);
  var n = 300, E = new Float64Array(n * n), D2 = M.sqdist(lin.X);
  for (var i = 0; i < n * n; i++) E[i] = Math.sqrt(D2[i]);
  var md = M.classicalMDS(E, n, 2), worst = 0;
  for (i = 0; i < n; i++) for (var j = 0; j < n; j += 7) {
    var a = Math.hypot(md.Y[i][0] - md.Y[j][0], md.Y[i][1] - md.Y[j][1]), b = Math.hypot(p.Y[i][0] - p.Y[j][0], p.Y[i][1] - p.Y[j][1]);
    worst = Math.max(worst, Math.abs(a - b));
  }
  ok('classical MDS on Euclidean distances = PCA', worst < 1e-8, worst.toExponential(1));
  var kp = M.kpca(lin.X, 30, D2), al = M.align(kp.Y, p.Y), pa = M.align(p.Y, p.Y), err = 0;
  for (i = 0; i < n; i++) err = Math.max(err, Math.hypot(al[i][0] - pa[i][0], al[i][1] - pa[i][1]));
  ok('kernel PCA with a very wide RBF → PCA', err < 0.01, f3(err));
  var sw = M.makeData('swiss', 400, 0, 1), ps = M.pca(sw.X);
  ok('Swiss roll: PCA stacks layers (distance corr < 0.5)', M.distCorr(sw.TD, ps.Y, 400) < 0.5, f3(M.distCorr(sw.TD, ps.Y, 400)));
  var kc = M.makeData('circle', 400, 0, 1), kk = M.kpca(kc.X, 1);
  ok('circle: kernel PCA succeeds (10-NN kept > 0.95)', M.neighborKeep(kc.TD, kk.Y, 400, 10) > 0.95);
  var ks = M.kpca(sw.X, 1);
  ok('Swiss roll: kernel PCA does not unroll (distance corr < 0.5)', M.distCorr(sw.TD, ks.Y, 400) < 0.5, f3(M.distCorr(sw.TD, ks.Y, 400)));
})();

console.log('\nIsomap (section 4 presets)');
(function () {
  var sw = M.makeData('swiss', 400, 0, 1), D2 = M.sqdist(sw.X);
  var good = M.isomap(sw.X, 8, D2), bad = M.isomap(sw.X, 22, D2);
  var cg = M.distCorr(sw.TD, good.Y, 400), cb = M.distCorr(sw.TD, bad.Y, 400);
  ok('k = 8 unrolls the Swiss roll (distance corr > 0.98)', cg > 0.98, f3(cg));
  ok('k = 22 short-circuits (distance corr < 0.7)', cb < 0.7, f3(cb));
  var lin = M.isomap(M.makeData('linear', 400, 0, 1).X, 10).spectrum;
  ok('flat plane: λ₃/λ₂ small (only L = 2 eigenvalues carry the geometry)', lin[2] / lin[1] < 0.1, f3(lin[2] / lin[1]));
  ok('Swiss roll k = 8: λ₂ ≫ λ₃ even though λ₁ ≫ λ₂ (long strip)', good.spectrum[1] > 2.5 * good.spectrum[2], f3(good.spectrum[2] / good.spectrum[1]));
  var nz = M.makeData('swiss', 400, 0.6, 1), cn = M.distCorr(nz.TD, M.isomap(nz.X, 8).Y, 400);
  ok('noise 0.6 with k = 8 breaks it (distance corr < 0.8)', cn < 0.8, f3(cn));
  var sparse = M.isomap(sw.X, 2, D2);
  ok('k = 2 disconnects the graph and warns', sparse.components > 1 && !!sparse.warn, sparse.components + ' pieces');
})();

console.log('\nGeodesics (section 3)');
(function () {
  var n = 500, ds = M.makeData('swiss', n, 0, 11), D2 = M.sqdist(ds.X), g = M.knnGraph(D2, n, 7);
  var r = M.rng(4), lo = Infinity, hi = 0, under = true;
  for (var t = 0; t < 40; t++) {
    var a = (r() * n) | 0, b = (r() * n) | 0; if (a === b) continue;
    var sp = M.shortestPath(g.adj, n, a, b), q = sp.dist / ds.TD[a * n + b];
    lo = Math.min(lo, q); hi = Math.max(hi, q);
    if (sp.dist + 1e-9 < Math.sqrt(D2[a * n + b])) under = false;
  }
  ok('graph distance is never shorter than the Euclidean chord', under);
  ok('k = 7: graph geodesic within [0.97, 1.4] × true geodesic (zig-zag only)', lo > 0.97 && hi < 1.4, f3(lo) + '–' + f3(hi));
  var firstShort = 0;
  for (var kk = 4; kk <= 40 && !firstShort; kk++) {
    var gk = M.knnGraph(D2, n, kk);
    gk.adj.forEach(function (e, u) { e.to.forEach(function (v) { var ee = Math.sqrt(D2[u * n + v]), tt = ds.TD[u * n + v]; if (u < v && tt > 2 * ee && tt - ee > 2) firstShort = kk; }); });
  }
  ok('N = 500: no short-circuit edges until k ≈ 20 (slider goes to 40)', firstShort >= 15 && firstShort <= 30, 'first at k = ' + firstShort);
  var target = M.ROLL.s0 + 0.6 * (M.ROLL.s1 - M.ROLL.s0), A = -1, bd = Infinity;
  for (var i = 0; i < n; i++) { var d = Math.abs(ds.T[i][0] - target) + Math.abs(ds.T[i][1] - M.ROLL.height / 2); if (d < bd) { bd = d; A = i; } }
  var B = -1; bd = Infinity;
  for (var j = 0; j < n; j++) { if (ds.TD[A * n + j] < 15) continue; if (D2[A * n + j] < bd) { bd = D2[A * n + j]; B = j; } }
  var ratio = ds.TD[A * n + B] / Math.sqrt(D2[A * n + B]);
  ok('"Neighbors across a layer": geodesic 5–10 × Euclidean', ratio > 5 && ratio < 10, f3(ratio));
  var big = M.knnGraph(D2, n, 32), spb = M.shortestPath(big.adj, n, A, B);
  ok('k = 32 short-circuits that pair (graph < half the true geodesic)', spb.dist < 0.5 * ds.TD[A * n + B], f3(spb.dist) + ' vs ' + f3(ds.TD[A * n + B]));
})();

console.log('\nLLE and Laplacian eigenmaps');
(function () {
  var sc = M.makeData('scurve', 400, 0, 1), D2 = M.sqdist(sc.X), W = M.lleWeights(sc.X, 12, 1e-3, D2);
  ok('LLE weights sum to 1', W.every(function (w) { return close(w.w.reduce(function (a, b) { return a + b; }, 0), 1, 1e-9); }));
  var rec = 0;
  W.forEach(function (w, i) { var x = [0, 0, 0]; w.nb.forEach(function (j, a) { x = M.add(x, M.scale(sc.X[j], w.w[a])); }); rec = Math.max(rec, M.norm(M.sub(x, sc.X[i]))); });
  ok('weights rebuild each point from its neighbours (max error < 0.05)', rec < 0.05, f3(rec));
  var good = M.lle(sc.X, 12, 1e-3, D2), bad = M.lle(sc.X, 4, 1e-3, D2);
  ok('LLE: M has eigenvalue ≈ 0 (constant vector)', Math.abs(good.spectrum[0]) < 1e-8, good.spectrum[0].toExponential(1));
  var kg = M.neighborKeep(sc.TD, good.Y, 400, 10), kb = M.neighborKeep(sc.TD, bad.Y, 400, 10);
  ok('S-curve: LLE k = 12 beats k = 4', kg > kb + 0.05 && M.distCorr(sc.TD, good.Y, 400) > 0.8, f3(kg) + ' vs ' + f3(kb));
  var cc = M.makeData('circle', 400, 0, 1), lc = M.laplacian(cc.X, 10, 1);
  ok('Laplacian: smallest eigenvalue 0', Math.abs(lc.spectrum[0]) < 1e-9);
  ok('circle: Laplacian eigenmaps succeeds (10-NN kept > 0.85)', M.neighborKeep(cc.TD, lc.Y, 400, 10) > 0.85, f3(M.neighborKeep(cc.TD, lc.Y, 400, 10)));
  var sw = M.makeData('swiss', 400, 0, 1), ls = M.laplacian(sw.X, 10, 1);
  ok('Swiss roll: Laplacian keeps under 60% of neighbourhoods (the arch)', M.neighborKeep(sw.TD, ls.Y, 400, 10) < 0.6, f3(M.neighborKeep(sw.TD, ls.Y, 400, 10)));
})();

console.log('\nt-SNE');
(function () {
  var sc = M.makeData('scurve', 300, 0, 1), D2 = M.sqdist(sc.X), Pr = M.tsneP(D2, 300, 20), worst = 0;
  for (var i = 0; i < 300; i++) worst = Math.max(worst, Math.abs(Pr.perplexities[i] - 20));
  ok('bisection hits the target perplexity for every point', worst < 1e-3, worst.toExponential(1));
  var s = 0, sym = 0;
  for (i = 0; i < 300; i++) for (var j = 0; j < 300; j++) { s += Pr.P[i * 300 + j]; sym = Math.max(sym, Math.abs(Pr.P[i * 300 + j] - Pr.P[j * 300 + i])); }
  ok('P is symmetric and sums to 1', close(s, 1, 1e-6) && sym < 1e-15);
  var S = M.tsneInit(sc.X, { perplexity: 30, iters: 1000 }, D2);
  while (S.iter < 200) M.tsneStep(S);
  var kl200 = S.kl;
  while (S.iter < 1000) M.tsneStep(S);
  ok('KL decreases after early exaggeration', S.kl < kl200, f3(kl200) + ' → ' + f3(S.kl));
  var Y = M.tsneY(S), keep = M.neighborKeep(sc.TD, Y, 300, 10);
  ok('perplexity 30 keeps local neighbourhoods (> 0.8)', keep > 0.8, f3(keep));
  var torn = M.tsne(sc.X, { perplexity: 2, iters: 1000 }, D2);
  ok('perplexity 2 loses global structure (distance corr < 0.5)', M.distCorr(sc.TD, torn.Y, 300) < 0.5, f3(M.distCorr(sc.TD, torn.Y, 300)));
})();

console.log('\nMetrics and alignment');
(function () {
  var ds = M.makeData('linear', 200, 0, 3), th = 1.1, c = Math.cos(th), s = Math.sin(th);
  var R = ds.T.map(function (t) { return [-(c * t[0] - s * t[1]) * 7 + 3, (s * t[0] + c * t[1]) * 7 - 1]; });
  var A = M.align(R, ds.T), B = M.normalize2(ds.T), e = 0;
  for (var i = 0; i < 200; i++) e = Math.max(e, Math.hypot(A[i][0] - B[i][0], A[i][1] - B[i][1]));
  ok('Procrustes undoes rotation + reflection + scale + shift', e < 1e-9, e.toExponential(1));
  ok('truth vs. itself: 10-NN kept = 1, distance corr = 1', close(M.neighborKeep(ds.TD, ds.T, 200, 10), 1) && close(M.distCorr(ds.TD, ds.T, 200), 1, 1e-9));
  var dp = M.dualPCA([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1]], 2), pz = M.pca([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1]]);
  ok('dual (Gram) PCA matches PCA eigenvalues', close(dp.spectrum[0] / 4, pz.spectrum[0], 1e-9) && close(dp.spectrum[1] / 4, pz.spectrum[1], 1e-9));
})();

console.log('\n' + pass + ' passed, ' + fail + ' failed');
process.exitCode = fail ? 1 : 0;
