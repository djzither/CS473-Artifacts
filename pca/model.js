/* ============================================================================
   PCA Interactive Learning Lab — the math.
   No DOM. Loaded by index.html and by test-math.js under Node.

   Conventions follow Murphy, PML §20.1:
     X  is N × D (rows are observations), x̄ is the column mean,
     Σ̂ = (1/N) Xcᵀ Xc  (the 1/N "MLE" covariance, as in the book),
     W  is D × L (orthonormal columns), Z = Xc W is N × L.
   Matrices are arrays of rows.
   ========================================================================= */

(function (global) {
  'use strict';

  /* ---------- random numbers (seeded, so every page load is reproducible) ---------- */
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
  function gauss(r) {
    var u = 0, v = 0;
    while (u === 0) u = r();
    while (v === 0) v = r();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
  }

  /* ---------- basic linear algebra ---------- */
  function zeros(r, c) { var M = []; for (var i = 0; i < r; i++) { M.push(new Array(c).fill(0)); } return M; }
  function eye(n) { var M = zeros(n, n); for (var i = 0; i < n; i++) M[i][i] = 1; return M; }
  function copy(A) { return A.map(function (r) { return r.slice(); }); }
  function shape(A) { return [A.length, A.length ? A[0].length : 0]; }
  function T(A) {
    var r = A.length, c = r ? A[0].length : 0, B = zeros(c, r);
    for (var i = 0; i < r; i++) for (var j = 0; j < c; j++) B[j][i] = A[i][j];
    return B;
  }
  function mul(A, B) {
    var n = A.length, m = B[0].length, k = B.length, C = zeros(n, m);
    for (var i = 0; i < n; i++) {
      var Ai = A[i], Ci = C[i];
      for (var t = 0; t < k; t++) { var a = Ai[t]; if (a === 0) continue; var Bt = B[t]; for (var j = 0; j < m; j++) Ci[j] += a * Bt[j]; }
    }
    return C;
  }
  function sub(A, B) { return A.map(function (r, i) { return r.map(function (v, j) { return v - B[i][j]; }); }); }
  function scale(A, s) { return A.map(function (r) { return r.map(function (v) { return v * s; }); }); }
  function dot(a, b) { var s = 0; for (var i = 0; i < a.length; i++) s += a[i] * b[i]; return s; }
  function norm(a) { return Math.sqrt(dot(a, a)); }
  function col(A, j) { return A.map(function (r) { return r[j]; }); }
  function cols(A, L) { return A.map(function (r) { return r.slice(0, L); }); }
  function frob2(A) { var s = 0; A.forEach(function (r) { r.forEach(function (v) { s += v * v; }); }); return s; }
  function diag(v) { var M = zeros(v.length, v.length); v.forEach(function (x, i) { M[i][i] = x; }); return M; }
  function sum(a) { var s = 0; for (var i = 0; i < a.length; i++) s += a[i]; return s; }

  /* ---------- centering, covariance, correlation ---------- */
  function colMeans(X) {
    var N = X.length, D = X[0].length, m = new Array(D).fill(0);
    X.forEach(function (r) { for (var j = 0; j < D; j++) m[j] += r[j]; });
    return m.map(function (v) { return v / N; });
  }
  function center(X) {
    var m = colMeans(X);
    return { Xc: X.map(function (r) { return r.map(function (v, j) { return v - m[j]; }); }), mean: m };
  }
  /* population standard deviation (divide by N), matching Σ̂ = (1/N) XcᵀXc */
  function colStds(X) {
    var c = center(X), N = X.length;
    return c.mean.map(function (_, j) { var s = 0; c.Xc.forEach(function (r) { s += r[j] * r[j]; }); return Math.sqrt(s / N); });
  }
  function standardize(X) {
    var c = center(X), sd = colStds(X);
    return { Z: c.Xc.map(function (r) { return r.map(function (v, j) { return sd[j] > 0 ? v / sd[j] : 0; }); }), mean: c.mean, sd: sd };
  }
  function cov(X) { var Xc = center(X).Xc; return scale(mul(T(Xc), Xc), 1 / X.length); }
  function corr(X) {
    var S = cov(X), d = S.map(function (r, i) { return Math.sqrt(r[i]); });
    return S.map(function (r, i) { return r.map(function (v, j) { return (d[i] > 0 && d[j] > 0) ? v / (d[i] * d[j]) : (i === j ? 1 : 0); }); });
  }

  /* ---------- symmetric eigendecomposition (cyclic Jacobi) ----------
     Returns eigenvalues in descending order and the matching unit eigenvectors
     as the COLUMNS of `vectors`. Sign convention: the largest-magnitude entry
     of each eigenvector is positive, so results are deterministic. */
  function eigSym(A0) {
    var n = A0.length, A = copy(A0), V = eye(n), p, q, k;
    var sc = 0; for (p = 0; p < n; p++) for (q = 0; q < n; q++) sc += A[p][q] * A[p][q];
    if (sc === 0) return { values: new Array(n).fill(0), vectors: V };
    for (var sweep = 0; sweep < 100; sweep++) {
      var off = 0;
      for (p = 0; p < n; p++) for (q = p + 1; q < n; q++) off += A[p][q] * A[p][q];
      if (off <= 1e-30 * sc) break;
      for (p = 0; p < n - 1; p++) {
        for (q = p + 1; q < n; q++) {
          var apq = A[p][q];
          if (Math.abs(apq) < 1e-300) continue;
          var theta = (A[q][q] - A[p][p]) / (2 * apq);
          var t = (theta >= 0 ? 1 : -1) / (Math.abs(theta) + Math.sqrt(theta * theta + 1));
          var c = 1 / Math.sqrt(t * t + 1), s = t * c;
          for (k = 0; k < n; k++) { var akp = A[k][p], akq = A[k][q]; A[k][p] = c * akp - s * akq; A[k][q] = s * akp + c * akq; }
          for (k = 0; k < n; k++) { var apk = A[p][k], aqk = A[q][k]; A[p][k] = c * apk - s * aqk; A[q][k] = s * apk + c * aqk; }
          for (k = 0; k < n; k++) { var vkp = V[k][p], vkq = V[k][q]; V[k][p] = c * vkp - s * vkq; V[k][q] = s * vkp + c * vkq; }
        }
      }
    }
    var idx = []; for (p = 0; p < n; p++) idx.push(p);
    idx.sort(function (i, j) { return A[j][j] - A[i][i]; });
    var vals = idx.map(function (i) { return A[i][i]; });
    var vecs = zeros(n, n);
    idx.forEach(function (src, dst) {
      var big = 0, sign = 1;
      for (k = 0; k < n; k++) if (Math.abs(V[k][src]) > big + 1e-12) { big = Math.abs(V[k][src]); sign = V[k][src] < 0 ? -1 : 1; }
      for (k = 0; k < n; k++) vecs[k][dst] = sign * V[k][src];
    });
    return { values: vals, vectors: vecs };
  }

  /* ---------- SVD of a (small) matrix: X = U S Vᵀ ----------
     Economy form with k = min(N, D): U is N×k, s has length k, V is D×k.
     V and s come from the eigendecomposition of XᵀX (eigenvalues s²); then
     u_j = X v_j / s_j. Columns of U whose s_j ≈ 0 are completed to an
     orthonormal set by Gram–Schmidt (they multiply a zero, so any choice works). */
  function svd(X) {
    var N = X.length, D = X[0].length, k = Math.min(N, D);
    var e = eigSym(mul(T(X), X));
    var s = e.values.slice(0, k).map(function (v) { return Math.sqrt(Math.max(v, 0)); });
    var V = cols(e.vectors, k), U = zeros(N, k), smax = s[0] || 0;
    var tol = 1e-7 * (smax || 1);   /* s comes from √(eigenvalue), so round-off is ~√ε relative */
    for (var j = 0; j < k; j++) {
      if (s[j] > tol) {
        var vj = col(V, j);
        for (var i = 0; i < N; i++) U[i][j] = dot(X[i], vj) / s[j];
      } else {
        s[j] = 0;
        var basis = 0, done = false;
        while (!done && basis < N) {
          var u = new Array(N).fill(0); u[basis++] = 1;
          for (var m = 0; m < j; m++) { var um = col(U, m), d = dot(u, um); for (i = 0; i < N; i++) u[i] -= d * um[i]; }
          var nu = norm(u);
          if (nu > 1e-6) { for (i = 0; i < N; i++) U[i][j] = u[i] / nu; done = true; }
        }
      }
    }
    return { U: U, s: s, V: V };
  }

  /* ---------- PCA proper ---------- */
  /* Fit PCA on X: eigenvectors of Σ̂ = (1/N) XcᵀXc. */
  function pcaFit(X) {
    var c = center(X), e = eigSym(scale(mul(T(c.Xc), c.Xc), 1 / X.length));
    return { mean: c.mean, W: e.vectors, lambdas: e.values.map(function (v) { return Math.max(v, 0); }) };
  }
  /* Reconstruct rows of X using the first L principal directions of a fit:
     x̂ = x̄ + W_L W_Lᵀ (x − x̄). */
  function reconstruct(fit, X, L) {
    var WL = cols(fit.W, L), m = fit.mean;
    return X.map(function (r) {
      var xc = r.map(function (v, j) { return v - m[j]; });
      var z = new Array(L).fill(0), j, l;
      for (l = 0; l < L; l++) for (j = 0; j < xc.length; j++) z[l] += WL[j][l] * xc[j];
      return xc.map(function (_, j2) { var v = m[j2]; for (l = 0; l < L; l++) v += WL[j2][l] * z[l]; return v; });
    });
  }
  /* Mean squared reconstruction error per example: (1/N) Σ ||x_n − x̂_n||². */
  function reconError(X, Xh) { var s = 0; X.forEach(function (r, i) { r.forEach(function (v, j) { var d = v - Xh[i][j]; s += d * d; }); }); return s / X.length; }

  /* ---------- the spectrum (§20.1.4) ---------- */
  /* Eq. 20.26: reconstruction error on the training set = sum of the discarded eigenvalues. */
  function residualFromEigs(lams, L) { return sum(lams.slice(L)); }
  /* Eq. 20.27: fraction of variance explained by the first L components. */
  function explained(lams, L) { var t = sum(lams); return t > 0 ? sum(lams.slice(0, L)) / t : 0; }
  function cumulative(lams) { var t = sum(lams), s = 0; return lams.map(function (v) { s += v; return t > 0 ? s / t : 0; }); }
  /* smallest L with F(L) ≥ target; small tolerance so 0.9 exactly counts as reaching 90% */
  function minComponents(lams, target) {
    var cum = cumulative(lams);
    for (var i = 0; i < cum.length; i++) if (cum[i] >= target - 1e-12) return i + 1;
    return lams.length;
  }

  /* ---------- profile likelihood (§20.1.4.3, Zhu & Ghodsi 2006) ----------
     Eigenvalues 1..L are N(μ1, σ²), eigenvalues L+1..Lmax are N(μ2, σ²),
     with ONE shared variance. Valid L: 1 ≤ L ≤ Lmax − 1 (both groups non-empty). */
  function logNormal(x, mu, s2) { return -0.5 * Math.log(2 * Math.PI * s2) - (x - mu) * (x - mu) / (2 * s2); }
  function profile(lams, L) {
    var m = lams.length, g1 = lams.slice(0, L), g2 = lams.slice(L);
    var mu1 = sum(g1) / g1.length, mu2 = sum(g2) / g2.length;
    var ss1 = 0, ss2 = 0;
    g1.forEach(function (v) { ss1 += (v - mu1) * (v - mu1); });
    g2.forEach(function (v) { ss2 += (v - mu2) * (v - mu2); });
    var sigma2 = (ss1 + ss2) / m;
    var floor = 1e-12 * Math.max(1e-300, sum(lams.map(function (v) { return v * v; })) / m);
    var degenerate = sigma2 < floor, s2 = Math.max(sigma2, floor);
    var ll1 = 0, ll2 = 0;
    g1.forEach(function (v) { ll1 += logNormal(v, mu1, s2); });
    g2.forEach(function (v) { ll2 += logNormal(v, mu2, s2); });
    return { L: L, mu1: mu1, mu2: mu2, ss1: ss1, ss2: ss2, sigma2: sigma2, ll1: ll1, ll2: ll2, ll: ll1 + ll2, degenerate: degenerate };
  }
  function profileAll(lams) {
    var out = [], best = null;
    for (var L = 1; L < lams.length; L++) {
      var p = profile(lams, L); out.push(p);
      if (!best || p.ll > best.ll + 1e-12) best = p;
    }
    return { all: out, best: best };
  }

  /* A simple "elbow" heuristic, for comparison only (not from Murphy):
     the eigenvalue farthest from the straight line joining the first and last
     points of the scree plot, after scaling both axes to [0, 1]. Returns how many
     components come before that bend point. */
  function elbow(lams) {
    var m = lams.length; if (m < 3) return 1;
    var hi = Math.max.apply(null, lams), lo = Math.min.apply(null, lams), span = (hi - lo) || 1;
    var x = function (i) { return i / (m - 1); }, y = function (i) { return (lams[i] - lo) / span; };
    var x1 = x(0), y1 = y(0), x2 = x(m - 1), y2 = y(m - 1), best = 0, bi = 0;
    for (var i = 1; i < m - 1; i++) {
      var d = Math.abs((y2 - y1) * x(i) - (x2 - x1) * y(i) + x2 * y1 - y2 * x1) / Math.hypot(y2 - y1, x2 - x1);
      if (d > best) { best = d; bi = i; }
    }
    return bi;   /* the bend is at λ_{bi+1}; keep the bi components before it */
  }

  /* ---------- the Gram-matrix trick (§20.1.3.2, Eq. 20.22) ----------
     Eigendecompose the N×N matrix X Xᵀ instead of the D×D matrix Xᵀ X.
     If (XXᵀ)U = UΛ then (XᵀX)(XᵀU) = (XᵀU)Λ, and ||Xᵀu_j||² = λ_j, so the
     normalized eigenvectors of XᵀX are V = Xᵀ U Λ^{-1/2}. Only eigenvalues
     above tol can be normalized. */
  function gramTrick(X, tol) {
    var G = mul(X, T(X)), e = eigSym(G), N = X.length, D = X[0].length;
    tol = tol === undefined ? 1e-9 * Math.max(1, Math.abs(e.values[0] || 0)) : tol;
    var keep = []; e.values.forEach(function (v, j) { if (v > tol) keep.push(j); });
    var U = e.vectors, XtU = mul(T(X), U);
    var V = zeros(D, keep.length);
    keep.forEach(function (j, c) { var r = Math.sqrt(e.values[j]); for (var i = 0; i < D; i++) V[i][c] = XtU[i][j] / r; });
    return { G: G, values: e.values, U: U, XtU: XtU, keep: keep, V: V, N: N, D: D };
  }

  /* ---------- data sets (all synthetic; labeled as such on the page) ---------- */
  /* N points from a 2D Gaussian with sds (sx, sy) and correlation rho. */
  function gaussian2D(seed, N, mean, sx, sy, rho) {
    var r = rng(seed), out = [];
    for (var i = 0; i < N; i++) {
      var z1 = gauss(r), z2 = gauss(r);
      out.push([mean[0] + sx * z1, mean[1] + sy * (rho * z1 + Math.sqrt(1 - rho * rho) * z2)]);
    }
    return out;
  }
  /* Random D×D orthogonal matrix (Gram–Schmidt on Gaussian columns). */
  function randomOrthogonal(D, r) {
    var Q = [];
    for (var j = 0; j < D; j++) {
      var v = []; for (var i = 0; i < D; i++) v.push(gauss(r));
      for (var m = 0; m < j; m++) { var d = dot(v, Q[m]); for (i = 0; i < D; i++) v[i] -= d * Q[m][i]; }
      var nv = norm(v); Q.push(v.map(function (x) { return x / nv; }));
    }
    return T(Q);   /* columns are the orthonormal vectors */
  }
  /* N samples x = R diag(√λ) z, z ~ N(0, I): population covariance has spectrum `lams`. */
  function sampleSpectrum(r, N, lams, R) {
    var D = lams.length, out = [];
    for (var n = 0; n < N; n++) {
      var z = lams.map(function (l) { return Math.sqrt(Math.max(l, 0)) * gauss(r); });
      var x = new Array(D).fill(0);
      for (var i = 0; i < D; i++) for (var j = 0; j < D; j++) x[i] += R[i][j] * z[j];
      out.push(x);
    }
    return out;
  }
  /* Train/test reconstruction error for L = 0..D (§20.1.4.1, Fig. 20.6 in spirit). */
  function trainTest(lams, Ntrain, Ntest, seed) {
    var r = rng(seed), R = randomOrthogonal(lams.length, r);
    var Xtr = sampleSpectrum(r, Ntrain, lams, R), Xte = sampleSpectrum(r, Ntest, lams, R);
    var fit = pcaFit(Xtr), train = [], test = [];
    for (var L = 0; L <= lams.length; L++) {
      train.push(reconError(Xtr, reconstruct(fit, Xtr, L)));
      test.push(reconError(Xte, reconstruct(fit, Xte, L)));
    }
    return { train: train, test: test, sampleLams: fit.lambdas };
  }

  /* A 64×64 grayscale test image, values in [0, 1]: sky gradient, sun, hills,
     a striped barn and a fence. Built from simple shapes so it has both smooth
     regions (few components) and sharp edges (many components). */
  function testImage(n) {
    n = n || 64;
    var img = zeros(n, n);
    for (var i = 0; i < n; i++) for (var j = 0; j < n; j++) {
      var y = i / (n - 1), x = j / (n - 1), v = 0.85 - 0.35 * y;
      if (Math.hypot(x - 0.75, y - 0.2) < 0.11) v = 1;
      var hill = 0.55 + 0.08 * Math.sin(6 * x) + 0.05 * Math.cos(11 * x + 1);
      if (y > hill) v = 0.35 + 0.15 * Math.sin(40 * x * y);
      if (x > 0.15 && x < 0.42 && y > 0.42 && y < 0.78) v = (Math.floor(x * 40) % 2) ? 0.15 : 0.55;
      var roof = 0.42 - (0.135 - Math.abs(x - 0.285)) * 0.8;
      if (x > 0.15 && x < 0.42 && y > roof && y <= 0.42) v = 0.05;
      if (y > 0.82 && y < 0.86) v = 0.95;
      if (y > 0.78 && y < 0.95 && Math.floor(x * 24) % 3 === 0) v = 0.95;
      img[i][j] = Math.min(1, Math.max(0, v));
    }
    return img;
  }

  var API = {
    rng: rng, gauss: gauss,
    zeros: zeros, eye: eye, copy: copy, shape: shape, T: T, mul: mul, sub: sub, scale: scale, dot: dot, norm: norm,
    col: col, cols: cols, frob2: frob2, diag: diag, sum: sum,
    colMeans: colMeans, center: center, colStds: colStds, standardize: standardize, cov: cov, corr: corr,
    eigSym: eigSym, svd: svd,
    pcaFit: pcaFit, reconstruct: reconstruct, reconError: reconError,
    residualFromEigs: residualFromEigs, explained: explained, cumulative: cumulative, minComponents: minComponents,
    profile: profile, profileAll: profileAll, elbow: elbow,
    gramTrick: gramTrick,
    gaussian2D: gaussian2D, randomOrthogonal: randomOrthogonal, sampleSpectrum: sampleSpectrum, trainTest: trainTest,
    testImage: testImage
  };
  if (typeof module !== 'undefined' && module.exports) module.exports = API;
  else global.PCA = API;
})(typeof window !== 'undefined' ? window : globalThis);
