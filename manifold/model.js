/* ============================================================================
   Manifold Learning Playground — the math.   CS 473 module 17 · Murphy §20.4
   No DOM. Loaded by index.html and by test-math.js under Node.

   Everything here is the real algorithm, run exactly at small N:
     PCA, kernel PCA (RBF), Isomap (kNN graph + Dijkstra + classical MDS),
     LLE, Laplacian eigenmaps, exact-gradient t-SNE.
   The one dense symmetric eigensolver (Householder tridiagonalization + QL,
   after JAMA / EISPACK tred2–tql2) does all the spectral work.
   ========================================================================= */

(function (global) {
  'use strict';
  var TAU = 2 * Math.PI;

  /* ---------------------------------------------------------------- random */
  function rng(seed) {
    var a = (seed >>> 0) || 1;
    return function () {
      a = (a + 0x6D2B79F5) | 0;
      var t = Math.imul(a ^ (a >>> 15), 1 | a);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }
  function randn(r) { var u = 1 - r(), v = r(); return Math.sqrt(-2 * Math.log(u)) * Math.cos(TAU * v); }

  /* ---------------------------------------------------------- small vectors */
  function dot(a, b) { var s = 0; for (var i = 0; i < a.length; i++) s += a[i] * b[i]; return s; }
  function sub(a, b) { return a.map(function (x, i) { return x - b[i]; }); }
  function add(a, b) { return a.map(function (x, i) { return x + b[i]; }); }
  function scale(a, s) { return a.map(function (x) { return x * s; }); }
  function norm(a) { return Math.sqrt(dot(a, a)); }
  function unit(a) { var n = norm(a) || 1; return scale(a, 1 / n); }
  function cross(a, b) { return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]; }

  /* ============================================================ Swiss roll arc
     The roll is the spiral (t cos t, t sin t) swept along the height axis.
     Arc length along the spiral is s(t) = ½[t√(1+t²) + asinh t], so (s, h)
     are exact isometric "unrolled" coordinates: the roll is a bent sheet of
     paper and geodesics are straight lines in (s, h). */
  function spiralArc(t) { var q = Math.sqrt(1 + t * t); return 0.5 * (t * q + Math.log(t + q)); }
  function spiralArcInv(s) {
    var t = Math.sqrt(Math.max(2 * s, 1e-9));
    for (var i = 0; i < 60; i++) {
      var dt = (spiralArc(t) - s) / Math.sqrt(1 + t * t);
      t -= dt;
      if (Math.abs(dt) < 1e-13) break;
    }
    return t;
  }
  /* Height 10 (sklearn uses 21) keeps the sheet dense enough at N ≈ 400 that a
     kNN graph with k ≲ 12 follows the sheet; k ≳ 20 jumps between layers. */
  var ROLL = { t0: 1.5 * Math.PI, t1: 4.5 * Math.PI, height: 10 };
  ROLL.s0 = spiralArc(ROLL.t0); ROLL.s1 = spiralArc(ROLL.t1);
  function rollPoint(s, h) { var t = spiralArcInv(s); return [t * Math.cos(t), h, t * Math.sin(t)]; }

  /* ================================================================ datasets
     Each dataset returns
       X   ambient points (N × 3)
       T   true intrinsic coordinates (N × 2), isometric where possible
       c   a color parameter in [0, 1]
       TD  true manifold (geodesic) distances, N × N  */
  var CIRCLE_A = unit([0.8, 0.0, 0.6]);
  var CIRCLE_B = unit(sub([0, 0.8, 0.6], scale(CIRCLE_A, dot([0, 0.8, 0.6], CIRCLE_A))));
  var PLANE_A = unit([0.9, 0.3, 0.3]);
  var PLANE_B = unit(sub([-0.2, 0.5, 0.85], scale(PLANE_A, dot([-0.2, 0.5, 0.85], PLANE_A))));

  var DATASETS = {
    swiss: {
      label: 'Swiss roll', dim: 2, noiseScale: 1.2, cyclic: false,
      coords: ['position along the roll', 'height'],
      sample: function (r, noise) {
        var s = ROLL.s0 + (ROLL.s1 - ROLL.s0) * r(), h = ROLL.height * r();
        var p = rollPoint(s, h);
        return { x: [p[0] + noise * randn(r), p[1] + noise * randn(r), p[2] + noise * randn(r)],
          t: [s, h], c: (s - ROLL.s0) / (ROLL.s1 - ROLL.s0), c2: h / ROLL.height };
      }
    },
    scurve: {
      label: 'S-curve', dim: 2, noiseScale: 0.12, cyclic: false,
      coords: ['position along the S', 'depth'],
      sample: function (r, noise) {
        var t = 3 * Math.PI * (r() - 0.5), y = 2 * r();
        var x = [Math.sin(t), y, (t < 0 ? -1 : 1) * (Math.cos(t) - 1)];
        return { x: [x[0] + noise * randn(r), x[1] + noise * randn(r), x[2] + noise * randn(r)],
          t: [t, y], c: t / (3 * Math.PI) + 0.5, c2: y / 2 };
      }
    },
    circle: {
      label: 'Circle', dim: 1, noiseScale: 0.1, cyclic: true,
      coords: ['angle'],
      sample: function (r, noise) {
        var th = TAU * r();
        var p = add(scale(CIRCLE_A, Math.cos(th)), scale(CIRCLE_B, Math.sin(th)));
        return { x: [p[0] + noise * randn(r), p[1] + noise * randn(r), p[2] + noise * randn(r)],
          t: [Math.cos(th), Math.sin(th)], c: th / TAU, c2: th / TAU, theta: th };
      }
    },
    linear: {
      label: 'Linear plane', dim: 2, noiseScale: 0.15, cyclic: false,
      coords: ['long side', 'short side'],
      sample: function (r, noise) {
        var u = 4 * r() - 2, v = 2 * r() - 1;
        var p = add(scale(PLANE_A, u), scale(PLANE_B, v));
        return { x: [p[0] + noise * randn(r), p[1] + noise * randn(r), p[2] + noise * randn(r)],
          t: [u, v], c: (u + 2) / 4, c2: (v + 1) / 2 };
      }
    }
  };

  function makeData(key, n, noise, seed) {
    var D = DATASETS[key], r = rng(seed), sd = noise * D.noiseScale;
    var X = [], T = [], c = [], c2 = [], th = [];
    for (var i = 0; i < n; i++) {
      var q = D.sample(r, sd);
      X.push(q.x); T.push(q.t); c.push(q.c); c2.push(q.c2); th.push(q.theta);
    }
    var TD = new Float64Array(n * n);
    for (i = 0; i < n; i++) for (var j = i + 1; j < n; j++) {
      var d;
      if (key === 'circle') { var a = Math.abs(th[i] - th[j]); d = Math.min(a, TAU - a); }
      else { var dx = T[i][0] - T[j][0], dy = T[i][1] - T[j][1]; d = Math.sqrt(dx * dx + dy * dy); }
      TD[i * n + j] = TD[j * n + i] = d;
    }
    return { key: key, label: D.label, dim: D.dim, cyclic: D.cyclic, coords: D.coords,
      n: n, noise: noise, seed: seed, X: X, T: T, c: c, c2: c2, TD: TD };
  }

  /* =========================================================== distances, kNN */
  function sqdist(X) {
    var n = X.length, d = X[0].length, D = new Float64Array(n * n);
    for (var i = 0; i < n; i++) {
      var xi = X[i];
      for (var j = i + 1; j < n; j++) {
        var xj = X[j], s = 0;
        for (var k = 0; k < d; k++) { var t = xi[k] - xj[k]; s += t * t; }
        D[i * n + j] = D[j * n + i] = s;
      }
    }
    return D;
  }

  /* k nearest neighbours of every row of a distance matrix (self excluded). */
  function knn(D, n, k) {
    var out = new Array(n), idx = new Array(n - 1);
    k = Math.min(k, n - 1);
    for (var i = 0; i < n; i++) {
      var m = 0, row = i * n;
      for (var j = 0; j < n; j++) if (j !== i) idx[m++] = j;
      idx.sort(function (a, b) { return D[row + a] - D[row + b]; });
      out[i] = idx.slice(0, k);
    }
    return out;
  }

  /* Symmetric kNN graph: an edge i–j when either is among the other's k
     nearest. Edge weight = Euclidean length. */
  function knnGraph(D2, n, k) {
    var nb = knn(D2, n, k), maps = [], i, j;
    for (i = 0; i < n; i++) maps.push(new Map());
    for (i = 0; i < n; i++) for (var a = 0; a < nb[i].length; a++) {
      j = nb[i][a];
      var w = Math.sqrt(D2[i * n + j]);
      maps[i].set(j, w); maps[j].set(i, w);
    }
    var adj = [], edges = 0;
    for (i = 0; i < n; i++) {
      var to = [], ws = [];
      maps[i].forEach(function (w, j) { to.push(j); ws.push(w); });
      adj.push({ to: to, w: ws });
      edges += to.length;
    }
    return { nb: nb, adj: adj, edges: edges / 2 };
  }

  function components(adj, n) {
    var lab = new Int32Array(n).fill(-1), c = 0, stack = [];
    for (var s = 0; s < n; s++) {
      if (lab[s] >= 0) continue;
      lab[s] = c; stack.push(s);
      while (stack.length) {
        var u = stack.pop(), e = adj[u];
        for (var a = 0; a < e.to.length; a++) if (lab[e.to[a]] < 0) { lab[e.to[a]] = c; stack.push(e.to[a]); }
      }
      c++;
    }
    return { count: c, label: lab };
  }

  /* Dijkstra with a binary heap. dist (and prev, if given) are filled in. */
  function dijkstra(adj, n, src, dist, prev) {
    dist.fill(Infinity); if (prev) prev.fill(-1);
    var hk = [], hv = [];
    function push(k, v) {
      var i = hk.length; hk.push(k); hv.push(v);
      while (i > 0) {
        var p = (i - 1) >> 1;
        if (hk[p] <= hk[i]) break;
        var tk = hk[p]; hk[p] = hk[i]; hk[i] = tk;
        var tv = hv[p]; hv[p] = hv[i]; hv[i] = tv;
        i = p;
      }
    }
    function pop() {
      var top = hv[0], lk = hk.pop(), lv = hv.pop();
      if (hk.length) {
        hk[0] = lk; hv[0] = lv;
        var i = 0, L = hk.length;
        for (;;) {
          var l = 2 * i + 1, r = l + 1, m = i;
          if (l < L && hk[l] < hk[m]) m = l;
          if (r < L && hk[r] < hk[m]) m = r;
          if (m === i) break;
          var tk = hk[m]; hk[m] = hk[i]; hk[i] = tk;
          var tv = hv[m]; hv[m] = hv[i]; hv[i] = tv;
          i = m;
        }
      }
      return top;
    }
    dist[src] = 0; push(0, src);
    while (hk.length) {
      var d = hk[0], u = pop();
      if (d > dist[u]) continue;
      var e = adj[u];
      for (var a = 0; a < e.to.length; a++) {
        var v = e.to[a], nd = d + e.w[a];
        if (nd < dist[v]) { dist[v] = nd; if (prev) prev[v] = u; push(nd, v); }
      }
    }
  }

  function allPairs(adj, n) {
    var G = new Float64Array(n * n);
    for (var i = 0; i < n; i++) dijkstra(adj, n, i, G.subarray(i * n, (i + 1) * n));
    return G;
  }

  function shortestPath(adj, n, a, b) {
    var dist = new Float64Array(n), prev = new Int32Array(n);
    dijkstra(adj, n, a, dist, prev);
    if (!isFinite(dist[b])) return { dist: Infinity, path: [] };
    var path = [b];
    while (path[path.length - 1] !== a) path.push(prev[path[path.length - 1]]);
    return { dist: dist[b], path: path.reverse() };
  }

  /* ===================================================== symmetric eigensolver
     eigSym(A, n): A is a symmetric n × n matrix (Float64Array, row-major).
     Returns eigenvalues ascending and the eigenvectors as rows:
       vectors[j] is the unit eigenvector for values[j]. */
  function tred2(V, d, e, n) {
    var i, j, k, f, g, h, hh, scl;
    for (j = 0; j < n; j++) d[j] = V[n - 1][j];
    for (i = n - 1; i > 0; i--) {
      scl = 0; h = 0;
      for (k = 0; k < i; k++) scl += Math.abs(d[k]);
      if (scl === 0) {
        e[i] = d[i - 1];
        for (j = 0; j < i; j++) { d[j] = V[i - 1][j]; V[i][j] = 0; V[j][i] = 0; }
      } else {
        for (k = 0; k < i; k++) { d[k] /= scl; h += d[k] * d[k]; }
        f = d[i - 1]; g = Math.sqrt(h);
        if (f > 0) g = -g;
        e[i] = scl * g; h = h - f * g; d[i - 1] = f - g;
        for (j = 0; j < i; j++) e[j] = 0;
        for (j = 0; j < i; j++) {
          f = d[j]; V[j][i] = f; g = e[j] + V[j][j] * f;
          for (k = j + 1; k <= i - 1; k++) { g += V[k][j] * d[k]; e[k] += V[k][j] * f; }
          e[j] = g;
        }
        f = 0;
        for (j = 0; j < i; j++) { e[j] /= h; f += e[j] * d[j]; }
        hh = f / (h + h);
        for (j = 0; j < i; j++) e[j] -= hh * d[j];
        for (j = 0; j < i; j++) {
          f = d[j]; g = e[j];
          for (k = j; k <= i - 1; k++) V[k][j] -= (f * e[k] + g * d[k]);
          d[j] = V[i - 1][j]; V[i][j] = 0;
        }
      }
      d[i] = h;
    }
    for (i = 0; i < n - 1; i++) {
      V[n - 1][i] = V[i][i]; V[i][i] = 1;
      h = d[i + 1];
      if (h !== 0) {
        for (k = 0; k <= i; k++) d[k] = V[k][i + 1] / h;
        for (j = 0; j <= i; j++) {
          g = 0;
          for (k = 0; k <= i; k++) g += V[k][i + 1] * V[k][j];
          for (k = 0; k <= i; k++) V[k][j] -= g * d[k];
        }
      }
      for (k = 0; k <= i; k++) V[k][i + 1] = 0;
    }
    for (j = 0; j < n; j++) { d[j] = V[n - 1][j]; V[n - 1][j] = 0; }
    V[n - 1][n - 1] = 1;
    e[0] = 0;
  }

  /* QL with implicit shifts. Z holds the columns of V as rows (Z[c][k] = V[k][c])
     so the inner rotation loop walks contiguous memory. */
  function tql2(Z, d, e, n) {
    var i, j, k, l, m, f = 0, tst1 = 0, eps = Math.pow(2, -52);
    for (i = 1; i < n; i++) e[i - 1] = e[i];
    e[n - 1] = 0;
    for (l = 0; l < n; l++) {
      tst1 = Math.max(tst1, Math.abs(d[l]) + Math.abs(e[l]));
      m = l;
      while (m < n) { if (Math.abs(e[m]) <= eps * tst1) break; m++; }
      if (m > l) {
        var iter = 0;
        do {
          iter++;
          var g = d[l], p = (d[l + 1] - g) / (2 * e[l]), r = Math.hypot(p, 1);
          if (p < 0) r = -r;
          d[l] = e[l] / (p + r);
          d[l + 1] = e[l] * (p + r);
          var dl1 = d[l + 1], h = g - d[l];
          for (i = l + 2; i < n; i++) d[i] -= h;
          f += h;
          p = d[m];
          var c = 1, c2 = c, c3 = c, el1 = e[l + 1], s = 0, s2 = 0;
          for (i = m - 1; i >= l; i--) {
            c3 = c2; c2 = c; s2 = s;
            g = c * e[i]; h = c * p; r = Math.hypot(p, e[i]);
            e[i + 1] = s * r; s = e[i] / r; c = p / r;
            p = c * d[i] - s * g;
            d[i + 1] = h + s * (c * g + s * d[i]);
            var zi = Z[i], zi1 = Z[i + 1];
            for (k = 0; k < n; k++) {
              h = zi1[k];
              zi1[k] = s * zi[k] + c * h;
              zi[k] = c * zi[k] - s * h;
            }
          }
          p = -s * s2 * c3 * el1 * e[l] / dl1;
          e[l] = s * p; d[l] = c * p;
        } while (Math.abs(e[l]) > eps * tst1 && iter < 60);
      }
      d[l] += f; e[l] = 0;
    }
    for (i = 0; i < n - 1; i++) {
      k = i; var pv = d[i];
      for (j = i + 1; j < n; j++) if (d[j] < pv) { k = j; pv = d[j]; }
      if (k !== i) { d[k] = d[i]; d[i] = pv; var tz = Z[i]; Z[i] = Z[k]; Z[k] = tz; }
    }
  }

  function eigSym(A, n) {
    var V = new Array(n), d = new Float64Array(n), e = new Float64Array(n), i, j;
    for (i = 0; i < n; i++) V[i] = Float64Array.from(A.subarray ? A.subarray(i * n, (i + 1) * n) : A.slice(i * n, (i + 1) * n));
    tred2(V, d, e, n);
    var Z = new Array(n);
    for (i = 0; i < n; i++) Z[i] = new Float64Array(n);
    for (i = 0; i < n; i++) for (j = 0; j < n; j++) Z[j][i] = V[i][j];
    tql2(Z, d, e, n);
    return { values: d, vectors: Z };
  }

  /* Double centring: B = -½ H A H when given squared distances (MDS), or
     H K H when given a kernel matrix (factor = 1). In place. */
  function doubleCenter(A, n, factor) {
    var r = new Float64Array(n), g = 0, i, j;
    for (i = 0; i < n; i++) { var s = 0; for (j = 0; j < n; j++) s += A[i * n + j]; r[i] = s / n; g += r[i]; }
    g /= n;
    for (i = 0; i < n; i++) for (j = 0; j < n; j++) A[i * n + j] = factor * (A[i * n + j] - r[i] - r[j] + g);
    return A;
  }

  function topK(E, n, L) {
    var Y = [], vals = [];
    for (var l = 0; l < L; l++) vals.push(E.values[n - 1 - l]);
    for (var i = 0; i < n; i++) {
      var y = [];
      for (l = 0; l < L; l++) y.push(E.vectors[n - 1 - l][i] * Math.sqrt(Math.max(vals[l], 0)));
      Y.push(y);
    }
    return Y;
  }
  function spectrumDesc(E, n, m) { var s = []; for (var l = 0; l < Math.min(m, n); l++) s.push(E.values[n - 1 - l]); return s; }

  function median(D, n) {
    var v = [];
    for (var i = 0; i < n; i++) for (var j = i + 1; j < n; j++) v.push(D[i * n + j]);
    v.sort(function (a, b) { return a - b; });
    return v[v.length >> 1];
  }

  /* ================================================================ the six */

  /* PCA: eigenvectors of the sample covariance; z = Wᵀ(x − μ). */
  function pca(X, L) {
    L = L || 2;
    var n = X.length, d = X[0].length, mu = new Array(d).fill(0), i, a, b;
    for (i = 0; i < n; i++) for (a = 0; a < d; a++) mu[a] += X[i][a] / n;
    var C = new Float64Array(d * d);
    for (i = 0; i < n; i++) for (a = 0; a < d; a++) for (b = 0; b < d; b++) C[a * d + b] += (X[i][a] - mu[a]) * (X[i][b] - mu[b]) / n;
    var E = eigSym(C, d), W = [];
    for (var l = 0; l < L; l++) W.push(Array.from(E.vectors[d - 1 - l]));
    var Y = X.map(function (x) { var xc = sub(x, mu); return W.map(function (w) { return dot(w, xc); }); });
    var spec = spectrumDesc(E, d, d), tot = spec.reduce(function (s, v) { return s + Math.max(v, 0); }, 0) || 1;
    return { Y: Y, W: W, mu: mu, spectrum: spec, ratio: spec.map(function (v) { return Math.max(v, 0) / tot; }) };
  }

  /* Classical (Torgerson) MDS on a matrix of distances. */
  function classicalMDS(Dist, n, L) {
    var B = new Float64Array(n * n);
    for (var i = 0; i < n * n; i++) B[i] = Dist[i] * Dist[i];
    doubleCenter(B, n, -0.5);
    var E = eigSym(B, n);
    return { Y: topK(E, n, L || 2), spectrum: spectrumDesc(E, n, 8) };
  }

  /* Kernel PCA with an RBF kernel, σ = bw × median pairwise distance. */
  function kpca(X, bw, D2) {
    var n = X.length; D2 = D2 || sqdist(X);
    var med = Math.sqrt(median(D2, n)), sigma = bw * med, K = new Float64Array(n * n);
    for (var i = 0; i < n * n; i++) K[i] = Math.exp(-D2[i] / (2 * sigma * sigma));
    doubleCenter(K, n, 1);
    var E = eigSym(K, n);
    return { Y: topK(E, n, 2), spectrum: spectrumDesc(E, n, 8), sigma: sigma };
  }

  /* Isomap: kNN graph → all-pairs shortest paths → classical MDS. */
  function isomap(X, k, D2) {
    var n = X.length; D2 = D2 || sqdist(X);
    var g = knnGraph(D2, n, k), comp = components(g.adj, n), G = allPairs(g.adj, n), warn = null;
    if (comp.count > 1) {
      var mx = 0; for (var i = 0; i < n * n; i++) if (isFinite(G[i]) && G[i] > mx) mx = G[i];
      for (i = 0; i < n * n; i++) if (!isFinite(G[i])) G[i] = 1.5 * mx;
      warn = 'The kNN graph has ' + comp.count + ' disconnected pieces. Infinite distances were replaced by 1.5 × the largest finite one, so the layout of separate pieces is arbitrary. Raise k.';
    }
    var m = classicalMDS(G, n, 2);
    return { Y: m.Y, spectrum: m.spectrum, G: G, graph: g, components: comp.count, warn: warn };
  }

  /* Solve a small dense system by Gaussian elimination with partial pivoting. */
  function solve(A, b, k) {
    var M = A.slice(), x = b.slice(), i, j, r;
    for (i = 0; i < k; i++) {
      var p = i; for (r = i + 1; r < k; r++) if (Math.abs(M[r * k + i]) > Math.abs(M[p * k + i])) p = r;
      if (p !== i) { for (j = 0; j < k; j++) { var t = M[i * k + j]; M[i * k + j] = M[p * k + j]; M[p * k + j] = t; } var tb = x[i]; x[i] = x[p]; x[p] = tb; }
      var piv = M[i * k + i] || 1e-300;
      for (r = i + 1; r < k; r++) {
        var f = M[r * k + i] / piv;
        if (!f) continue;
        for (j = i; j < k; j++) M[r * k + j] -= f * M[i * k + j];
        x[r] -= f * x[i];
      }
    }
    for (i = k - 1; i >= 0; i--) { var s = x[i]; for (j = i + 1; j < k; j++) s -= M[i * k + j] * x[j]; x[i] = s / (M[i * k + i] || 1e-300); }
    return x;
  }

  /* LLE reconstruction weights: for each point, the affine combination of its
     k neighbours that best rebuilds it (Σ_j w_ij = 1), with Tikhonov
     regularisation reg × trace(C) on the local Gram matrix. */
  function lleWeights(X, k, reg, D2) {
    var n = X.length, d = X[0].length; D2 = D2 || sqdist(X);
    var nb = knn(D2, n, k), W = [];
    for (var i = 0; i < n; i++) {
      var N = nb[i], kk = N.length, Z = [], a, b;
      for (a = 0; a < kk; a++) Z.push(sub(X[N[a]], X[i]));
      var C = new Array(kk * kk), tr = 0;
      for (a = 0; a < kk; a++) for (b = 0; b < kk; b++) C[a * kk + b] = dot(Z[a], Z[b]);
      for (a = 0; a < kk; a++) tr += C[a * kk + a];
      var R = tr > 0 ? reg * tr : reg;
      for (a = 0; a < kk; a++) C[a * kk + a] += R;
      var w = solve(C, new Array(kk).fill(1), kk), s = w.reduce(function (p, q) { return p + q; }, 0);
      W.push({ nb: N, w: w.map(function (q) { return q / s; }) });
    }
    return W;
  }

  /* LLE: bottom eigenvectors of M = (I − W)ᵀ(I − W), skipping the constant one. */
  function lle(X, k, reg, D2) {
    var n = X.length, W = lleWeights(X, k, reg, D2), M = new Float64Array(n * n);
    for (var i = 0; i < n; i++) {
      var N = W[i].nb, w = W[i].w;
      M[i * n + i] += 1;
      for (var a = 0; a < N.length; a++) {
        M[i * n + N[a]] -= w[a]; M[N[a] * n + i] -= w[a];
        for (var b = 0; b < N.length; b++) M[N[a] * n + N[b]] += w[a] * w[b];
      }
    }
    var E = eigSym(M, n), sq = Math.sqrt(n), Y = [];
    for (i = 0; i < n; i++) Y.push([E.vectors[1][i] * sq, E.vectors[2][i] * sq]);
    var spec = []; for (i = 0; i < 8; i++) spec.push(E.values[i]);
    var warn = null;
    if (k <= X[0].length) warn = 'k ≤ D: the local Gram matrix is not singular, so the weights are not unique in the usual sense. LLE needs k > intrinsic dimension; k > ambient D is the usual setting.';
    return { Y: Y, spectrum: spec, weights: W, warn: warn };
  }

  /* Laplacian eigenmaps: heat-kernel weights on the kNN graph, then the
     generalised problem L y = λ D y, solved through the symmetric normalised
     Laplacian I − D^{-1/2} W D^{-1/2}. heat = Infinity gives 0/1 weights. */
  function laplacian(X, k, heat, D2) {
    var n = X.length; D2 = D2 || sqdist(X);
    var g = knnGraph(D2, n, k), i, a, msum = 0, cnt = 0;
    for (i = 0; i < n; i++) for (a = 0; a < g.adj[i].to.length; a++) { msum += g.adj[i].w[a]; cnt++; }
    var sigma = heat * msum / cnt, Wt = [], deg = new Float64Array(n);
    for (i = 0; i < n; i++) {
      var e = g.adj[i], ws = [];
      for (a = 0; a < e.to.length; a++) {
        var w = isFinite(sigma) ? Math.exp(-e.w[a] * e.w[a] / (2 * sigma * sigma)) : 1;
        ws.push(w); deg[i] += w;
      }
      Wt.push(ws);
    }
    var Ls = new Float64Array(n * n);
    for (i = 0; i < n; i++) {
      Ls[i * n + i] = 1;
      for (a = 0; a < g.adj[i].to.length; a++) { var j = g.adj[i].to[a]; Ls[i * n + j] -= Wt[i][a] / Math.sqrt(deg[i] * deg[j]); }
    }
    var E = eigSym(Ls, n), Y = [], zero = 0;
    for (i = 0; i < n; i++) if (E.values[i] < 1e-9) zero++;
    for (i = 0; i < n; i++) { var q = Math.sqrt(n / deg[i]); Y.push([E.vectors[1][i] * q, E.vectors[2][i] * q]); }
    var spec = []; for (i = 0; i < 8; i++) spec.push(E.values[i]);
    var comp = components(g.adj, n).count;
    return { Y: Y, spectrum: spec, sigma: sigma, components: comp,
      warn: comp > 1 ? 'The kNN graph has ' + comp + ' disconnected pieces, so λ = 0 repeats ' + comp + ' times and the bottom eigenvectors just label the pieces. Raise k.' : null };
  }

  /* ----------------------------------------------------------------- t-SNE
     Exact O(N²) gradient (no Barnes–Hut), with early exaggeration, momentum
     and per-parameter gains, as in van der Maaten & Hinton (2008). */
  function tsneP(D2, n, perplexity) {
    var P = new Float64Array(n * n), target = Math.log(perplexity), row = new Float64Array(n), perps = new Float64Array(n);
    for (var i = 0; i < n; i++) {
      var beta = 1, lo = -Infinity, hi = Infinity, mn = Infinity, j, H = 0, sum = 0;
      for (j = 0; j < n; j++) if (j !== i && D2[i * n + j] < mn) mn = D2[i * n + j];
      for (var it = 0; it < 200; it++) {
        sum = 0; var sdp = 0;
        for (j = 0; j < n; j++) {
          if (j === i) { row[j] = 0; continue; }
          var dd = D2[i * n + j] - mn, p = Math.exp(-dd * beta);
          row[j] = p; sum += p; sdp += dd * p;
        }
        H = Math.log(sum) + beta * sdp / sum;
        if (Math.abs(H - target) < 1e-6) break;
        if (H > target) { lo = beta; beta = hi === Infinity ? beta * 2 : (beta + hi) / 2; }
        else { hi = beta; beta = lo === -Infinity ? beta / 2 : (beta + lo) / 2; }
      }
      for (j = 0; j < n; j++) P[i * n + j] = row[j] / sum;
      perps[i] = Math.exp(H);
    }
    var S = new Float64Array(n * n);
    for (i = 0; i < n; i++) for (j = 0; j < n; j++) S[i * n + j] = Math.max((P[i * n + j] + P[j * n + i]) / (2 * n), 1e-12);
    for (i = 0; i < n; i++) S[i * n + i] = 0;
    return { P: S, cond: P, perplexities: perps };
  }

  function tsneInit(X, opts, D2) {
    var n = X.length; D2 = D2 || sqdist(X);
    var o = { perplexity: 30, lr: 200, exag: 12, exagIters: 100, iters: 1000, seed: 7 };
    for (var key in opts) if (opts[key] !== undefined) o[key] = opts[key];
    var r = rng(o.seed), Y = new Float64Array(2 * n);
    for (var i = 0; i < 2 * n; i++) Y[i] = 1e-4 * randn(r);
    var Pr = tsneP(D2, n, Math.min(o.perplexity, (n - 1) / 3));
    return { n: n, o: o, P: Pr.P, Y: Y, iY: new Float64Array(2 * n), gains: new Float64Array(2 * n).fill(1),
      num: new Float64Array(n * n), grad: new Float64Array(2 * n), iter: 0, kl: NaN, klHist: [] };
  }

  function tsneStep(S) {
    var n = S.n, Y = S.Y, P = S.P, num = S.num, g = S.grad, o = S.o, i, j;
    var ex = S.iter < o.exagIters ? o.exag : 1, sumQ = 0;
    for (i = 0; i < n; i++) {
      var yi0 = Y[2 * i], yi1 = Y[2 * i + 1];
      for (j = i + 1; j < n; j++) {
        var dx = yi0 - Y[2 * j], dy = yi1 - Y[2 * j + 1], q = 1 / (1 + dx * dx + dy * dy);
        num[i * n + j] = q; num[j * n + i] = q; sumQ += 2 * q;
      }
    }
    for (i = 0; i < n; i++) {
      var gx = 0, gy = 0, a0 = Y[2 * i], a1 = Y[2 * i + 1], row = i * n;
      for (j = 0; j < n; j++) {
        if (j === i) continue;
        var qq = num[row + j], m = (ex * P[row + j] - qq / sumQ) * qq;
        gx += m * (a0 - Y[2 * j]); gy += m * (a1 - Y[2 * j + 1]);
      }
      g[2 * i] = 4 * gx; g[2 * i + 1] = 4 * gy;
    }
    var mom = S.iter < 250 ? 0.5 : 0.8, mx = 0, my = 0;
    for (i = 0; i < 2 * n; i++) {
      var same = (g[i] > 0) === (S.iY[i] > 0);
      S.gains[i] = Math.max(same ? S.gains[i] * 0.8 : S.gains[i] + 0.2, 0.01);
      S.iY[i] = mom * S.iY[i] - o.lr * S.gains[i] * g[i];
      Y[i] += S.iY[i];
    }
    for (i = 0; i < n; i++) { mx += Y[2 * i]; my += Y[2 * i + 1]; }
    mx /= n; my /= n;
    for (i = 0; i < n; i++) { Y[2 * i] -= mx; Y[2 * i + 1] -= my; }
    S.iter++;
    if (S.iter % 10 === 0 || S.iter === o.iters) {
      var kl = 0;
      for (i = 0; i < n; i++) for (j = 0; j < n; j++) if (i !== j) { var p = P[i * n + j]; kl += p * Math.log(p / Math.max(num[i * n + j] / sumQ, 1e-300)); }
      S.kl = kl; S.klHist.push([S.iter, kl]);
    }
    return S;
  }
  function tsneY(S) { var Y = []; for (var i = 0; i < S.n; i++) Y.push([S.Y[2 * i], S.Y[2 * i + 1]]); return Y; }
  function tsne(X, opts, D2) {
    var S = tsneInit(X, opts, D2);
    while (S.iter < S.o.iters) tsneStep(S);
    return { Y: tsneY(S), kl: S.kl, state: S };
  }

  /* ================================================================ metrics */

  /* Fraction of each point's k true manifold neighbours that are also among
     its k nearest neighbours in the embedding. 1 = every neighbourhood kept. */
  function neighborKeep(TD, Y, n, k, trueNb) {
    trueNb = trueNb || knn(TD, n, k);
    var eNb = knn(sqdist(Y), n, k), s = 0;
    for (var i = 0; i < n; i++) {
      var set = new Set(trueNb[i]), c = 0;
      for (var a = 0; a < k; a++) if (set.has(eNb[i][a])) c++;
      s += c / k;
    }
    return s / n;
  }

  /* Pearson correlation between true manifold distances and embedding distances. */
  function distCorr(TD, Y, n) {
    var sx = 0, sy = 0, sxx = 0, syy = 0, sxy = 0, m = 0;
    for (var i = 0; i < n; i++) for (var j = i + 1; j < n; j++) {
      var a = TD[i * n + j], dx = Y[i][0] - Y[j][0], dy = Y[i][1] - Y[j][1], b = Math.sqrt(dx * dx + dy * dy);
      sx += a; sy += b; sxx += a * a; syy += b * b; sxy += a * b; m++;
    }
    var cov = sxy / m - sx * sy / (m * m), vx = sxx / m - sx * sx / (m * m), vy = syy / m - sy * sy / (m * m);
    return vx > 0 && vy > 0 ? cov / Math.sqrt(vx * vy) : 0;
  }

  /* Centre, scale to unit RMS radius. */
  function normalize2(Y) {
    var n = Y.length, mx = 0, my = 0, i;
    for (i = 0; i < n; i++) { mx += Y[i][0]; my += Y[i][1]; }
    mx /= n; my /= n;
    var ss = 0; for (i = 0; i < n; i++) { var a = Y[i][0] - mx, b = Y[i][1] - my; ss += a * a + b * b; }
    var s = Math.sqrt(ss / n) || 1;
    return Y.map(function (y) { return [(y[0] - mx) / s, (y[1] - my) / s]; });
  }

  /* Orthogonal Procrustes: the rotation (or reflection) of Y that best matches T.
     Changes nothing about the embedding's shape; it only stops the plots from
     spinning or flipping arbitrarily between methods. */
  function align(Y, T) {
    var A = normalize2(Y), B = normalize2(T);
    function best(Aa) {
      var n00 = 0, n01 = 0, n10 = 0, n11 = 0;
      for (var i = 0; i < Aa.length; i++) { n00 += B[i][0] * Aa[i][0]; n01 += B[i][0] * Aa[i][1]; n10 += B[i][1] * Aa[i][0]; n11 += B[i][1] * Aa[i][1]; }
      var th = Math.atan2(n10 - n01, n00 + n11);
      return { th: th, val: Math.hypot(n10 - n01, n00 + n11) };
    }
    var F = A.map(function (y) { return [y[0], -y[1]]; });
    var p = best(A), q = best(F), src = q.val > p.val ? F : A, th = q.val > p.val ? q.th : p.th;
    var c = Math.cos(th), s = Math.sin(th);
    return src.map(function (y) { return [c * y[0] - s * y[1], s * y[0] + c * y[1]]; });
  }

  /* Dual (Gram-matrix) PCA for N images in a high-dimensional pixel space. */
  function dualPCA(X, L) {
    var n = X.length, d = X[0].length, mu = new Float64Array(d), i, j, a;
    for (i = 0; i < n; i++) for (a = 0; a < d; a++) mu[a] += X[i][a] / n;
    var Xc = X.map(function (x) { var y = new Float64Array(d); for (var a = 0; a < d; a++) y[a] = x[a] - mu[a]; return y; });
    var G = new Float64Array(n * n);
    for (i = 0; i < n; i++) for (j = i; j < n; j++) { var s = 0, xi = Xc[i], xj = Xc[j]; for (a = 0; a < d; a++) s += xi[a] * xj[a]; G[i * n + j] = G[j * n + i] = s; }
    var E = eigSym(G, n), comps = [];
    for (var l = 0; l < L; l++) {
      var lam = Math.max(E.values[n - 1 - l], 1e-12), u = E.vectors[n - 1 - l], w = new Float64Array(d);
      for (i = 0; i < n; i++) for (a = 0; a < d; a++) w[a] += Xc[i][a] * u[i];
      for (a = 0; a < d; a++) w[a] /= Math.sqrt(lam);
      comps.push(w);
    }
    var spec = []; for (i = n - 1; i >= 0; i--) spec.push(Math.max(E.values[i], 0));
    function project(x) { return comps.map(function (w) { var s = 0; for (var a = 0; a < d; a++) s += (x[a] - mu[a]) * w[a]; return s; }); }
    return { mean: mu, comps: comps, spectrum: spec, project: project, Z: Xc.map(function (x) { return project(Array.from(x).map(function (v, a) { return v + mu[a]; })); }) };
  }

  /* ================================================== surfaces (section 1)
     Parametric surfaces f(u, v) ∈ ℝ³ with numerical differential geometry:
     the tangent plane, the metric tensor g (first fundamental form) and the
     Gaussian curvature K = (LN − M²)/(EG − F²). */
  var SURFACES = {
    sphere: { label: 'Sphere', u: [0, TAU], v: [0.08, Math.PI - 0.08], nu: 48, nv: 24, wrapU: true,
      f: function (u, v) { return [3 * Math.sin(v) * Math.cos(u), 3 * Math.cos(v), 3 * Math.sin(v) * Math.sin(u)]; }, start: [0.6, 1.0] },
    swiss: { label: 'Swiss roll', u: [ROLL.t0, ROLL.t1], v: [-2.6, 2.6], nu: 90, nv: 12,
      f: function (t, h) { return [0.26 * t * Math.cos(t), h, 0.26 * t * Math.sin(t)]; }, start: [3.1 * Math.PI, 0.4] },
    saddle: { label: 'Saddle', u: [-2.6, 2.6], v: [-2.6, 2.6], nu: 30, nv: 30,
      f: function (x, z) { return [x, (x * x - z * z) / 3.5, z]; }, start: [0, 0] },
    torus: { label: 'Torus', u: [0, TAU], v: [0, TAU], nu: 56, nv: 22, wrapU: true, wrapV: true,
      f: function (u, v) { var R = 2.6, r = 1.0; return [(R + r * Math.cos(v)) * Math.cos(u), r * Math.sin(v), (R + r * Math.cos(v)) * Math.sin(u)]; }, start: [0.5, 0.3] }
  };

  function surfGeom(S, u, v) {
    var f = S.f, h = 1e-4, h2 = 1e-3;
    var p = f(u, v);
    var fu = scale(sub(f(u + h, v), f(u - h, v)), 1 / (2 * h));
    var fv = scale(sub(f(u, v + h), f(u, v - h)), 1 / (2 * h));
    var fuu = scale(add(sub(f(u + h2, v), scale(p, 2)), f(u - h2, v)), 1 / (h2 * h2));
    var fvv = scale(add(sub(f(u, v + h2), scale(p, 2)), f(u, v - h2)), 1 / (h2 * h2));
    var fuv = scale(sub(sub(f(u + h2, v + h2), f(u + h2, v - h2)), sub(f(u - h2, v + h2), f(u - h2, v - h2))), 1 / (4 * h2 * h2));
    var n = unit(cross(fu, fv));
    var E = dot(fu, fu), F = dot(fu, fv), G = dot(fv, fv), L = dot(fuu, n), M = dot(fuv, n), N = dot(fvv, n);
    var det = E * G - F * F;
    var e1 = unit(fu), e2 = unit(cross(n, e1));
    return { p: p, fu: fu, fv: fv, n: n, e1: e1, e2: e2, E: E, F: F, G: G,
      K: (L * N - M * M) / det, H: (E * N - 2 * F * M + G * L) / (2 * det) };
  }

  /* Points of the surface within ambient distance r of f(u0, v0), found on a
     parameter window around (u0, v0) so other sheets of the roll never leak in.
     Returns tangent-plane coordinates (a, b), height off the plane, and δ. */
  function patch(S, u0, v0, r, res) {
    res = res || 41;
    var g = surfGeom(S, u0, v0), du = 2.2 * r / Math.sqrt(g.E), dv = 2.2 * r / Math.sqrt(g.G);
    if (!S.wrapU) du = Math.min(du, S.u[1] - S.u[0]);
    if (!S.wrapV) dv = Math.min(dv, S.v[1] - S.v[0]);
    var pts = [], delta = 0;
    for (var i = 0; i < res; i++) for (var j = 0; j < res; j++) {
      var u = u0 + du * (2 * i / (res - 1) - 1), v = v0 + dv * (2 * j / (res - 1) - 1);
      if (!S.wrapU && (u < S.u[0] || u > S.u[1])) continue;
      if (!S.wrapV && (v < S.v[0] || v > S.v[1])) continue;
      var q = sub(S.f(u, v), g.p);
      if (norm(q) > r) continue;
      var ht = dot(q, g.n);
      pts.push([dot(q, g.e1), dot(q, g.e2), ht]);
      if (Math.abs(ht) > delta) delta = Math.abs(ht);
    }
    return { geom: g, pts: pts, delta: delta };
  }

  var API = {
    TAU: TAU, rng: rng, randn: randn, dot: dot, sub: sub, add: add, scale: scale, norm: norm, unit: unit, cross: cross,
    ROLL: ROLL, spiralArc: spiralArc, spiralArcInv: spiralArcInv, rollPoint: rollPoint,
    DATASETS: DATASETS, makeData: makeData,
    sqdist: sqdist, knn: knn, knnGraph: knnGraph, components: components, dijkstra: dijkstra, allPairs: allPairs, shortestPath: shortestPath,
    eigSym: eigSym, doubleCenter: doubleCenter, median: median, solve: solve,
    pca: pca, classicalMDS: classicalMDS, kpca: kpca, isomap: isomap, lleWeights: lleWeights, lle: lle, laplacian: laplacian,
    tsneP: tsneP, tsneInit: tsneInit, tsneStep: tsneStep, tsneY: tsneY, tsne: tsne,
    neighborKeep: neighborKeep, distCorr: distCorr, normalize2: normalize2, align: align, dualPCA: dualPCA,
    SURFACES: SURFACES, surfGeom: surfGeom, patch: patch
  };

  if (typeof module !== 'undefined' && module.exports) module.exports = API;
  else global.ML = API;
})(typeof window !== 'undefined' ? window : globalThis);
