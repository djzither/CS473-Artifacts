/* ============================================================================
   PCA Interactive Learning Lab — exam-practice question generators.
   No DOM: each generator returns plain data plus HTML strings, so test-math.js
   can check that every answer agrees with its own worked solution.

   Question shape:
     { topic, kind: 'num' | 'mc' | 'dim', prompt, hint, steps[],
       answer (num) | [rows, cols] (dim), tol, percentOK,
       options[], correct (mc) }
   TeX goes in <span data-tex="…"> and is rendered by the page with KaTeX.
   ========================================================================= */

(function (global) {
  'use strict';

  function esc(s) { return String(s).replace(/&/g, '&amp;').replace(/"/g, '&quot;').replace(/</g, '&lt;'); }
  function tx(s) { return '<span class="tex" data-tex="' + esc(s) + '"></span>'; }
  function td(s) { return '<div class="tex-block" data-tex="' + esc(s) + '"></div>'; }
  function ri(r, a, b) { return a + Math.floor(r() * (b - a + 1)); }
  function pick(r, arr) { return arr[Math.floor(r() * arr.length)]; }
  function shuffle(r, arr) { var a = arr.slice(); for (var i = a.length - 1; i > 0; i--) { var j = Math.floor(r() * (i + 1)); var t = a[i]; a[i] = a[j]; a[j] = t; } return a; }
  function fmt(x, d) { d = d === undefined ? 4 : d; var v = +(+x).toFixed(d); return String(Object.is(v, -0) ? 0 : v); }
  function sum(a) { return a.reduce(function (s, v) { return s + v; }, 0); }
  function plus(a) { return a.map(function (v) { return fmt(v); }).join(' + '); }
  function vecTex(a) { return '(' + a.map(function (v) { return fmt(v); }).join(',\\ ') + ')'; }

  /* descending positive integer eigenvalues */
  function eigs(r, m, lo, hi) {
    var a = []; for (var i = 0; i < m; i++) a.push(ri(r, lo, hi));
    return a.sort(function (x, y) { return y - x; });
  }
  /* a "signal + noise" spectrum with a clear elbow after k */
  function elbowSpectrum(r, k, m) {
    var a = [];
    for (var i = 0; i < k; i++) a.push(ri(r, 14, 30));
    for (i = k; i < m; i++) a.push(+(ri(r, 5, 25) / 10).toFixed(1));
    return a.sort(function (x, y) { return y - x; });
  }

  /* small inline SVG scree plot (as a string) */
  function screeSVG(lams, mark) {
    var W = 340, H = 170, L = 34, R = 10, Tt = 12, B = 30, m = lams.length, top = Math.max.apply(null, lams) * 1.08;
    var x = function (i) { return L + (i + 0.5) * (W - L - R) / m; }, y = function (v) { return Tt + (1 - v / top) * (H - Tt - B); };
    var s = '<svg class="qplot" viewBox="0 0 ' + W + ' ' + H + '" role="img" aria-label="Scree plot of ' + m + ' eigenvalues">';
    s += '<line class="ax" x1="' + L + '" x2="' + (W - R) + '" y1="' + y(0) + '" y2="' + y(0) + '"/>';
    s += '<line class="ax" x1="' + L + '" x2="' + L + '" y1="' + Tt + '" y2="' + y(0) + '"/>';
    [0, top / 2].forEach(function (v) { s += '<text class="tick" x="' + (L - 5) + '" y="' + (y(v) + 4) + '" text-anchor="end">' + fmt(v, 1) + '</text>'; });
    var d = '';
    lams.forEach(function (v, i) { d += (i ? 'L' : 'M') + x(i).toFixed(1) + ',' + y(v).toFixed(1); });
    s += '<path class="qline" d="' + d + '"/>';
    lams.forEach(function (v, i) {
      s += '<circle class="qdot" cx="' + x(i).toFixed(1) + '" cy="' + y(v).toFixed(1) + '" r="4.5"><title>λ' + (i + 1) + ' = ' + fmt(v) + '</title></circle>';
      s += '<text class="tick" x="' + x(i).toFixed(1) + '" y="' + (H - B + 16) + '" text-anchor="middle">' + (i + 1) + '</text>';
    });
    s += '<text class="axt" x="' + ((L + W - R) / 2) + '" y="' + (H - 2) + '" text-anchor="middle">component j</text>';
    if (mark) s += '';
    return s + '</svg>';
  }

  /* ---------- the generators, one per topic ---------- */
  var GEN = {};

  GEN.explained = function (r) {
    var m = ri(r, 5, 7), lams = eigs(r, m, 1, 20), L = ri(r, 1, m - 2), tot = sum(lams), kept = sum(lams.slice(0, L));
    return {
      kind: 'num', percentOK: true, answer: kept / tot, tol: 0.0051,
      prompt: '<p>The covariance matrix of a centered data set has eigenvalues</p>' + td('\\lambda = ' + vecTex(lams)) +
        '<p>What fraction of the variance is explained by the first ' + tx('L=' + L) + ' principal components? Give a decimal (e.g. 0.73) or a percent.</p>',
      hint: 'Eq. 20.27 divides the sum of the <b>kept</b> eigenvalues by the sum of <b>all</b> of them.',
      steps: [
        'Each eigenvalue ' + tx('\\lambda_j') + ' is the variance of the data along principal direction ' + tx('j') + ', and the total variance is ' + tx('\\operatorname{tr}\\hat\\Sigma = \\sum_j \\lambda_j') + '.',
        'Kept: ' + tx('\\sum_{j=1}^{' + L + '}\\lambda_j = ' + plus(lams.slice(0, L)) + ' = ' + fmt(kept)) + '.',
        'Total: ' + tx('\\sum_{j=1}^{' + m + '}\\lambda_j = ' + fmt(tot)) + '.',
        td('F(' + L + ') = \\frac{' + fmt(kept) + '}{' + fmt(tot) + '} = ' + fmt(kept / tot) + '\\ (' + fmt(100 * kept / tot, 2) + '\\%)')
      ]
    };
  };

  GEN.recon = function (r) {
    if (r() < 0.6) {
      var m = ri(r, 5, 7), lams = eigs(r, m, 1, 18), L = ri(r, 1, m - 2), rest = lams.slice(L);
      return {
        kind: 'num', answer: sum(rest), tol: 1e-6,
        prompt: '<p>PCA is fit to a training set whose sample covariance ' + tx('\\hat\\Sigma = \\tfrac1N X^\\top X') + ' has eigenvalues</p>' + td('\\lambda = ' + vecTex(lams)) +
          '<p>Keeping ' + tx('L = ' + L) + ' components, what is the average squared reconstruction error on the <b>training</b> set, ' + tx('\\tfrac1N\\sum_n \\lVert x_n - \\hat x_n\\rVert^2') + '?</p>',
        hint: 'Eq. 20.26: the error is whatever variance you threw away.',
        steps: [
          'The reconstruction ' + tx('\\hat x_n = W_L W_L^\\top x_n') + ' keeps the component of ' + tx('x_n') + ' in the span of the first ' + L + ' eigenvectors and drops the rest.',
          'The dropped part has squared length ' + tx('\\sum_{j>L} (u_j^\\top x_n)^2') + '; averaging over ' + tx('n') + ' turns each ' + tx('\\tfrac1N\\sum_n (u_j^\\top x_n)^2') + ' into ' + tx('u_j^\\top\\hat\\Sigma u_j = \\lambda_j') + '.',
          td('\\mathcal L_{' + L + '} = \\sum_{j=' + (L + 1) + '}^{' + m + '} \\lambda_j = ' + plus(rest) + ' = ' + fmt(sum(rest)))
        ]
      };
    }
    var total = pick(r, [20, 40, 50, 80, 120, 200]), F = pick(r, [0.6, 0.75, 0.8, 0.85, 0.9, 0.95]);
    return {
      kind: 'num', answer: total * (1 - F), tol: 1e-6,
      prompt: '<p>The total variance of a data set is ' + tx('\\operatorname{tr}\\hat\\Sigma = ' + total) + '. The first ' + tx('L') + ' components explain ' + fmt(100 * F) + '% of it. What is the training reconstruction error ' + tx('\\mathcal L_L') + '?</p>',
      hint: 'Explained + unexplained variance = total variance.',
      steps: [
        'Eq. 20.26 says ' + tx('\\mathcal L_L = \\sum_{j>L}\\lambda_j') + ': the variance not explained.',
        'Explained variance: ' + tx(fmt(F) + '\\times ' + total + ' = ' + fmt(F * total)) + '.',
        td('\\mathcal L_L = ' + total + ' - ' + fmt(F * total) + ' = ' + fmt(total * (1 - F)))
      ]
    };
  };

  GEN.minL = function (r) {
    var m = ri(r, 6, 8), lams = eigs(r, m, 1, 25), target = pick(r, [0.8, 0.9, 0.95, 0.99]), tot = sum(lams);
    var cum = 0, rows = [], ans = m;
    for (var i = 0; i < m; i++) { cum += lams[i]; rows.push(tx('L=' + (i + 1) + ':\\ ' + fmt(cum) + '/' + fmt(tot) + ' = ' + fmt(cum / tot, 3))); if (cum / tot >= target - 1e-12 && ans === m) ans = Math.min(ans, i + 1); }
    var first = rows.slice(0, Math.min(m, ans + 1));
    return {
      kind: 'num', answer: ans, tol: 0,
      prompt: '<p>Eigenvalues of ' + tx('\\hat\\Sigma') + ':</p>' + td('\\lambda = ' + vecTex(lams)) +
        '<p>What is the smallest number of components ' + tx('L') + ' that explains at least ' + fmt(100 * target) + '% of the variance?</p>',
      hint: 'Accumulate the eigenvalues from largest to smallest and stop at the first cumulative fraction ≥ ' + fmt(target) + '.',
      steps: [
        'Total variance ' + tx('= ' + fmt(tot)) + '. Target: ' + tx(fmt(target) + '\\times' + fmt(tot) + ' = ' + fmt(target * tot)) + '.',
        'Cumulative fractions (Eq. 20.27): ' + first.join(', ') + (first.length < m ? ', …' : '') + '.',
        'The first ' + tx('L') + ' with ' + tx('F(L) \\ge ' + fmt(target)) + ' is ' + tx('L = ' + ans) + '.'
      ]
    };
  };

  GEN.scree = function (r) {
    if (r() < 0.65) {
      var k = ri(r, 2, 4), m = ri(r, 8, 10), lams = elbowSpectrum(r, k, m);
      return {
        kind: 'num', answer: k, tol: 0,
        prompt: '<p>Here is the scree plot of a data set (eigenvalues in decreasing order; hover a point for its value).</p>' + screeSVG(lams) +
          '<p>Where is the elbow? Enter the number of components you would keep.</p>',
        hint: 'Look for where the curve stops falling steeply and flattens into a slowly decaying tail.',
        steps: [
          'The values are ' + tx(vecTex(lams)) + '.',
          'The first ' + k + ' are large (' + tx('\\ge ' + fmt(lams[k - 1])) + '); from ' + tx('j = ' + (k + 1)) + ' on they are all small (' + tx('\\le ' + fmt(lams[k])) + ') and roughly level.',
          'The flat tail looks like noise: directions with similar small variance. The elbow is at ' + tx('L = ' + k) + '.',
          'Check: ' + tx('F(' + k + ') = ' + fmt(sum(lams.slice(0, k)) / sum(lams), 3)) + ' of the variance.'
        ]
      };
    }
    var lams2 = []; var v = ri(r, 18, 26);
    for (var i = 0; i < 9; i++) { lams2.push(+v.toFixed(1)); v *= 0.72 + 0.06 * r(); }
    var opts = [
      'There is no clear elbow, so the scree plot alone doesn\'t pick L. Use a variance threshold (Eq. 20.27) or profile likelihood instead.',
      'Keep 1 component, because λ₁ is the largest.',
      'Keep all 9 components, because every eigenvalue is positive.',
      'The data must not have been centered.'
    ];
    var order = shuffle(r, [0, 1, 2, 3]);
    return {
      kind: 'mc', options: order.map(function (j) { return opts[j]; }), correct: order.indexOf(0),
      prompt: '<p>A scree plot whose eigenvalues shrink by a roughly constant factor each step:</p>' + screeSVG(lams2) + '<p>What should you conclude?</p>',
      hint: 'An elbow is a sharp change in slope. Is there one?',
      steps: [
        'Values: ' + tx(vecTex(lams2.map(function (x) { return +x.toFixed(1); }))) + '. Each is about 0.72–0.78 times the one before: a geometric decay with no change in slope.',
        'Without a sharp bend, "visual elbow detection" is subjective. Any L you pick is arbitrary.',
        'Use a quantitative rule instead: choose the smallest L with ' + tx('F(L)\\ge 0.9') + ' (or 0.95), or maximize the profile likelihood (§20.1.4.3).'
      ]
    };
  };

  GEN.dims = function (r) {
    var D = ri(r, 3, 12), N = D + ri(r, 3, 40), L = ri(r, 1, D - 1);
    var items = [
      ['X', 'the data matrix', [N, D], 'one row per observation, one column per feature.'],
      ['\\hat\\Sigma = \\tfrac1N X^\\top X', 'the sample covariance', [D, D], 'feature-by-feature: (D×N)(N×D).'],
      ['X X^\\top', 'the Gram matrix', [N, N], 'observation-by-observation: (N×D)(D×N).'],
      ['W', 'the PCA weight (loading) matrix', [D, L], 'one column per kept principal direction, each a vector in feature space.'],
      ['Z = XW', 'the latent scores', [N, L], 'one L-dimensional code per observation: (N×D)(D×L).'],
      ['\\hat X = Z W^\\top', 'the reconstruction', [N, D], 'back in feature space: (N×L)(L×D).'],
      ['U', 'U in the economy SVD X = USV^\\top', [N, D], 'with N > D the economy SVD keeps min(N, D) = D left singular vectors, each of length N.'],
      ['S', 'S in the economy SVD', [D, D], 'a diagonal matrix of the D singular values.'],
      ['V', 'V in the SVD', [D, D], 'its columns are the right singular vectors = principal directions in feature space.'],
      ['U_L S_L', 'the truncated product U_L S_L', [N, L], 'this is Z: (N×L)(L×L).']
    ];
    var it = pick(r, items);
    return {
      kind: 'dim', answer: it[2],
      prompt: '<p>A data set has ' + tx('N = ' + N) + ' observations and ' + tx('D = ' + D) + ' features, and PCA keeps ' + tx('L = ' + L) + ' components. What are the dimensions of ' + tx(it[0]) + ' (' + it[1] + ')? Answer as <i>rows × cols</i>, e.g. <code>5 x 3</code>.</p>',
      hint: 'Write each factor\'s shape and check that the inner dimensions match.',
      steps: [
        'Murphy\'s convention: ' + tx('X \\in \\mathbb R^{N\\times D}') + ', ' + tx('W\\in\\mathbb R^{D\\times L}') + ', ' + tx('Z\\in\\mathbb R^{N\\times L}') + '.',
        tx(it[0]) + ' is ' + it[3],
        'So ' + tx(it[0]) + ' is ' + tx(it[2][0] + '\\times' + it[2][1]) + '.'
      ]
    };
  };

  GEN.gram = function (r) {
    var wide = r() < 0.65;
    var N = wide ? pick(r, [50, 100, 200, 400]) : pick(r, [5000, 10000, 60000]);
    var D = wide ? pick(r, [4096, 10000, 65536, 1000000]) : pick(r, [20, 50, 100, 784]);
    if (r() < 0.4) {
      var which = pick(r, ['gram', 'cov']), ans = which === 'gram' ? N * N : D * D;
      return {
        kind: 'num', answer: ans, tol: 0,
        prompt: '<p>' + tx('N = ' + N) + ', ' + tx('D = ' + D) + '. How many entries does ' + (which === 'gram' ? 'the Gram matrix ' + tx('XX^\\top') : tx('X^\\top X')) + ' have?</p>',
        hint: 'First get its shape from (rows of the left factor) × (columns of the right factor).',
        steps: [which === 'gram' ? tx('XX^\\top') + ' is ' + tx('(N\\times D)(D\\times N) = N\\times N') + '.' : tx('X^\\top X') + ' is ' + tx('(D\\times N)(N\\times D) = D\\times D') + '.',
          'Entries: ' + tx((which === 'gram' ? N : D) + '^2 = ' + ans.toLocaleString('en-US').replace(/,/g, '{,}')) + '.']
      };
    }
    var opts = ['Eigendecompose the D×D matrix XᵀX', 'Eigendecompose the N×N Gram matrix XXᵀ, then map back with V = XᵀUΛ^(−1/2)', 'It makes no difference: both cost the same'];
    var corr = D > N ? 1 : 0;
    return {
      kind: 'mc', options: opts, correct: corr,
      prompt: '<p>You have ' + tx('N = ' + N.toLocaleString('en-US').replace(/,/g, '{,}')) + ' centered observations with ' + tx('D = ' + D.toLocaleString('en-US').replace(/,/g, '{,}')) + ' features. Which is cheaper for finding the principal directions?</p>',
      hint: 'A dense symmetric eigendecomposition of a k×k matrix costs O(k³). Which k is smaller?',
      steps: [
        tx('X^\\top X') + ' is ' + tx(D + '\\times' + D) + ': ' + tx('O(D^3)') + ' to eigendecompose. ' + tx('XX^\\top') + ' is ' + tx(N + '\\times' + N) + ': ' + tx('O(N^3)') + '.',
        D > N ? 'Here ' + tx('N \\ll D') + ', so use the Gram matrix (§20.1.3.2). It has the same nonzero eigenvalues, and Eq. 20.22 recovers the unit eigenvectors: ' + tx('V = X^\\top U\\Lambda^{-1/2}') + '.'
              : 'Here ' + tx('D \\ll N') + ', so the covariance route is cheaper. The Gram trick pays off only when ' + tx('D > N') + ' (images, genes, text).',
        'At most ' + tx('\\min(N, D)') + ' eigenvalues are nonzero either way (' + tx('\\min(N-1, D)') + ' after centering), so the small matrix loses nothing.'
      ]
    };
  };

  GEN.svdeig = function (r) {
    var N = pick(r, [4, 5, 8, 10, 16, 20, 25]), s = [ri(r, 8, 14), ri(r, 4, 7), ri(r, 1, 3)], j = ri(r, 1, 3), v = ri(r, 0, 2);
    if (v === 0) return {
      kind: 'num', answer: s[j - 1] * s[j - 1] / N, tol: 0.0051,
      prompt: '<p>A centered ' + tx(N + '\\times 3') + ' data matrix has singular values ' + tx('s = ' + vecTex(s)) + '. What is the eigenvalue ' + tx('\\lambda_' + j) + ' of the covariance matrix ' + tx('\\hat\\Sigma = \\tfrac1N X^\\top X') + '?</p>',
      hint: 'Substitute X = USVᵀ into XᵀX. What happens to UᵀU?',
      steps: [
        tx('X^\\top X = VSU^\\top USV^\\top = VS^2V^\\top') + ', because ' + tx('U^\\top U = I') + '.',
        'So ' + tx('\\hat\\Sigma = V\\big(\\tfrac1N S^2\\big)V^\\top') + ': eigenvectors ' + tx('V') + ', eigenvalues ' + tx('\\lambda_j = s_j^2/N') + '.',
        td('\\lambda_' + j + ' = \\frac{' + s[j - 1] + '^2}{' + N + '} = \\frac{' + s[j - 1] * s[j - 1] + '}{' + N + '} = ' + fmt(s[j - 1] * s[j - 1] / N))
      ]
    };
    if (v === 1) {
      var lam = s[j - 1] * s[j - 1] / N;
      return {
        kind: 'num', answer: s[j - 1], tol: 0.0051,
        prompt: '<p>The covariance matrix ' + tx('\\hat\\Sigma = \\tfrac1N X^\\top X') + ' of a centered data set with ' + tx('N = ' + N) + ' has eigenvalue ' + tx('\\lambda_' + j + ' = ' + fmt(lam)) + '. What is the singular value ' + tx('s_' + j) + ' of ' + tx('X') + '?</p>',
        hint: 'Invert λ = s²/N. Singular values are non-negative.',
        steps: ['From ' + tx('\\lambda_j = s_j^2/N') + ': ' + tx('s_j = \\sqrt{N\\lambda_j}') + '.', td('s_' + j + ' = \\sqrt{' + N + '\\times' + fmt(lam) + '} = \\sqrt{' + fmt(N * lam) + '} = ' + fmt(s[j - 1]))]
      };
    }
    var sq = s.map(function (x) { return x * x; });
    return {
      kind: 'num', percentOK: true, answer: sq[0] / sum(sq), tol: 0.0051,
      prompt: '<p>A centered data matrix has singular values ' + tx('s = ' + vecTex(s)) + '. What fraction of the variance does the first principal component explain?</p>',
      hint: 'Variance goes with λ ∝ s², not with s. The 1/N cancels.',
      steps: ['Each ' + tx('\\lambda_j = s_j^2/N') + ', and the factor ' + tx('1/N') + ' cancels in the ratio.',
        'Squares: ' + tx(vecTex(sq)) + ', total ' + tx(fmt(sum(sq))) + '.',
        td('F(1) = \\frac{' + sq[0] + '}{' + sum(sq) + '} = ' + fmt(sq[0] / sum(sq))),
        'Using ' + tx('s_1/\\sum s_j = ' + fmt(s[0] / sum(s))) + ' instead is the classic mistake.']
    };
  };

  GEN.pcdir = function (r) {
    var M;
    if (r() < 0.5) { var a = 0.6, b = 0.8; M = [[a, -b], [b, a]]; }
    else M = [[1, 2, 2], [2, 1, -2], [2, -2, 1]].map(function (row) { return row.map(function (v) { return v / 3; }); });
    var D = M.length, perm = shuffle(r, M.map(function (_, i) { return i; }));
    var V = M.map(function (row) { return perm.map(function (p) { return row[p]; }); });
    var signs = perm.map(function () { return r() < 0.5 ? -1 : 1; });
    V = V.map(function (row) { return row.map(function (v, c) { return v * signs[c]; }); });
    var showT = r() < 0.5, which = ri(r, 1, Math.min(2, D)), comp = ri(r, 1, D);
    var shown = showT ? V[0].map(function (_, c) { return V.map(function (row) { return row[c]; }); }) : V;
    var texM = '\\begin{bmatrix}' + shown.map(function (row) { return row.map(function (v) { return fmt(v, 4); }).join(' & '); }).join('\\\\') + '\\end{bmatrix}';
    var w = V.map(function (row) { return row[which - 1]; });
    return {
      kind: 'num', answer: w[comp - 1], tol: 0.002,
      prompt: '<p>The SVD of a centered data matrix ' + tx('X = USV^\\top') + ' (singular values sorted in decreasing order) has</p>' +
        td((showT ? 'V^\\top' : 'V') + ' = ' + texM) +
        '<p>What is component ' + comp + ' of the principal direction ' + tx('w_' + which) + '?</p>',
      hint: 'The principal directions are the columns of V. Read carefully whether you were given V or Vᵀ.',
      steps: [
        'PCA takes ' + tx('W = V_L') + ': the ' + tx('j') + '-th principal direction is the ' + tx('j') + '-th <b>column of ' + tx('V') + '</b>, a unit vector in feature space (length ' + tx('D=' + D) + ').',
        showT ? 'You were given ' + tx('V^\\top') + ', so the columns of ' + tx('V') + ' are the <b>rows</b> of the matrix shown.' : 'You were given ' + tx('V') + ' itself, so read down column ' + which + '.',
        tx('w_' + which + ' = ' + vecTex(w.map(function (v) { return +v.toFixed(4); }))) + ', so component ' + comp + ' is ' + tx(fmt(w[comp - 1])) + '.',
        'Not ' + tx('U') + ': its columns live in ' + tx('\\mathbb R^N') + ' (one entry per observation). ' + tx('US') + ' gives the scores ' + tx('Z') + ', not the directions.'
      ]
    };
  };

  var STD_MC = [
    { q: 'Height is recorded in millimetres instead of metres; weight stays in kilograms. What happens to PCA on the <b>covariance</b> matrix?',
      a: ['PC1 swings to point almost entirely along height, because height\'s variance is multiplied by 10⁶.', 'Nothing. PCA finds the same directions in any units.', 'PC1 swings toward weight, because height\'s numbers become harder to compare.', 'All eigenvalues are multiplied by 10⁶, but the eigenvectors stay the same.'],
      s: ['Scaling feature 1 by c turns ' + tx('\\hat\\Sigma_{11}') + ' into ' + tx('c^2\\hat\\Sigma_{11}') + ' and ' + tx('\\hat\\Sigma_{12}') + ' into ' + tx('c\\,\\hat\\Sigma_{12}') + '. With ' + tx('c = 1000') + ', the (1,1) entry dominates.',
        'The top eigenvector of a matrix dominated by one diagonal entry is nearly that coordinate axis, so PC1 ≈ "height".',
        'PCA maximizes variance in the units it is given, so it "finds" whatever is measured with the biggest numbers. Standardizing (correlation-matrix PCA) removes this (§20.1.3.1).'] },
    { q: 'You standardize every feature to mean 0 and variance 1 before PCA. Which matrix are you now eigendecomposing?',
      a: ['The correlation matrix of the original features', 'The covariance matrix of the original features', 'The Gram matrix XXᵀ', 'The identity matrix'],
      s: ['After standardization, ' + tx('\\tilde x_{nd} = (x_{nd} - \\bar x_d)/\\sigma_d') + '.', 'Its covariance is ' + tx('\\tfrac1N\\sum_n \\tilde x_{ni}\\tilde x_{nj} = \\hat\\Sigma_{ij}/(\\sigma_i\\sigma_j) = \\rho_{ij}') + ': the correlation matrix.', 'So "PCA on standardized data" and "PCA on the correlation matrix" are the same thing.'] },
    { q: 'When is it usually <b>better not</b> to standardize before PCA?',
      a: ['When all features share the same units and their variances are meaningful (e.g. pixel intensities)', 'When the features are measured in different units', 'When one feature has a much larger numeric range only because of its units', 'Never. Standardizing is always better.'],
      s: ['Standardizing gives every feature variance 1 and so equal say.', 'For pixels in an image (MNIST, faces), variance differences are real: border pixels are almost constant and should contribute little. Standardizing would inflate the noise in those nearly constant pixels.', 'Standardize when units are arbitrary or mixed; keep the covariance when units are shared and meaningful.'] },
    { q: 'Feature 2 is rescaled by a factor of 50 and PCA is then run on the <b>correlation</b> matrix. What changes?',
      a: ['Nothing: the correlation matrix, its eigenvalues and its eigenvectors are unchanged', 'The eigenvalues are multiplied by 50', 'PC1 rotates toward feature 2', 'The correlation between the features is multiplied by 50'],
      s: [tx('\\rho_{12} = \\hat\\Sigma_{12}/(\\sigma_1\\sigma_2)') + ': rescaling feature 2 by ' + tx('c') + ' multiplies both numerator and ' + tx('\\sigma_2') + ' by ' + tx('c') + ', which cancel.', 'The diagonal stays 1. Same matrix, same eigendecomposition: correlation-based PCA is invariant to the units of each feature.'] }
  ];
  GEN.standardize = function (r) {
    var v = ri(r, 0, 5);
    if (v === 4) {
      var rho = pick(r, [-0.9, -0.6, -0.3, 0.2, 0.5, 0.7, 0.8, 0.95]);
      return {
        kind: 'num', answer: 1 + Math.abs(rho), tol: 0.0051,
        prompt: '<p>Two standardized features have correlation ' + tx('\\rho = ' + rho) + '. What is the largest eigenvalue of their correlation matrix ' + tx('R = \\begin{bmatrix}1&\\rho\\\\\\rho&1\\end{bmatrix}') + '?</p>',
        hint: 'Try the vectors (1, 1) and (1, −1).',
        steps: [tx('R(1,1)^\\top = (1+\\rho)(1,1)^\\top') + ' and ' + tx('R(1,-1)^\\top = (1-\\rho)(1,-1)^\\top') + '.',
          'So the eigenvalues are ' + tx('1\\pm\\rho') + ' and the eigenvectors are always the diagonals ' + tx('\\tfrac{1}{\\sqrt2}(1,\\pm1)') + ', whatever ' + tx('\\rho') + ' is.',
          'Largest: ' + tx('1 + |\\rho| = ' + fmt(1 + Math.abs(rho))) + (rho < 0 ? ' (along ' + tx('(1,-1)/\\sqrt2') + ' because ' + tx('\\rho<0') + ').' : '.')]
      };
    }
    if (v === 5) {
      var D = ri(r, 3, 30);
      return {
        kind: 'num', answer: D, tol: 0,
        prompt: '<p>A data set with ' + tx('D = ' + D) + ' features is standardized and PCA is run on its correlation matrix. What is the sum of all ' + D + ' eigenvalues?</p>',
        hint: 'The sum of the eigenvalues is the trace.',
        steps: [tx('\\sum_j\\lambda_j = \\operatorname{tr} R') + ', and every diagonal entry of a correlation matrix is 1.', 'So the total is ' + tx('D = ' + D) + '. An eigenvalue above 1 means that component carries more variance than any single original feature (Kaiser\'s "eigenvalue > 1" rule of thumb comes from this).']
      };
    }
    var it = STD_MC[v], order = shuffle(r, it.a.map(function (_, i) { return i; }));
    return { kind: 'mc', prompt: '<p>' + it.q + '</p>', options: order.map(function (i) { return it.a[i]; }), correct: order.indexOf(0), hint: 'Ask what rescaling does to each entry of the matrix being eigendecomposed.', steps: it.s };
  };

  var WHY_MC = [
    { q: 'Why does the <b>training</b> reconstruction error never increase as you keep more components?',
      a: ['The subspaces are nested: span(w₁…w_L) ⊂ span(w₁…w_{L+1}), so the closest point can only get closer, and the error Σ_{j>L} λ_j loses a non-negative term each step', 'Because the eigenvalues are sorted in decreasing order', 'Because PCA is fit by gradient descent, which always lowers the loss', 'It does increase once L passes the elbow'],
      s: ['Projecting onto a bigger subspace that contains the old one can never move the projection farther from ' + tx('x') + '.', 'In numbers: ' + tx('\\mathcal L_L - \\mathcal L_{L+1} = \\lambda_{L+1} \\ge 0') + ' (Eq. 20.26). Sorting decides <i>how fast</i> it falls, not <i>that</i> it falls.'] },
    { q: 'Why doesn\'t the <b>test</b> reconstruction error of PCA show the usual U shape as L grows?',
      a: ['PCA is not a generative model that can overfit test data: the reconstruction just projects the test point onto a nested, ever larger subspace, and at L = D it is exact', 'Because the test set is always smaller than the training set', 'Because PCA uses the 1/N covariance instead of 1/(N−1)', 'It does show a U shape; the curve in the book is smoothed'],
      s: ['A test point ' + tx('x_*') + ' is reconstructed as ' + tx('\\hat x_* = W_LW_L^\\top x_*') + ': the orthogonal projection onto the span of the training directions.', 'The spans are nested, so the error on <i>any</i> fixed point is non-increasing in ' + tx('L') + ', and it is 0 at ' + tx('L = D') + ' (the span is everything).', 'Murphy §20.1.4.1: reconstruction error does not penalize complexity, so it cannot pick ' + tx('L') + '. A proper likelihood (probabilistic PCA, §20.2) does show a U shape on test data.'] },
    { q: 'Which statement about choosing L by <b>test reconstruction error</b> is correct?',
      a: ['It always prefers the largest L, so you need something else (scree plot, variance threshold, profile likelihood, or a probabilistic model\'s likelihood)', 'Pick the L that minimizes test error; that is the bias–variance optimum', 'Pick the L where test error first exceeds training error', 'Test error is undefined for PCA'],
      s: ['Because test error is monotone non-increasing in ' + tx('L') + ', "minimize test error" just returns ' + tx('L = D') + '.', 'That is why §20.1.4 turns to scree plots (Eq. 20.26–20.27) and profile likelihood (Eqs. 20.28–20.31).'] }
  ];
  GEN.whydecrease = function (r) {
    var it = pick(r, WHY_MC), order = shuffle(r, it.a.map(function (_, i) { return i; }));
    return { kind: 'mc', prompt: '<p>' + it.q + '</p>', options: order.map(function (i) { return it.a[i]; }), correct: order.indexOf(0), hint: 'Think about where x̂ lives as L goes from L to L+1.', steps: it.s };
  };

  var PROF_MC = [
    { q: 'In the profile-likelihood method, why do both groups of eigenvalues share <b>one</b> variance σ²?',
      a: ['With separate variances, a group with a single eigenvalue gets σ² → 0 and infinite likelihood, so every split would be degenerate; a shared σ² makes splits comparable', 'To make the math simpler; separate variances give the same answer', 'Because eigenvalues of a covariance matrix always have equal variance', 'Because the two means are also equal'],
      s: ['A Gaussian fit to one point with its own variance has ' + tx('\\sigma^2 = 0') + ' and ' + tx('\\log\\mathcal N \\to +\\infty') + '.', 'Pooling the variance (Eq. 20.31) gives each split a single, finite ' + tx('\\sigma^2(L)') + '. Then ' + tx('\\ell(L) = -\\tfrac{L_{max}}2\\log(2\\pi\\sigma^2(L)) - \\tfrac{L_{max}}2') + ', so the best ' + tx('L') + ' is the split with the smallest pooled within-group spread.'] },
    { q: 'Maximizing the profile log likelihood ℓ(L) over L is equivalent to…',
      a: ['minimizing the pooled within-group sum of squares Σ_{k≤L}(λ_k−μ₁)² + Σ_{k>L}(λ_k−μ₂)²', 'maximizing the explained variance F(L)', 'minimizing the reconstruction error Σ_{k>L} λ_k', 'maximizing the gap λ_L − λ_{L+1}'],
      s: ['Plug the MLEs into ' + tx('\\ell') + ': the quadratic terms sum to ' + tx('\\mathrm{SS}/(2\\sigma^2) = L_{max}/2') + ', a constant.', 'What is left is ' + tx('-\\tfrac{L_{max}}2\\log(2\\pi\\,\\mathrm{SS}/L_{max})') + ', which decreases in SS. So the best split minimizes SS: a 1-D two-cluster problem on the eigenvalues.'] }
  ];
  GEN.profile = function (r) {
    var v = ri(r, 0, 4);
    if (v >= 3) {
      var it = pick(r, PROF_MC), order = shuffle(r, it.a.map(function (_, i) { return i; }));
      return { kind: 'mc', prompt: '<p>' + it.q + '</p>', options: order.map(function (i) { return it.a[i]; }), correct: order.indexOf(0), hint: 'Write ℓ(L) with the maximum-likelihood μ₁, μ₂, σ² plugged in.', steps: it.s };
    }
    var k = ri(r, 2, 3), m = 6, lams = [];
    for (var i = 0; i < k; i++) lams.push(ri(r, 9, 14));
    for (i = k; i < m; i++) lams.push(ri(r, 1, 4));
    lams.sort(function (a, b) { return b - a; });
    var L = ri(r, 2, 4), g1 = lams.slice(0, L), g2 = lams.slice(L);
    var mu1 = sum(g1) / L, mu2 = sum(g2) / (m - L);
    var ss1 = sum(g1.map(function (x) { return (x - mu1) * (x - mu1); })), ss2 = sum(g2.map(function (x) { return (x - mu2) * (x - mu2); }));
    var s2 = (ss1 + ss2) / m, target = ['mu1', 'mu2', 'sigma2'][v];
    var ans = target === 'mu1' ? mu1 : target === 'mu2' ? mu2 : s2;
    var name = target === 'mu1' ? '\\mu_1(L)' : target === 'mu2' ? '\\mu_2(L)' : '\\sigma^2(L)';
    return {
      kind: 'num', answer: ans, tol: 0.0051,
      prompt: '<p>Eigenvalues ' + tx('\\lambda = ' + vecTex(lams)) + ' (' + tx('L_{max} = ' + m) + '). For the split ' + tx('L = ' + L) + ', compute ' + tx(name) + '.</p>',
      hint: target === 'sigma2' ? 'Pool the squared deviations of BOTH groups (each from its own mean) and divide by L_max.' : 'It is the plain average of the eigenvalues in that group.',
      steps: [
        'Group 1 (kept): ' + tx(vecTex(g1)) + '. Group 2 (discarded): ' + tx(vecTex(g2)) + '.',
        'Eqs. 20.29–20.30: ' + tx('\\mu_1 = \\tfrac{' + sum(g1) + '}{' + L + '} = ' + fmt(mu1)) + ', ' + tx('\\mu_2 = \\tfrac{' + sum(g2) + '}{' + (m - L) + '} = ' + fmt(mu2)) + '.',
        target === 'sigma2' ? 'Eq. 20.31: ' + tx('\\sigma^2 = \\tfrac{1}{' + m + '}\\big[' + fmt(ss1) + ' + ' + fmt(ss2) + '\\big] = ' + fmt(s2)) + ', where ' + fmt(ss1) + ' and ' + fmt(ss2) + ' are the within-group sums of squares.'
          : 'Answer: ' + tx(name + ' = ' + fmt(ans)) + '.'
      ]
    };
  };

  var TOPICS = [
    { key: 'explained', name: 'Explained variance (Eq. 20.27)' },
    { key: 'recon', name: 'Reconstruction error from eigenvalues (Eq. 20.26)' },
    { key: 'minL', name: 'Minimum components for a target' },
    { key: 'scree', name: 'Interpreting scree plots' },
    { key: 'dims', name: 'Matrix dimensions' },
    { key: 'gram', name: 'Covariance vs. Gram matrix' },
    { key: 'svdeig', name: 'Singular values → eigenvalues' },
    { key: 'pcdir', name: 'Principal directions from an SVD' },
    { key: 'standardize', name: 'Effects of standardization' },
    { key: 'whydecrease', name: 'Why reconstruction error decreases' },
    { key: 'profile', name: 'Profile likelihood' }
  ];

  function make(key, r) { var q = GEN[key](r); q.topic = key; return q; }

  /* Grade a typed answer. Returns true/false. */
  function grade(q, text) {
    text = String(text || '').trim();
    if (q.kind === 'dim') {
      var m = text.replace(/\s+/g, ' ').match(/^\(?\s*(\d+)\s*(?:x|×|\*|by|,)\s*(\d+)\s*\)?$/i);
      return !!m && +m[1] === q.answer[0] && +m[2] === q.answer[1];
    }
    if (q.kind === 'num') {
      var pct = /%\s*$/.test(text);
      var v = parseFloat(text.replace(/[%,\s]/g, ''));
      if (/^\s*-?\d+\s*\/\s*\d+\s*$/.test(text)) { var f = text.split('/'); v = +f[0] / +f[1]; }
      if (!isFinite(v)) return false;
      if (q.percentOK && (pct || (v > 1.0001 && v <= 100))) v /= 100;
      var tol = q.tol === undefined ? 1e-6 : q.tol;
      return Math.abs(v - q.answer) <= Math.max(tol, 1e-9) + 1e-9 * Math.abs(q.answer);
    }
    return false;
  }

  var API = { TOPICS: TOPICS, make: make, grade: grade, tx: tx, td: td, fmt: fmt };
  if (typeof module !== 'undefined' && module.exports) module.exports = API;
  else global.PCAPractice = API;
})(typeof window !== 'undefined' ? window : globalThis);
