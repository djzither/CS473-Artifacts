/* Module 4 — PCA using SVD (§20.1.3.3). */
(function () {
  'use strict';
  var P = window.PCA, U = window.UI, $ = U.$;
  var EXAMPLE = [[2, 0, 1], [1, 3, 2], [4, 1, 0], [0, 2, 5], [3, 3, 3]];

  UI.module('m4', function () {
    var seed = 40, ed, Ls = $('#m4L'), sN = $('#m4N'), sD = $('#m4D');
    function randInts(N, D) { var r = P.rng(seed++), M = []; for (var i = 0; i < N; i++) { var row = []; for (var j = 0; j < D; j++) row.push(Math.floor(r() * 10)); M.push(row); } return M; }
    function rank1(N, D) {
      var r = P.rng(seed++), b = []; for (var j = 0; j < D; j++) b.push(1 + Math.floor(r() * 4));
      return Array.from({ length: N }, function () { var t = P.gauss(r) * 1.5; return b.map(function (bj) { return +(t * bj + 0.3 * P.gauss(r)).toFixed(1); }); });
    }

    function draw(X) {
      var N = X.length, D = X[0].length, c = P.center(X), Xc = c.Xc, sv = P.svd(Xc), k = sv.s.length;
      var rank = sv.s.filter(function (v) { return v > 1e-9 * (sv.s[0] || 1); }).length;
      Ls.max = k; if (+Ls.value > k) Ls.value = k;
      var L = +Ls.value;
      $('#m4Lv').textContent = L;
      $('#m4Lnote').textContent = '(of k = ' + k + '; rank of X_c = ' + rank + (rank < k ? ', so s_' + k + ' = 0' : '') + ')';

      U.html($('#m4mean'), U.matrix([c.mean], { name: '\\bar x^\\top', digits: 3, note: 'column means' }));
      U.html($('#m4xc'), U.matrix(Xc, { name: 'X_c = X - \\mathbf 1\\bar x^\\top', digits: 3, note: 'centered' }));

      var S = P.diag(sv.s), Vt = P.T(sv.V);
      U.html($('#m4svd'),
        U.matrix(Xc, { name: 'X_c', digits: 2 }) + U.op('=') +
        U.matrix(sv.U, { name: 'U', digits: 3, hlCols: L }) + U.op('×') +
        U.matrix(S, { name: 'S', digits: 3, diagHL: L }) + U.op('×') +
        U.matrix(Vt, { name: 'V^\\top', digits: 3, hlRows: L }) +
        '<span class="mop">&nbsp;</span>' +
        U.matrix(sv.V, { name: 'V', digits: 3, hlCols: L, note: 'columns = principal directions' }));

      var WL = P.cols(sv.V, L), UL = P.cols(sv.U, L), SL = P.diag(sv.s.slice(0, L));
      var Z = P.mul(UL, SL), Z2 = P.mul(Xc, WL), Xhc = P.mul(Z, P.T(WL));
      var Xh = Xhc.map(function (r) { return r.map(function (v, j) { return v + c.mean[j]; }); });
      var Res = P.sub(Xc, Xhc);
      U.html($('#m4pca'),
        U.matrix(WL, { name: 'W = V_L', digits: 3 }) +
        U.matrix(Z, { name: 'Z = U_LS_L', digits: 3 }) +
        U.matrix(Xhc, { name: '\\hat X_c = ZW^\\top', digits: 3 }) +
        U.matrix(Xh, { name: '\\hat X = \\hat X_c + \\mathbf 1\\bar x^\\top', digits: 3 }) +
        U.matrix(Res, { name: 'X_c - \\hat X_c', digits: 3, note: 'residual' }));

      var dz = 0; Z.forEach(function (r, i) { r.forEach(function (v, j) { dz = Math.max(dz, Math.abs(v - Z2[i][j])); }); });
      var fro = P.frob2(Res), tail = sv.s.slice(L), tail2 = tail.map(function (v) { return v * v; });
      U.html($('#m4check'),
        '<p>' + U.tx('Z = U_LS_L') + ' equals ' + U.tx('X_cW') + ' (largest difference ' + (dz < 1e-9 ? '0' : dz.toExponential(1)) + ') <span class="okmark">✓</span>. Each row of ' + U.tx('Z') + ' is one observation&rsquo;s ' + L + '-number code.</p>' +
        U.td('\\lVert X_c-\\hat X_c\\rVert_F^2 = ' + U.fx(fro, 4) + (tail.length ? ' = \\sum_{j>' + L + '} s_j^2 = ' + tail2.map(function (v) { return U.fx(v, 3); }).join(' + ') : '') ) +
        U.td('\\tfrac1N\\lVert X_c-\\hat X_c\\rVert_F^2 = \\tfrac{' + U.fx(fro, 3) + '}{' + N + '} = ' + U.fx(fro / N, 4) + ' = \\textstyle\\sum_{j>' + L + '}\\lambda_j\\quad\\text{(Eq. 20.26)}'));

      var eC = P.eigSym(P.cov(X)).values, tot = P.sum(sv.s.map(function (v) { return v * v; }));
      var h = '<div class="tablewrap"><table class="data"><tr><th>j</th><th>s<sub>j</sub></th><th>s<sub>j</sub>²</th><th>s<sub>j</sub>²/N</th><th>eig of ' + U.tx('\\hat\\Sigma') + ' (direct)</th><th>match</th><th>F(j)</th></tr>', cum = 0;
      sv.s.forEach(function (s, j) {
        cum += s * s;
        var lam = s * s / N, ok = Math.abs(lam - eC[j]) < 1e-8 * Math.max(1, eC[0]);
        h += '<tr class="' + (j < L ? 'sel' : '') + '"><td>' + (j + 1) + (j < L ? ' · kept' : '') + '</td><td>' + U.fx(s, 4) + '</td><td>' + U.fx(s * s, 4) + '</td><td>' + U.fx(lam, 4) + '</td><td>' + U.fx(eC[j], 4) + '</td><td class="okmark">' + (ok ? '✓' : '✗') + '</td><td>' + (tot > 0 ? (100 * cum / tot).toFixed(1) + '%' : '—') + '</td></tr>';
      });
      for (var j = k; j < D; j++) h += '<tr><td>' + (j + 1) + '</td><td colspan="3" class="muted">no singular value (N &lt; D)</td><td>' + U.fx(eC[j], 4) + '</td><td></td><td></td></tr>';
      U.html($('#m4eig'), h + '</table></div><p class="note">The &ldquo;direct&rdquo; column eigendecomposes ' + U.tx('\\hat\\Sigma=\\tfrac1N X_c^\\top X_c') + ' independently. It agrees with ' + U.tx('s_j^2/N') + ' every time. Explained fractions use ' + U.tx('s_j^2') + ', not ' + U.tx('s_j') + '.</p>');
    }

    function rebuild(M) { ed = U.editor($('#m4edit'), M, draw); draw(ed.get()); }
    Ls.addEventListener('input', function () { draw(ed.get()); });
    $('#m4gen').addEventListener('click', function () { rebuild(randInts(+sN.value, +sD.value)); });
    $('#m4corr').addEventListener('click', function () { rebuild(rank1(+sN.value, +sD.value)); });
    $('#m4ex').addEventListener('click', function () { sN.value = 5; sD.value = 3; rebuild(EXAMPLE); });
    [sN, sD].forEach(function (s) { s.addEventListener('change', function () { rebuild(randInts(+sN.value, +sD.value)); }); });
    rebuild(EXAMPLE);
  });
})();
