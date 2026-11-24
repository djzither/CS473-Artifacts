/* Module 3 — High-dimensional data and the Gram matrix (§20.1.3.2, Eq. 20.22). */
(function () {
  'use strict';
  var P = window.PCA, U = window.UI, $ = U.$;

  UI.module('m3', function () { sizes(); example(); });

  function sizes() {
    var sN = $('#m3N'), sD = $('#m3D');
    function val(s) { return Math.round(Math.pow(10, +s.value)); }
    function draw() {
      var N = Math.max(1, val(sN)), D = Math.max(1, val(sD));
      $('#m3Nv').textContent = U.big(N); $('#m3Dv').textContent = U.big(D);
      var svg = $('#m3shapes'); U.clear(svg);
      var k = function (n) { return 22 + 30 * Math.log10(n); };   /* log-scale side length */
      var hN = k(N), wD = k(D), y0 = 40, gap = 34, x = 14;
      function box(x, w, h, cls, name, dims, entries) {
        U.el('rect', { x: x, y: y0, width: w, height: h, rx: 4, fill: cls === 1 ? 'var(--pc1-soft)' : cls === 2 ? 'var(--pc2-soft)' : 'var(--sunken)', stroke: cls === 1 ? 'var(--pc1)' : cls === 2 ? 'var(--pc2)' : 'var(--ink-2)', 'stroke-width': 1.5, 'data-tip': name + ': ' + dims + '<br>' + U.big(entries) + ' entries' }, svg);
        U.el('text', { x: x + w / 2, y: y0 - 22, 'text-anchor': 'middle', 'class': 'plab' }, svg).textContent = name;
        U.el('text', { x: x + w / 2, y: y0 - 8, 'text-anchor': 'middle', 'class': 'tick' }, svg).textContent = dims;
      }
      box(x, wD, hN, 0, 'X', U.big(N) + ' × ' + U.big(D), N * D); x += wD + gap;
      box(x, wD, wD, 1, 'XᵀX', U.big(D) + ' × ' + U.big(D), D * D); x += wD + gap;
      box(x, hN, hN, 2, 'XXᵀ (Gram)', U.big(N) + ' × ' + U.big(N), N * N);
      var W = x + hN + 14, H = Math.max(hN, wD) + y0 + 14;
      svg.setAttribute('viewBox', '0 0 ' + Math.max(W, 300) + ' ' + H);

      var rows = [
        ['entries', U.big(N * D), U.big(D * D), U.big(N * N)],
        ['memory (float64)', U.bytes(8 * N * D), U.bytes(8 * D * D), U.bytes(8 * N * N)],
        ['cost to form', '—', U.tx('ND^2') + ' ≈ ' + U.sci(N * D * D), U.tx('N^2D') + ' ≈ ' + U.sci(N * N * D)],
        ['cost to eigendecompose', '—', U.tx('D^3') + ' ≈ ' + U.sci(D * D * D), U.tx('N^3') + ' ≈ ' + U.sci(N * N * N)],
        ['nonzero eigenvalues (centered)', '—', '≤ ' + U.big(Math.min(N - 1, D)), '≤ ' + U.big(Math.min(N - 1, D))]
      ];
      var h = '<div class="tablewrap"><table class="data wrap"><tr><th></th><th>' + U.tx('X') + '</th><th>' + U.tx('X^\\top X') + '</th><th>' + U.tx('XX^\\top') + '</th></tr>';
      rows.forEach(function (r) { h += '<tr><td>' + r[0] + '</td><td>' + r[1] + '</td><td>' + r[2] + '</td><td>' + r[3] + '</td></tr>'; });
      U.html($('#m3table'), h + '</table></div>');
      var ratio = Math.pow(Math.max(N, D) / Math.min(N, D), 3);
      U.html($('#m3rec'), D > N
        ? '<b>Use the Gram matrix.</b> ' + U.tx('D > N') + ', so ' + U.tx('XX^\\top') + ' (' + U.big(N) + '×' + U.big(N) + ') is the small one: about <b>' + U.sci(ratio) + '×</b> cheaper to eigendecompose. Recover unit eigenvectors with Eq. 20.22, ' + U.tx('V = X^\\top U\\Lambda^{-1/2}') + '. Images and genomics data look like this.'
        : D < N ? '<b>Use the covariance matrix.</b> ' + U.tx('D < N') + ', so ' + U.tx('X^\\top X') + ' (' + U.big(D) + '×' + U.big(D) + ') is the small one: about <b>' + U.sci(ratio) + '×</b> cheaper. Most tabular data looks like this.'
        : '<b>Either.</b> With ' + U.tx('N = D') + ' both matrices are the same size.');
    }
    [sN, sD].forEach(function (s) { s.addEventListener('input', draw); });
    U.$$('[data-nd]').forEach(function (b) { b.addEventListener('click', function () { var v = b.dataset.nd.split(','); sN.value = Math.log10(+v[0]); sD.value = Math.log10(+v[1]); draw(); }); });
    draw();
  }

  function example() {
    var seed = 12, shown = 1, ed;
    function rand() { var r = P.rng(seed), M = []; for (var i = 0; i < 4; i++) { var row = []; for (var j = 0; j < 6; j++) row.push(Math.floor(r() * 7)); M.push(row); } return M; }
    function draw(X) {
      var N = 4, c = P.center(X), Xc = c.Xc, g = P.gramTrick(Xc), lam = g.values, k = g.keep.length;
      var big = P.eigSym(P.mul(P.T(Xc), Xc)), XtX = P.mul(P.T(Xc), Xc);
      var lamTex = lam.map(function (v) { return U.fx(v, 3); }).join(',\\ ');
      var st = [];
      st.push(['Center the columns', 'Subtract each column mean ' + U.tx('\\bar x = ' + U.texMat([c.mean], 2)) + '. Every PCA formula assumes centered data. The four centered rows now sum to zero, so they span at most 3 dimensions.',
        U.matrix(Xc, { name: 'X_c', digits: 2 })]);
      st.push(['Form the small Gram matrix', U.tx('G = X_cX_c^\\top') + ' is ' + U.tx('4\\times4') + '. Entry ' + U.tx('(n,m)') + ' is the inner product ' + U.tx('x_n^\\top x_m') + ' between two observations. (' + U.tx('X_c^\\top X_c') + ' would be ' + U.tx('6\\times6') + '.)',
        U.matrix(g.G, { name: 'G = X_cX_c^\\top', digits: 2 })]);
      st.push(['Eigendecompose the Gram matrix', U.tx('GU = U\\Lambda') + ' with ' + U.tx('\\lambda = (' + lamTex + ')') + '. The last eigenvalue is 0 (up to round-off) because centering made the rows linearly dependent: ' + U.tx('\\mathbf 1^\\top X_c = 0') + ', so ' + U.tx('\\tfrac12(1,1,1,1)') + ' is always a zero-eigenvalue eigenvector of ' + U.tx('G') + '.',
        U.matrix(g.U, { name: 'U', digits: 3, hlCols: k }) + U.matrix(P.diag(lam), { name: '\\Lambda', digits: 3, diagOnly: true })]);
      var norms = g.keep.map(function (j) { return P.norm(P.col(g.XtU, j)) ** 2; });
      st.push(['Map each u back to feature space with Xᵀ', 'Multiply ' + U.tx('X_c^\\top U') + ' (' + U.tx('6\\times4') + '). Each column is an eigenvector of ' + U.tx('X_c^\\top X_c') + ' (Eq. 20.21), but its squared length is ' + U.tx('\\lVert X_c^\\top u_j\\rVert^2 = u_j^\\top G u_j = \\lambda_j') + ', not 1. Squared column lengths here: ' + U.tx('(' + norms.map(function (v) { return U.fx(v, 3); }).join(',\\ ') + ')') + ', matching ' + U.tx('\\lambda_1,\\dots,\\lambda_' + k) + '.',
        U.matrix(g.XtU, { name: 'X_c^\\top U', digits: 3, hlCols: k })]);
      st.push(['Normalize: V = XᵀUΛ^{-1/2} (Eq. 20.22)', 'Divide column ' + U.tx('j') + ' by ' + U.tx('\\sqrt{\\lambda_j}') + ' for the ' + k + ' nonzero eigenvalues: ' + g.keep.map(function (j) { return U.tx('\\sqrt{' + U.fx(lam[j], 3) + '} = ' + U.fx(Math.sqrt(lam[j]), 3)); }).join(', ') + '. ' + (4 - k === 1 ? 'The zero-eigenvalue column cannot be normalized and carries no variance, so it is dropped.' : 'The ' + (4 - k) + ' zero-eigenvalue columns cannot be normalized and carry no variance, so they are dropped.') + '',
        U.matrix(g.V, { name: 'V = X_c^\\top U\\Lambda^{-1/2}', digits: 3 })]);
      var checks = g.keep.map(function (j, cc) {
        var v = P.col(g.V, cc), Av = XtX.map(function (r) { return P.dot(r, v); });
        var err = Math.max.apply(null, Av.map(function (a, i) { return Math.abs(a - lam[j] * v[i]); }));
        return U.tx('\\lVert X_c^\\top X_c v_' + (cc + 1) + ' - ' + U.fx(lam[j], 3) + ' v_' + (cc + 1) + '\\rVert_\\infty = ' + (err < 1e-9 ? '0' : err.toExponential(1))) + ' <span class="okmark">✓</span>';
      });
      st.push(['Check against the big matrix', 'Eigendecomposing the ' + U.tx('6\\times6') + ' matrix ' + U.tx('X_c^\\top X_c') + ' directly gives ' + U.tx('(' + big.values.map(function (v) { return U.fx(v, 3); }).join(',\\ ') + ')') + ': the same nonzero eigenvalues plus ' + (6 - k) + ' extra zeros. Each recovered ' + U.tx('v_j') + ' satisfies the eigen-equation: ' + checks.join('; ') + '. Unit length: ' + g.keep.map(function (_, cc) { return U.tx('\\lVert v_' + (cc + 1) + '\\rVert = ' + U.fx(P.norm(P.col(g.V, cc)), 3)); }).join(', ') + '.',
        U.matrix(XtX, { name: 'X_c^\\top X_c', digits: 2 })]);
      st.push(['Convert to covariance eigenvalues', 'Murphy&rsquo;s ' + U.tx('\\hat\\Sigma = \\tfrac1N X_c^\\top X_c') + ' has the same eigenvectors and eigenvalues divided by ' + U.tx('N = 4') + ': ' + U.tx('(' + g.keep.map(function (j) { return U.fx(lam[j] / N, 3); }).join(',\\ ') + ')') + '. The principal directions are the columns of ' + U.tx('V') + ': ' + U.tx('W = V_L') + '.', '']);
      var h = '';
      st.forEach(function (s, i) { h += '<li' + (i < shown ? '' : ' hidden') + '><h4>' + s[0] + '</h4><p>' + s[1] + '</p>' + (s[2] ? '<div class="matrow">' + s[2] + '</div>' : '') + '</li>'; });
      U.html($('#m3steps'), h);
      $('#m3next').disabled = shown >= st.length;
    }
    ed = U.editor($('#m3edit'), rand(), draw);
    draw(ed.get());
    $('#m3gen').addEventListener('click', function () { seed++; ed.set(rand()); });
    $('#m3next').addEventListener('click', function () { shown++; draw(ed.get()); });
    $('#m3all').addEventListener('click', function () { shown = 99; draw(ed.get()); });
  }
})();
