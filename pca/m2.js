/* Module 2 — Covariance vs. correlation (§20.1.3.1). */
(function () {
  'use strict';
  var P = window.PCA, U = window.UI, $ = U.$;

  UI.module('m2', function () {
    var base = P.gaussian2D(3, 120, [172, 72], 9, 13, 0.6);
    var view = 'centered';
    var s1 = $('#m2s1'), s2 = $('#m2s2'), eq = $('#m2eq');
    var U1 = [[-2, 'm'], [0, 'cm'], [1, 'mm'], [-0.40483, 'in']], U2 = [[0, 'kg'], [3, 'g'], [0.34333, 'lb'], [-3, 't']];
    function unit(v, tab, baseName) {
      for (var i = 0; i < tab.length; i++) if (Math.abs(v - tab[i][0]) < 0.006) return tab[i][1];
      return baseName + ' × ' + fmtScale(Math.pow(10, v));
    }
    function fmtScale(c) { return c >= 100 || c < 0.01 ? c.toExponential(1).replace('e+', 'e') : (+c.toPrecision(3)).toString(); }

    function eig2(S) {
      var a = S[0][0], b = S[0][1], d = S[1][1], m = (a + d) / 2, r = Math.sqrt(((a - d) / 2) * ((a - d) / 2) + b * b);
      return { a: a, b: b, d: d, m: m, r: r, e: P.eigSym(S) };
    }
    function eigHtml(S, name) {
      var q = eig2(S), e = q.e, v1 = [e.vectors[0][0], e.vectors[1][0]], v2 = [e.vectors[0][1], e.vectors[1][1]];
      var frac = e.values[0] / (e.values[0] + e.values[1]);
      return U.td(name + ' = ' + U.texMat(S, S[0][0] > 1e4 ? 1 : 4)) +
        U.td('\\lambda_{1,2} = \\tfrac{a+d}{2} \\pm \\sqrt{\\big(\\tfrac{a-d}{2}\\big)^2 + b^2} = ' + U.fmt(q.m, 4) + ' \\pm ' + U.fmt(q.r, 4)) +
        U.td('\\lambda_1 = ' + U.fmt(e.values[0], 4) + ',\\ u_1 = ' + U.texVec(v1, 3) + '\\qquad \\lambda_2 = ' + U.fmt(e.values[1], 4) + ',\\ u_2 = ' + U.texVec(v2, 3)) +
        '<p class="note">PC1 explains <b>' + (100 * frac).toFixed(1) + '%</b> of the variance. Loadings on (height, weight): (' + v1[0].toFixed(3) + ', ' + v1[1].toFixed(3) + ').</p>';
    }

    function draw() {
      var c1 = Math.pow(10, +s1.value), c2 = Math.pow(10, +s2.value);
      var X = base.map(function (p) { return [p[0] * c1, p[1] * c2]; });
      var cen = P.center(X), S = P.cov(X), R = P.corr(X), sd = P.colStds(X);
      var eS = P.eigSym(S), eR = P.eigSym(R);
      var n1 = unit(+s1.value, U1, 'cm'), n2 = unit(+s2.value, U2, 'kg');
      $('#m2s1v').textContent = fmtScale(c1) + '  (' + n1 + ')';
      $('#m2s2v').textContent = fmtScale(c2) + '  (' + n2 + ')';

      var D, origin, e, ghost, ghostLabel;
      if (view === 'std') {
        D = P.standardize(X).Z; origin = [0, 0]; e = eR;
        var g = [eS.vectors[0][0] / sd[0], eS.vectors[1][0] / sd[1]], ng = Math.hypot(g[0], g[1]); ghost = [g[0] / ng, g[1] / ng]; ghostLabel = 'covariance PC1';
      } else {
        D = view === 'raw' ? X : cen.Xc; origin = view === 'raw' ? cen.mean : [0, 0]; e = eS;
        var h = [eR.vectors[0][0] * sd[0], eR.vectors[1][0] * sd[1]], nh = Math.hypot(h[0], h[1]); ghost = [h[0] / nh, h[1] / nh]; ghostLabel = 'correlation PC1';
      }
      var xs = D.map(function (p) { return p[0]; }), ys = D.map(function (p) { return p[1]; });
      var xr = [Math.min.apply(null, xs), Math.max.apply(null, xs)], yr = [Math.min.apply(null, ys), Math.max.apply(null, ys)];
      if (view === 'raw') { xr[0] = Math.min(xr[0], 0); yr[0] = Math.min(yr[0], 0); xr[1] = Math.max(xr[1], 0); yr[1] = Math.max(yr[1], 0); }
      var px = (xr[1] - xr[0]) * 0.1 || 1, py = (yr[1] - yr[0]) * 0.1 || 1;
      var unitLab = view === 'std' ? ['height (standardized)', 'weight (standardized)'] : ['height (' + n1 + ')', 'weight (' + n2 + ')'];
      var svg = $('#m2plot');
      var f = U.frame(svg, { W: 480, H: 400, x: [xr[0] - px, xr[1] + px], y: [yr[0] - py, yr[1] + py], equal: eq.checked, xlabel: unitLab[0], ylabel: unitLab[1], m: { l: 54, r: 12, t: 12, b: 38 }, nx: 5 });
      if (view !== 'raw') {
        U.el('line', { x1: f.sx(0), x2: f.sx(0), y1: f.m.t, y2: f.m.t + f.ph, 'class': 'grid' }, f.g);
        U.el('line', { x1: f.m.l, x2: f.m.l + f.pw, y1: f.sy(0), y2: f.sy(0), 'class': 'grid' }, f.g);
      }
      D.forEach(function (p) { U.el('circle', { cx: f.sx(p[0]), cy: f.sy(p[1]), r: 3.2, fill: 'var(--ink-2)', 'fill-opacity': 0.55 }, f.g); });
      if (view === 'raw') {
        /* uncentered "PCA": top eigenvector of (1/N) XᵀX on raw X points at the mean */
        var M2 = P.scale(P.mul(P.T(X), X), 1 / X.length), un = P.eigSym(M2).vectors, uu = [un[0][0], un[1][0]];
        var far = 3 * Math.hypot(cen.mean[0], cen.mean[1]);
        U.el('line', { x1: f.sx(0), y1: f.sy(0), x2: f.sx(uu[0] * far), y2: f.sy(uu[1] * far), 'class': 'faint' }, f.g);
        U.el('text', { x: f.sx(0) + 6, y: f.sy(0) - 6, 'class': 'plab' }, f.top).textContent = 'origin';
        var lab = [cen.mean[0] * 0.45, cen.mean[1] * 0.45];
        U.el('text', { x: f.sx(lab[0]) + 8, y: f.sy(lab[1]), 'class': 'plab', fill: 'var(--muted)', 'data-tip': 'Top eigenvector of (1/N)XᵀX without centering. It points at the mean, not along the spread.' }, f.top).textContent = 'uncentered PC1';
      }
      var spanX = (f.x1 - f.x0), Lg = spanX * 0.6;
      U.el('line', { x1: f.sx(origin[0] - Lg * ghost[0]), y1: f.sy(origin[1] - Lg * ghost[1]), x2: f.sx(origin[0] + Lg * ghost[0]), y2: f.sy(origin[1] + Lg * ghost[1]), 'class': 'ghost', 'data-tip': ghostLabel + ', drawn in these coordinates' }, f.g);
      [0, 1].forEach(function (k) {
        var v = [e.vectors[0][k], e.vectors[1][k]], len = 2 * Math.sqrt(Math.max(e.values[k], 0));
        U.el('line', { x1: f.sx(origin[0]), y1: f.sy(origin[1]), x2: f.sx(origin[0] + len * v[0]), y2: f.sy(origin[1] + len * v[1]), 'class': k ? 'arrow2' : 'arrow1', 'stroke-linecap': 'round' }, f.top);
        U.el('text', { x: f.sx(origin[0] + len * v[0]) + 6, y: f.sy(origin[1] + len * v[1]) - 6, 'class': 'plab ' + (k ? 'pc2' : 'pc1') }, f.top).textContent = 'PC' + (k + 1);
      });

      U.html($('#m2cov'), eigHtml(S, '\\hat\\Sigma'));
      U.html($('#m2cor'), eigHtml(R, 'R') + '<p class="note">For 2&times;2 the eigenvalues are ' + U.tx('1\\pm\\rho = 1 \\pm ' + U.fx(R[0][1], 3)) + ' and the eigenvectors are always ' + U.tx('\\tfrac1{\\sqrt2}(1,\\pm1)') + '.</p>');

      var vS = [eS.vectors[0][0], eS.vectors[1][0]], vR = [eR.vectors[0][0], eR.vectors[1][0]];
      var fS = eS.values[0] / (eS.values[0] + eS.values[1]), fR = eR.values[0] / (eR.values[0] + eR.values[1]);
      var dom = Math.abs(vS[0]) > 0.98 ? 'height' : Math.abs(vS[1]) > 0.98 ? 'weight' : null;
      U.html($('#m2cmp'),
        '<table class="data"><tr><th>PC1 from</th><th>height loading</th><th>weight loading</th><th>variance explained</th></tr>' +
        '<tr><td>covariance ' + U.tx('\\hat\\Sigma') + '</td><td>' + vS[0].toFixed(3) + '</td><td>' + vS[1].toFixed(3) + '</td><td>' + (100 * fS).toFixed(1) + '%</td></tr>' +
        '<tr><td>correlation ' + U.tx('R') + '</td><td>' + vR[0].toFixed(3) + '</td><td>' + vR[1].toFixed(3) + '</td><td>' + (100 * fR).toFixed(1) + '%</td></tr></table>' +
        '<div class="callout' + (dom ? '' : ' good') + '">' + (dom
          ? 'In these units the covariance PC1 is essentially the <b>' + dom + '</b> axis: its numbers are so much larger that its variance swamps everything. The correlation PC1 hasn&rsquo;t moved.'
          : 'Height variance ' + U.tx('=' + U.fmt(S[0][0], 3)) + ', weight variance ' + U.tx('=' + U.fmt(S[1][1], 3)) + '. Change the units and watch the covariance loadings move; the correlation row never changes.') + '</div>');
    }

    $('#m2view').addEventListener('click', function (ev) {
      var b = ev.target.closest('button'); if (!b) return; view = b.dataset.v;
      U.$$('#m2view button').forEach(function (x) { x.setAttribute('aria-pressed', x === b ? 'true' : 'false'); });
      draw();
    });
    U.$$('[data-s1]').forEach(function (b) { b.addEventListener('click', function () { s1.value = b.dataset.s1; draw(); }); });
    U.$$('[data-s2]').forEach(function (b) { b.addEventListener('click', function () { s2.value = b.dataset.s2; draw(); }); });
    [s1, s2, eq].forEach(function (x) { x.addEventListener('input', draw); });
    draw();
  });
})();
