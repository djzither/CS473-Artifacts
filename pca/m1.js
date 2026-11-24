/* Module 1 — Understanding PCA (§20.1.1). */
(function () {
  'use strict';
  var P = window.PCA, U = window.UI, $ = U.$;

  UI.module('m1', function () {
    projection();
    imageDemo();
    setTimeout(digitDemo, 30);
  });

  /* ------------------------------------------------------------ projection */
  function projection() {
    var svg = $('#m1plot'), curve = $('#m1curve');
    var th = $('#m1th'), rho = $('#m1rho'), seed = 4;
    var pts, theta = 20;
    var DOM = { x: [-6, 6], y: [-5, 5] };

    function gen() { pts = P.gaussian2D(seed, 36, [0, 0], 2.1, 1.4, +rho.value); }
    function stats() {
      var c = P.center(pts), S = P.cov(pts), e = P.eigSym(S);
      return { mean: c.mean, Xc: c.Xc, S: S, e: e, tr: S[0][0] + S[1][1] };
    }
    function w() { var a = theta * Math.PI / 180; return [Math.cos(a), Math.sin(a)]; }
    function angOf(v) { var a = Math.atan2(v[1], v[0]) * 180 / Math.PI; a = ((a % 180) + 180) % 180; return a; }

    function draw() {
      var s = stats(), wv = w(), N = pts.length;
      var f = U.frame(svg, { W: 480, H: 420, x: DOM.x, y: DOM.y, equal: true, xlabel: 'feature 1', ylabel: 'feature 2', m: { l: 40, r: 10, t: 10, b: 36 } });
      var g = f.g, m = s.mean, R = 20;
      /* PC1 reference */
      var u1 = [s.e.vectors[0][0], s.e.vectors[1][0]];
      U.el('line', { x1: f.sx(m[0] - R * u1[0]), y1: f.sy(m[1] - R * u1[1]), x2: f.sx(m[0] + R * u1[0]), y2: f.sy(m[1] + R * u1[1]), 'class': 'faint' }, g);
      /* projection line */
      U.el('line', { x1: f.sx(m[0] - R * wv[0]), y1: f.sy(m[1] - R * wv[1]), x2: f.sx(m[0] + R * wv[0]), y2: f.sy(m[1] + R * wv[1]), 'class': 'wline' }, g);
      var pv = 0, err = 0;
      pts.forEach(function (p, i) {
        var xc = s.Xc[i], z = xc[0] * wv[0] + xc[1] * wv[1];
        var q = [m[0] + z * wv[0], m[1] + z * wv[1]];
        pv += z * z / N; err += ((xc[0] - z * wv[0]) ** 2 + (xc[1] - z * wv[1]) ** 2) / N;
        U.el('line', { x1: f.sx(p[0]), y1: f.sy(p[1]), x2: f.sx(q[0]), y2: f.sy(q[1]), 'class': 'resid' }, g);
        U.el('circle', { cx: f.sx(q[0]), cy: f.sy(q[1]), r: 2.6, 'class': 'pdot' }, g);
      });
      pts.forEach(function (p, i) {
        U.el('circle', { cx: f.sx(p[0]), cy: f.sy(p[1]), r: 5, 'class': 'dot', 'data-i': i,
          'data-tip': 'x = (' + U.fx(p[0], 2) + ', ' + U.fx(p[1], 2) + ')<br>score z = wᵀ(x − x̄) = ' + U.fx(s.Xc[i][0] * wv[0] + s.Xc[i][1] * wv[1], 2) }, g);
      });
      /* arrow + handle */
      var A = 3.2, tip = [m[0] + A * wv[0], m[1] + A * wv[1]];
      U.el('line', { x1: f.sx(m[0]), y1: f.sy(m[1]), x2: f.sx(tip[0]), y2: f.sy(tip[1]), 'class': 'arrow1' }, f.top);
      U.el('circle', { cx: f.sx(m[0]), cy: f.sy(m[1]), r: 4, fill: 'var(--ink)' }, f.top);
      U.el('circle', { cx: f.sx(tip[0]), cy: f.sy(tip[1]), r: 9, 'class': 'handle', 'data-handle': 1 }, f.top);
      U.el('text', { x: f.sx(tip[0]) + 12, y: f.sy(tip[1]) - 8, 'class': 'plab pc1' }, f.top).textContent = 'w';
      var lp = [m[0] + 5.2 * u1[0], m[1] + 5.2 * u1[1]];
      U.el('text', { x: f.sx(lp[0]), y: f.sy(lp[1]), 'class': 'plab', 'text-anchor': 'middle', fill: 'var(--muted)' }, f.top).textContent = 'PC1';
      svg._f = f;

      var lam1 = s.e.values[0], dAng = Math.abs(theta - angOf(u1)); dAng = Math.min(dAng, 180 - dAng);
      th.value = theta; $('#m1thv').textContent = theta.toFixed(1) + '°';
      $('#m1rhov').textContent = (+rho.value).toFixed(2);
      U.html($('#m1calc'),
        U.td('\\hat\\Sigma = ' + U.texMat(s.S, 3) + ',\\quad w = ' + U.texVec(wv, 3)) +
        U.td('\\underbrace{' + U.fx(s.tr, 3) + '}_{\\operatorname{tr}\\hat\\Sigma} = \\underbrace{' + U.fx(pv, 3) + '}_{w^\\top\\hat\\Sigma w} + \\underbrace{' + U.fx(err, 3) + '}_{\\frac1N\\sum\\lVert x_n-\\hat x_n\\rVert^2}') +
        '<p>Here ' + U.tx('w^\\top\\hat\\Sigma w = ' + U.fx(s.S[0][0], 3) + '\\cos^2\\theta + 2(' + U.fx(s.S[0][1], 3) + ')\\cos\\theta\\sin\\theta + ' + U.fx(s.S[1][1], 3) + '\\sin^2\\theta') + '.</p>');
      U.html($('#m1stats'),
        U.stat('projected variance', U.fx(pv, 3), 'c1') + U.stat('reconstruction error', U.fx(err, 3), 'c2') +
        U.stat('best possible = λ₁', U.fx(lam1, 3)) + U.stat('angle from PC1', dAng.toFixed(1) + '°'));
      drawCurve(s, angOf(u1));
    }

    function drawCurve(s, a1) {
      var tr = s.tr, f = U.frame(curve, { W: 480, H: 230, x: [0, 180], y: [0, tr * 1.08], xt: [0, 30, 60, 90, 120, 150, 180], xfmt: function (v) { return v + '°'; }, xlabel: 'angle of w', ylabel: 'variance', m: { l: 46, r: 12, t: 12, b: 36 } });
      var d1 = '', d2 = '';
      for (var a = 0; a <= 180; a += 1) {
        var c = Math.cos(a * Math.PI / 180), sn = Math.sin(a * Math.PI / 180);
        var pv = s.S[0][0] * c * c + 2 * s.S[0][1] * c * sn + s.S[1][1] * sn * sn;
        d1 += (a ? 'L' : 'M') + f.sx(a).toFixed(1) + ',' + f.sy(pv).toFixed(1);
        d2 += (a ? 'L' : 'M') + f.sx(a).toFixed(1) + ',' + f.sy(tr - pv).toFixed(1);
      }
      U.el('line', { x1: f.sx(0), x2: f.sx(180), y1: f.sy(tr), y2: f.sy(tr), 'class': 'lt' }, f.g);
      U.el('path', { d: d1, 'class': 'l1' }, f.g);
      U.el('path', { d: d2, 'class': 'l2' }, f.g);
      U.el('line', { x1: f.sx(a1), x2: f.sx(a1), y1: f.sy(0), y2: f.sy(tr), 'class': 'faint' }, f.g);
      U.el('text', { x: f.sx(a1), y: f.sy(tr) - 4, 'text-anchor': 'middle', 'class': 'plab' }, f.top).textContent = 'PC1';
      U.el('line', { x1: f.sx(theta), x2: f.sx(theta), y1: f.sy(0), y2: f.sy(tr * 1.08), 'class': 'marker' }, f.top);
      var c0 = Math.cos(theta * Math.PI / 180), s0 = Math.sin(theta * Math.PI / 180);
      var pv0 = s.S[0][0] * c0 * c0 + 2 * s.S[0][1] * c0 * s0 + s.S[1][1] * s0 * s0;
      U.el('circle', { cx: f.sx(theta), cy: f.sy(pv0), r: 5, fill: 'var(--pc1)', stroke: '#fff', 'stroke-width': 2 }, f.top);
      U.el('circle', { cx: f.sx(theta), cy: f.sy(tr - pv0), r: 5, fill: 'var(--pc2)', stroke: '#fff', 'stroke-width': 2 }, f.top);
      U.el('rect', { x: f.m.l, y: f.m.t, width: f.pw, height: f.ph, 'class': 'hit', 'data-tip': 'click or drag to set the angle' }, f.top);
      curve._f = f;
    }

    U.drag(svg, function (ev, pt) {
      var f = svg._f; if (!f) return null;
      var t = ev.target;
      if (t.getAttribute('data-handle')) {
        return function (p) { var m = stats().mean, x = f.ix(p.x) - m[0], y = f.iy(p.y) - m[1]; theta = Math.round(angOfXY(x, y) * 2) / 2; draw(); };
      }
      if (t.getAttribute('data-i') !== null) {
        var i = +t.getAttribute('data-i');
        return function (p) { pts[i] = [Math.max(DOM.x[0], Math.min(DOM.x[1], f.ix(p.x))), Math.max(DOM.y[0] - 1, Math.min(DOM.y[1] + 1, f.iy(p.y)))]; draw(); };
      }
      return null;
    });
    function angOfXY(x, y) { var a = Math.atan2(y, x) * 180 / Math.PI; return ((a % 180) + 180) % 180; }
    U.drag(curve, function (ev, pt) {
      var f = curve._f; if (!f) return null;
      var set = function (p) { theta = Math.max(0, Math.min(180, Math.round(f.ix(p.x) * 2) / 2)); draw(); };
      set(pt); return set;
    });
    th.addEventListener('input', function () { theta = +th.value; draw(); });
    rho.addEventListener('input', function () { gen(); draw(); });
    $('#m1new').addEventListener('click', function () { seed++; gen(); draw(); });
    $('#m1pc1').addEventListener('click', function () { var e = stats().e; theta = angOfXY(e.vectors[0][0], e.vectors[1][0]); draw(); });
    $('#m1pc2').addEventListener('click', function () { var e = stats().e; theta = angOfXY(e.vectors[0][1], e.vectors[1][1]); draw(); });
    gen(); draw();
  }

  /* ------------------------------------------------------------ image compression */
  function imageDemo() {
    var img, fit, slider = $('#m1L');
    function load(I) { img = I; fit = P.pcaFit(img); U.drawGray($('#m1orig'), img); draw(); }
    function draw() {
      var L = +slider.value, N = img.length, D = img[0].length;
      var rec = P.reconstruct(fit, img, L), res = P.sub(img, rec);
      U.drawGray($('#m1rec'), rec); U.drawDiverging($('#m1res'), res, 0.35);
      $('#m1Lv').textContent = L; $('#m1reccap').textContent = 'reconstruction, L = ' + L;
      var mse = P.reconError(img, rec) / D, F = P.explained(fit.lambdas, L);
      var store = N * L + D * L + D, full = N * D;
      U.html($('#m1imgstats'),
        U.stat('variance explained F(L)', (100 * F).toFixed(1) + '%', 'c1') +
        U.stat('mean sq. error / pixel', mse.toFixed(5), 'c2') +
        U.stat('numbers stored', U.big(store) + ' / ' + U.big(full)) +
        U.stat('of the original size', (100 * store / full).toFixed(0) + '%'));
      spec(L);
    }
    function spec(L) {
      var svg = $('#m1spec'), cum = P.cumulative(fit.lambdas), D = cum.length;
      var f = U.frame(svg, { W: 720, H: 150, x: [0, D], y: [0, 1], yt: [0, 0.5, 1], xt: [0, 8, 16, 24, 32, 40, 48, 56, 64].filter(function (v) { return v <= D; }), yfmt: function (v) { return Math.round(v * 100) + '%'; }, xlabel: 'components kept L', m: { l: 46, r: 12, t: 10, b: 34 } });
      var d = 'M' + f.sx(0) + ',' + f.sy(0);
      cum.forEach(function (v, i) { d += 'L' + f.sx(i + 1).toFixed(1) + ',' + f.sy(v).toFixed(1); });
      U.el('path', { d: d, 'class': 'l1' }, f.g);
      U.el('line', { x1: f.sx(L), x2: f.sx(L), y1: f.sy(0), y2: f.sy(1), 'class': 'marker' }, f.top);
      U.el('circle', { cx: f.sx(L), cy: f.sy(L ? cum[L - 1] : 0), r: 5, fill: 'var(--pc1)', stroke: '#fff', 'stroke-width': 2 }, f.top);
      U.el('text', { x: f.sx(L) + 8, y: f.sy(L ? cum[L - 1] : 0) + 16, 'class': 'plab' }, f.top).textContent = 'F(' + L + ') = ' + (100 * (L ? cum[L - 1] : 0)).toFixed(1) + '%';
    }
    slider.addEventListener('input', draw);
    $('#m1reset').addEventListener('click', function () { load(P.testImage(64)); });
    $('#m1file').addEventListener('change', function (ev) {
      var file = ev.target.files && ev.target.files[0]; if (!file) return;
      var url = URL.createObjectURL(file), im = new Image();
      im.onload = function () {
        var c = document.createElement('canvas'); c.width = c.height = 64;
        var ctx = c.getContext('2d'), s = Math.min(im.width, im.height);
        ctx.drawImage(im, (im.width - s) / 2, (im.height - s) / 2, s, s, 0, 0, 64, 64);
        var d = ctx.getImageData(0, 0, 64, 64).data, out = P.zeros(64, 64);
        for (var i = 0; i < 64; i++) for (var j = 0; j < 64; j++) { var k = 4 * (i * 64 + j); out[i][j] = (0.299 * d[k] + 0.587 * d[k + 1] + 0.114 * d[k + 2]) / 255; }
        URL.revokeObjectURL(url); load(out);
      };
      im.src = url;
    });
    load(P.testImage(64));
  }

  /* ------------------------------------------------------------ synthetic digits */
  function digitDemo() {
    var S = 20, N = 150, r = P.rng(31), c = document.createElement('canvas'); c.width = c.height = S;
    var ctx = c.getContext('2d', { willReadFrequently: true });
    var fonts = ['serif', 'sans-serif', 'Georgia', 'Arial', 'Verdana', 'Times New Roman', 'Trebuchet MS', 'Courier New'];
    var X = [];
    for (var n = 0; n < N; n++) {
      ctx.setTransform(1, 0, 0, 1, 0, 0); ctx.fillStyle = '#000'; ctx.fillRect(0, 0, S, S);
      ctx.translate(S / 2 + (r() - 0.5) * 3, S / 2 + (r() - 0.5) * 3);
      ctx.rotate((r() - 0.5) * 0.5); ctx.transform(1, 0, (r() - 0.5) * 0.6, 1, 0, 0);
      var size = 14 + r() * 5;
      ctx.font = (r() < 0.5 ? 'bold ' : '') + size.toFixed(1) + 'px ' + fonts[Math.floor(r() * fonts.length)];
      ctx.textAlign = 'center'; ctx.textBaseline = 'middle'; ctx.fillStyle = '#fff';
      ctx.fillText('3', 0, 1);
      if (r() < 0.4) { ctx.strokeStyle = '#fff'; ctx.lineWidth = 0.8 + r(); ctx.strokeText('3', 0, 1); }
      var d = ctx.getImageData(0, 0, S, S).data, row = [];
      for (var k = 0; k < S * S; k++) row.push(d[4 * k] / 255);
      X.push(row);
    }
    $('#m1dN').textContent = 'N = ' + N;
    var cen = P.center(X), g = P.gramTrick(cen.Xc), V = g.V, lam = g.keep.map(function (j) { return g.values[j] / N; });
    var tot = P.sum(lam);
    var Z = cen.Xc.map(function (x) { return [0, 1].map(function (l) { var s = 0; for (var i = 0; i < x.length; i++) s += x[i] * V[i][l]; return s; }); });
    var toImg = function (v) { var out = []; for (var i = 0; i < S; i++) out.push(v.slice(i * S, (i + 1) * S)); return out; };

    /* mean + first eigen-images */
    var strip = $('#m1eig'), h = '<figure><canvas id="m1mean"></canvas><figcaption>mean x̄</figcaption></figure>';
    for (var l = 0; l < 5; l++) h += '<figure><canvas id="m1e' + l + '"></canvas><figcaption>w' + (l + 1) + ' · ' + (100 * lam[l] / tot).toFixed(1) + '%</figcaption></figure>';
    strip.innerHTML = h;
    U.drawGray($('#m1mean'), toImg(cen.mean));
    for (l = 0; l < 5; l++) {
      var v = P.col(V, l), amp = Math.max.apply(null, v.map(Math.abs));
      U.drawDiverging($('#m1e' + l), toImg(v), amp);
    }
    var sel = 0, sl = $('#m1dL'); sl.max = Math.min(40, V[0].length);
    function rec() {
      var L = +sl.value, x = cen.Xc[sel], z = [];
      for (var q = 0; q < L; q++) { var s = 0; for (var i = 0; i < x.length; i++) s += x[i] * V[i][q]; z.push(s); }
      var xh = cen.mean.map(function (m, i) { var v = m; for (var q2 = 0; q2 < L; q2++) v += z[q2] * V[i][q2]; return v; });
      U.drawGray($('#m1dx'), toImg(X[sel])); U.drawGray($('#m1dxh'), toImg(xh));
      $('#m1dLv').textContent = L;
      $('#m1dxhcap').textContent = 'L = ' + L + ' · F = ' + (100 * P.sum(lam.slice(0, L)) / tot).toFixed(0) + '%';
    }
    function map() {
      var svg = $('#m1map'), xs = Z.map(function (z) { return z[0]; }), ys = Z.map(function (z) { return z[1]; });
      var pad = function (a, b) { var s = (b - a) * 0.08; return [a - s, b + s]; };
      var f = U.frame(svg, { W: 300, H: 260, x: pad(Math.min.apply(null, xs), Math.max.apply(null, xs)), y: pad(Math.min.apply(null, ys), Math.max.apply(null, ys)), xlabel: 'z₁ (score on w₁)', ylabel: 'z₂', nx: 4, ny: 4, m: { l: 40, r: 8, t: 8, b: 34 } });
      Z.forEach(function (z, i) {
        U.el('circle', { cx: f.sx(z[0]), cy: f.sy(z[1]), r: i === sel ? 6 : 4, 'class': 'mapdot' + (i === sel ? ' sel' : ''), 'data-k': i, 'data-tip': 'digit #' + (i + 1) + '<br>z = (' + U.fx(z[0], 2) + ', ' + U.fx(z[1], 2) + ')' }, f.g);
      });
    }
    $('#m1map').addEventListener('pointerover', function (ev) { var k = ev.target.getAttribute && ev.target.getAttribute('data-k'); if (k !== null && k !== undefined && +k !== sel) { sel = +k; map(); rec(); } });
    $('#m1map').addEventListener('click', function (ev) { var k = ev.target.getAttribute && ev.target.getAttribute('data-k'); if (k !== null) { sel = +k; map(); rec(); } });
    sl.addEventListener('input', rec);
    map(); rec();
  }
})();
