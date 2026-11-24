/* Module 5 — Choosing the number of components (§20.1.4, Eqs. 20.26–20.27). */
(function () {
  'use strict';
  var P = window.PCA, U = window.UI, $ = U.$;
  var PRESETS = {
    decay: [8, 5.2, 3.4, 2.2, 1.4, 0.9, 0.6, 0.4, 0.25, 0.15],
    elbow: [9.5, 7.8, 6.1, 1.0, 0.8, 0.7, 0.55, 0.45, 0.35, 0.3],
    flat: [1.4, 1.3, 1.2, 1.15, 1.1, 1.0, 0.95, 0.9, 0.85, 0.8],
    one: [12, 1.2, 0.9, 0.7, 0.5, 0.4, 0.3, 0.2]
  };
  window.M5 = { get: function () { return null; } };

  UI.module('m5', function () {
    var lams = PRESETS.decay.slice(), L = 3, target = 0.95, seed = 5;
    var Ls = $('#m5L'), Ns = $('#m5N'), chips, tt = null, frozenTop = null;
    window.M5.get = function () { return lams.slice(); };

    function setL(v) { L = Math.max(1, Math.min(lams.length, v)); drawAll(false); }

    function drawScree() {
      var svg = $('#m5scree'), m = lams.length, top = frozenTop || Math.max.apply(null, lams) * 1.12 || 1;
      var f = U.frame(svg, { W: 520, H: 300, x: [0.5, m + 0.5], y: [0, top], xt: lams.map(function (_, i) { return i + 1; }), xlabel: 'component j', ylabel: 'eigenvalue λⱼ', m: { l: 46, r: 12, t: 16, b: 38 } });
      var bw = f.pw / m - 4;
      lams.forEach(function (v, i) {
        var x = f.sx(i + 1) - bw / 2, y = f.sy(v), h = Math.max(1, f.sy(0) - y), kept = i < L;
        var r = Math.min(4, bw / 2, h);
        var d = 'M' + x + ',' + f.sy(0) + 'V' + (y + r) + 'Q' + x + ',' + y + ' ' + (x + r) + ',' + y + 'H' + (x + bw - r) + 'Q' + (x + bw) + ',' + y + ' ' + (x + bw) + ',' + (y + r) + 'V' + f.sy(0) + 'Z';
        U.el('path', { d: d, 'class': kept ? 'bar1' : 'bar2', 'data-bar': i, 'data-tip': 'λ' + (i + 1) + ' = ' + U.fmt(v, 4) + '<br>' + (kept ? 'retained' : 'discarded') + ' · ' + (100 * v / P.sum(lams)).toFixed(1) + '% of variance<br>drag to change' }, f.g);
        if (m <= 12) U.el('text', { x: f.sx(i + 1), y: y - 5, 'text-anchor': 'middle', 'class': 'tick' }, f.top).textContent = U.fmt(v, 2);
      });
      var xs = f.sx(L + 0.5);
      U.el('line', { x1: xs, x2: xs, y1: f.m.t, y2: f.sy(0), 'class': 'split' }, f.top);
      U.el('text', { x: xs + 6, y: f.m.t + 10, 'class': 'plab' }, f.top).textContent = 'L = ' + L;
      svg._f = f;
    }

    function drawCalc() {
      var kept = lams.slice(0, L), disc = lams.slice(L), tot = P.sum(lams), F = P.explained(lams, L), E = P.residualFromEigs(lams, L);
      var list = function (a) { return a.length > 6 ? a.slice(0, 5).map(function (v) { return U.fmt(v, 3); }).join('+') + '+\\cdots' : a.map(function (v) { return U.fmt(v, 3); }).join('+'); };
      U.html($('#m5calc'),
        U.td('F(' + L + ') = \\sum_{j=1}^{' + L + '}\\lambda_j \\Big/ \\sum_{j\'=1}^{' + lams.length + '}\\lambda_{j\'} =\\dfrac{' + list(kept) + '}{' + U.fmt(tot, 4) + '} = \\dfrac{' + U.fmt(P.sum(kept), 4) + '}{' + U.fmt(tot, 4) + '} = \\mathbf{' + (100 * F).toFixed(2) + '\\%}\\quad(20.27)') +
        U.td('\\mathcal L_{' + L + '} = \\sum_{j=' + (L + 1) + '}^{' + lams.length + '}\\lambda_j = ' + (disc.length ? list(disc) + ' = ' : '') + '\\mathbf{' + U.fmt(E, 4) + '}\\quad(20.26)') +
        '<p>Check: ' + U.tx('\\mathcal L_L/\\operatorname{tr}\\hat\\Sigma = ' + U.fmt(E / tot, 4)) + ' and ' + U.tx('F(L) + \\mathcal L_L/\\operatorname{tr}\\hat\\Sigma = 1') + '. Keeping one more component would lower the error by exactly ' + U.tx('\\lambda_{' + (L + 1) + '}' + (L < lams.length ? ' = ' + U.fmt(lams[L], 4) : '\\ (\\text{none left})')) + '.</p>');
    }

    function drawCum() {
      var svg = $('#m5cum'), m = lams.length, cum = P.cumulative(lams), need = P.minComponents(lams, target);
      var lo = Math.min(cum[0], target) - 0.08; lo = Math.max(0, Math.floor(lo * 10) / 10);
      var f = U.frame(svg, { W: 520, H: 260, x: [0.5, m + 0.5], y: [lo, 1.02], xt: cum.map(function (_, i) { return i + 1; }), yfmt: function (v) { return Math.round(v * 100) + '%'; }, xlabel: 'components L', ylabel: 'F(L)', m: { l: 46, r: 12, t: 12, b: 38 } });
      U.el('line', { x1: f.m.l, x2: f.m.l + f.pw, y1: f.sy(target), y2: f.sy(target), 'class': 'lt' }, f.g);
      U.el('text', { x: f.m.l + 8, y: f.sy(target) + 16, 'text-anchor': 'start', 'class': 'plab' }, f.top).textContent = 'target ' + (100 * target).toFixed(1).replace(/\.0$/, '') + '%';
      var d = ''; cum.forEach(function (v, i) { d += (i ? 'L' : 'M') + f.sx(i + 1).toFixed(1) + ',' + f.sy(v).toFixed(1); });
      U.el('path', { d: d, 'class': 'l1' }, f.g);
      cum.forEach(function (v, i) {
        var isNeed = i + 1 === need;
        U.el('circle', { cx: f.sx(i + 1), cy: f.sy(v), r: isNeed ? 7 : 4.5, 'class': isNeed ? 'best' : '', fill: isNeed ? null : (i < L ? 'var(--pc1)' : 'var(--pc2)'), stroke: '#fff', 'stroke-width': 1.5, 'data-tip': 'F(' + (i + 1) + ') = ' + (100 * v).toFixed(2) + '%' + (isNeed ? '<br>first L reaching the target' : '') }, f.top);
      });
      U.el('circle', { cx: f.sx(L), cy: f.sy(cum[L - 1]), r: 10, 'class': 'cur' }, f.top);
      var ok = cum[L - 1] >= target - 1e-12;
      U.html($('#m5target'),
        'To explain at least <b>' + (100 * target).toFixed(1).replace(/\.0$/, '') + '%</b> you need <b>L = ' + need + '</b> (green dot): ' + U.tx('F(' + need + ') = ' + (100 * cum[need - 1]).toFixed(2) + '\\%') +
        (need > 1 ? ' but ' + U.tx('F(' + (need - 1) + ') = ' + (100 * cum[need - 2]).toFixed(2) + '\\%') : '') + '. Your current ' + U.tx('L = ' + L) + (ok ? ' meets the target.' : ' falls short.') +
        ' <button type="button" class="btn btn-quiet" id="m5use">Set L = ' + need + '</button>');
      $('#m5use').addEventListener('click', function () { setL(need); });
    }

    function drawTT(resample) {
      if (resample || !tt) tt = P.trainTest(lams, +Ns.value, 1000, seed);
      $('#m5Nv').textContent = Ns.value;
      var svg = $('#m5tt'), m = lams.length, top = Math.max(tt.train[0], tt.test[0]) * 1.08 || 1;
      var f = U.frame(svg, { W: 520, H: 280, x: [0, m], y: [0, top], xt: Array.from({ length: m + 1 }, function (_, i) { return i; }), xlabel: 'components L', ylabel: 'mean squared error', m: { l: 50, r: 12, t: 12, b: 38 } });
      ['train', 'test'].forEach(function (k, c) {
        var d = ''; tt[k].forEach(function (v, i) { d += (i ? 'L' : 'M') + f.sx(i).toFixed(1) + ',' + f.sy(v).toFixed(1); });
        U.el('path', { d: d, 'class': c ? 'l2' : 'l1' }, f.g);
        tt[k].forEach(function (v, i) { U.el('circle', { cx: f.sx(i), cy: f.sy(v), r: 4, fill: c ? 'var(--pc2)' : 'var(--pc1)', stroke: '#fff', 'stroke-width': 1.5, 'data-tip': k + ' error at L = ' + i + ': ' + U.fmt(v, 4) + (c ? '' : '<br>= Σ of discarded sample eigenvalues') }, f.top); });
      });
      U.el('line', { x1: f.sx(L), x2: f.sx(L), y1: f.m.t, y2: f.sy(0), 'class': 'marker' }, f.g);
      U.el('text', { x: f.sx(tt.test.length - 1) - 4, y: f.sy(0) - 34, 'text-anchor': 'end', 'class': 'plab' }, f.top).textContent = 'both = 0 at L = D';
    }

    function drawAll(resample) {
      Ls.max = lams.length; if (L > lams.length) L = lams.length; Ls.value = L; $('#m5Lv').textContent = L;
      drawScree(); drawCalc(); drawCum(); drawTT(resample);
    }

    chips = UI.chips($('#m5chips'), lams, function (v) { lams = v; drawAll(true); });
    Ls.addEventListener('input', function () { setL(+Ls.value); });
    Ns.addEventListener('input', function () { drawTT(true); });
    $('#m5re').addEventListener('click', function () { seed++; drawTT(true); });
    U.$$('[data-p5]').forEach(function (b) { b.addEventListener('click', function () { chips.set(PRESETS[b.dataset.p5]); }); });
    U.$$('#m5targets [data-t]').forEach(function (b) {
      b.addEventListener('click', function () { target = +b.dataset.t; $('#m5tc').value = target * 100; marks(); drawCum(); });
    });
    $('#m5tc').addEventListener('input', function () { var v = +$('#m5tc').value; if (v > 0 && v <= 100) { target = v / 100; marks(); drawCum(); } });
    function marks() { U.$$('#m5targets [data-t]').forEach(function (b) { b.setAttribute('aria-pressed', Math.abs(+b.dataset.t - target) < 1e-9 ? 'true' : 'false'); }); }

    /* drag bars; neighbours clamp the value so the list stays sorted */
    var svg = $('#m5scree');
    U.drag(svg, function (ev) {
      var i = ev.target.getAttribute('data-bar'); if (i === null) return null; i = +i;
      frozenTop = svg._f.y1;
      var mv = function (p) {
        var f = svg._f, v = Math.max(0, f.iy(p.y));
        var hi = i > 0 ? lams[i - 1] : Infinity, lo = i < lams.length - 1 ? lams[i + 1] : 0;
        lams[i] = +Math.min(hi, Math.max(lo, v)).toFixed(3);
        drawScree(); drawCalc(); drawCum();
      };
      mv.end = function () { frozenTop = null; chips.setQuiet(lams); drawScree(); drawTT(true); };
      return mv;
    });
    drawAll(true);
  });
})();
