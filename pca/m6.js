/* Module 6 — Profile likelihood (§20.1.4.3, Eqs. 20.28–20.31). */
(function () {
  'use strict';
  var P = window.PCA, U = window.UI, $ = U.$;
  var PRESETS = {
    elbow: [10, 9.2, 8.1, 7.5, 2.1, 1.7, 1.4, 1.1, 0.9, 0.6],
    decay: [8, 5.2, 3.4, 2.2, 1.4, 0.9, 0.6, 0.4, 0.25, 0.15],
    two: [12, 11, 6, 5.5, 5, 1, 0.8, 0.6, 0.5, 0.4]
  };

  UI.module('m6', function () {
    var lams = PRESETS.elbow.slice(), L = 2, Ls = $('#m6L'), chips;

    function drawScree(pr) {
      var svg = $('#m6scree'), m = lams.length, top = Math.max.apply(null, lams) * 1.15 || 1;
      var f = U.frame(svg, { W: 520, H: 300, x: [0.5, m + 0.5], y: [0, top], xt: lams.map(function (_, i) { return i + 1; }), xlabel: 'component k', ylabel: 'eigenvalue λₖ', m: { l: 46, r: 12, t: 16, b: 38 } });
      var sd = Math.sqrt(pr.sigma2);
      [[1, L, pr.mu1, 1], [L + 1, m, pr.mu2, 2]].forEach(function (g) {
        var x1 = f.sx(g[0] - 0.5), x2 = f.sx(g[1] + 0.5);
        U.el('rect', { x: x1, width: x2 - x1, y: f.sy(g[2] + sd), height: Math.max(0, f.sy(Math.max(0, g[2] - sd)) - f.sy(g[2] + sd)), 'class': 'band' + g[3] }, f.g);
        U.el('line', { x1: x1, x2: x2, y1: f.sy(g[2]), y2: f.sy(g[2]), 'class': 'mean' + g[3] }, f.g);
        U.el('text', { x: (x1 + x2) / 2, y: Math.max(f.m.t + 12, f.sy(g[2] + sd) - 6), 'text-anchor': 'middle', 'class': 'plab pc' + g[3] }, f.top).textContent = 'μ' + g[3] + ' = ' + U.fmt(g[2], 3);
      });
      lams.forEach(function (v, i) {
        var g = i < L ? 1 : 2;
        U.el('line', { x1: f.sx(i + 1), x2: f.sx(i + 1), y1: f.sy(0), y2: f.sy(v), stroke: g === 1 ? 'var(--pc1)' : 'var(--pc2)', 'stroke-width': 2 }, f.g);
        U.el('circle', { cx: f.sx(i + 1), cy: f.sy(v), r: 6, fill: g === 1 ? 'var(--pc1)' : 'var(--pc2)', stroke: '#fff', 'stroke-width': 2,
          'data-tip': 'λ' + (i + 1) + ' = ' + U.fmt(v, 4) + '<br>group ' + g + ', deviation from μ' + g + ' = ' + U.fmt(v - (g === 1 ? pr.mu1 : pr.mu2), 3) }, f.top);
      });
      var xs = f.sx(L + 0.5);
      U.el('line', { x1: xs, x2: xs, y1: f.m.t, y2: f.sy(0), 'class': 'split' }, f.top);
      U.el('rect', { x: f.m.l, y: f.m.t, width: f.pw, height: f.ph, 'class': 'hit', 'data-tip': 'click to move the change point' }, f.g);
      svg._f = f;
    }

    function drawLL(all) {
      var svg = $('#m6ll'), lls = all.all.map(function (p) { return p.degenerate ? NaN : p.ll; }).filter(isFinite);
      var lo = Math.min.apply(null, lls), hi = Math.max.apply(null, lls), pad = (hi - lo) * 0.12 || 1;
      var m = lams.length;
      var f = U.frame(svg, { W: 520, H: 260, x: [0.5, m - 0.5], y: [lo - pad, hi + pad], xt: all.all.map(function (p) { return p.L; }), xlabel: 'change point L', ylabel: 'ℓ(L)', m: { l: 54, r: 12, t: 14, b: 38 } });
      var d = ''; all.all.forEach(function (p, i) { if (!p.degenerate) d += (d ? 'L' : 'M') + f.sx(p.L).toFixed(1) + ',' + f.sy(p.ll).toFixed(1); });
      U.el('path', { d: d, 'class': 'l1' }, f.g);
      all.all.forEach(function (p) {
        var y = p.degenerate ? f.m.t + 6 : f.sy(p.ll), isBest = p.L === all.best.L;
        U.el('circle', { cx: f.sx(p.L), cy: y, r: isBest ? 8 : 5, 'class': isBest ? 'best' : '', fill: isBest ? null : 'var(--pc1)', stroke: '#fff', 'stroke-width': 1.5 }, f.top);
        U.el('circle', { cx: f.sx(p.L), cy: y, r: 16, 'class': 'hit', 'data-L': p.L, 'data-tip': 'L = ' + p.L + '<br>ℓ = ' + (p.degenerate ? '+∞ (σ² = 0)' : U.fmt(p.ll, 4)) + '<br>σ² = ' + U.fmt(p.sigma2, 4) + '<br>click to select' }, f.top);
      });
      var cy = all.all[L - 1].degenerate ? f.m.t + 6 : f.sy(all.all[L - 1].ll);
      U.el('circle', { cx: f.sx(L), cy: cy, r: 12, 'class': 'cur' }, f.top);
      U.el('text', { x: f.sx(all.best.L) + (all.best.L > (m - 1) / 2 ? -14 : 14), y: (all.best.degenerate ? f.m.t + 6 : f.sy(all.best.ll)) + 5, 'text-anchor': all.best.L > (m - 1) / 2 ? 'end' : 'start', 'class': 'plab' }, f.top).textContent = 'max: L* = ' + all.best.L;
    }

    function drawCalc(pr) {
      var m = lams.length, g1 = lams.slice(0, L), g2 = lams.slice(L);
      var ls = function (a) { return a.map(function (v) { return U.fmt(v, 3); }).join('+'); };
      var dev = function (a, mu) { return a.map(function (v) { return '(' + U.fmt(v, 3) + '-' + U.fmt(mu, 3) + ')^2'; }); };
      var short = function (terms) { return terms.length > 3 ? terms.slice(0, 2).join('+') + '+\\cdots' : terms.join('+'); };
      U.html($('#m6calc'),
        '<p><b>Eq. 20.29 &ndash; mean of group 1</b> (the ' + L + ' &ldquo;signal&rdquo; eigenvalues):</p>' +
        U.td('\\mu_1(L) = \\frac{\\sum_{k\\le L}\\lambda_k}{L} = \\frac{' + ls(g1) + '}{' + L + '} = ' + U.fmt(pr.mu1, 4)) +
        '<p><b>Eq. 20.30 &ndash; mean of group 2</b> (the ' + (m - L) + ' &ldquo;noise&rdquo; eigenvalues):</p>' +
        U.td('\\mu_2(L) = \\frac{\\sum_{k>L}\\lambda_k}{L_{max}-L} = \\frac{' + ls(g2) + '}{' + (m - L) + '} = ' + U.fmt(pr.mu2, 4)) +
        '<p><b>Eq. 20.31 &ndash; shared variance</b>: pool both groups&rsquo; squared deviations, divide by ' + U.tx('L_{max} = ' + m) + ':</p>' +
        U.td('\\sigma^2(L) = \\frac{\\overbrace{' + short(dev(g1, pr.mu1)) + '}^{' + U.fmt(pr.ss1, 4) + '} + \\overbrace{' + short(dev(g2, pr.mu2)) + '}^{' + U.fmt(pr.ss2, 4) + '}}{' + m + '} = ' + U.fmt(pr.sigma2, 4)) +
        '<p><b>Eq. 20.28 &ndash; profile log likelihood</b>: sum the Gaussian log densities with these plugged in:</p>' +
        U.td('\\ell(L) = \\underbrace{\\sum_{k\\le L}\\log\\mathcal N(\\lambda_k\\mid\\mu_1,\\sigma^2)}_{' + U.fmt(pr.ll1, 4) + '} + \\underbrace{\\sum_{k>L}\\log\\mathcal N(\\lambda_k\\mid\\mu_2,\\sigma^2)}_{' + U.fmt(pr.ll2, 4) + '} = ' + (pr.degenerate ? '+\\infty' : '\\mathbf{' + U.fmt(pr.ll, 4) + '}')) +
        (pr.degenerate ? '<p class="note">Both groups are perfectly constant, so σ² = 0 and the likelihood is unbounded. Nudge an eigenvalue.</p>'
          : '<p>Shortcut: ' + U.tx('-\\tfrac{L_{max}}{2}\\big[\\log(2\\pi\\sigma^2)+1\\big] = -' + (m / 2) + '\\,[\\log(2\\pi\\cdot' + U.fmt(pr.sigma2, 4) + ')+1] = ' + U.fmt(-m / 2 * (Math.log(2 * Math.PI * pr.sigma2) + 1), 4)) + ' <span class="okmark">✓</span></p>'));
    }

    function drawTable(all) {
      var h = '<table class="data click"><tr><th>L</th><th>μ₁</th><th>μ₂</th><th>SS₁+SS₂</th><th>σ²</th><th>ℓ(L)</th></tr>';
      all.all.forEach(function (p) {
        h += '<tr data-L="' + p.L + '" class="' + (p.L === all.best.L ? 'best ' : '') + (p.L === L ? 'sel' : '') + '"><td>' + p.L + (p.L === all.best.L ? ' ★' : '') + '</td><td>' + U.fmt(p.mu1, 3) + '</td><td>' + U.fmt(p.mu2, 3) + '</td><td>' + U.fmt(p.ss1 + p.ss2, 3) + '</td><td>' + U.fmt(p.sigma2, 4) + '</td><td>' + (p.degenerate ? '+∞' : U.fmt(p.ll, 3)) + '</td></tr>';
      });
      U.html($('#m6table'), h + '</table><p class="note">The row with the smallest pooled SS always has the largest ℓ: same ranking, because ' + U.tx('\\ell = -\\tfrac{L_{max}}2[\\log(2\\pi\\,\\mathrm{SS}/L_{max})+1]') + '.</p>');
    }

    function drawCmp(all) {
      var el = P.elbow(lams), t = [0.9, 0.95, 0.99].map(function (x) { return P.minComponents(lams, x); });
      var Fb = P.explained(lams, all.best.L);
      U.html($('#m6cmp'),
        '<table class="data"><tr><th>method</th><th>L</th><th>F(L)</th></tr>' +
        '<tr class="best"><td>profile likelihood (argmax ℓ)</td><td>' + all.best.L + '</td><td>' + (100 * Fb).toFixed(1) + '%</td></tr>' +
        '<tr><td>visual elbow (max distance to chord)<sup>†</sup></td><td>' + el + '</td><td>' + (100 * P.explained(lams, el)).toFixed(1) + '%</td></tr>' +
        ['90%', '95%', '99%'].map(function (s, i) { return '<tr><td>explained variance ≥ ' + s + '</td><td>' + t[i] + '</td><td>' + (100 * P.explained(lams, t[i])).toFixed(1) + '%</td></tr>'; }).join('') +
        '</table>' +
        '<div class="why-body flat"><p><b>Visual elbow:</b> fast and intuitive, but subjective; with a smooth decay there&rsquo;s no clear bend and two people pick different L. <b>Variance threshold:</b> objective and easy to compute by hand, but the threshold (90? 95? 99?) is arbitrary and often keeps many noise components. <b>Profile likelihood:</b> automates the elbow. It picks the split that makes the eigenvalues look most like &ldquo;two flat clusters&rdquo;, with no threshold to tune, but it assumes that two-cluster shape. With two gaps (try the preset) it picks whichever gap gives the tighter clusters.</p>' +
        '<p class="note"><sup>†</sup> The elbow column uses a simple automatic heuristic (the point farthest from the line joining the first and last eigenvalues). It stands in for eyeballing and is not from Murphy.</p></div>');
    }

    function draw() {
      var m = lams.length; Ls.max = m - 1; if (L > m - 1) L = m - 1; if (L < 1) L = 1; Ls.value = L; $('#m6Lv').textContent = L;
      var all = P.profileAll(lams), pr = all.all[L - 1];
      drawScree(pr); drawLL(all); drawCalc(pr); drawTable(all); drawCmp(all);
      U.html($('#m6best'), 'The profile likelihood is maximized at <b>L* = ' + all.best.L + '</b>' + (L === all.best.L ? ', which is your current split.' : '. You are at L = ' + L + '. <button type="button" class="btn btn-quiet" id="m6go">Jump to L*</button>'));
      var go = $('#m6go'); if (go) go.addEventListener('click', function () { L = all.best.L; draw(); });
    }

    chips = UI.chips($('#m6chips'), lams, function (v) { lams = v; draw(); }, { min: 3 });
    Ls.addEventListener('input', function () { L = +Ls.value; draw(); });
    U.$$('[data-p6]').forEach(function (b) { b.addEventListener('click', function () { chips.set(PRESETS[b.dataset.p6]); }); });
    $('#m6copy').addEventListener('click', function () { var v = window.M5 && window.M5.get(); if (v && v.length >= 3) chips.set(v); else { $('#m6copy').textContent = 'Open Module 5 first'; } });
    $('#m6ll').addEventListener('click', function (ev) { var l = ev.target.getAttribute('data-L'); if (l) { L = +l; draw(); } });
    $('#m6table').addEventListener('click', function (ev) { var tr = ev.target.closest('tr[data-L]'); if (tr) { L = +tr.dataset.L; draw(); } });
    $('#m6scree').addEventListener('click', function (ev) { var f = $('#m6scree')._f; if (!f) return; var p = UI.svgPt($('#m6scree'), ev), x = f.ix(p.x); L = Math.max(1, Math.min(lams.length - 1, Math.round(x - 0.5))); draw(); });
    draw();
  });
})();
