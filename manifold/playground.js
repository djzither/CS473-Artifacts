/* ============================================================================
   Manifold Learning Playground — section 4 (playground) and section 5 (tabs).
   ========================================================================= */

(function () {
  'use strict';
  var M = window.ML, P = window.Plot, U = window.UI, $ = U.$, $$ = U.$$, fmt = U.fmt;

  var METHODS = [
    { key: 'truth', label: 'Ground truth', short: 'Truth', params: [] },
    { key: 'pca', label: 'PCA', short: 'PCA', params: [] },
    { key: 'kpca', label: 'Kernel PCA (RBF)', short: 'Kernel PCA', params: [
      { key: 'bw', label: 'RBF width σ', sub: '× median pairwise distance', min: 0.05, max: 2, step: 0.05, def: 1, d: 2 }] },
    { key: 'isomap', label: 'Isomap', short: 'Isomap', params: [
      { key: 'k', label: 'neighbors k', min: 3, max: 40, step: 1, def: 8, d: 0 }] },
    { key: 'lle', label: 'Locally linear embedding', short: 'LLE', params: [
      { key: 'k', label: 'neighbors k', min: 3, max: 30, step: 1, def: 12, d: 0 },
      { key: 'reg', label: 'regularizer log₁₀ ε', sub: 'ε × tr(C) added to the local Gram matrix', min: -5, max: 0, step: 0.25, def: -3, d: 2 }] },
    { key: 'lap', label: 'Laplacian eigenmaps', short: 'Laplacian', params: [
      { key: 'k', label: 'neighbors k', min: 3, max: 30, step: 1, def: 10, d: 0 },
      { key: 'heat', label: 'heat-kernel width σ', sub: '× mean edge length', min: 0.25, max: 5, step: 0.25, def: 1, d: 2 }] },
    { key: 'tsne', label: 't-SNE (exact gradient)', short: 't-SNE', params: [
      { key: 'perp', label: 'perplexity', sub: 'effective number of neighbors', min: 2, max: 100, step: 1, def: 30, d: 0 },
      { key: 'lr', label: 'learning rate η', min: 10, max: 1000, step: 10, def: 200, d: 0 },
      { key: 'exag', label: 'early exaggeration', sub: 'first 100 iterations', min: 1, max: 30, step: 1, def: 12, d: 0 },
      { key: 'iters', label: 'iterations', min: 100, max: 2000, step: 100, def: 1000, d: 0 }] }
  ];
  var BY = {}; METHODS.forEach(function (m) { BY[m.key] = m; });
  var TIPS = {
    truth: 'The coordinates that generated the data: (arc length, height) for the roll, the angle for the circle. A perfect method reproduces this picture up to rotation and scale.',
    pca: 'PCA can only rotate and drop an axis. On the Swiss roll, look for colors from different layers piled on top of each other. On the linear plane it is perfect.',
    kpca: 'Try σ from 0.1 to 2. Small σ makes every kernel eigenvector a local bump; large σ turns kernel PCA back into PCA. Neither unrolls the roll.',
    isomap: 'Slide k upward on the Swiss roll. Somewhere near k ≈ 15–20, edges jump across layers and the unrolled sheet collapses. Section 3 shows why.',
    lle: 'Watch the bottom eigenvalues: the first is 0 (the constant vector) and is skipped. Try k = 4 to see the manifold shatter, or a large ε to over-smooth.',
    lap: 'The output keeps neighbors together but not distances. On the long Swiss roll the second eigenvector is often a harmonic of the first, which gives an arch instead of a rectangle.',
    tsne: 'Neighborhoods are kept (high 10-NN score) but global distances are not (lower distance corr.). Try perplexity 2 to tear the manifold into islands, or 100 for a smoother global layout.'
  };

  var st = { data: 'swiss', n: 400, noise: 0, seed: 1, colorBy: 'c', focus: 'isomap', hover: -1, params: {}, res: {}, ver: {}, gen: 0 };
  METHODS.forEach(function (m) { st.params[m.key] = {}; m.params.forEach(function (p) { st.params[m.key][p.key] = p.def; }); st.ver[m.key] = 0; });

  /* ---------- views ---------- */
  var V = P.Viewer3D($('#s4-3d'), { yaw: 0.35, pitch: 1.1 });
  var focus = P.Scatter2D($('#s4-focus'), { r: 3.2, pad: 16 });
  var thumbs = {};
  var tbox = $('#s4-thumbs');
  METHODS.forEach(function (m) {
    var d = document.createElement('div'); d.className = 'thumb'; d.setAttribute('role', 'listitem');
    d.innerHTML = '<button type="button" class="thumb-btn" aria-pressed="false"><span class="thumb-name">' + m.short + '</span>' +
      '<canvas class="c-thumb" aria-hidden="true"></canvas><span class="thumb-score num">—</span></button>';
    tbox.appendChild(d);
    var plot = P.Scatter2D($('canvas', d), { r: 1.7, pad: 6, ms: 500 });
    thumbs[m.key] = { el: d, btn: $('button', d), plot: plot, score: $('.thumb-score', d) };
    $('button', d).addEventListener('click', function () { setFocus(m.key); });
    plot.onHover = function (i) { setHover(i); };
  });
  focus.onHover = function (i) { setHover(i); };

  /* ---------- data ---------- */
  function regenerate() {
    st.gen++;
    st.ds = M.makeData(st.data, st.n, st.noise, st.seed);
    st.D2 = M.sqdist(st.ds.X);
    st.trueNb = M.knn(st.ds.TD, st.n, 10);
    st.res = {};
    st.hover = -1;
    recolor();
    V.boundsOf(st.ds.X);
    $('#s4-nv').textContent = st.n; $('#s4-noisev').textContent = st.noise.toFixed(2);
    $('#s4-dims').innerHTML = 'ambient D = 3 · intrinsic L = ' + st.ds.dim;
    var sel = $('#s4-color'); sel.innerHTML = '';
    st.ds.coords.forEach(function (c, i) { var o = document.createElement('option'); o.value = i ? 'c2' : 'c'; o.textContent = c; sel.appendChild(o); });
    if (st.ds.coords.length < 2) st.colorBy = 'c';
    sel.value = st.colorBy;
    finish('truth', { Y: st.ds.T }, 0);
    METHODS.forEach(function (m) { if (m.key !== 'truth') { thumbs[m.key].plot.clear('waiting…'); thumbs[m.key].score.textContent = '—'; } });
    var order = [st.focus].concat(METHODS.map(function (m) { return m.key; }).filter(function (k) { return k !== st.focus && k !== 'truth' && k !== 'tsne'; }));
    if (st.focus !== 'tsne') order.push('tsne');
    queue = []; enqueue(order.filter(function (k) { return k !== 'truth'; }), false);
    drawFocus(false); V.render();
  }
  function recolor() {
    var src = st.ds[st.colorBy];
    st.colors = src.map(function (t) { return st.ds.cyclic ? P.sinebow(t) : P.viridis(t); });
    METHODS.forEach(function (m) { thumbs[m.key].plot.colors = st.colors; thumbs[m.key].plot.render(); });
    focus.colors = st.colors; focus.render(); V.render();
  }

  /* ---------- job queue: one method at a time, t-SNE animated ---------- */
  var queue = [], running = null;
  function enqueue(keys, front) {
    keys.forEach(function (k) { var i = queue.indexOf(k); if (i >= 0) queue.splice(i, 1); st.ver[k]++; });
    queue = front ? keys.concat(queue) : queue.concat(keys);
    keys.forEach(function (k) { status(k, 'computing…'); });
    pump();
  }
  function pump() {
    if (running || !queue.length) return;
    var k = queue.shift(), v = st.ver[k], g = st.gen;
    running = k;
    if (k === st.focus) $('#s4-status').textContent = 'computing…';
    setTimeout(function () { run(k, v, g, function () { running = null; pump(); }); }, 20);
  }
  function stale(k, v, g) { return st.ver[k] !== v || st.gen !== g; }

  function run(k, v, g, done) {
    if (stale(k, v, g)) return done();
    var p = st.params[k], ds = st.ds, D2 = st.D2, t0 = performance.now(), r;
    if (k === 'tsne') {
      var S = M.tsneInit(ds.X, { perplexity: p.perp, lr: p.lr, exag: p.exag, iters: p.iters, seed: 7 + st.seed }, D2);
      (function frame() {
        if (stale(k, v, g)) return done();
        var end = performance.now() + 14;
        while (S.iter < S.o.iters && performance.now() < end) M.tsneStep(S);
        var Y = M.align(M.tsneY(S), ds.T);
        if (S.iter >= S.o.iters) { finish(k, { Y: M.tsneY(S), kl: S.kl, klHist: S.klHist.slice(), iters: S.iter }, performance.now() - t0); return done(); }
        st.res[k] = { Y: Y, partial: true, kl: S.kl, klHist: S.klHist.slice(), iters: S.iter };
        thumbs[k].plot.set(Y, st.colors, false);
        thumbs[k].score.textContent = 'iter ' + S.iter;
        if (st.focus === k) { focus.set(Y, st.colors, false); $('#s4-status').textContent = 'iteration ' + S.iter + ' / ' + S.o.iters; drawSide(); }
        requestAnimationFrame(frame);
      })();
      return;
    }
    try {
      if (k === 'pca') r = M.pca(ds.X);
      else if (k === 'kpca') r = M.kpca(ds.X, p.bw, D2);
      else if (k === 'isomap') r = M.isomap(ds.X, p.k, D2);
      else if (k === 'lle') r = M.lle(ds.X, p.k, Math.pow(10, p.reg), D2);
      else if (k === 'lap') r = M.laplacian(ds.X, p.k, p.heat, D2);
      finish(k, r, performance.now() - t0);
    } catch (e) {
      console.error(e);
      st.res[k] = { error: String(e) }; status(k, 'error');
    }
    done();
  }

  function finish(k, r, ms) {
    var ds = st.ds, Y = M.align(r.Y, ds.T), bad = Y.some(function (y) { return !isFinite(y[0]) || !isFinite(y[1]); });
    if (bad) Y = Y.map(function () { return [0, 0]; });
    var res = { Y: Y, ms: ms, spectrum: r.spectrum, ratio: r.ratio, warn: r.warn, sigma: r.sigma, kl: r.kl, klHist: r.klHist, iters: r.iters };
    res.keep = M.neighborKeep(ds.TD, Y, ds.n, 10, st.trueNb);
    res.corr = M.distCorr(ds.TD, Y, ds.n);
    st.res[k] = res;
    thumbs[k].plot.set(Y, st.colors, true);
    thumbs[k].score.textContent = k === 'truth' ? 'reference' : 'keep ' + res.keep.toFixed(2) + ' · corr ' + res.corr.toFixed(2);
    thumbs[k].el.classList.toggle('has-warn', !!res.warn);
    if (k === st.focus) drawFocus(true);
  }
  function status(k, s) { if (thumbs[k] && !(st.res[k] && !st.res[k].partial)) thumbs[k].score.textContent = s; if (k === st.focus) $('#s4-status').textContent = s; }

  /* ---------- focus panel ---------- */
  function setFocus(k) {
    st.focus = k;
    METHODS.forEach(function (m) { thumbs[m.key].btn.setAttribute('aria-pressed', m.key === k ? 'true' : 'false'); });
    buildParams(); if (st.ds) drawFocus(true);
  }
  function drawFocus(animate) {
    var k = st.focus, r = st.res[k], m = BY[k];
    $('#s4-ftitle').textContent = m.label + ' · ' + st.ds.label;
    if (!r || r.error) { focus.clear(r && r.error ? 'error: ' + r.error : 'computing…'); $('#s4-status').textContent = r && r.error ? 'error' : 'computing…'; $('#s4-metrics').innerHTML = ''; drawSide(); return; }
    focus.set(r.Y, st.colors, animate && !r.partial);
    if (!r.partial) $('#s4-status').textContent = k === 'truth' ? '' : fmt(r.ms, 0) + ' ms';
    var html = '';
    if (!r.partial && k !== 'truth') {
      html += metric('10-NN kept', r.keep, 'local: how many of each point’s 10 true neighbors stay neighbors');
      html += metric('Distance corr.', r.corr, 'global: do far points stay far, near points near?');
    } else if (k === 'truth') html = '<p class="small muted">The reference picture. Hover a point to light up its 10 true neighbors, then look for them in the other plots.</p>';
    if (r.warn) html += '<div class="warn">' + r.warn + '</div>';
    $('#s4-metrics').innerHTML = html;
    drawSide();
  }
  function metric(name, v, sub) {
    var c = Math.max(0, Math.min(1, v)), cls = c > 0.85 ? 'good' : c > 0.6 ? 'mid' : 'bad';
    return '<div class="metric ' + cls + '"><div class="m-top"><span>' + name + '</span><b class="num">' + v.toFixed(3) + '</b></div>' +
      '<div class="m-bar"><i style="width:' + (c * 100).toFixed(1) + '%"></i></div><small>' + sub + '</small></div>';
  }

  function drawSide() {
    var k = st.focus, r = st.res[k], cv = $('#s4-spec'), note = $('#s4-spec-note'), head = $('#s4-spec-h');
    $('#s4-tip').innerHTML = '<b>Try:</b> ' + TIPS[k];
    var wrap = $('.spec-wrap');
    if (k === 'truth' || !r || r.error) { wrap.hidden = true; return; }
    wrap.hidden = false;
    if (k === 'tsne') {
      head.textContent = 'KL(P ‖ Q) by iteration';
      var c = P.setupCanvas(cv), ctx = c.ctx, W = c.w, H = c.h, h = r.klHist || [];
      ctx.clearRect(0, 0, W, H);
      if (h.length > 1) {
        var mx = Math.max.apply(null, h.map(function (q) { return q[1]; })), it = st.params.tsne.iters;
        ctx.strokeStyle = '#c3cede'; ctx.beginPath(); ctx.moveTo(30, 6); ctx.lineTo(30, H - 16); ctx.lineTo(W - 4, H - 16); ctx.stroke();
        ctx.strokeStyle = '#1d5fd6'; ctx.lineWidth = 2; ctx.beginPath();
        h.forEach(function (q, i) { var x = 30 + (q[0] / it) * (W - 34), y = H - 16 - (q[1] / mx) * (H - 24); if (i) ctx.lineTo(x, y); else ctx.moveTo(x, y); });
        ctx.stroke();
        ctx.fillStyle = '#7b879c'; ctx.font = '400 10.5px "IBM Plex Mono", monospace'; ctx.fillText(mx.toFixed(1), 0, 14); ctx.fillText('0', 18, H - 16);
        var ex = 30 + (100 / it) * (W - 34); ctx.strokeStyle = '#e0701a'; ctx.setLineDash([3, 3]); ctx.beginPath(); ctx.moveTo(ex, 6); ctx.lineTo(ex, H - 16); ctx.stroke(); ctx.setLineDash([]);
        ctx.fillStyle = '#e0701a'; ctx.fillText('exaggeration ends', Math.min(ex + 4, W - 110), 14);
      }
      note.innerHTML = 'KL = <b class="num">' + (isFinite(r.kl) ? r.kl.toFixed(3) : '…') + '</b> after ' + (r.iters || 0) + ' iterations. The jump at iteration 100 is the end of early exaggeration (P is multiplied by ' + st.params.tsne.exag + ' before that).';
      return;
    }
    var s = r.spectrum || [], lab, vals, col, txt;
    if (k === 'pca') {
      head.textContent = 'Explained variance';
      vals = r.ratio; lab = ['PC1', 'PC2', 'PC3']; col = function (i) { return i < 2 ? '#1d5fd6' : '#c3cede'; };
      txt = 'The two kept components explain <b>' + ((vals[0] + vals[1]) * 100).toFixed(1) + '%</b> of the variance. The dropped third keeps ' + (vals[2] * 100).toFixed(1) + '%.' +
        (st.data === 'swiss' ? ' A high number does not mean the layers are separated.' : '');
    } else if (k === 'kpca' || k === 'isomap') {
      head.textContent = (k === 'isomap' ? 'Top eigenvalues of B (MDS)' : 'Top eigenvalues of the centered kernel') + ', shown as √λ';
      vals = s.slice(0, 8).map(function (v) { return Math.sqrt(Math.max(v, 0)); }); col = function (i) { return i < 2 ? '#1d5fd6' : '#c3cede'; };
      var neg = s.slice(0, 8).some(function (v) { return v < -1e-9; });
      txt = k === 'isomap'
        ? 'Bar height √λ is the spread of the embedding along each axis. λ₃/λ₂ = <b class="num">' + fmt(s[2] / s[1], 2) + '</b>. If graph distances were exact geodesics on a flat manifold, only L = ' + st.ds.dim + ' eigenvalue' + (st.ds.dim > 1 ? 's' : '') + ' would be nonzero, and the rest measure zig-zag, noise and short circuits. (A long, thin manifold has λ₁ ≫ λ₂, so look at the drop <i>after</i> λ<sub>L</sub>.)' + (neg ? ' Some eigenvalues are negative, so these distances are not exactly Euclidean.' : '')
        : 'σ = ' + fmt(r.sigma || st.params.kpca.bw, 2) + ' in data units. The spectrum decays slowly, with no sharp elbow at L. The kernel spreads variance over many smooth eigenfunctions.';
    } else {
      head.textContent = k === 'lle' ? 'Bottom eigenvalues of M' : 'Bottom eigenvalues of the normalized Laplacian';
      vals = s.slice(0, 8); col = function (i) { return i === 0 ? '#c3cede' : i < 3 ? '#1d5fd6' : '#dbe2ee'; };
      txt = 'λ₀ = ' + s[0].toExponential(1) + ' belongs to the constant vector, which is skipped. Eigenvectors 1–2 (blue) become the coordinates. A <i>repeated</i> near-zero eigenvalue would mean the graph has come apart.';
    }
    P.bars(cv, vals, { labels: lab, color: col, max: Math.max.apply(null, vals.map(Math.abs)) || 1 });
    note.innerHTML = txt;
  }

  function buildParams() {
    var m = BY[st.focus], box = $('#s4-params');
    $('#s4-ptitle').textContent = m.params.length ? 'Hyperparameters · ' + m.short : m.short;
    box.innerHTML = m.params.length ? '' : '<p class="small muted">' + (st.focus === 'pca' ? 'PCA has no hyperparameters: center, eigendecompose, project.' : 'No hyperparameters.') + '</p>';
    var tmr = 0;
    m.params.forEach(function (p) {
      var w = document.createElement('div'); w.className = 'ctrl';
      var val = st.params[m.key][p.key];
      w.innerHTML = '<label for="pp-' + p.key + '">' + p.label + ' <output class="num">' + show(p, val) + '</output>' + (p.sub ? '<small>' + p.sub + '</small>' : '') + '</label>' +
        '<input type="range" id="pp-' + p.key + '" min="' + p.min + '" max="' + p.max + '" step="' + p.step + '" value="' + val + '">';
      box.appendChild(w);
      $('input', w).addEventListener('input', function (e) {
        st.params[m.key][p.key] = +e.target.value; $('output', w).textContent = show(p, +e.target.value);
        clearTimeout(tmr); tmr = setTimeout(function () { enqueue([m.key], true); }, m.key === 'tsne' ? 350 : 220);
      });
    });
    if (m.params.length) {
      var b = document.createElement('button'); b.type = 'button'; b.className = 'btn btn-quiet'; b.textContent = 'Reset to defaults';
      b.addEventListener('click', function () { m.params.forEach(function (p) { st.params[m.key][p.key] = p.def; }); buildParams(); enqueue([m.key], true); });
      box.appendChild(b);
    }
  }
  function show(p, v) { return p.key === 'reg' ? 'ε = 10^' + v.toFixed(2) : v.toFixed(p.d); }

  /* ---------- hover linking ---------- */
  function setHover(i) {
    if (i === st.hover) return;
    st.hover = i;
    var nb = i >= 0 ? st.trueNb[i] : null;
    [focus].concat(METHODS.map(function (m) { return thumbs[m.key].plot; })).forEach(function (p) { p.hi = i; p.nbrs = nb; p.render(); });
    V.render();
  }
  V.scene = function (api) {
    if (!st.ds) return;
    var X = st.ds.X, hi = st.hover, nb = hi >= 0 ? new Set(st.trueNb[hi]) : null;
    for (var i = 0; i < X.length; i++) {
      if (nb && (i === hi || nb.has(i))) continue;
      api.point(X[i], 2.7, st.colors[i]);
    }
    if (nb) {
      nb.forEach(function (j) { api.point(X[j], 4.5, st.colors[j], { top: true, stroke: '#14213d', lw: 1.4 }); });
      api.point(X[hi], 7, st.colors[hi], { top: true, stroke: '#14213d', lw: 2.5 });
    }
  };
  V.onHover = function (x, y) { setHover(x < 0 ? -1 : P.pick(V, st.ds.X, x, y, 8)); };

  /* ---------- toolbar ---------- */
  U.seg($('#s4-data'), Object.keys(M.DATASETS).map(function (k) { return { key: k, label: M.DATASETS[k].label }; }), st.data, function (k) { st.data = k; regenerate(); });
  var tm = 0;
  $('#s4-n').addEventListener('input', function (e) { st.n = +e.target.value; $('#s4-nv').textContent = st.n; clearTimeout(tm); tm = setTimeout(regenerate, 350); });
  $('#s4-noise').addEventListener('input', function (e) { st.noise = +e.target.value; $('#s4-noisev').textContent = st.noise.toFixed(2); clearTimeout(tm); tm = setTimeout(regenerate, 350); });
  $('#s4-color').addEventListener('change', function (e) { st.colorBy = e.target.value; recolor(); });
  $('#s4-seed').addEventListener('click', function () { st.seed++; regenerate(); });

  /* ---------- section 5: tabs and "see it" buttons ---------- */
  var tabs = $('#s5-tabs'), panels = $$('.algo');
  var NAMES = { pca: 'PCA', kpca: 'Kernel PCA', isomap: 'Isomap', lle: 'LLE', lap: 'Laplacian eigenmaps', tsne: 't-SNE' };
  panels.forEach(function (pnl, i) {
    var key = pnl.dataset.algo, b = document.createElement('button');
    b.type = 'button'; b.className = 'tab'; b.textContent = NAMES[key]; b.setAttribute('role', 'tab');
    b.id = 'tab-' + key; pnl.setAttribute('aria-labelledby', b.id);
    b.setAttribute('aria-selected', i === 0 ? 'true' : 'false');
    b.addEventListener('click', function () {
      $$('.tab', tabs).forEach(function (t) { t.setAttribute('aria-selected', t === b ? 'true' : 'false'); });
      panels.forEach(function (q) { q.hidden = q !== pnl; });
    });
    tabs.appendChild(b);
  });
  $$('[data-try]').forEach(function (b) {
    b.addEventListener('click', function () {
      var parts = b.dataset.try.split('|'), data = parts[0], meth = parts[1];
      METHODS.forEach(function (m) { m.params.forEach(function (p) { st.params[m.key][p.key] = p.def; }); });
      st.noise = 0; st.n = 400; st.seed = 1;
      parts.slice(2).forEach(function (kv) {
        var a = kv.split('='), name = a[0], val = +a[1];
        if (name === 'noise') st.noise = val; else st.params[meth][name] = val;
      });
      $('#s4-n').value = st.n; $('#s4-noise').value = st.noise;
      st.data = data;
      $$('#s4-data .seg-btn').forEach(function (x) { x.setAttribute('aria-pressed', x.dataset.key === data ? 'true' : 'false'); });
      st.focus = meth; setFocus(meth);
      regenerate();
      $('#s4').scrollIntoView({ behavior: matchMedia('(prefers-reduced-motion: reduce)').matches ? 'auto' : 'smooth', block: 'start' });
    });
  });

  window.addEventListener('resize', function () { drawSide(); });
  setFocus(st.focus);
  regenerate();
})();
