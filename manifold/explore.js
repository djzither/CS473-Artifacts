/* ============================================================================
   Manifold Learning Playground — sections 1–3 (intuition).
     1. Surfaces, tangent planes, curvature
     2. The manifold hypothesis on a drawn digit
     3. Euclidean vs. geodesic distance on the Swiss roll
   ========================================================================= */

(function () {
  'use strict';
  var M = window.ML, P = window.Plot;
  var $ = function (s, r) { return (r || document).querySelector(s); };
  var $$ = function (s, r) { return Array.prototype.slice.call((r || document).querySelectorAll(s)); };

  /* ---------- shared helpers, exposed for the other scripts ---------- */
  function tex(node, src, display) {
    if (window.katex) { try { katex.render(src, node, { throwOnError: false, displayMode: !!display }); return; } catch (e) { /* fall through */ } }
    node.textContent = src; node.classList.add('tex-raw');
  }
  function texIn(root) {
    $$('.tex', root).forEach(function (n) { if (!n.dataset.done) { tex(n, n.dataset.tex, false); n.dataset.done = 1; } });
    $$('.tex-block', root).forEach(function (n) { if (!n.dataset.done) { tex(n, n.dataset.tex, true); n.dataset.done = 1; } });
  }
  function fmt(v, d) { if (!isFinite(v)) return '∞'; var s = v.toFixed(d == null ? 2 : d); return s === '-0.00' || s === '-0.000' ? s.slice(1) : s; }
  function seg(container, items, active, onPick) {
    container.innerHTML = '';
    items.forEach(function (it) {
      var b = document.createElement('button');
      b.type = 'button'; b.className = 'seg-btn'; b.textContent = it.label; b.dataset.key = it.key;
      b.setAttribute('aria-pressed', it.key === active ? 'true' : 'false');
      b.addEventListener('click', function () {
        $$('.seg-btn', container).forEach(function (x) { x.setAttribute('aria-pressed', x === b ? 'true' : 'false'); });
        onPick(it.key);
      });
      container.appendChild(b);
    });
  }
  window.UI = { tex: tex, texIn: texIn, fmt: fmt, seg: seg, $: $, $$: $$ };
  texIn(document);

  /* ======================================================== 1. surfaces */
  (function () {
    var canvas = $('#s1-canvas'), V = P.Viewer3D(canvas, { yaw: -0.55, pitch: 0.42 });
    var st = { key: 'sphere', u: 0, v: 0, r: 1.2, plane: true, mesh: null, pt: null };
    var LIGHT = M.unit([0.35, 0.85, 0.4]);

    seg($('#s1-surf'), Object.keys(M.SURFACES).map(function (k) { return { key: k, label: M.SURFACES[k].label }; }), st.key, function (k) { setSurface(k); });

    function setSurface(k) {
      st.key = k; var S = M.SURFACES[k];
      st.u = S.start[0]; st.v = S.start[1];
      var G = [], all = [];
      for (var i = 0; i <= S.nu; i++) {
        var row = [];
        for (var j = 0; j <= S.nv; j++) {
          var u = S.u[0] + (S.u[1] - S.u[0]) * i / S.nu, v = S.v[0] + (S.v[1] - S.v[0]) * j / S.nv, p = S.f(u, v);
          row.push({ p: p, u: u, v: v }); all.push(p);
        }
        G.push(row);
      }
      st.mesh = G; st.all = all;
      V.boundsOf(all); V.zoom = 1;
      update(true);
    }

    function update(curveToo) {
      var S = M.SURFACES[st.key], pt = M.patch(S, st.u, st.v, st.r, 45);
      st.pt = pt;
      var g = pt.geom;
      $('#s1-rv').textContent = fmt(st.r);
      $('#s1-delta').textContent = fmt(pt.delta, 3);
      $('#s1-ratio').textContent = fmt(pt.delta / st.r, 3);
      var K = Math.abs(g.K) < 5e-4 ? 0 : g.K;
      $('#s1-K').textContent = fmt(K, 3) + (K === 0 ? '  (flat)' : K > 0 ? '  (dome)' : '  (saddle)');
      var names = { sphere: ['\\theta', '\\phi'], swiss: ['t', 'h'], saddle: ['x', 'z'], torus: ['\\theta', '\\phi'] }[st.key];
      tex($('#s1-metric'),
        '\\text{At } p = (' + g.p.map(function (x) { return fmt(x); }).join(',\\ ') + '):\\quad ' +
        'g = \\begin{pmatrix} E & F \\\\ F & G \\end{pmatrix} = \\begin{pmatrix} ' + fmt(g.E) + ' & ' + fmt(g.F) + ' \\\\ ' + fmt(g.F) + ' & ' + fmt(g.G) + ' \\end{pmatrix}' +
        '\\ \\text{in coordinates } (' + names[0] + ',' + names[1] + '),\\qquad ds^2 = E\\,d' + names[0] + '^2 + 2F\\,d' + names[0] + 'd' + names[1] + ' + G\\,d' + names[1] + '^2', true);
      drawChart();
      if (curveToo) drawCurve();
      else drawCurveMarker();
      V.render();
    }

    /* ---- the 3-D scene ---- */
    V.scene = function (api) {
      var S = M.SURFACES[st.key], G = st.mesh, g = st.pt.geom, p0 = g.p;
      var du = 2.2 * st.r / Math.sqrt(g.E), dv = 2.2 * st.r / Math.sqrt(g.G);
      function inPatch(c, u, v) {
        var a = Math.abs(u - st.u), b = Math.abs(v - st.v);
        if (S.wrapU) a = Math.min(a, M.TAU - a);
        if (S.wrapV) b = Math.min(b, M.TAU - b);
        return a <= du && b <= dv && M.norm(M.sub(c, p0)) < st.r;
      }
      for (var i = 0; i < G.length - 1; i++) for (var j = 0; j < G[i].length - 1; j++) {
        var a = G[i][j], b = G[i + 1][j], c = G[i + 1][j + 1], d = G[i][j + 1];
        var n = M.cross(M.sub(b.p, a.p), M.sub(d.p, a.p)), nn = M.norm(n);
        var shade = nn > 0 ? Math.abs(M.dot(n, LIGHT)) / nn : 0.5;
        var cen = [(a.p[0] + c.p[0]) / 2, (a.p[1] + c.p[1]) / 2, (a.p[2] + c.p[2]) / 2];
        var hit = inPatch(cen, (a.u + c.u) / 2, (a.v + c.v) / 2), k = 0.42 + 0.58 * shade;
        var base = hit ? [40, 165, 140] : [168, 184, 210];
        api.quad([a.p, b.p, c.p, d.p], P.rgbStr([base[0] * k, base[1] * k, base[2] * k]), hit ? null : 'rgba(70,83,107,0.25)');
      }
      if (st.plane) {
        var s = Math.min(Math.max(st.r * 1.3, 0.7), V.radius * 0.85), m = 6;
        for (var x = 0; x < m; x++) for (var y = 0; y < m; y++) {
          var a0 = -s + 2 * s * x / m, a1 = -s + 2 * s * (x + 1) / m, b0 = -s + 2 * s * y / m, b1 = -s + 2 * s * (y + 1) / m;
          var q = function (A, B) { return M.add(p0, M.add(M.scale(g.e1, A), M.scale(g.e2, B))); };
          api.quad([q(a0, b0), q(a1, b0), q(a1, b1), q(a0, b1)], 'rgba(224,112,26,0.22)', 'rgba(224,112,26,0.25)');
        }
        var corner = function (A, B) { return M.add(p0, M.add(M.scale(g.e1, A), M.scale(g.e2, B))); };
        api.polyline([corner(-s, -s), corner(s, -s), corner(s, s), corner(-s, s), corner(-s, -s)], 'rgba(200,90,10,0.9)', 1.6, { top: true });
        api.line(p0, M.add(p0, M.scale(g.e1, s * 0.7)), '#1d5fd6', 2.6, { top: true });
        api.line(p0, M.add(p0, M.scale(g.e2, s * 0.7)), '#7b3fe4', 2.6, { top: true });
        api.line(p0, M.add(p0, M.scale(g.n, s * 0.8)), '#138a5a', 2.2, { top: true, dash: [4, 3] });
        api.label(M.add(p0, M.scale(g.e1, s * 0.7)), '∂/∂' + ({ sphere: 'θ', swiss: 't', saddle: 'x', torus: 'θ' })[st.key], '#1d5fd6');
        api.label(M.add(p0, M.scale(g.n, s * 0.8)), 'normal', '#138a5a');
      }
      api.point(p0, 5.5, '#14213d', { top: true, stroke: '#fff', lw: 2 });
      api.label(p0, 'p', '#14213d');
    };

    V.onClick = function (x, y) {
      var flat = [], idx = [];
      st.mesh.forEach(function (row, i) { row.forEach(function (c, j) { flat.push(c.p); idx.push([i, j]); }); });
      var k = P.pick(V, flat, x, y, 16);
      if (k < 0) return;
      var c = st.mesh[idx[k][0]][idx[k][1]], S = M.SURFACES[st.key];
      st.u = Math.min(Math.max(c.u, S.u[0] + 1e-3), S.u[1] - 1e-3);
      st.v = Math.min(Math.max(c.v, S.v[0] + 1e-3), S.v[1] - 1e-3);
      update(true);
    };

    /* ---- the chart inset: the patch seen from above its tangent plane ---- */
    function drawChart() {
      var cv = $('#s1-chart'), c = P.setupCanvas(cv), ctx = c.ctx, W = c.w, H = c.h, r = st.r, pts = st.pt.pts;
      ctx.clearRect(0, 0, W, H);
      var sc = (Math.min(W, H) / 2 - 10) / r, cx = W / 2, cy = H / 2;
      ctx.strokeStyle = '#dbe2ee'; ctx.lineWidth = 1;
      ctx.beginPath(); ctx.moveTo(cx - r * sc, cy); ctx.lineTo(cx + r * sc, cy); ctx.moveTo(cx, cy - r * sc); ctx.lineTo(cx, cy + r * sc); ctx.stroke();
      var norm = 0.5 * r, dot = Math.max(1.6, sc * r / 26);
      pts.forEach(function (q) {
        var t = Math.max(-1, Math.min(1, q[2] / norm)), a = Math.abs(t);
        var col = t >= 0 ? [255 - a * (255 - 224), 255 - a * (255 - 112), 255 - a * (255 - 26)] : [255 - a * (255 - 29), 255 - a * (255 - 95), 255 - a * (255 - 214)];
        ctx.fillStyle = P.rgbStr(col); ctx.fillRect(cx + q[0] * sc - dot / 2, cy - q[1] * sc - dot / 2, dot, dot);
      });
      ctx.strokeStyle = '#7b879c'; ctx.setLineDash([4, 3]);
      ctx.beginPath(); ctx.arc(cx, cy, r * sc, 0, 2 * Math.PI); ctx.stroke(); ctx.setLineDash([]);
      ctx.fillStyle = '#14213d'; ctx.beginPath(); ctx.arc(cx, cy, 3.5, 0, 2 * Math.PI); ctx.fill();
      ctx.font = '500 11px "IBM Plex Mono", monospace'; ctx.fillStyle = '#7b879c';
      ctx.fillText('r = ' + fmt(r), 6, 14); ctx.fillText('color scale ±' + fmt(norm), 6, H - 6);
    }

    /* ---- δ/r as a function of r ---- */
    var curve = [];
    function drawCurve() {
      var S = M.SURFACES[st.key]; curve = [];
      for (var i = 1; i <= 36; i++) { var r = 3 * i / 36, p = M.patch(S, st.u, st.v, r, 23); curve.push([r, p.delta / r]); }
      drawCurveMarker();
    }
    function drawCurveMarker() {
      var cv = $('#s1-curve'), c = P.setupCanvas(cv), ctx = c.ctx, W = c.w, H = c.h, L = 34, B = 20, T = 8, R = 8;
      ctx.clearRect(0, 0, W, H);
      var ymax = Math.max(0.2, Math.ceil(Math.max.apply(null, curve.map(function (q) { return q[1]; })) * 10) / 10);
      var sx = function (r) { return L + (r / 3) * (W - L - R); }, sy = function (v) { return H - B - (v / ymax) * (H - B - T); };
      ctx.strokeStyle = '#c3cede'; ctx.beginPath(); ctx.moveTo(L, T); ctx.lineTo(L, H - B); ctx.lineTo(W - R, H - B); ctx.stroke();
      ctx.font = '400 10.5px "IBM Plex Mono", monospace'; ctx.fillStyle = '#7b879c';
      ctx.fillText('0', L - 12, H - B + 3); ctx.fillText(ymax.toFixed(1), 2, T + 8); ctx.fillText('r →', W - R - 22, H - 5); ctx.fillText('3', sx(3) - 4, H - B + 13);
      ctx.strokeStyle = '#1d5fd6'; ctx.lineWidth = 2; ctx.beginPath();
      curve.forEach(function (q, i) { var x = sx(q[0]), y = sy(q[1]); if (i) ctx.lineTo(x, y); else ctx.moveTo(sx(0), sy(0)), ctx.lineTo(x, y); });
      ctx.stroke();
      var now = st.pt.delta / st.r;
      ctx.strokeStyle = '#e0701a'; ctx.setLineDash([3, 3]); ctx.beginPath(); ctx.moveTo(sx(st.r), T); ctx.lineTo(sx(st.r), H - B); ctx.stroke(); ctx.setLineDash([]);
      ctx.fillStyle = '#e0701a'; ctx.beginPath(); ctx.arc(sx(st.r), sy(now), 4.5, 0, 2 * Math.PI); ctx.fill();
    }

    $('#s1-r').addEventListener('input', function (e) { st.r = +e.target.value; update(false); });
    $('#s1-plane').addEventListener('change', function (e) { st.plane = e.target.checked; V.render(); });
    $('#s1-zoom').addEventListener('click', function () { V.animateTo(st.pt.geom.p, Math.min(12, Math.max(1.2, V.radius / (st.r * 1.25)))); });
    $('#s1-home').addEventListener('click', function () { V.home(); });
    $('#s1-spin').addEventListener('click', function (e) { var on = !V.spinning(); V.setSpin(on); e.target.setAttribute('aria-pressed', on); });
    window.addEventListener('resize', function () { drawChart(); drawCurveMarker(); });
    setSurface('sphere');
  })();

  /* ================================================= 2. manifold hypothesis */
  (function () {
    var DIGITS = {
      '2': function (c) { c.moveTo(-0.55, -0.42); c.bezierCurveTo(-0.5, -0.95, 0.55, -0.95, 0.5, -0.33); c.bezierCurveTo(0.45, 0.02, -0.15, 0.3, -0.6, 0.75); c.lineTo(0.62, 0.72); },
      '3': function (c) { c.moveTo(-0.5, -0.68); c.bezierCurveTo(-0.1, -1.0, 0.6, -0.85, 0.45, -0.36); c.bezierCurveTo(0.36, -0.07, 0.02, -0.03, -0.15, -0.03); c.bezierCurveTo(0.3, -0.03, 0.66, 0.2, 0.5, 0.55); c.bezierCurveTo(0.35, 0.9, -0.25, 0.95, -0.56, 0.62); },
      '7': function (c) { c.moveTo(-0.56, -0.74); c.lineTo(0.56, -0.74); c.bezierCurveTo(0.3, -0.3, 0.02, 0.3, -0.12, 0.86); c.moveTo(-0.22, 0.02); c.lineTo(0.38, 0.02); }
    };
    var VARS = [
      { key: 'rot', label: 'Rotation', min: -180, max: 180, step: 1, def: 0, unit: '°', d: 0 },
      { key: 'tx', label: 'Shift x', min: -6, max: 6, step: 0.25, def: 0, unit: ' px', d: 2 },
      { key: 'ty', label: 'Shift y', min: -6, max: 6, step: 0.25, def: 0, unit: ' px', d: 2 },
      { key: 'scale', label: 'Scale', min: 0.6, max: 1.4, step: 0.01, def: 1, unit: '×', d: 2 },
      { key: 'thick', label: 'Stroke width', min: 1.2, max: 3.6, step: 0.05, def: 2.2, unit: ' px', d: 2 }
    ];
    var st = { digit: '2', p: {}, free: { rot: true }, noise: false, noiseVec: null, fam: null, trail: [], sweeping: 0 };
    VARS.forEach(function (v) { st.p[v.key] = v.def; });

    var off = document.createElement('canvas'); off.width = off.height = 28;
    var octx = off.getContext('2d', { willReadFrequently: true });
    function render(p, d) {
      octx.setTransform(1, 0, 0, 1, 0, 0); octx.fillStyle = '#000'; octx.fillRect(0, 0, 28, 28);
      octx.translate(14 + p.tx, 14 + p.ty); octx.rotate(p.rot * Math.PI / 180);
      var s = 10 * p.scale; octx.scale(s, s);
      octx.lineWidth = p.thick / s; octx.lineCap = 'round'; octx.lineJoin = 'round'; octx.strokeStyle = '#fff';
      octx.beginPath(); DIGITS[d](octx); octx.stroke();
      var data = octx.getImageData(0, 0, 28, 28).data, out = new Float64Array(784);
      for (var i = 0; i < 784; i++) out[i] = data[4 * i] / 255;
      return out;
    }

    /* sliders */
    var box = $('#s2-sliders');
    VARS.forEach(function (v) {
      var w = document.createElement('div'); w.className = 'ctrl';
      w.innerHTML = '<label for="s2-' + v.key + '">' + v.label + ' <output class="num" id="s2-' + v.key + 'v"></output></label>' +
        '<input type="range" id="s2-' + v.key + '" min="' + v.min + '" max="' + v.max + '" step="' + v.step + '" value="' + v.def + '">';
      box.appendChild(w);
      $('#s2-' + v.key, w).addEventListener('input', function (e) { st.p[v.key] = +e.target.value; st.noise = false; syncNoiseBtn(); draw(true); });
    });
    seg($('#s2-digit'), [{ key: '2', label: 'Digit 2' }, { key: '3', label: 'Digit 3' }, { key: '7', label: 'Digit 7' }], st.digit, function (k) { st.digit = k; st.trail = []; buildFamily(); draw(false); });

    var chips = $('#s2-free');
    VARS.forEach(function (v) {
      var l = document.createElement('label'); l.className = 'chip';
      l.innerHTML = '<input type="checkbox" ' + (st.free[v.key] ? 'checked' : '') + '> ' + v.label;
      $('input', l).addEventListener('change', function (e) {
        st.free[v.key] = e.target.checked;
        if (!Object.keys(st.free).some(function (k) { return st.free[k]; })) { st.free[v.key] = true; e.target.checked = true; }
        st.trail = []; buildFamily(); draw(false);
      });
      chips.appendChild(l);
    });

    /* the family of images and its PCA shadow */
    var famPlot = P.Scatter2D($('#s2-pca'), { r: 2.6, pad: 18, fadeOnHover: false });
    var hoverIdx = -1, hoverPos = null;
    function buildFamily() {
      var r = M.rng(5 + st.digit.charCodeAt(0)), N = 300, X = [], params = [];
      var freeKeys = VARS.filter(function (v) { return st.free[v.key]; });
      for (var i = 0; i < N; i++) {
        var p = {}; VARS.forEach(function (v) { p[v.key] = st.free[v.key] ? v.min + (v.max - v.min) * r() : v.def; });
        params.push(p); X.push(render(p, st.digit));
      }
      var pc = M.dualPCA(X, 2), first = freeKeys[0];
      var colors = params.map(function (p) {
        var t = (p[first.key] - first.min) / (first.max - first.min);
        return first.key === 'rot' ? P.sinebow(t) : P.viridis(t);
      });
      var b = [Infinity, -Infinity, Infinity, -Infinity];
      pc.Z.forEach(function (z) { b[0] = Math.min(b[0], z[0]); b[1] = Math.max(b[1], z[0]); b[2] = Math.min(b[2], z[1]); b[3] = Math.max(b[3], z[1]); });
      var px = 0.12 * (b[1] - b[0]), py = 0.12 * (b[3] - b[2]);
      famPlot.bounds = [b[0] - px, b[1] + px, b[2] - py, b[3] + py];
      st.fam = { X: X, params: params, pc: pc, L: freeKeys.length, colorBy: first };
      famPlot.set(pc.Z, colors, false);
      drawSpectrum();
    }

    famPlot.overlay = function (ctx, toS) {
      if (st.trail.length > 1) {
        ctx.strokeStyle = 'rgba(20,33,61,0.55)'; ctx.lineWidth = 1.6; ctx.beginPath();
        st.trail.forEach(function (z, i) { var q = toS(z[0], z[1]); if (i) ctx.lineTo(q[0], q[1]); else ctx.moveTo(q[0], q[1]); });
        ctx.stroke();
      }
      if (!st.noise && st.cur) {
        var q = toS(st.cur[0], st.cur[1]);
        ctx.beginPath(); ctx.arc(q[0], q[1], 8, 0, 2 * Math.PI); ctx.fillStyle = '#e0701a'; ctx.fill();
        ctx.lineWidth = 2.5; ctx.strokeStyle = '#fff'; ctx.stroke();
      }
      if (hoverIdx >= 0 && hoverPos) {
        var img = st.fam.X[hoverIdx], s = 2, x0 = Math.min(hoverPos[0] + 12, ctx.canvas.clientWidth - 62), y0 = Math.max(hoverPos[1] - 64, 4);
        ctx.fillStyle = '#fff'; ctx.fillRect(x0 - 3, y0 - 3, 28 * s + 6, 28 * s + 6);
        ctx.strokeStyle = '#c3cede'; ctx.strokeRect(x0 - 3, y0 - 3, 28 * s + 6, 28 * s + 6);
        for (var i = 0; i < 784; i++) { if (img[i] < 0.03) continue; var g = 255 - img[i] * 235; ctx.fillStyle = 'rgb(' + (g * 0.95 | 0) + ',' + (g * 0.97 | 0) + ',' + g + ')'; ctx.fillRect(x0 + (i % 28) * s, y0 + ((i / 28) | 0) * s, s, s); }
      }
    };
    famPlot.onHover = function (i, pos) { hoverIdx = i; hoverPos = pos; famPlot.render(); };

    function drawSpectrum() {
      var cv = $('#s2-spec'), c = P.setupCanvas(cv), ctx = c.ctx, W = c.w, H = c.h, L = 34, B = 20, T = 8, R = 8;
      var spec = st.fam.pc.spectrum, tot = spec.reduce(function (a, b) { return a + b; }, 0) || 1, cum = [], s = 0, K = 40;
      for (var i = 0; i < K; i++) { s += spec[i] || 0; cum.push(s / tot); }
      var m95 = cum.findIndex(function (v) { return v >= 0.95; }) + 1 || K + '+';
      ctx.clearRect(0, 0, W, H);
      var sx = function (k) { return L + (k / K) * (W - L - R); }, sy = function (v) { return H - B - v * (H - B - T); };
      ctx.strokeStyle = '#c3cede'; ctx.beginPath(); ctx.moveTo(L, T); ctx.lineTo(L, H - B); ctx.lineTo(W - R, H - B); ctx.stroke();
      ctx.font = '400 10.5px "IBM Plex Mono", monospace'; ctx.fillStyle = '#7b879c';
      ctx.fillText('100%', 0, T + 8); ctx.fillText('0', L - 10, H - B + 3); ctx.fillText('# components →', W - R - 96, H - 5);
      ctx.strokeStyle = '#7b879c'; ctx.setLineDash([3, 3]); ctx.beginPath(); ctx.moveTo(L, sy(0.95)); ctx.lineTo(W - R, sy(0.95)); ctx.stroke();
      ctx.strokeStyle = '#e0701a'; ctx.beginPath(); ctx.moveTo(sx(st.fam.L), T); ctx.lineTo(sx(st.fam.L), H - B); ctx.stroke(); ctx.setLineDash([]);
      ctx.fillStyle = '#7b879c'; ctx.fillText('95%', W - R - 26, sy(0.95) - 4);
      ctx.fillStyle = '#e0701a'; ctx.fillText('L = ' + st.fam.L, sx(st.fam.L) + 4, T + 10);
      ctx.strokeStyle = '#1d5fd6'; ctx.lineWidth = 2; ctx.beginPath(); ctx.moveTo(sx(0), sy(0));
      cum.forEach(function (v, k) { ctx.lineTo(sx(k + 1), sy(v)); }); ctx.stroke();
      $('#s2-spec-note').innerHTML = 'Cumulative variance explained by linear PCA. The family has <b>L = ' + st.fam.L + '</b> free variable' + (st.fam.L > 1 ? 's' : '') +
        ', but PCA needs <b>' + m95 + '</b> components to reach 95%. The manifold is low-dimensional but <i>curved</i>, so no flat subspace of dimension ' + st.fam.L +
        ' holds it. That gap is the case for nonlinear methods: intrinsic ' + st.fam.L + ' &lt; linear ' + m95 + ' &lt; ambient 784.';
    }

    /* main image + vector strip */
    var vecCanvas = document.createElement('canvas'); vecCanvas.width = 784; vecCanvas.height = 1;
    function draw(addTrail) {
      VARS.forEach(function (v) { $('#s2-' + v.key + 'v').textContent = st.p[v.key].toFixed(v.d) + v.unit; $('#s2-' + v.key).value = st.p[v.key]; });
      var x = st.noise ? st.noiseVec : render(st.p, st.digit);
      var cv = $('#s2-img'), ctx = cv.getContext('2d'), cell = cv.width / 28;
      for (var i = 0; i < 784; i++) {
        var g = 255 - x[i] * 235;
        ctx.fillStyle = 'rgb(' + (g * 0.95 | 0) + ',' + (g * 0.97 | 0) + ',' + (g | 0) + ')';
        ctx.fillRect((i % 28) * cell, ((i / 28) | 0) * cell, cell, cell);
      }
      ctx.strokeStyle = 'rgba(123,135,156,0.18)'; ctx.lineWidth = 1; ctx.beginPath();
      for (var k = 1; k < 28; k++) { ctx.moveTo(k * cell + 0.5, 0); ctx.lineTo(k * cell + 0.5, cv.height); ctx.moveTo(0, k * cell + 0.5); ctx.lineTo(cv.width, k * cell + 0.5); }
      ctx.stroke();
      var vctx = vecCanvas.getContext('2d'), id = vctx.createImageData(784, 1);
      for (i = 0; i < 784; i++) { var gg = 255 - x[i] * 235; id.data[4 * i] = gg * 0.95; id.data[4 * i + 1] = gg * 0.97; id.data[4 * i + 2] = gg; id.data[4 * i + 3] = 255; }
      vctx.putImageData(id, 0, 0);
      var sc = P.setupCanvas($('#s2-vec')), c2 = sc.ctx;
      c2.clearRect(0, 0, sc.w, sc.h); c2.imageSmoothingEnabled = false;
      c2.drawImage(vecCanvas, 0, 0, sc.w, sc.h - 14);
      c2.fillStyle = '#7b879c'; c2.font = '400 10px "IBM Plex Mono", monospace';
      for (k = 0; k <= 28; k += 7) { var xx = Math.min(sc.w - 1, k * 28 * sc.w / 784); c2.fillRect(xx, sc.h - 14, 1, 4); c2.fillText('row ' + k, Math.min(xx + 2, sc.w - 40), sc.h - 2); }
      if (!st.noise && st.fam) {
        st.cur = st.fam.pc.project(x);
        if (addTrail) { st.trail.push(st.cur); if (st.trail.length > 600) st.trail.shift(); }
      }
      famPlot.render();
    }

    function syncNoiseBtn() { $('#s2-noise').setAttribute('aria-pressed', st.noise); $('#s2-noise-note').hidden = !st.noise; }
    $('#s2-noise').addEventListener('click', function () {
      st.noise = !st.noise;
      if (st.noise) { var r = M.rng((Math.random() * 1e9) | 0); st.noiseVec = new Float64Array(784).map(function () { return r(); }); }
      syncNoiseBtn(); draw(false);
    });
    $('#s2-reset').addEventListener('click', function () { VARS.forEach(function (v) { st.p[v.key] = v.def; }); st.noise = false; syncNoiseBtn(); draw(false); });
    $('#s2-clear').addEventListener('click', function () { st.trail = []; famPlot.render(); });
    $('#s2-sweep').addEventListener('click', function () {
      if (st.sweeping) { cancelAnimationFrame(st.sweeping); st.sweeping = 0; return; }
      st.noise = false; syncNoiseBtn(); st.trail = [];
      var t0 = performance.now(), dur = 4200, btn = $('#s2-sweep');
      btn.textContent = 'Stop';
      (function step(t) {
        var a = Math.min(1, (t - t0) / dur);
        st.p.rot = Math.round(-180 + 360 * a); draw(true);
        if (a < 1) st.sweeping = requestAnimationFrame(step);
        else { st.sweeping = 0; btn.innerHTML = 'Sweep rotation &minus;180&deg; &rarr; 180&deg;'; }
      })(t0);
    });
    window.addEventListener('resize', function () { draw(false); if (st.fam) drawSpectrum(); });
    buildFamily(); draw(false);
  })();

  /* ============================================== 3. geodesic vs Euclidean */
  (function () {
    var V = P.Viewer3D($('#s3-canvas'), { yaw: 0.35, pitch: 1.1 });
    var flat = P.Scatter2D($('#s3-flat'), { r: 2.4, pad: 14, fadeOnHover: false, bounds: [M.ROLL.s0, M.ROLL.s1, 0, M.ROLL.height] });
    var st = { n: 500, k: 7, noise: 0, A: -1, B: -1, next: 'A', edges: false, ds: null, D2: null, g: null, sp: null, short: null, nShort: 0 };
    var COL = { euc: '#e0701a', graph: '#1d5fd6', tru: '#138a5a', bad: '#d6293a' };

    function isShort(u, v) { var n = st.n, e = Math.sqrt(st.D2[u * n + v]), t = st.ds.TD[u * n + v]; return t > 2 * e && t - e > 2; }

    function rebuildData() {
      st.ds = M.makeData('swiss', st.n, st.noise, 11); st.D2 = M.sqdist(st.ds.X);
      st.colors = st.ds.c.map(function (c) { return P.viridis(c); });
      if (!st.boundsSet) { V.boundsOf(st.ds.X); st.boundsSet = true; }
      flat.set(st.ds.T, st.colors, false);
      rebuildGraph();
    }
    function rebuildGraph() {
      st.g = M.knnGraph(st.D2, st.n, st.k); st.comp = M.components(st.g.adj, st.n);
      var c = 0; st.shortEdges = [];
      st.g.adj.forEach(function (e, u) { e.to.forEach(function (v) { if (u < v && isShort(u, v)) { c++; st.shortEdges.push([u, v]); } }); });
      st.nShort = c;
      if (st.A < 0 || st.A >= st.n || st.B >= st.n) presetLayers(); else solve();
    }
    function solve() {
      if (st.A < 0 || st.B < 0) { st.sp = null; refresh(); return; }
      st.sp = M.shortestPath(st.g.adj, st.n, st.A, st.B);
      refresh();
    }
    function presetLayers() {
      var ds = st.ds, n = st.n, best = -1, bd = Infinity, target = M.ROLL.s0 + 0.6 * (M.ROLL.s1 - M.ROLL.s0);
      for (var i = 0; i < n; i++) { var d = Math.abs(ds.T[i][0] - target) + Math.abs(ds.T[i][1] - M.ROLL.height / 2); if (d < bd) { bd = d; best = i; } }
      st.A = best; bd = Infinity;
      for (var j = 0; j < n; j++) { if (ds.TD[best * n + j] < 15) continue; var e = st.D2[best * n + j]; if (e < bd) { bd = e; st.B = j; } }
      st.next = 'A'; solve();
    }
    function randomPair() {
      var r = Math.random; st.A = (r() * st.n) | 0; do { st.B = (r() * st.n) | 0; } while (st.B === st.A); st.next = 'A'; solve();
    }
    function place(i) {
      if (i < 0) return;
      if (st.next === 'A') { st.A = i; st.next = 'B'; } else { if (i === st.A) return; st.B = i; st.next = 'A'; }
      solve();
    }

    function refresh() {
      $('#s3-kv').textContent = st.k; $('#s3-nv').textContent = st.noise.toFixed(2); $('#s3-Nv').textContent = st.n;
      $('#s3-next').innerHTML = 'Next click places <b class="' + (st.next === 'A' ? 'pa' : 'pb') + '">' + st.next + '</b>';
      var n = st.n, A = st.A, B = st.B, rows = '', warn = [];
      if (A >= 0 && B >= 0) {
        var euc = Math.sqrt(st.D2[A * n + B]), tru = st.ds.TD[A * n + B], gd = st.sp.dist, hops = st.sp.path.length - 1;
        st.pathShort = [];
        for (var i = 0; i + 1 < st.sp.path.length; i++) if (isShort(st.sp.path[i], st.sp.path[i + 1])) st.pathShort.push(i);
        rows =
          row(COL.euc, 'Euclidean', '‖x<sub>A</sub> − x<sub>B</sub>‖', fmt(euc)) +
          row(COL.graph, 'Graph geodesic', isFinite(gd) ? hops + ' hops' : 'no path', fmt(gd)) +
          row(COL.tru, 'True geodesic', 'straight line in (s, h)', fmt(tru)) +
          '<div class="dt-sep"></div>' +
          row(null, 'Euclidean ÷ true', '', fmt(euc / tru)) +
          row(null, 'Graph ÷ true', '', isFinite(gd) ? fmt(gd / tru) : '—');
        if (!isFinite(gd)) warn.push('<b>No path.</b> A and B sit in different pieces of the graph (' + st.comp.count + ' pieces in total). Raise <i>k</i>.');
        else if (st.pathShort.length) warn.push('<b>Short circuit.</b> This path uses ' + st.pathShort.length + ' edge' + (st.pathShort.length > 1 ? 's that jump' : ' that jumps') + ' between layers (red), so the graph distance badly underestimates the geodesic. Lower <i>k</i> or the noise.');
        else if (gd / tru > 1.15) warn.push('The graph path zig-zags, so it runs a little long. More points or a slightly larger <i>k</i> straighten it.');
      }
      if (st.comp.count > 1 && !(warn.length && warn[0].indexOf('No path') >= 0)) warn.push('The graph has ' + st.comp.count + ' disconnected pieces.');
      if (st.nShort && !(st.pathShort && st.pathShort.length)) warn.push(st.nShort + ' short-circuit edge' + (st.nShort > 1 ? 's' : '') + ' exist in the graph. Isomap would be corrupted even though this particular path avoids them.');
      $('#s3-table').innerHTML = rows || '<p class="small muted">Click two points.</p>';
      var w = $('#s3-warn'); w.hidden = !warn.length; w.innerHTML = warn.join('<br>');
      V.render(); flat.render();
    }
    function row(color, name, sub, val) {
      return '<div class="dt-row"><span class="dt-name"><span class="dt-main">' + (color ? '<i class="dot" style="background:' + color + '"></i>' : '') + name + '</span>' + (sub ? '<small>' + sub + '</small>' : '') + '</span><b class="num">' + val + '</b></div>';
    }

    V.scene = function (api) {
      var X = st.ds.X, n = st.n;
      for (var i = 0; i < n; i++) api.point(X[i], 2.4, st.colors[i]);
      if (st.edges) st.g.adj.forEach(function (e, u) { e.to.forEach(function (v) { if (u < v && !isShort(u, v)) api.line(X[u], X[v], 'rgba(70,83,107,0.22)', 0.8); }); });
      st.shortEdges.forEach(function (e) { if (st.edges) api.line(X[e[0]], X[e[1]], 'rgba(214,41,58,0.75)', 1.4); });
      if (st.A >= 0 && st.B >= 0) {
        var a = st.ds.T[st.A], b = st.ds.T[st.B], geo = [];
        for (var k = 0; k <= 80; k++) { var t = k / 80; geo.push(M.rollPoint(a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1]))); }
        api.polyline(geo, COL.tru, 2.4, { top: true, dash: [6, 4] });
        if (st.sp && st.sp.path.length) for (var j = 0; j + 1 < st.sp.path.length; j++)
          api.line(X[st.sp.path[j]], X[st.sp.path[j + 1]], st.pathShort && st.pathShort.indexOf(j) >= 0 ? COL.bad : COL.graph, 3.2, { top: true });
        api.line(X[st.A], X[st.B], COL.euc, 3, { top: true });
      }
      if (st.A >= 0) { api.point(X[st.A], 7, COL.euc, { top: true, stroke: '#fff', lw: 2 }); api.label(X[st.A], 'A', '#14213d'); }
      if (st.B >= 0) { api.point(X[st.B], 7, COL.euc, { top: true, stroke: '#fff', lw: 2 }); api.label(X[st.B], 'B', '#14213d'); }
    };
    V.onClick = function (x, y) { place(P.pick(V, st.ds.X, x, y, 10)); };

    flat.overlay = function (ctx, toS) {
      var T = st.ds.T;
      if (st.edges) {
        ctx.lineWidth = 0.6; ctx.strokeStyle = 'rgba(70,83,107,0.2)'; ctx.beginPath();
        st.g.adj.forEach(function (e, u) { e.to.forEach(function (v) { if (u < v && !isShort(u, v)) { var p = toS(T[u][0], T[u][1]), q = toS(T[v][0], T[v][1]); ctx.moveTo(p[0], p[1]); ctx.lineTo(q[0], q[1]); } }); });
        ctx.stroke();
        ctx.strokeStyle = 'rgba(214,41,58,0.55)'; ctx.lineWidth = 1; ctx.beginPath();
        st.shortEdges.forEach(function (e) { var p = toS(T[e[0]][0], T[e[0]][1]), q = toS(T[e[1]][0], T[e[1]][1]); ctx.moveTo(p[0], p[1]); ctx.lineTo(q[0], q[1]); });
        ctx.stroke();
      }
      if (st.A < 0 || st.B < 0) return;
      var a = toS(T[st.A][0], T[st.A][1]), b = toS(T[st.B][0], T[st.B][1]);
      ctx.setLineDash([6, 4]); ctx.strokeStyle = COL.tru; ctx.lineWidth = 2.4; ctx.beginPath(); ctx.moveTo(a[0], a[1]); ctx.lineTo(b[0], b[1]); ctx.stroke(); ctx.setLineDash([]);
      if (st.sp && st.sp.path.length) {
        ctx.lineWidth = 3;
        for (var j = 0; j + 1 < st.sp.path.length; j++) {
          var p = toS(T[st.sp.path[j]][0], T[st.sp.path[j]][1]), q = toS(T[st.sp.path[j + 1]][0], T[st.sp.path[j + 1]][1]);
          ctx.strokeStyle = st.pathShort && st.pathShort.indexOf(j) >= 0 ? COL.bad : COL.graph;
          ctx.beginPath(); ctx.moveTo(p[0], p[1]); ctx.lineTo(q[0], q[1]); ctx.stroke();
        }
      }
      [[a, 'A'], [b, 'B']].forEach(function (m) {
        ctx.beginPath(); ctx.arc(m[0][0], m[0][1], 7, 0, 2 * Math.PI); ctx.fillStyle = COL.euc; ctx.fill(); ctx.strokeStyle = '#fff'; ctx.lineWidth = 2; ctx.stroke();
        ctx.font = '600 12px "IBM Plex Mono", monospace'; ctx.fillStyle = '#14213d'; ctx.fillText(m[1], m[0][0] + 9, m[0][1] - 8);
      });
      ctx.font = '400 10.5px "IBM Plex Mono", monospace'; ctx.fillStyle = '#7b879c';
      ctx.fillText('s (arc length along the roll) →', 14, ctx.canvas.clientHeight - 4);
    };
    flat.onClick = function (i) { place(i); };

    $('#s3-k').addEventListener('input', function (e) { st.k = +e.target.value; rebuildGraph(); });
    var tn = 0;
    function later(fn) { clearTimeout(tn); tn = setTimeout(fn, 120); }
    $('#s3-noise').addEventListener('input', function (e) { st.noise = +e.target.value; $('#s3-nv').textContent = st.noise.toFixed(2); later(function () { var a = st.A, b = st.B; rebuildData(); st.A = a; st.B = b; solve(); }); });
    $('#s3-n').addEventListener('input', function (e) { st.n = +e.target.value; $('#s3-Nv').textContent = st.n; later(function () { st.A = -1; rebuildData(); }); });
    $('#s3-edges').addEventListener('change', function (e) { st.edges = e.target.checked; refresh(); });
    $('#s3-layers').addEventListener('click', presetLayers);
    $('#s3-rand').addEventListener('click', randomPair);
    /* The unrolled sheet is ~9× longer than tall; size its canvas to that
       aspect so the chart stays isometric without wasted space. */
    function sizeFlat() {
      var cv = $('#s3-flat'), w = cv.clientWidth, asp = M.ROLL.height / (M.ROLL.s1 - M.ROLL.s0);
      cv.style.height = Math.max(90, Math.round((w - 28) * asp + 40)) + 'px';
      flat.render();
    }
    window.addEventListener('resize', sizeFlat);
    sizeFlat();
    rebuildData();
  })();
})();
