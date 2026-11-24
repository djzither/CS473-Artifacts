/* ============================================================================
   PCA Interactive Learning Lab — shared drawing helpers and tab navigation.
   ========================================================================= */

(function () {
  'use strict';
  var NS = 'http://www.w3.org/2000/svg';
  var UI = {};

  UI.$ = function (s, r) { return (r || document).querySelector(s); };
  UI.$$ = function (s, r) { return Array.prototype.slice.call((r || document).querySelectorAll(s)); };

  UI.el = function (tag, attrs, parent) {
    var e = document.createElementNS(NS, tag);
    for (var k in attrs) if (attrs[k] !== undefined && attrs[k] !== null) e.setAttribute(k, attrs[k]);
    if (parent) parent.appendChild(e);
    return e;
  };
  UI.clear = function (n) { while (n.firstChild) n.removeChild(n.firstChild); };

  UI.esc = function (s) { return String(s).replace(/&/g, '&amp;').replace(/"/g, '&quot;').replace(/</g, '&lt;'); };
  UI.tx = function (s) { return '<span class="tex" data-tex="' + UI.esc(s) + '"></span>'; };
  UI.td = function (s) { return '<div class="tex-block" data-tex="' + UI.esc(s) + '"></div>'; };
  UI.tex = function (node, src, display) {
    if (window.katex) { try { katex.render(src, node, { throwOnError: false, displayMode: !!display }); return; } catch (e) { /* fall through */ } }
    node.textContent = src; node.classList.add('tex-raw');
  };
  UI.renderTex = function (root) {
    UI.$$('[data-tex]', root || document).forEach(function (n) {
      if (n.getAttribute('data-done') === n.getAttribute('data-tex')) return;
      UI.tex(n, n.getAttribute('data-tex'), n.classList.contains('tex-block'));
      n.setAttribute('data-done', n.getAttribute('data-tex'));
    });
  };
  /* set innerHTML and render any TeX inside */
  UI.html = function (node, h) { node.innerHTML = h; UI.renderTex(node); };

  UI.fmt = function (x, d) {
    if (!isFinite(x)) return x > 0 ? '∞' : x < 0 ? '−∞' : '—';
    d = d === undefined ? 3 : d;
    if (Math.abs(x) >= 1e5 || (Math.abs(x) < 1e-3 && x !== 0 && Math.abs(x) > 1e-12)) return x.toExponential(2).replace('e+', 'e');
    var v = +x.toFixed(d); if (Object.is(v, -0) || Math.abs(v) < Math.pow(10, -d) / 2) v = 0;
    return String(v);
  };
  UI.fx = function (x, d) { d = d === undefined ? 3 : d; var v = Math.abs(x) < 0.5 * Math.pow(10, -d) ? 0 : x; return v.toFixed(d); };
  UI.big = function (n) {
    if (n < 1e4) return String(Math.round(n));
    return Math.round(n).toLocaleString('en-US');
  };
  UI.sci = function (n) {
    if (n < 1e6) return UI.big(n);
    var e = Math.floor(Math.log10(n)), m = n / Math.pow(10, e);
    return m.toFixed(2) + ' × 10<sup>' + e + '</sup>';
  };
  UI.bytes = function (n) {
    var u = ['B', 'KB', 'MB', 'GB', 'TB', 'PB', 'EB'], i = 0;
    while (n >= 1024 && i < u.length - 1) { n /= 1024; i++; }
    return (n >= 100 ? n.toFixed(0) : n.toFixed(1)) + ' ' + u[i];
  };

  /* nice tick values */
  UI.ticks = function (a, b, n) {
    n = n || 5; var span = b - a; if (span <= 0) return [a];
    var step = Math.pow(10, Math.floor(Math.log10(span / n))), err = n / span * step;
    if (err <= 0.15) step *= 10; else if (err <= 0.35) step *= 5; else if (err <= 0.75) step *= 2;
    var out = [], t = Math.ceil(a / step) * step;
    for (; t <= b + step * 1e-9; t += step) out.push(+t.toFixed(12));
    return out;
  };

  /* A plot frame with axes. opts: W,H, m{l,r,t,b}, x[a,b], y[a,b], xlabel, ylabel, equal, xt, yt (tick arrays), grid */
  UI.frame = function (svg, o) {
    UI.clear(svg);
    var m = o.m || { l: 46, r: 14, t: 14, b: 38 };
    var W = o.W, H = o.H, x0 = o.x[0], x1 = o.x[1], y0 = o.y[0], y1 = o.y[1];
    var pw = W - m.l - m.r, ph = H - m.t - m.b;
    if (o.equal) {
      var kx = pw / (x1 - x0), ky = ph / (y1 - y0), k = Math.min(kx, ky);
      var cx = (x0 + x1) / 2, cy = (y0 + y1) / 2;
      x0 = cx - pw / k / 2; x1 = cx + pw / k / 2; y0 = cy - ph / k / 2; y1 = cy + ph / k / 2;
    }
    var sx = function (v) { return m.l + (v - x0) / (x1 - x0) * pw; };
    var sy = function (v) { return m.t + (1 - (v - y0) / (y1 - y0)) * ph; };
    var g = UI.el('g', {}, svg);
    var xt = o.xt || UI.ticks(x0, x1, o.nx || 6), yt = o.yt || UI.ticks(y0, y1, o.ny || 5);
    if (o.grid !== false) {
      yt.forEach(function (v) { UI.el('line', { x1: m.l, x2: W - m.r, y1: sy(v), y2: sy(v), 'class': 'grid' }, g); });
    }
    UI.el('line', { x1: m.l, x2: W - m.r, y1: m.t + ph, y2: m.t + ph, 'class': 'ax' }, g);
    UI.el('line', { x1: m.l, x2: m.l, y1: m.t, y2: m.t + ph, 'class': 'ax' }, g);
    var xf = o.xfmt || function (v) { return UI.fmt(v, 3); }, yf = o.yfmt || function (v) { return UI.fmt(v, 3); };
    xt.forEach(function (v) { if (v < x0 - 1e-9 || v > x1 + 1e-9) return; UI.el('text', { x: sx(v), y: m.t + ph + 15, 'text-anchor': 'middle', 'class': 'tick' }, g).textContent = xf(v); });
    yt.forEach(function (v) { if (v < y0 - 1e-9 || v > y1 + 1e-9) return; UI.el('text', { x: m.l - 6, y: sy(v) + 4, 'text-anchor': 'end', 'class': 'tick' }, g).textContent = yf(v); });
    if (o.xlabel) UI.el('text', { x: m.l + pw / 2, y: H - 4, 'text-anchor': 'middle', 'class': 'axt' }, g).textContent = o.xlabel;
    if (o.ylabel) UI.el('text', { x: 12, y: m.t + ph / 2, 'text-anchor': 'middle', 'class': 'axt', transform: 'rotate(-90 12 ' + (m.t + ph / 2) + ')' }, g).textContent = o.ylabel;
    var clipId = 'c' + Math.random().toString(36).slice(2);
    var defs = UI.el('defs', {}, svg), cp = UI.el('clipPath', { id: clipId }, defs);
    UI.el('rect', { x: m.l, y: m.t, width: pw, height: ph }, cp);
    var plot = UI.el('g', { 'clip-path': 'url(#' + clipId + ')' }, svg);
    var top = UI.el('g', {}, svg);
    return { sx: sx, sy: sy, ix: function (px) { return x0 + (px - m.l) / pw * (x1 - x0); }, iy: function (py) { return y0 + (1 - (py - m.t) / ph) * (y1 - y0); },
      g: plot, top: top, m: m, pw: pw, ph: ph, x0: x0, x1: x1, y0: y0, y1: y1, W: W, H: H };
  };

  /* Convert a pointer event to SVG viewBox coordinates. */
  UI.svgPt = function (svg, ev) {
    var pt = svg.createSVGPoint(); pt.x = ev.clientX; pt.y = ev.clientY;
    var m = svg.getScreenCTM(); if (!m) return { x: 0, y: 0 };
    var p = pt.matrixTransform(m.inverse()); return { x: p.x, y: p.y };
  };
  /* Generic drag helper: onStart(ev) returns a move(pt) function or null. */
  UI.drag = function (svg, onStart) {
    var move = null;
    svg.addEventListener('pointerdown', function (ev) {
      move = onStart(ev, UI.svgPt(svg, ev));
      if (move) { svg.setPointerCapture(ev.pointerId); ev.preventDefault(); }
    });
    svg.addEventListener('pointermove', function (ev) { if (move) move(UI.svgPt(svg, ev)); });
    var end = function () { if (move && move.end) move.end(); move = null; };
    svg.addEventListener('pointerup', end); svg.addEventListener('pointercancel', end);
  };

  /* tooltips: any element with data-tip */
  var tip = null;
  function showTip(ev, html) {
    tip = tip || document.getElementById('tip');
    tip.innerHTML = html; tip.hidden = false;
    var x = ev.clientX + 14, y = ev.clientY + 14, r = tip.getBoundingClientRect();
    if (x + r.width > window.innerWidth - 8) x = ev.clientX - r.width - 14;
    if (y + r.height > window.innerHeight - 8) y = ev.clientY - r.height - 14;
    tip.style.left = x + 'px'; tip.style.top = y + 'px';
  }
  document.addEventListener('pointerover', function (ev) {
    var t = ev.target.closest && ev.target.closest('[data-tip]');
    if (t) showTip(ev, t.getAttribute('data-tip'));
  });
  document.addEventListener('pointermove', function (ev) {
    var t = ev.target.closest && ev.target.closest('[data-tip]');
    if (t) showTip(ev, t.getAttribute('data-tip')); else if (tip) tip.hidden = true;
  });
  document.addEventListener('pointerdown', function () { if (tip) tip.hidden = true; });

  /* ---------- matrices ---------- */
  /* opts: name (TeX), digits, hlCols (count kept, highlights first n columns and fades the rest),
     hlRows, faded (bool), note */
  UI.matrix = function (M, o) {
    o = o || {};
    var d = o.digits === undefined ? 2 : o.digits, r = M.length, c = r ? M[0].length : 0;
    var h = '<figure class="mat"><figcaption>' + (o.name ? UI.tx(o.name) : '') + '<span class="dim">' + r + '×' + c + '</span>' + (o.note ? '<span class="muted">' + o.note + '</span>' : '') + '</figcaption><div class="mwrap"><table class="mtab">';
    for (var i = 0; i < r; i++) {
      h += '<tr>';
      for (var j = 0; j < c; j++) {
        var cls = '';
        if (o.hlCols !== undefined) cls = j < o.hlCols ? 'hl' : 'fd';
        if (o.hlRows !== undefined) cls = i < o.hlRows ? 'hl' : 'fd';
        if (o.diagOnly && i !== j) cls = (cls ? cls + ' ' : '') + 'fd';
        if (o.diagHL !== undefined) cls = (i === j && j < o.diagHL) ? 'hl' : 'fd';
        h += '<td class="' + cls + '">' + UI.fx(M[i][j], d) + '</td>';
      }
      h += '</tr>';
    }
    return h + '</table></div></figure>';
  };
  UI.op = function (s) { return '<span class="mop">' + s + '</span>'; };
  UI.texMat = function (M, d) {
    d = d === undefined ? 3 : d;
    return '\\begin{bmatrix}' + M.map(function (r) { return r.map(function (v) { return UI.fx(v, d); }).join(' & '); }).join('\\\\') + '\\end{bmatrix}';
  };
  UI.texVec = function (v, d) { return UI.texMat(v.map(function (x) { return [x]; }), d); };

  /* editable matrix: returns {get(), set(M)} and calls onChange(M) */
  UI.editor = function (host, M, onChange) {
    var cur = M.map(function (r) { return r.slice(); });
    function build() {
      var h = '<table><tr><th></th>';
      for (var j = 0; j < cur[0].length; j++) h += '<th>x' + (j + 1) + '</th>';
      h += '</tr>';
      cur.forEach(function (row, i) {
        h += '<tr><th>n=' + (i + 1) + '</th>';
        row.forEach(function (v, j) { h += '<td><input type="text" inputmode="decimal" data-i="' + i + '" data-j="' + j + '" value="' + v + '" aria-label="X row ' + (i + 1) + ' column ' + (j + 1) + '"></td>'; });
        h += '</tr>';
      });
      host.innerHTML = h + '</table>';
    }
    if (host._edh) host.removeEventListener('input', host._edh);
    host._edh = function (ev) {
      var t = ev.target; if (!t.dataset || t.dataset.i === undefined) return;
      var v = parseFloat(t.value);
      if (isFinite(v)) { cur[+t.dataset.i][+t.dataset.j] = v; t.style.borderColor = ''; onChange(cur.map(function (r) { return r.slice(); })); }
      else t.style.borderColor = 'var(--bad)';
    };
    host.addEventListener('input', host._edh);
    build();
    return { get: function () { return cur.map(function (r) { return r.slice(); }); }, set: function (N) { cur = N.map(function (r) { return r.slice(); }); build(); onChange(this.get()); } };
  };

  /* eigenvalue chip editor (Modules 5, 6). onChange(sortedValues). */
  UI.chips = function (host, values, onChange, opts) {
    opts = opts || {};
    var vals = values.slice();
    function sorted() { return vals.slice().sort(function (a, b) { return b - a; }); }
    function build() {
      vals = sorted();
      var h = '';
      vals.forEach(function (v, i) {
        h += '<span class="chip"><label for="' + host.id + 'c' + i + '">λ' + (i + 1) + '</label><input id="' + host.id + 'c' + i + '" type="number" min="0" step="any" value="' + (+v.toFixed(4)) + '" data-i="' + i + '">' +
          (vals.length > (opts.min || 2) ? '<button type="button" data-del="' + i + '" aria-label="remove λ' + (i + 1) + '">×</button>' : '') + '</span>';
      });
      if (vals.length < (opts.max || 16)) h += '<button type="button" class="add" data-add="1" aria-label="add an eigenvalue">+</button>';
      host.innerHTML = h;
    }
    host.addEventListener('input', function (ev) {
      var t = ev.target; if (t.dataset.i === undefined) return;
      var v = parseFloat(t.value); if (!isFinite(v) || v < 0) return;
      var copy = vals.slice(); copy[+t.dataset.i] = v;
      onChange(copy.slice().sort(function (a, b) { return b - a; }));
    });
    host.addEventListener('change', function (ev) {
      var t = ev.target; if (t.dataset.i === undefined) return;
      var v = parseFloat(t.value); if (isFinite(v) && v >= 0) vals[+t.dataset.i] = v;
      build(); onChange(vals.slice());
    });
    host.addEventListener('click', function (ev) {
      var t = ev.target.closest('button'); if (!t) return;
      if (t.dataset.del !== undefined) vals.splice(+t.dataset.del, 1);
      else if (t.dataset.add) vals.push(+(Math.max(0.05, vals[vals.length - 1] * 0.6)).toFixed(3));
      build(); onChange(vals.slice());
    });
    build();
    return { set: function (v) { vals = v.slice(); build(); onChange(vals.slice()); }, get: function () { return vals.slice(); },
      setQuiet: function (v) { vals = v.slice(); build(); } };
  };

  /* a "stat" tile */
  UI.stat = function (label, value, cls) { return '<div class="stat ' + (cls || '') + '"><span>' + label + '</span><b>' + value + '</b></div>'; };

  /* grayscale image (rows of values, 0..1 or arbitrary range) to canvas */
  UI.drawGray = function (canvas, img, lo, hi) {
    var h = img.length, w = img[0].length; canvas.width = w; canvas.height = h;
    var ctx = canvas.getContext('2d'), id = ctx.createImageData(w, h);
    lo = lo === undefined ? 0 : lo; hi = hi === undefined ? 1 : hi;
    for (var i = 0; i < h; i++) for (var j = 0; j < w; j++) {
      var v = Math.round(255 * Math.min(1, Math.max(0, (img[i][j] - lo) / (hi - lo)))), k = 4 * (i * w + j);
      id.data[k] = id.data[k + 1] = id.data[k + 2] = v; id.data[k + 3] = 255;
    }
    ctx.putImageData(id, 0, 0);
  };
  /* diverging image: blue (negative) ↔ light grey (0) ↔ orange (positive) */
  UI.drawDiverging = function (canvas, img, amp) {
    var h = img.length, w = img[0].length; canvas.width = w; canvas.height = h;
    var ctx = canvas.getContext('2d'), id = ctx.createImageData(w, h);
    var mid = [236, 239, 244], neg = [29, 95, 214], pos = [217, 104, 26];
    for (var i = 0; i < h; i++) for (var j = 0; j < w; j++) {
      var t = Math.max(-1, Math.min(1, img[i][j] / (amp || 1))), c = t < 0 ? neg : pos, a = Math.abs(t), k = 4 * (i * w + j);
      for (var q = 0; q < 3; q++) id.data[k + q] = Math.round(mid[q] + (c[q] - mid[q]) * a);
      id.data[k + 3] = 255;
    }
    ctx.putImageData(id, 0, 0);
  };

  /* ---------- tabs ---------- */
  var inits = {};
  UI.module = function (id, init) { inits[id] = { init: init, done: false }; };
  function show(id) {
    if (!document.getElementById(id)) id = 'm1';
    UI.$$('.module').forEach(function (s) { s.hidden = s.id !== id; });
    UI.$$('.tabs a').forEach(function (a) { if (a.dataset.tab === id) a.setAttribute('aria-current', 'page'); else a.removeAttribute('aria-current'); });
    var m = inits[id];
    if (m && !m.done) { m.done = true; m.init(); }
    UI.renderTex(document.getElementById(id));
  }
  UI.go = function (id) { if (location.hash !== '#' + id) location.hash = id; else show(id); };
  function start() {
    UI.renderTex(UI.$('.masthead'));
    window.addEventListener('hashchange', function () { show(location.hash.slice(1)); window.scrollTo({ top: UI.$('.tabs').offsetTop - 4 }); });
    show(location.hash.slice(1) || 'm1');
  }
  UI.start = start;
  window.UI = UI;
  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', function () { setTimeout(start, 0); });
  else setTimeout(start, 0);
})();
