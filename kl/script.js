/* ============================================================================
   The Wrong Codebook — entropy, cross entropy and KL divergence as bits.
   CS 473 module 4 · Murphy, PML §6.1.1–6.1.2, §6.2

   Layout of this file
     PART 1  Math          pure, DOM-free, exported for the Node test suite
     PART 2  State         the two distributions, the fixed draws, presets
     PART 3  Editors       the draggable p and q bars
     PART 4  Area chart    width = p, height = code length, area = bits
     PART 5  Readouts      the ledger, the plain-language story, the codebook
     PART 6  Message       a stream of weather encoded with both codebooks
     PART 7  Controls      buttons, presets, keyboard

   PART 1 runs under Node (`require('./script.js')` returns the math only); the
   rest is skipped when there is no document. See test-math.js.
   ========================================================================= */

(function (global) {
  'use strict';

  /* ==========================================================================
     PART 1 — MATH
     Everything here is a pure function of its arguments. No DOM, no state.
     ====================================================================== */

  var LN2 = Math.LN2;

  function log2(x) { return Math.log(x) / LN2; }

  function mulberry32(seed) {
    var a = seed >>> 0;
    return function () {
      a = (a + 0x6D2B79F5) >>> 0;
      var t = a;
      t = Math.imul(t ^ (t >>> 15), 1 | t);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }

  /* H(p) = −Σ p log₂ p  (Eq. 6.1). Terms with p = 0 contribute 0. */
  function entropy(p) {
    var h = 0;
    for (var i = 0; i < p.length; i++) if (p[i] > 0) h -= p[i] * log2(p[i]);
    return h;
  }

  /* H(p, q) = −Σ p log₂ q  (Eq. 6.13): bits per symbol when data from p is
     written with a code built for q. Infinite if q rules out something p allows. */
  function crossEntropy(p, q) {
    var h = 0;
    for (var i = 0; i < p.length; i++) {
      if (p[i] <= 0) continue;
      if (q[i] <= 0) return Infinity;
      h -= p[i] * log2(q[i]);
    }
    return h;
  }

  /* D_KL(p‖q) = Σ p log₂(p/q)  (Eq. 6.29) = H(p, q) − H(p). */
  function kl(p, q) {
    var d = 0;
    for (var i = 0; i < p.length; i++) {
      if (p[i] <= 0) continue;
      if (q[i] <= 0) return Infinity;
      d += p[i] * log2(p[i] / q[i]);
    }
    return d;
  }

  /* Set entry i to v and rescale the others so everything still sums to 1.
     Each entry keeps at least `floor`; the others share what is left in
     proportion to how far each one sits above the floor. */
  function setShare(arr, i, v, floor) {
    var n = arr.length, f = floor || 0, j;
    v = Math.min(1 - f * (n - 1), Math.max(f, v));
    var free = 1 - v - f * (n - 1);
    var ex = 0;
    for (j = 0; j < n; j++) if (j !== i) ex += Math.max(0, arr[j] - f);
    var out = new Array(n);
    for (j = 0; j < n; j++) {
      if (j === i) out[j] = v;
      else out[j] = f + (ex > 1e-12 ? Math.max(0, arr[j] - f) * free / ex : free / (n - 1));
    }
    return out;
  }

  /* Normalise, then lift anything below `floor` up to it, paying for the lift
     out of the other entries in proportion to their excess. */
  function withFloor(arr, floor) {
    var n = arr.length, s = 0, i;
    for (i = 0; i < n; i++) s += arr[i];
    var a = arr.map(function (x) { return x / s; });
    var need = 0, ex = 0;
    for (i = 0; i < n; i++) {
      if (a[i] < floor) need += floor - a[i];
      else ex += a[i] - floor;
    }
    if (need === 0) return a;
    return a.map(function (x) { return x < floor ? floor : floor + (x - floor) * (ex - need) / ex; });
  }

  /* A Huffman code for weights w: the best code that uses whole bits.
     Ties break by symbol order, so the textbook case (½, ¼, ⅛, ⅛) comes out
     as 0, 10, 110, 111. */
  function huffman(w) {
    var n = w.length;
    var code = w.map(function () { return ''; });
    if (n === 1) return ['0'];
    var nodes = w.map(function (x, i) { return { w: x, k: i, leaves: [i] }; });
    while (nodes.length > 1) {
      nodes.sort(function (a, b) { return a.w - b.w || a.k - b.k; });
      var a = nodes.shift(), b = nodes.shift();
      a.leaves.forEach(function (i) { code[i] = '0' + code[i]; });
      b.leaves.forEach(function (i) { code[i] = '1' + code[i]; });
      nodes.push({ w: a.w + b.w, k: Math.min(a.k, b.k), leaves: a.leaves.concat(b.leaves) });
    }
    return code;
  }

  /* Inverse-CDF draw: a fixed uniform u becomes a symbol under p. Never
     returns a symbol with p = 0. */
  function sampleIndex(u, p) {
    var c = 0, last = -1;
    for (var i = 0; i < p.length; i++) {
      if (p[i] <= 0) continue;
      last = i;
      c += p[i];
      if (u < c) return i;
    }
    return last;
  }

  /* The area chart, as numbers. Column i spans [x0, x0 + p_i]. Its codeword
     costs −log₂ q_i bits; a perfect code would cost −log₂ p_i. Wasted and
     saved are the areas between those two heights, and wasted − saved is
     exactly D_KL(p‖q). */
  function columns(p, q) {
    var x0 = 0, out = [];
    for (var i = 0; i < p.length; i++) {
      var w = p[i];
      var spent = -log2(q[i]);
      var ideal = w > 0 ? -log2(w) : Infinity;
      out.push({
        i: i, x0: x0, w: w, spent: spent, ideal: ideal,
        extra: w > 0 ? spent - ideal : 0,
        wasted: w > 0 ? w * Math.max(0, spent - ideal) : 0,
        saved: w > 0 ? w * Math.max(0, ideal - spent) : 0
      });
      x0 += w;
    }
    return out;
  }

  var MATH = {
    log2: log2, mulberry32: mulberry32, entropy: entropy, crossEntropy: crossEntropy,
    kl: kl, setShare: setShare, withFloor: withFloor, huffman: huffman,
    sampleIndex: sampleIndex, columns: columns
  };

  if (typeof module !== 'undefined' && module.exports) module.exports = MATH;
  if (typeof document === 'undefined') return;


  /* ==========================================================================
     PART 2 — STATE
     ====================================================================== */

  var SYMBOLS = [
    { name: 'Sun',   glyph: '☀︎' },
    { name: 'Cloud', glyph: '☁︎' },
    { name: 'Rain',  glyph: '☂︎' },
    { name: 'Snow',  glyph: '❄︎' }
  ];
  var N_SYM = SYMBOLS.length;

  var Q_FLOOR = 0.01;          // a belief of exactly 0 would need an infinite codeword
  var START_N = 40;
  var MAX_N = 20000;

  var PRESETS = {
    match:   null,                                     // q := p, filled in at click time
    ruleout: { p: [0.40, 0.30, 0.10, 0.20], q: [0.47, 0.36, 0.16, 0.01] },
    reverse: { p: [0.47, 0.36, 0.16, 0.01], q: [0.40, 0.30, 0.10, 0.20] },
    shrug:   { p: [0.5, 0.25, 0.125, 0.125], q: [0.25, 0.25, 0.25, 0.25] },
    flat:    { p: [0.25, 0.25, 0.25, 0.25], q: [0.25, 0.25, 0.25, 0.25] }
  };

  var S = {
    p: [0.5, 0.25, 0.125, 0.125],
    q: [0.25, 0.25, 0.25, 0.25],
    n: START_N,
    timer: null
  };

  /* Fixed uniform draws. The message is always draws[0..n) pushed through the
     current p, so dragging p morphs the same message instead of rerolling it. */
  var draws = new Float64Array(MAX_N);
  (function () { var r = mulberry32(4736); for (var i = 0; i < MAX_N; i++) draws[i] = r(); })();

  var $ = function (id) { return document.getElementById(id); };
  var SVGNS = 'http://www.w3.org/2000/svg';

  function fmt(x, d) {
    if (!isFinite(x)) return '∞';
    if (Math.abs(x) < 0.5 * Math.pow(10, -(d == null ? 2 : d))) x = 0;
    return x.toFixed(d == null ? 2 : d);
  }
  function pct(x) { return Math.round(x * 100) + '%'; }


  /* ==========================================================================
     PART 3 — EDITORS
     ====================================================================== */

  var ED = { W: 320, H: 184, top: 26, base: 142, slot: 80, bar: 46 };

  function el(tag, attrs, parent) {
    var e = document.createElementNS(SVGNS, tag);
    for (var k in attrs) e.setAttribute(k, attrs[k]);
    if (parent) parent.appendChild(e);
    return e;
  }

  function buildEditor(svg, key) {
    svg.setAttribute('viewBox', '0 0 ' + ED.W + ' ' + ED.H);
    var h = ED.base - ED.top;
    [0.25, 0.5, 0.75, 1].forEach(function (g) {
      var y = ED.base - g * h;
      el('line', { x1: 4, x2: ED.W - 4, y1: y, y2: y, 'class': 'ed-grid' }, svg);
    });
    el('line', { x1: 4, x2: ED.W - 4, y1: ED.base, y2: ED.base, 'class': 'ed-axis' }, svg);

    var parts = [];
    for (var i = 0; i < N_SYM; i++) {
      var cx = i * ED.slot + ED.slot / 2;
      var g = el('g', {
        'class': 'ed-slider', tabindex: 0, role: 'slider',
        'aria-label': SYMBOLS[i].name + (key === 'p' ? ', how often it happens' : ', how often you expect it'),
        'aria-valuemin': 0, 'aria-valuemax': 100, 'data-i': i
      }, svg);
      el('rect', { x: cx - ED.slot / 2 + 2, y: ED.top - 22, width: ED.slot - 4, height: ED.base - ED.top + 22, 'class': 'ed-hit' }, g);
      var bar = el('rect', { x: cx - ED.bar / 2, width: ED.bar, rx: 3, 'class': 'ed-bar ed-bar-' + key }, g);
      var cap = el('rect', { x: cx - ED.bar / 2 - 3, width: ED.bar + 6, height: 5, rx: 2.5, 'class': 'ed-cap ed-cap-' + key }, g);
      var val = el('text', { x: cx, 'text-anchor': 'middle', 'class': 'ed-val' }, g);
      var glyph = el('text', { x: cx, y: ED.base + 22, 'text-anchor': 'middle', 'class': 'ed-glyph' }, g);
      glyph.textContent = SYMBOLS[i].glyph;
      var name = el('text', { x: cx, y: ED.base + 38, 'text-anchor': 'middle', 'class': 'ed-name' }, g);
      name.textContent = SYMBOLS[i].name;
      parts.push({ g: g, bar: bar, cap: cap, val: val });
    }
    return parts;
  }

  function paintEditor(parts, arr) {
    var h = ED.base - ED.top;
    for (var i = 0; i < N_SYM; i++) {
      var v = arr[i], y = ED.base - v * h;
      parts[i].bar.setAttribute('y', y);
      parts[i].bar.setAttribute('height', Math.max(0, ED.base - y));
      parts[i].cap.setAttribute('y', y - 2.5);
      parts[i].val.setAttribute('y', y - 9);
      parts[i].val.textContent = pct(v);
      parts[i].g.setAttribute('aria-valuenow', Math.round(v * 100));
      parts[i].g.setAttribute('aria-valuetext', pct(v));
    }
  }

  function setValue(key, i, v) {
    S[key] = setShare(S[key], i, v, key === 'q' ? Q_FLOOR : 0);
    render();
  }

  function wireEditor(svg, key) {
    var active = -1;
    function locate(ev) {
      var r = svg.getBoundingClientRect();
      var x = (ev.clientX - r.left) * ED.W / r.width;
      var y = (ev.clientY - r.top) * ED.H / r.height;
      return {
        i: Math.max(0, Math.min(N_SYM - 1, Math.floor(x / ED.slot))),
        v: (ED.base - y) / (ED.base - ED.top)
      };
    }
    svg.addEventListener('pointerdown', function (ev) {
      var at = locate(ev);
      active = at.i;
      svg.setPointerCapture(ev.pointerId);
      svg.classList.add('dragging');
      setValue(key, active, at.v);
      ev.preventDefault();
    });
    svg.addEventListener('pointermove', function (ev) {
      if (active < 0) return;
      setValue(key, active, locate(ev).v);
    });
    function end() { active = -1; svg.classList.remove('dragging'); }
    svg.addEventListener('pointerup', end);
    svg.addEventListener('pointercancel', end);

    svg.addEventListener('keydown', function (ev) {
      var g = ev.target.closest && ev.target.closest('.ed-slider');
      if (!g) return;
      var i = +g.getAttribute('data-i');
      var step = ev.shiftKey ? 0.05 : 0.01;
      if (ev.key === 'ArrowUp' || ev.key === 'ArrowRight') { setValue(key, i, S[key][i] + step); ev.preventDefault(); }
      if (ev.key === 'ArrowDown' || ev.key === 'ArrowLeft') { setValue(key, i, S[key][i] - step); ev.preventDefault(); }
    });
  }


  /* ==========================================================================
     PART 4 — AREA CHART
     ====================================================================== */

  var CH = { W: 680, H: 330, l: 46, r: 12, t: 16, b: 50 };

  function renderChart(cols) {
    var pw = CH.W - CH.l - CH.r, ph = CH.H - CH.t - CH.b;
    var top = 0;
    cols.forEach(function (c) { if (c.w > 0) top = Math.max(top, c.spent, Math.min(c.ideal, 9)); });
    var yMax = Math.max(3, Math.min(8, Math.ceil(top + 0.2)));
    var sx = function (x) { return CH.l + x * pw; };
    var sy = function (b) { return CH.t + ph - Math.min(b, yMax + 0.5) / yMax * ph; };
    var base = sy(0);

    var s = [];
    s.push('<defs>' +
      '<pattern id="klHatch" width="7" height="7" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">' +
      '<rect width="7" height="7" class="hatch-bg"/><line x1="0" y1="0" x2="0" y2="7" class="hatch-line"/></pattern>' +
      '<clipPath id="klClip"><rect x="' + CH.l + '" y="' + CH.t + '" width="' + pw + '" height="' + ph + '"/></clipPath>' +
      '</defs>');

    for (var b = 0; b <= yMax; b++) {
      var y = sy(b);
      s.push('<line class="ch-grid" x1="' + CH.l + '" x2="' + (CH.W - CH.r) + '" y1="' + y + '" y2="' + y + '"/>');
      s.push('<text class="ch-tick" x="' + (CH.l - 8) + '" y="' + (y + 4) + '" text-anchor="end">' + b + '</text>');
    }
    s.push('<text class="ch-axis-label" transform="translate(13 ' + (CH.t + ph / 2) + ') rotate(-90)" text-anchor="middle">bits per codeword</text>');

    s.push('<g clip-path="url(#klClip)">');
    cols.forEach(function (c) {
      if (c.w <= 0) return;
      var x0 = sx(c.x0), x1 = sx(c.x0 + c.w), w = x1 - x0;
      var ya = sy(c.spent), yb = sy(c.ideal), yLow = Math.max(ya, yb);
      s.push('<rect class="col-base" x="' + x0 + '" y="' + yLow + '" width="' + w + '" height="' + (base - yLow) + '"/>');
      if (c.spent > c.ideal) s.push('<rect class="col-wasted" x="' + x0 + '" y="' + ya + '" width="' + w + '" height="' + (yb - ya) + '"/>');
      if (c.spent < c.ideal) s.push('<rect class="col-saved" x="' + x0 + '" y="' + yb + '" width="' + w + '" height="' + (ya - yb) + '"/>');
      s.push('<line class="ln-p" x1="' + x0 + '" x2="' + x1 + '" y1="' + yb + '" y2="' + yb + '"/>');
      s.push('<line class="ln-q" x1="' + x0 + '" x2="' + x1 + '" y1="' + ya + '" y2="' + ya + '"/>');
      if (w > 52) {
        var above = Math.min(ya, yb) - 7;
        s.push('<text class="col-bits" x="' + (x0 + w / 2) + '" y="' + Math.max(CH.t + 11, above) + '" text-anchor="middle">' +
          fmt(c.spent) + ' bits</text>');
      }
    });
    s.push('</g>');

    cols.forEach(function (c) {
      if (c.w <= 0) return;
      var x1 = sx(c.x0 + c.w);
      if (c.x0 + c.w < 0.999) s.push('<line class="col-sep" x1="' + x1 + '" x2="' + x1 + '" y1="' + CH.t + '" y2="' + base + '"/>');
    });
    s.push('<line class="ch-axis" x1="' + CH.l + '" x2="' + (CH.W - CH.r) + '" y1="' + base + '" y2="' + base + '"/>');

    cols.forEach(function (c) {
      if (c.w <= 0) return;
      var x0 = sx(c.x0), w = sx(c.x0 + c.w) - x0, cx = x0 + w / 2;
      if (w > 15) s.push('<text class="col-glyph" x="' + cx + '" y="' + (base + 20) + '" text-anchor="middle">' + SYMBOLS[c.i].glyph + '</text>');
      if (w > 44) s.push('<text class="col-p" x="' + cx + '" y="' + (base + 37) + '" text-anchor="middle">' + SYMBOLS[c.i].name + ' ' + pct(c.w) + '</text>');
    });

    $('chart').innerHTML = s.join('');
  }


  /* ==========================================================================
     PART 5 — READOUTS
     ====================================================================== */

  function renderLedger(H, Hce, D) {
    $('vHce').textContent = fmt(Hce);
    $('vH').textContent = fmt(H);
    $('vKL').textContent = fmt(D);

    var max = Math.max(3, Math.ceil(Hce));
    $('segH').style.width = (H / max * 100) + '%';
    $('segKL').style.width = (Math.max(0, D) / max * 100) + '%';
    var ticks = [];
    for (var t = 0; t <= max; t++) {
      ticks.push('<span style="left:' + (t / max * 100) + '%">' + t + '</span>');
    }
    $('ledgerScale').innerHTML = ticks.join('');

    var rev = kl(S.q, S.p);
    $('vRev').textContent = fmt(rev);
    $('revNote').textContent = !isFinite(rev)
      ? 'Infinite, because your belief allows weather that never happens. Reality would have zero probability for something you expect.'
      : (Math.abs(rev - D) < 0.005
        ? 'The same as before, which happens only in special cases.'
        : 'A different number, because KL is not symmetric. Which distribution plays reality matters.');
  }

  function renderStory(cols, H, Hce, D) {
    var story = $('story');
    if (D < 0.005) {
      story.innerHTML = 'Your codebook matches reality. Every codeword is exactly as long as it should be, ' +
        'so you spend <b>' + fmt(H) + ' bits</b> per symbol, the entropy, and nothing is wasted.';
      return;
    }
    var worst = null;
    cols.forEach(function (c) { if (c.w > 0 && (!worst || c.wasted > worst.wasted)) worst = c; });
    var sym = SYMBOLS[worst.i].name;
    story.innerHTML =
      'You spend <b>' + fmt(Hce) + ' bits</b> per symbol where a perfect code needs <b>' + fmt(H) + '</b>. ' +
      'That is <b class="t-waste">' + fmt(D) + ' wasted bits</b> on every symbol, ' +
      Math.round(D / Math.max(H, 1e-9) * 100) + '% more than necessary. ' +
      'The biggest leak is <b>' + sym + '</b>: it happens ' + pct(worst.w) + ' of the time but you expect it only ' +
      pct(S.q[worst.i]) + ', so each one costs ' + fmt(worst.spent) + ' bits instead of ' + fmt(worst.ideal) + '.';
  }

  function renderCodebook(cols) {
    var codes = huffman(S.q);
    var rows = cols.map(function (c) {
      var extra = c.w > 0
        ? '<span class="' + (c.extra > 0.005 ? 't-waste' : c.extra < -0.005 ? 't-saved' : '') + '">' +
          (c.extra > 0.005 ? '+' : '') + fmt(c.extra) + '</span>'
        : '<span class="t-muted">never sent</span>';
      return '<tr>' +
        '<th scope="row"><span class="cb-glyph">' + SYMBOLS[c.i].glyph + '</span> ' + SYMBOLS[c.i].name + '</th>' +
        '<td class="num c-p">' + pct(c.w) + '</td>' +
        '<td class="num c-q">' + pct(S.q[c.i]) + '</td>' +
        '<td class="num">' + (c.w > 0 ? fmt(c.ideal) : '—') + '</td>' +
        '<td class="num">' + fmt(c.spent) + '</td>' +
        '<td class="num">' + extra + '</td>' +
        '<td class="num cb-code">' + codes[c.i] + '</td>' +
        '</tr>';
    });
    $('codebook').innerHTML = rows.join('');
  }


  /* ==========================================================================
     PART 6 — MESSAGE
     ====================================================================== */

  var PX_PER_BIT = 13;
  var WINDOW = 14;

  function renderMessage(cols, D) {
    var your = 0, perfect = 0, k, sym = new Array(S.n);
    for (k = 0; k < S.n; k++) {
      var i = sampleIndex(draws[k], S.p);
      sym[k] = i;
      your += cols[i].spent;
      perfect += cols[i].ideal;
    }

    var from = Math.max(0, S.n - WINDOW);
    var tq = [], tp = [];
    for (k = from; k < S.n; k++) {
      var c = cols[sym[k]], g = SYMBOLS[sym[k]].glyph, nm = SYMBOLS[sym[k]].name;
      tq.push('<span class="blk blk-q" style="width:' + (c.spent * PX_PER_BIT) + 'px" title="' + nm + ': ' + fmt(c.spent) + ' bits">' + (c.spent * PX_PER_BIT > 14 ? g : '') + '</span>');
      tp.push('<span class="blk blk-p" style="width:' + (c.ideal * PX_PER_BIT) + 'px" title="' + nm + ': ' + fmt(c.ideal) + ' bits">' + (c.ideal * PX_PER_BIT > 14 ? g : '') + '</span>');
    }
    $('tapeQ').innerHTML = tq.join('');
    $('tapeP').innerHTML = tp.join('');

    $('nSent').textContent = S.n.toLocaleString();
    $('bitsYour').textContent = Math.round(your).toLocaleString();
    $('bitsPerfect').textContent = Math.round(perfect).toLocaleString();
    $('bitsExtra').textContent = Math.round(your - perfect).toLocaleString();
    $('extraRate').textContent = fmt((your - perfect) / Math.max(1, S.n));
    $('klRate').textContent = fmt(D);
  }


  /* ==========================================================================
     PART 7 — CONTROLS
     ====================================================================== */

  var edP, edQ;

  function render() {
    var cols = columns(S.p, S.q);
    var H = entropy(S.p), Hce = crossEntropy(S.p, S.q), D = Hce - H;
    paintEditor(edP, S.p);
    paintEditor(edQ, S.q);
    renderChart(cols);
    renderLedger(H, Hce, D);
    renderStory(cols, H, Hce, D);
    renderCodebook(cols);
    renderMessage(cols, D);
  }

  function stop() {
    if (S.timer) clearInterval(S.timer);
    S.timer = null;
    $('btnPlay').textContent = 'Play';
    $('btnPlay').setAttribute('aria-pressed', 'false');
  }

  function play() {
    if (S.timer) return stop();
    $('btnPlay').textContent = 'Pause';
    $('btnPlay').setAttribute('aria-pressed', 'true');
    S.timer = setInterval(function () {
      if (S.n >= MAX_N) return stop();
      S.n++;
      render();
    }, 120);
  }

  function boot() {
    edP = buildEditor($('edP'), 'p');
    edQ = buildEditor($('edQ'), 'q');
    wireEditor($('edP'), 'p');
    wireEditor($('edQ'), 'q');

    $('btnPlay').addEventListener('click', play);
    $('btnAdd').addEventListener('click', function () { S.n = Math.min(MAX_N, S.n + 500); render(); });
    $('btnReset').addEventListener('click', function () { stop(); S.n = START_N; render(); });
    $('btnSwap').addEventListener('click', function () {
      var p = S.p;
      S.p = S.q.slice();
      S.q = withFloor(p, Q_FLOOR);
      render();
    });

    Array.prototype.forEach.call(document.querySelectorAll('[data-preset]'), function (btn) {
      btn.addEventListener('click', function () {
        var k = btn.getAttribute('data-preset');
        if (k === 'match') S.q = withFloor(S.p, Q_FLOOR);
        else { S.p = PRESETS[k].p.slice(); S.q = PRESETS[k].q.slice(); }
        render();
      });
    });

    render();
  }

  boot();

})(typeof globalThis !== 'undefined' ? globalThis : this);
