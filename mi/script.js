/* ============================================================================
   Copies of Copies — mutual information, the chain rule and the data
   processing inequality.
   CS 473 module 5 · Murphy, PML §6.3

   The model: X is a fair bit. Y is X flipped with probability p. Z is Y
   flipped with probability q, independently. So X → Y → Z is a Markov chain
   of two binary symmetric channels, and everything has a closed form.

   Layout of this file
     PART 1  Math          pure, DOM-free, exported for the Node test suite
     PART 2  State         the two noise levels and the fixed sample draws
     PART 3  Pipeline      the X → Y → Z channel diagram
     PART 4  Readouts      the bars, the story, the sample tape, the numbers
     PART 5  Controls      sliders and presets

   PART 1 runs under Node (`require('./script.js')` returns the math only); the
   rest is skipped when there is no document. See test-math.js.
   ========================================================================= */

(function (global) {
  'use strict';

  /* ==========================================================================
     PART 1 — MATH
     Everything here is a pure function of its arguments. No DOM, no state.
     ====================================================================== */

  function log2(x) { return Math.log(x) / Math.LN2; }

  /* Binary entropy h₂(t) = −t log₂ t − (1−t) log₂(1−t), with h₂(0) = h₂(1) = 0. */
  function h2(t) {
    if (t <= 0 || t >= 1) return 0;
    return -t * log2(t) - (1 - t) * log2(1 - t);
  }

  /* Two flips in a row: X and Z disagree when exactly one channel flips.
     r = p(1−q) + (1−p)q = p + q − 2pq. Both flipping cancels out. */
  function compose(p, q) { return p + q - 2 * p * q; }

  /* Every quantity on the page, in bits. */
  function measures(p, q) {
    var r = compose(p, q), hp = h2(p), hr = h2(r);
    return {
      r: r,
      hp: hp,
      hr: hr,
      Ixy: 1 - hp,           // I(X;Y)
      Ixz: 1 - hr,           // I(X;Z)
      IxyGz: hr - hp,        // I(X;Y|Z): what Y still adds once Z is known
      IxzGy: 0,              // I(X;Z|Y): Z is made from Y alone
      IxYZ: 1 - hp           // I(X;Y,Z) = I(X;Y) + I(X;Z|Y) = I(X;Y)
    };
  }

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

  /* n fixed triples of uniforms: one for X, one for each channel. */
  function draws(n, seed) {
    var rnd = mulberry32(seed), out = [];
    for (var i = 0; i < n; i++) out.push({ ux: rnd(), u1: rnd(), u2: rnd() });
    return out;
  }

  /* Push the fixed draws through both channels. A channel flips when its
     uniform falls below the noise level, so raising p or q only ever adds
     flips; it never reshuffles the ones already there. */
  function transmit(ds, p, q) {
    return ds.map(function (d) {
      var x = d.ux < 0.5 ? 1 : 0;
      var f1 = d.u1 < p, f2 = d.u2 < q;
      var y = f1 ? 1 - x : x;
      var z = f2 ? 1 - y : y;
      return { x: x, y: y, z: z, f1: f1, f2: f2 };
    });
  }

  var MATH = {
    log2: log2, h2: h2, compose: compose, measures: measures,
    mulberry32: mulberry32, draws: draws, transmit: transmit
  };

  if (typeof module !== 'undefined' && module.exports) module.exports = MATH;
  if (typeof document === 'undefined') return;


  /* ==========================================================================
     PART 2 — STATE
     ====================================================================== */

  var N_DRAWS = 40;
  var EPS = 1e-9;

  var S = {
    p: 0.1,
    q: 0,
    ds: draws(N_DRAWS, 473005)
  };

  function $(id) { return document.getElementById(id); }

  function fmt(x, d) {
    var s = x.toFixed(d === undefined ? 3 : d);
    return s === '-0.000' ? '0.000' : s;
  }

  var SVGNS = 'http://www.w3.org/2000/svg';

  function el(tag, attrs, text) {
    var e = document.createElementNS(SVGNS, tag);
    for (var k in attrs) e.setAttribute(k, attrs[k]);
    if (text !== undefined) e.textContent = text;
    return e;
  }


  /* ==========================================================================
     PART 3 — PIPELINE
     Three stages, each a bit drawn as two dots (0 and 1). Each channel is the
     textbook binary-symmetric-channel picture: straight lines keep the bit,
     crossing lines flip it, and line thickness is the probability.
     ====================================================================== */

  var STAGES = [
    { x: 70,  name: 'X', role: 'fair bit' },
    { x: 340, name: 'Y', role: 'observation' },
    { x: 610, name: 'Z', role: 'processed copy' }
  ];
  var Y0 = 72, Y1 = 142, R = 16;

  function strokeFor(prob) { return 0.75 + 9 * prob; }

  function drawChannel(g, a, b, noise, sym) {
    var x0 = a.x + R + 4, x1 = b.x - R - 4, mid = (x0 + x1) / 2;
    var keep = 1 - noise;

    // keep: 0 → 0 and 1 → 1
    [Y0, Y1].forEach(function (y) {
      if (keep > 0) g.appendChild(el('line', { x1: x0, y1: y, x2: x1, y2: y, class: 'pp-keep', 'stroke-width': strokeFor(keep) }));
    });
    // flip: 0 → 1 and 1 → 0
    if (noise > 0) {
      g.appendChild(el('line', { x1: x0, y1: Y0, x2: x1, y2: Y1, class: 'pp-flip', 'stroke-width': strokeFor(noise) }));
      g.appendChild(el('line', { x1: x0, y1: Y1, x2: x1, y2: Y0, class: 'pp-flip', 'stroke-width': strokeFor(noise) }));
    }

    g.appendChild(el('text', { x: mid, y: Y0 - 12, 'text-anchor': 'middle', class: 'pp-prob pp-prob-keep' },
      'keep  1−' + sym + ' = ' + fmt(keep)));
    g.appendChild(el('text', { x: mid, y: Y1 + 22, 'text-anchor': 'middle', class: 'pp-prob pp-prob-flip' },
      'flip  ' + sym + ' = ' + fmt(noise)));
    g.appendChild(el('text', { x: mid, y: 200, 'text-anchor': 'middle', class: 'pp-chan' },
      sym === 'p' ? 'observe' : 'process'));
  }

  function renderPipe() {
    var svg = $('pipe');
    while (svg.firstChild) svg.removeChild(svg.firstChild);

    var g = el('g', {});
    drawChannel(g, STAGES[0], STAGES[1], S.p, 'p');
    drawChannel(g, STAGES[1], STAGES[2], S.q, 'q');
    svg.appendChild(g);

    STAGES.forEach(function (s) {
      svg.appendChild(el('text', { x: s.x, y: 30, 'text-anchor': 'middle', class: 'pp-name' }, s.name));
      [[Y0, '0'], [Y1, '1']].forEach(function (b) {
        svg.appendChild(el('circle', { cx: s.x, cy: b[0], r: R, class: 'pp-node' }));
        svg.appendChild(el('text', { x: s.x, y: b[0] + 5, 'text-anchor': 'middle', class: 'pp-bit' }, b[1]));
      });
      svg.appendChild(el('text', { x: s.x, y: 200, 'text-anchor': 'middle', class: 'pp-role' }, s.role));
    });
  }


  /* ==========================================================================
     PART 4 — READOUTS
     ====================================================================== */

  function pct(bits) { return (Math.max(0, bits) * 100) + '%'; }

  function renderBars(m) {
    // Bar 1 is literally bar 2 and bar 3 laid end to end.
    $('b1keep').style.width = pct(m.Ixz);
    $('b1lost').style.width = pct(m.IxyGz);
    $('b2keep').style.width = pct(m.Ixz);
    $('b3gap').style.width = pct(m.Ixz);
    $('b3lost').style.width = pct(m.IxyGz);

    $('vIxy').textContent = fmt(m.Ixy);
    $('vIxz').textContent = fmt(m.Ixz);
    $('vIxyGz').textContent = fmt(m.IxyGz);
  }

  function renderNumbers(m) {
    $('vR').textContent = fmt(m.r);
    $('cHp').textContent = fmt(m.hp);
    $('cR').textContent = fmt(m.r);
    $('cHr').textContent = fmt(m.hr);
    $('cIxy').textContent = fmt(m.Ixy);
    $('cIxz').textContent = fmt(m.Ixz);
    $('cIxyGz').textContent = fmt(m.IxyGz);
  }

  function renderStory(m) {
    var a = fmt(m.Ixy), b = fmt(m.Ixz), c = fmt(m.IxyGz), html;

    if (S.p >= 0.5 - EPS) {
      html = 'With <b>p = 0.5</b>, Y is already a fair coin flip that ignores X. Y knows nothing about X, ' +
             'so there is nothing for processing to lose: <b>every bar is 0</b>.';
    } else if (S.q <= EPS) {
      html = 'With <b>q = 0</b>, Z is an exact copy of Y, so <b>no information is lost</b>. ' +
             'Y tells you <b>' + a + '</b> bits about X and so does Z: <b class="t-keep">I(X;Z) = I(X;Y)</b>. ' +
             'Once you have Z, Y adds nothing: <b class="t-lost">I(X;Y|Z) = 0</b>.';
    } else if (S.q >= 0.5 - EPS) {
      html = 'With <b>q = 0.5</b>, Z is a fresh coin flip no matter what Y was, so <b>Z is independent of X</b> ' +
             'and <b class="t-keep">I(X;Z) = 0</b>. All <b>' + a + '</b> bits that Y carried are lost. ' +
             'If you hold Z, Y still tells you everything it ever did: <b class="t-lost">I(X;Y|Z) = ' + a + '</b>.';
    } else {
      var kept = m.Ixy > EPS ? Math.round(100 * m.Ixz / m.Ixy) : 0;
      html = 'Y tells you <b>' + a + '</b> bits about X. After processing, Z keeps <b class="t-keep">' + b +
             '</b> of them (' + kept + '%). The other <b class="t-lost">' + c + '</b> bits are lost to Z, ' +
             'which is exactly how much extra Y would still tell you if you already had Z.';
    }
    $('story').innerHTML = html;
  }

  function renderTape(m) {
    var rows = transmit(S.ds, S.p, S.q);
    var tx = $('tapeX'), ty = $('tapeY'), tz = $('tapeZ');
    [tx, ty, tz].forEach(function (t) { t.textContent = ''; });

    var dY = 0, dZ = 0, cancel = 0;
    rows.forEach(function (r) {
      tx.appendChild(cell(r.x, false));
      ty.appendChild(cell(r.y, r.f1));
      tz.appendChild(cell(r.z, r.f2));
      if (r.y !== r.x) dY++;
      if (r.z !== r.x) dZ++;
      if (r.f1 && r.f2) cancel++;
    });

    var note = 'In these ' + rows.length + ' draws, <b>Y differs from X ' + dY + '</b> times (expected ' +
               fmt(rows.length * S.p, 1) + ') and <b>Z differs from X ' + dZ + '</b> times (expected ' +
               fmt(rows.length * m.r, 1) + '). Red marks a flip in that channel.';
    if (cancel) {
      note += ' ' + cancel + (cancel === 1 ? ' bit was' : ' bits were') + ' flipped twice and came back right, ' +
              'which is why r = p + q &minus; 2pq rather than p + q.';
    }
    $('tapeNote').innerHTML = note;
  }

  function cell(bit, flipped) {
    var d = document.createElement('span');
    d.className = 'bit' + (bit ? ' one' : '') + (flipped ? ' flip' : '');
    d.textContent = bit;
    return d;
  }


  /* ==========================================================================
     PART 5 — CONTROLS
     ====================================================================== */

  function render() {
    var m = measures(S.p, S.q);
    $('oP').textContent = fmt(S.p);
    $('oQ').textContent = fmt(S.q);
    renderPipe();
    renderBars(m);
    renderNumbers(m);
    renderStory(m);
    renderTape(m);

    Array.prototype.forEach.call(document.querySelectorAll('[data-p]'), function (btn) {
      var on = Math.abs(+btn.getAttribute('data-p') - S.p) < EPS && Math.abs(+btn.getAttribute('data-q') - S.q) < EPS;
      btn.classList.toggle('is-on', on);
      btn.setAttribute('aria-pressed', on ? 'true' : 'false');
    });
  }

  function boot() {
    var sP = $('sP'), sQ = $('sQ');
    S.p = +sP.value;
    S.q = +sQ.value;

    sP.addEventListener('input', function () { S.p = +sP.value; render(); });
    sQ.addEventListener('input', function () { S.q = +sQ.value; render(); });

    Array.prototype.forEach.call(document.querySelectorAll('[data-p]'), function (btn) {
      btn.addEventListener('click', function () {
        S.p = +btn.getAttribute('data-p');
        S.q = +btn.getAttribute('data-q');
        sP.value = S.p;
        sQ.value = S.q;
        render();
      });
    });

    render();
  }

  boot();

})(typeof globalThis !== 'undefined' ? globalThis : this);
