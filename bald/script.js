/* ============================================================================
   Two Kinds of 50/50 — drawing and interaction. The math lives in model.js.
   ========================================================================= */

(function () {
  'use strict';
  var M = window.BALD, W = M.PRIOR;
  var NS = 'http://www.w3.org/2000/svg';
  var $ = function (s, r) { return (r || document).querySelector(s); };

  function el(tag, attrs, parent) {
    var e = document.createElementNS(NS, tag);
    for (var k in attrs) e.setAttribute(k, attrs[k]);
    if (parent) parent.appendChild(e);
    return e;
  }
  function tex(node, src, display) {
    if (window.katex) { try { katex.render(src, node, { throwOnError: false, displayMode: !!display }); return; } catch (e) { /* fall through */ } }
    node.textContent = src; node.classList.add('tex-raw');
  }
  function f3(v) { return (Math.abs(v) < 5e-4 ? 0 : v).toFixed(3); }
  function pct(p) { return Math.round(p * 100) + '%'; }
  function pct1(p) { var s = (p * 100).toFixed(1); return s.replace(/\.0$/, '') + '%'; }

  var SCEN = [
    { key: 'disagree', kicker: 'Reply 1', title: 'Confident disagreement',
      reply: '“Sounds good. Circle back after the holidays?”',
      note: 'Model A reads it as a warm lead; Model B reads it as a polite brush-off.' },
    { key: 'shared', kicker: 'Reply 2', title: 'Shared uncertainty',
      reply: '“ok”',
      note: 'Neither model can tell; there just isn’t much in the reply.' }
  ];

  /* ---- entropy plot geometry ---- */
  var PW = 360, PH = 250, L = 42, R = 14, T = 36, B = 40;
  function sx(p) { return L + p * (PW - L - R); }
  function sy(h) { return T + (1 - h) * (PH - T - B); }
  var CURVE = (function () { var d = ''; for (var i = 0; i <= 200; i++) { var p = i / 200; d += (i ? 'L' : 'M') + sx(p).toFixed(1) + ',' + sy(M.H2(p)).toFixed(1); } return d; })();

  function Card(cfg) {
    var root = document.createElement('article');
    root.className = 'card';
    root.innerHTML =
      '<div class="card-head"><span class="kicker">' + cfg.kicker + '</span><h2>' + cfg.title + '</h2>' +
        '<p class="reply">' + cfg.reply + '</p></div>' +
      '<div class="bars">' +
        row('a', 'Model A', 'weight ½') + row('b', 'Model B', 'weight ½') +
        '<div class="brow avgrow"><span class="blab avg">Averaged<small>½ A + ½ B</small></span>' +
          '<div class="track"><div class="fill avg"></div><i class="mid"></i></div><span class="bval" data-v="avg"></span></div>' +
      '</div>' +
      '<svg class="plot" viewBox="0 0 ' + PW + ' ' + PH + '" role="img" aria-label="Binary entropy curve with both model probabilities and their average"></svg>' +
      '<div class="calc"><div class="tex-block" data-calc></div>' +
        '<div class="score"><b data-score></b><span>expected information gain from one human label, before seeing it</span></div></div>' +
      '<div class="reveal"><span class="reveal-h">Get the human label</span>' +
        '<div class="btn-row"><button type="button" class="btn" data-y="1">Reveal interested</button>' +
        '<button type="button" class="btn" data-y="0">Reveal not interested</button>' +
        '<button type="button" class="btn btn-quiet" data-preset>Reset to preset</button></div>' +
        '<div class="result" aria-live="polite"></div></div>';

    function row(k, name, sub) {
      return '<div class="brow"><span class="blab ' + k + '">' + name + '<small>' + sub + '</small></span>' +
        '<div class="track"><div class="fill ' + k + '"></div><i class="mid"></i></div><span class="bval" data-v="' + k + '"></span>' +
        '<input type="range" class="' + k + '" min="0" max="100" step="1" aria-label="' + name + ' probability of interested"></div>';
    }

    var ps, revealed = null;
    var sliders = root.querySelectorAll('input[type=range]');
    var fills = { a: $('.fill.a', root), b: $('.fill.b', root), avg: $('.fill.avg', root) };
    var vals = { a: $('[data-v=a]', root), b: $('[data-v=b]', root), avg: $('[data-v=avg]', root) };
    var svg = $('.plot', root), calc = $('[data-calc]', root), scoreEl = $('[data-score]', root), result = $('.result', root);
    var btnY = { 1: $('[data-y="1"]', root), 0: $('[data-y="0"]', root) };

    sliders.forEach(function (s, m) {
      s.addEventListener('input', function () { ps[m] = +s.value / 100; revealed = null; draw(); });
    });
    [1, 0].forEach(function (y) { btnY[y].addEventListener('click', function () { reveal(y); }); });
    $('[data-preset]', root).addEventListener('click', reset);

    function reset() { ps = M.PRESETS[cfg.key].slice(); revealed = null; draw(); }

    function draw() {
      var r = M.bald(ps, W);
      sliders.forEach(function (s, m) { s.value = Math.round(ps[m] * 100); });
      ['a', 'b'].forEach(function (k, m) { fills[k].style.width = ps[m] * 100 + '%'; vals[k].textContent = pct(ps[m]); });
      fills.avg.style.width = r.pbar * 100 + '%'; vals.avg.textContent = pct1(r.pbar);
      drawPlot(r);
      tex(calc,
        '\\begin{aligned}\\text{BALD} &= H(\\bar p) - \\tfrac12\\big[H(p_A) + H(p_B)\\big]\\\\' +
        '&= H(' + r.pbar.toFixed(3).replace(/0+$/, '').replace(/\.$/, '') + ') - \\tfrac12\\big[' + f3(r.ents[0]) + ' + ' + f3(r.ents[1]) + '\\big]\\\\' +
        '&= ' + f3(r.Hbar) + ' - ' + f3(r.avgH) + ' = \\mathbf{' + f3(r.score) + '}\\ \\text{bits}\\end{aligned}', true);
      scoreEl.textContent = f3(r.score) + ' bits';
      [1, 0].forEach(function (y) {
        var possible = M.posterior(ps, W, y) !== null;
        btnY[y].disabled = !possible;
        btnY[y].title = possible ? '' : 'Both models give this answer 0% probability.';
        btnY[y].setAttribute('aria-pressed', revealed === y ? 'true' : 'false');
      });
      drawResult(r);
    }

    function drawPlot(r) {
      while (svg.firstChild) svg.removeChild(svg.firstChild);
      [0, 0.5, 1].forEach(function (v) {
        el('line', { x1: sx(v), x2: sx(v), y1: sy(0), y2: sy(0) + 4, 'class': 'ax' }, svg);
        el('text', { x: sx(v), y: sy(0) + 16, 'text-anchor': 'middle', 'class': 'tick' }, svg).textContent = v;
        el('line', { x1: sx(0) - 4, x2: sx(0), y1: sy(v), y2: sy(v), 'class': 'ax' }, svg);
        el('text', { x: sx(0) - 7, y: sy(v) + 4, 'text-anchor': 'end', 'class': 'tick' }, svg).textContent = v;
      });
      el('line', { x1: sx(0), x2: sx(1), y1: sy(0), y2: sy(0), 'class': 'ax' }, svg);
      el('line', { x1: sx(0), x2: sx(0), y1: sy(0), y2: sy(1), 'class': 'ax' }, svg);
      el('text', { x: sx(0.5), y: PH - 4, 'text-anchor': 'middle', 'class': 'axt' }, svg).textContent = 'P(interested)';
      el('text', { x: 12, y: sy(0.5), 'text-anchor': 'middle', 'class': 'axt', transform: 'rotate(-90 12 ' + sy(0.5) + ')' }, svg).textContent = 'entropy H (bits)';
      el('path', { d: CURVE, 'class': 'curve' }, svg);

      var pA = ps[0], pB = ps[1], hA = r.ents[0], hB = r.ents[1];
      [[pA, hA, 'var(--a)'], [pB, hB, 'var(--b)']].forEach(function (q) {
        el('line', { x1: sx(q[0]), x2: sx(q[0]), y1: sy(q[1]), y2: sy(0), 'class': 'guide', stroke: q[2] }, svg);
      });
      el('line', { x1: sx(pA), y1: sy(hA), x2: sx(pB), y2: sy(hB), 'class': 'chord' }, svg);
      if (r.score > 0.002) el('line', { x1: sx(r.pbar), x2: sx(r.pbar), y1: sy(r.avgH), y2: sy(r.Hbar), 'class': 'gap' }, svg);
      var same = Math.abs(pA - pB) < 0.04;
      /* when A and B coincide, draw them as rings so neither hides under the average */
      el('circle', { cx: sx(pA), cy: sy(hA), r: same ? 13 : 7, 'class': 'dot-a' }, svg);
      el('circle', { cx: sx(pB), cy: sy(hB), r: same ? 10 : 7, 'class': 'dot-b' }, svg);
      el('circle', { cx: sx(r.pbar), cy: sy(r.avgH), r: 5.5, 'class': 'dot-mean' }, svg);
      el('circle', { cx: sx(r.pbar), cy: sy(r.Hbar), r: 6.5, 'class': 'dot-avg' }, svg);

      /* labels: A and B beside their dots, avg above the curve point, "avg of H" below the chord point */
      if (!same) {
        lab(sx(pA) + (pA > 0.85 ? -12 : 12), sy(hA) + 4, 'a', 'A', pA > 0.85 ? 'end' : 'start');
        lab(sx(pB) + (pB > 0.85 ? -12 : 12), sy(hB) + 4, 'b', 'B', pB > 0.85 ? 'end' : 'start');
      }
      lab(Math.max(sx(0) + 64, Math.min(sx(1) - 64, sx(r.pbar))), sy(r.Hbar) - 18, 'avg', 'H(average) = ' + f3(r.Hbar), 'middle');
      if (!same) {
        var right = r.pbar < 0.55;
        lab(sx(r.pbar) + (right ? 10 : -10), sy(r.avgH) - 9, 'ink', 'average of H = ' + f3(r.avgH), right ? 'start' : 'end');
        if (r.score > 0.08) lab(sx(r.pbar) + 8, (sy(r.avgH) + sy(r.Hbar)) / 2 + 4, 'avg', 'BALD', 'start');
      } else {
        lab(sx(0.5), sy(0.42), 'ink', 'A = B = average', 'middle');
        lab(sx(0.5), sy(0.42) + 18, 'avg', 'no gap: BALD = ' + f3(r.score), 'middle');
      }
    }
    function lab(x, y, cls, s, anchor) {
      el('text', { x: x, y: y, 'class': 'plab ' + cls, 'text-anchor': anchor }, svg).textContent = s;
    }

    function reveal(y) {
      if (!M.posterior(ps, W, y)) return;
      revealed = y; draw();
    }

    function drawResult(r) {
      if (revealed === null) {
        result.innerHTML = '<p class="note">' + cfg.note + ' Reveal an answer to see what one label would teach us about which model to trust.</p>';
        return;
      }
      var g = M.realizedGain(ps, W, revealed), word = revealed ? 'interested' : 'not interested';
      var other = M.realizedGain(ps, W, 1 - revealed), pY = g.evidence, pO = 1 - pY;
      var h = '<p>The human says <b>' + word + '</b>. How much probability each model gave that answer:</p>' +
        '<div class="lik"><span class="blab a">Model A</span><b class="num">' + g.lik[0].toFixed(2) + '</b>' +
        '<span class="blab b">Model B</span><b class="num">' + g.lik[1].toFixed(2) + '</b></div>' +
        '<p>Bayes: new weight ∝ old weight × that probability. Model weights, prior → posterior:</p>' +
        wrow('a', 'Model A', g.post[0]) + wrow('b', 'Model B', g.post[1]) +
        '<div class="gainbox">Entropy of the model weights: <b class="num">' + f3(g.before) + '</b> → <b class="num">' + f3(g.after) +
          '</b> bits. Realized reduction: <b class="big">' + f3(g.gain) + ' bits</b>.</div>';
      var exp = pY * g.gain + (other ? pO * other.gain : 0);
      h += '<p>BALD (still <b class="num">' + f3(r.score) + '</b> bits above) was the <b>expected</b> reduction <i>before</i> the answer. It averages the realized reduction over both possible labels, weighted by how likely each was:</p>' +
        '<p class="num" style="font-size:13.5px">' + pY.toFixed(3) + ' × ' + f3(g.gain) + ' (' + word + ')' +
        (other ? ' + ' + pO.toFixed(3) + ' × ' + f3(other.gain) + ' (' + (revealed ? 'not interested' : 'interested') + ')' : ' (the other answer was impossible)') +
        ' = <b>' + f3(exp) + '</b> bits</p>';
      result.innerHTML = h;
      /* animate weights from the prior to the posterior */
      var ws = result.querySelectorAll('.wrow .fill');
      ws.forEach(function (f) { f.style.width = '50%'; });
      result.offsetWidth;
      requestAnimationFrame(function () { requestAnimationFrame(function () {
        ws.forEach(function (f, m) { f.style.width = g.post[m] * 100 + '%'; });
      }); });
    }
    function wrow(k, name, w) {
      return '<div class="wrow"><span class="blab ' + k + '">' + name + '</span><div class="track"><div class="fill ' + k + '" style="width:50%"></div><i class="mid"></i></div>' +
        '<span class="bval">' + pct1(w) + '</span></div>';
    }

    reset();
    return root;
  }

  /* ---- quiz ---- */
  function quiz() {
    var check = $('#qCheck'), fb = $('#qFeedback'), ex = $('#qExplain');
    var radios = document.querySelectorAll('input[name=q]');
    radios.forEach(function (r) { r.addEventListener('change', function () { check.disabled = false; }); });
    check.addEventListener('click', function () {
      var v = ($('input[name=q]:checked') || {}).value;
      if (!v) return;
      document.querySelectorAll('.opt').forEach(function (o) {
        var val = $('input', o).value;
        o.classList.toggle('is-right', val === 'a');
        o.classList.toggle('is-wrong', val === v && v !== 'a');
      });
      fb.className = 'feedback ' + (v === 'a' ? 'ok' : 'no');
      fb.textContent = v === 'a' ? 'Right. The label resolves a disagreement between models.'
        : v === 'b' ? 'Not quite. Hard for both models is not the same as informative: both models already agree it’s a coin flip.'
        : 'Not quite. The overall 50% hides two different situations: total entropy is equal, BALD is not.';
      ex.hidden = false;
    });
    $('#qShow').addEventListener('click', function () { ex.hidden = false; });
  }

  function start() {
    document.querySelectorAll('[data-tex]').forEach(function (n) { tex(n, n.getAttribute('data-tex'), n.classList.contains('tex-block')); });
    var pair = $('#pair');
    SCEN.forEach(function (c) { pair.appendChild(Card(c)); });
    quiz();
  }
  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', start); else start();
})();
