/* ============================================================================
   Same Votes, Different Teachers — naive Bayes and logistic regression share
   one prediction form. CS 473 module 7 · Murphy, PML §9.3–9.4, §10.2

   The model: a spam filter that looks at two binary features, whether an email
   contains "free" and whether it contains "winner". Naive Bayes, with the
   words conditionally independent given the class, gives

     posterior odds = prior odds × Π_j (likelihood ratio of feature j)

   and taking logs turns that product into a sum of votes, z. P(spam) is then
   σ(z) = 1 / (1 + e^−z), which is exactly logistic regression's form.

   Layout of this file
     PART 1  Math          pure, DOM-free, exported for the Node test suite
     PART 2  State         which words are present, which sub-step we're on
     PART 3  Rendering     the three step panels and the comparison card
     PART 4  Controls      checkboxes, Next step, Start over

   PART 1 runs under Node (`require('./script.js')` returns the math only); the
   rest is skipped when there is no document. See test-math.js.
   ========================================================================= */

(function (global) {
  'use strict';

  /* ==========================================================================
     PART 1 — MATH
     ====================================================================== */

  var PRIOR = 0.2;                                    // P(spam)
  var WORDS = [
    { key: 'free',   pSpam: 0.6, pHam: 0.1  },        // P(word | spam), P(word | not spam)
    { key: 'winner', pSpam: 0.5, pHam: 0.05 }
  ];

  function priorOdds() { return PRIOR / (1 - PRIOR); }

  /* How many times more likely this observation is under spam than not spam.
     An absent word uses 1 − p: its absence is evidence too. */
  function likelihoodRatio(word, present) {
    return present ? word.pSpam / word.pHam : (1 - word.pSpam) / (1 - word.pHam);
  }

  function sigmoid(z) {
    return z >= 0 ? 1 / (1 + Math.exp(-z)) : Math.exp(z) / (1 + Math.exp(z));
  }

  /* Route 1: multiply. x is [freePresent, winnerPresent] as booleans.
     Returns the odds after each factor, starting with the prior odds. */
  function multiplyRoute(x) {
    var odds = priorOdds(), chain = [odds], ratios = [];
    WORDS.forEach(function (w, j) {
      var r = likelihoodRatio(w, x[j]);
      ratios.push(r);
      odds *= r;
      chain.push(odds);
    });
    return { ratios: ratios, chain: chain, odds: odds, p: odds / (1 + odds) };
  }

  /* Route 2: add logs. Each factor becomes a vote; z is their sum. */
  function logRoute(x) {
    var votes = [Math.log(priorOdds())];
    WORDS.forEach(function (w, j) { votes.push(Math.log(likelihoodRatio(w, x[j]))); });
    var z = 0, running = [];
    votes.forEach(function (v) { z += v; running.push(z); });
    return { votes: votes, running: running, z: z, p: sigmoid(z) };
  }

  /* Route 3, for the tests only: Bayes' rule directly, no odds and no logs.
     P(spam | x) = P(spam) Π P(x_j | spam) / Σ_classes P(class) Π P(x_j | class). */
  function bayesDirect(x) {
    var s = PRIOR, h = 1 - PRIOR;
    WORDS.forEach(function (w, j) {
      s *= x[j] ? w.pSpam : 1 - w.pSpam;
      h *= x[j] ? w.pHam : 1 - w.pHam;
    });
    return s / (s + h);
  }

  /* Naive Bayes rewritten in logistic regression's form,
       z = w0 + Σ_j w_j x_j  with x_j ∈ {0, 1}.
     w0 collects the prior vote and every word's ABSENT vote; w_j is how much
     word j's vote changes when it appears. */
  function nbWeights() {
    var w0 = Math.log(priorOdds()), w = [];
    WORDS.forEach(function (word) {
      var absent = Math.log(likelihoodRatio(word, false));
      w0 += absent;
      w.push(Math.log(likelihoodRatio(word, true)) - absent);
    });
    return { w0: w0, w: w };
  }

  /* A mailbox of 1,000 emails whose per-class word rates match WORDS and
     PRIOR exactly, but in which "winner" never appears without "free". Each
     cell is one combination of the two words, with its spam / not-spam count. */
  var MAILBOX = [
    { x: [1, 1], spam: 100, ham: 40  },
    { x: [1, 0], spam: 20,  ham: 40  },
    { x: [0, 1], spam: 0,   ham: 0   },
    { x: [0, 0], spam: 80,  ham: 720 }
  ];

  /* What naive Bayes estimates from a mailbox: class share and per-class word
     rates. Nothing about which words appear together survives this step. */
  function nbFromCounts(cells) {
    var nS = 0, nH = 0, cS = [0, 0], cH = [0, 0];
    cells.forEach(function (c) {
      nS += c.spam; nH += c.ham;
      c.x.forEach(function (v, j) { if (v) { cS[j] += c.spam; cH[j] += c.ham; } });
    });
    return {
      prior: nS / (nS + nH),
      pSpam: cS.map(function (k) { return k / nS; }),
      pHam: cH.map(function (k) { return k / nH; })
    };
  }

  /* Logistic regression by Newton's method on grouped counts: maximise
       Σ_cells [ spam·log σ(z) + ham·log(1 − σ(z)) ],  z = w0 + w·x. */
  function fitLogistic(cells, iters) {
    var b = [0, 0, 0];
    for (var it = 0; it < (iters || 50); it++) {
      var g = [0, 0, 0], H = [[0, 0, 0], [0, 0, 0], [0, 0, 0]];
      cells.forEach(function (c) {
        var n = c.spam + c.ham;
        if (!n) return;
        var f = [1, c.x[0], c.x[1]];
        var p = sigmoid(b[0] + b[1] * f[1] + b[2] * f[2]);
        for (var i = 0; i < 3; i++) {
          g[i] += (c.spam - n * p) * f[i];
          for (var k = 0; k < 3; k++) H[i][k] += n * p * (1 - p) * f[i] * f[k];
        }
      });
      var step = solve3(H, g);
      for (var i = 0; i < 3; i++) b[i] += step[i];
      if (Math.abs(step[0]) + Math.abs(step[1]) + Math.abs(step[2]) < 1e-13) break;
    }
    return { w0: b[0], w: [b[1], b[2]] };
  }

  /* Solve A s = g for a 3×3 system (Gaussian elimination, partial pivoting).
     A tiny ridge keeps an empty-cell direction from making A singular. */
  function solve3(A, g) {
    var M = A.map(function (row, i) { return row.map(function (v, k) { return v + (i === k ? 1e-12 : 0); }).concat(g[i]); });
    for (var c = 0; c < 3; c++) {
      var piv = c;
      for (var r = c + 1; r < 3; r++) if (Math.abs(M[r][c]) > Math.abs(M[piv][c])) piv = r;
      var t = M[c]; M[c] = M[piv]; M[piv] = t;
      for (r = c + 1; r < 3; r++) {
        var f = M[r][c] / M[c][c];
        for (var k = c; k < 4; k++) M[r][k] -= f * M[c][k];
      }
    }
    var s = [0, 0, 0];
    for (var i = 2; i >= 0; i--) {
      var v = M[i][3];
      for (k = i + 1; k < 3; k++) v -= M[i][k] * s[k];
      s[i] = v / M[i][i];
    }
    return s;
  }

  function predictLinear(wts, x) {
    return sigmoid(wts.w0 + wts.w[0] * (x[0] ? 1 : 0) + wts.w[1] * (x[1] ? 1 : 0));
  }

  var MATH = {
    PRIOR: PRIOR, WORDS: WORDS, MAILBOX: MAILBOX,
    priorOdds: priorOdds, likelihoodRatio: likelihoodRatio, sigmoid: sigmoid,
    multiplyRoute: multiplyRoute, logRoute: logRoute, bayesDirect: bayesDirect,
    nbWeights: nbWeights, nbFromCounts: nbFromCounts, fitLogistic: fitLogistic,
    predictLinear: predictLinear
  };

  if (typeof module !== 'undefined' && module.exports) module.exports = MATH;
  if (typeof document === 'undefined') return;


  /* ==========================================================================
     PART 2 — STATE
     Sub-steps behind the Next step button:
       0  starting odds          (step 1)
       1  × "free"
       2  × "winner"  → final odds
       3  starting vote          (step 2)
       4  + "free" vote
       5  + "winner" vote → total z
       6  probability            (step 3)
     ====================================================================== */

  var LAST = 6;
  var BAR_MAX = 3.5;          // |vote| that fills half a bar track
  var DUR = 600;              // ms for number tweens

  var reduceMotion = window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches;

  var S = {
    x: [true, true],
    stage: 0,
    shown: { odds: priorOdds(), z: Math.log(priorOdds()) },   // values currently on screen
    tween: null
  };

  function $(id) { return document.getElementById(id); }

  /* Readable number: about three significant figures, trailing zeros dropped. */
  function nice(x) {
    var a = Math.abs(x);
    var s = a >= 100 ? x.toFixed(0) : a >= 10 ? x.toFixed(1) : a >= 0.1 ? x.toFixed(2) : x.toFixed(3);
    if (s.indexOf('.') >= 0) s = s.replace(/0+$/, '').replace(/\.$/, '');
    return s;
  }

  /* Signed with a real minus sign. */
  function signed(x, d) {
    var s = Math.abs(x).toFixed(d === undefined ? 2 : d);
    if (+s === 0) return s;
    return (x < 0 ? '−' : '+') + s;
  }

  function pct(p, d) { return (100 * p).toFixed(d === undefined ? 2 : d) + '%'; }

  function oddsInWords(o) {
    if (Math.abs(o - 1) < 1e-9) return 'even: 1 spam for every 1 not-spam';
    return o > 1 ? 'about ' + nice(o) + ' spam for every 1 not-spam'
                 : 'about 1 spam for every ' + nice(1 / o) + ' not-spam';
  }

  function quote(w) { return '“' + w.key + '”'; }


  /* ==========================================================================
     PART 3 — RENDERING
     ====================================================================== */

  function current() { return { mul: multiplyRoute(S.x), log: logRoute(S.x) }; }

  /* The odds and z the screen should be showing at this sub-step. */
  function targets(c) {
    var k1 = Math.min(S.stage, 2);
    var k2 = S.stage < 3 ? 0 : Math.min(S.stage - 3, 2);
    return { odds: c.mul.chain[k1], z: c.log.running[k2] };
  }

  function setText(id, s) { $(id).textContent = s; }

  function hide(id, hidden) { $(id).classList.toggle('is-hidden', hidden); }

  function pulse(id) {
    if (reduceMotion) return;
    var el = $(id);
    el.classList.add('is-pulse');
    setTimeout(function () { el.classList.remove('is-pulse'); }, 380);
  }

  function renderChrome() {
    var step = S.stage <= 2 ? 1 : S.stage <= 5 ? 2 : 3;
    [1, 2, 3].forEach(function (k) {
      $('chip' + k).classList.toggle('is-on', k === step);
      $('chip' + k).classList.toggle('is-done', k < step);
      $('panel' + k).classList.toggle('is-locked', k > step);
      $('panel' + k).classList.toggle('is-on', k === step);
    });
    var btn = $('btnNext');
    btn.disabled = S.stage >= LAST;
    btn.textContent = S.stage >= LAST ? 'Done: try the checkboxes' : 'Next step';
  }

  function renderStep1(c) {
    setText('tPriorV', nice(priorOdds()));
    ['Free', 'Winner'].forEach(function (name, j) {
      var w = WORDS[j], on = S.x[j], r = c.mul.ratios[j];
      var a = on ? w.pSpam : 1 - w.pSpam, b = on ? w.pHam : 1 - w.pHam;
      var tile = $('t' + name);
      tile.classList.toggle('for-spam', r > 1);
      tile.classList.toggle('for-ham', r < 1);
      $('t' + name + 'K').textContent = (on ? 'contains ' : 'no ') + quote(w);
      setText('t' + name + 'V', '×' + nice(r));
      setText('t' + name + 'D', quote(w) + ' is ' + (on ? 'in ' : 'missing from ') +
        Math.round(100 * a) + '% of spam vs ' + Math.round(100 * b) + '% of not-spam: ' +
        Math.round(100 * a) + ' ÷ ' + Math.round(100 * b) + ' = ×' + nice(r));
      hide('t' + name, S.stage < j + 1);
      hide('op' + (j + 1), S.stage < j + 1);
    });
    hide('op3', S.stage < 2);
    $('tOdds').querySelector('.tile-k').textContent = S.stage >= 2 ? 'Final odds' : 'Odds so far';
  }

  function placeBar(id, v) {
    var bar = $(id), half = Math.min(Math.abs(v), BAR_MAX) / BAR_MAX * 50;
    bar.classList.toggle('pos', v > 0);
    bar.classList.toggle('neg', v < 0);
    bar.style.left = (v >= 0 ? 50 : 50 - half) + '%';
    bar.style.width = half + '%';
  }

  function renderStep2(c) {
    var rows = ['Prior', 'Free', 'Winner'];
    rows.forEach(function (name, i) {
      var v = c.log.votes[i];
      var visible = S.stage >= 3 + i;
      hide('v' + name, !visible);
      placeBar('v' + name + 'B', visible ? v : 0);
      setText('v' + name + 'V', signed(v));
      var ratio = i === 0 ? priorOdds() : c.mul.ratios[i - 1];
      setText('v' + name + 'S', 'log ' + nice(ratio));
      $('v' + name).title = (i === 0 ? 'Starting odds ' : (S.x[i - 1] ? 'contains ' : 'no ') + quote(WORDS[i - 1]) + ': multiplier ') +
        nice(ratio) + ' → vote ' + signed(v) + (v > 0 ? ' (for spam)' : ' (for not spam)');
    });
    ['Free', 'Winner'].forEach(function (name, j) {
      $('v' + name + 'N').firstChild.textContent = (S.x[j] ? '' : 'no ') + quote(WORDS[j]);
    });
    hide('vTotal', S.stage < 3);
    setText('vTotalL', S.stage >= 5 ? 'Total' : 'Total so far');
  }

  /* Values that tween (odds so far, z so far) and everything drawn from them. */
  function renderShown() {
    var o = S.shown.odds, z = S.shown.z;
    setText('tOddsV', nice(o));
    setText('tOddsD', oddsInWords(o));
    $('balSpam').style.width = (100 * o / (1 + o)) + '%';
    setText('balText', 'odds ' + nice(o));
    placeBar('vTotalB', S.stage >= 3 ? z : 0);
    setText('vTotalV', signed(z));
    drawSigmoid(S.stage >= 6 ? z : null);
  }

  function renderStep3(c) {
    var o = c.mul.odds, z = c.log.z;
    $('routeA').innerHTML = nice(o) + ' ÷ (1 + ' + nice(o) + ') = <b>' + pct(c.mul.p) + '</b>';
    $('routeB').innerHTML = '1 ÷ (1 + e<sup>' + (z > 0 ? '−' : '') + Math.abs(z).toFixed(2) + '</sup>) = <b>' + pct(c.log.p) + '</b>';
    var spam = c.log.p >= 0.5;
    $('verdict').innerHTML =
      '<span class="big">Same answer: ' + pct(c.log.p) + '</span>' +
      'The two routes differ by less than 0.000001%. With a 50% cut-off, this email is ' +
      '<span class="tag"><i class="sw ' + (spam ? 'sw-spam' : 'sw-ham') + '"></i><b>' + (spam ? 'spam' : 'not spam') + '</b></span>.';
  }

  /* ---------------------------------------------------------- the S curve -- */

  var SV = { w: 560, h: 300, l: 58, r: 18, t: 18, b: 56, zMin: -6, zMax: 6 };

  function sx(z) { return SV.l + (z - SV.zMin) / (SV.zMax - SV.zMin) * (SV.w - SV.l - SV.r); }
  function sy(p) { return SV.t + (1 - p) * (SV.h - SV.t - SV.b); }

  function drawSigmoid(z) {
    var h = [], i;
    for (i = SV.zMin; i <= SV.zMax; i += 2) {
      h.push('<line class="grid" x1="' + sx(i) + '" x2="' + sx(i) + '" y1="' + sy(1) + '" y2="' + sy(0) + '"/>');
      h.push('<text class="tick-l" x="' + sx(i) + '" y="' + (sy(0) + 20) + '" text-anchor="middle">' + (i < 0 ? '−' + -i : i > 0 ? '+' + i : 0) + '</text>');
    }
    [0, 0.5, 1].forEach(function (p) {
      h.push('<text class="tick-l" x="' + (SV.l - 8) + '" y="' + (sy(p) + 4) + '" text-anchor="end">' + (p * 100) + '%</text>');
    });
    h.push('<line class="half" x1="' + sx(SV.zMin) + '" x2="' + sx(SV.zMax) + '" y1="' + sy(0.5) + '" y2="' + sy(0.5) + '"/>');
    h.push('<line class="ax" x1="' + SV.l + '" x2="' + (SV.w - SV.r) + '" y1="' + sy(0) + '" y2="' + sy(0) + '"/>');
    h.push('<line class="ax" x1="' + SV.l + '" x2="' + SV.l + '" y1="' + sy(0) + '" y2="' + sy(1) + '"/>');
    h.push('<text class="ax-l" x="' + sx(0) + '" y="' + (SV.h - 8) + '" text-anchor="middle">total vote z</text>');
    h.push('<text class="ax-l" x="' + (SV.l + 8) + '" y="' + (sy(1) + 4) + '">P(spam)</text>');
    /* Each zone label sits on the empty side of the 50% line, away from the curve. */
    h.push('<text class="zone-l" x="' + (sx(SV.zMin) + 8) + '" y="' + (sy(0.5) - 10) + '">← leans not spam</text>');
    h.push('<text class="zone-l" x="' + (sx(SV.zMax) - 8) + '" y="' + (sy(0.5) + 22) + '" text-anchor="end">leans spam →</text>');

    var d = '';
    for (i = 0; i <= 160; i++) {
      var zz = SV.zMin + (SV.zMax - SV.zMin) * i / 160;
      d += (i ? 'L' : 'M') + sx(zz).toFixed(1) + ' ' + sy(sigmoid(zz)).toFixed(1);
    }
    h.push('<path class="sig" d="' + d + '"/>');

    if (z !== null) {
      var p = sigmoid(z), X = sx(Math.max(SV.zMin, Math.min(SV.zMax, z))), Y = sy(p);
      h.push('<line class="guide" x1="' + X + '" x2="' + X + '" y1="' + sy(0) + '" y2="' + Y + '"/>');
      h.push('<line class="guide" x1="' + SV.l + '" x2="' + X + '" y1="' + Y + '" y2="' + Y + '"/>');
      h.push('<circle class="pt" cx="' + X + '" cy="' + Y + '" r="9" fill="var(' + (z >= 0 ? '--spam' : '--ham') + ')"/>');
      /* The curve rises left to right, so just above-left of the point is
         always clear of it. */
      h.push('<text class="pt-l" x="' + (X - 14) + '" y="' + (Y - 14) + '" text-anchor="end">' + pct(p, 1) + '</text>');
    }
    $('plotSig').innerHTML = h.join('');
  }

  /* ------------------------------------------------------------ captions -- */

  function caption(c) {
    var w = WORDS, m = c.mul, l = c.log;
    function wordLine(j) {
      var r = m.ratios[j];
      return S.x[j]
        ? quote(w[j]) + ' appears, and it is ' + nice(r) + '× more common in spam.'
        : quote(w[j]) + ' is missing, which is more typical of not-spam (×' + nice(r) + ').';
    }
    switch (S.stage) {
      case 0: return 'Before reading the email, the odds of spam are ' + nice(m.chain[0]) + ': 1 spam for every 4 not-spam.';
      case 1: return wordLine(0) + ' Odds: ' + nice(m.chain[0]) + ' × ' + nice(m.ratios[0]) + ' = ' + nice(m.chain[1]) + '.';
      case 2: return wordLine(1) + ' Final odds: ' + nice(m.chain[1]) + ' × ' + nice(m.ratios[1]) + ' = ' + nice(m.chain[2]) + '.';
      case 3: return 'Take the log of each number. The starting odds of ' + nice(m.chain[0]) + ' become a vote of ' + signed(l.votes[0]) + ', which leans not spam.';
      case 4: return quote(w[0]) + ' votes ' + signed(l.votes[1]) + ' (log ' + nice(m.ratios[0]) + '). Running total: ' + signed(l.running[1]) + '.';
      case 5: return quote(w[1]) + ' votes ' + signed(l.votes[2]) + '. Total z = ' + signed(l.z) + ', and undoing the log gives back ' + nice(Math.exp(l.z)) + ', the final odds from step 1.';
      default: return 'z = ' + signed(l.z) + ' gives P(spam) = ' + pct(l.p) + ', the same probability the odds from step 1 give.';
    }
  }

  /* ---------------------------------------------------- comparison card -- */

  function renderComparison() {
    var nb = nbWeights(), lr = fitLogistic(MAILBOX);
    var rows = [
      ['baseline w<sub>0</sub>', nb.w0, lr.w0],
      ['w<sub>free</sub>', nb.w[0], lr.w[0]],
      ['w<sub>winner</sub>', nb.w[1], lr.w[1]]
    ].map(function (r) {
      return '<tr><td>' + r[0] + '</td><td>' + signed(r[1]) + '</td><td>' + signed(r[2]) + '</td></tr>';
    });
    var both = [true, true], neither = [false, false];
    rows.push('<tr class="cmp-p"><td>P(spam | both words)</td><td>' + pct(predictLinear(nb, both), 1) + '</td><td>' + pct(predictLinear(lr, both), 1) + '</td></tr>');
    rows.push('<tr><td>P(spam | neither)</td><td>' + pct(predictLinear(nb, neither), 1) + '</td><td>' + pct(predictLinear(lr, neither), 1) + '</td></tr>');
    $('cmpTable').querySelector('tbody').innerHTML = rows.join('');

    var cell = MAILBOX[0];
    $('cmpNote').innerHTML =
      'In this mailbox, ' + cell.spam + ' of the ' + (cell.spam + cell.ham) + ' emails with both words are spam: <b>' +
      pct(cell.spam / (cell.spam + cell.ham), 1) + '</b>. Naive Bayes says ' + pct(predictLinear(nb, both), 1) +
      ' because it counts the overlapping clue twice. Logistic regression learns smaller word weights and matches the data.';

    $('nbLine').innerHTML = 'z = ' + signed(nb.w0) + ' ' + signed(nb.w[0]).replace(/^([+−])/, '$1 ') + '·[free] ' +
      signed(nb.w[1]).replace(/^([+−])/, '$1 ') + '·[winner]';
  }

  /* -------------------------------------------------------------- render -- */

  function ease(t) { return t < 0.5 ? 2 * t * t : 1 - Math.pow(-2 * t + 2, 2) / 2; }

  function render() {
    var c = current();
    renderChrome();
    renderStep1(c);
    renderStep2(c);
    renderStep3(c);
    setText('caption', ['Step 1', 'Step 1', 'Step 1', 'Step 2', 'Step 2', 'Step 2', 'Step 3'][S.stage] + ' · ' + caption(c));

    var to = targets(c), from = { odds: S.shown.odds, z: S.shown.z };
    if (reduceMotion || (from.odds === to.odds && from.z === to.z)) {
      S.shown = to; S.tween = null; renderShown(); return;
    }
    /* Tween odds on a log scale so ×10 and ×0.1 feel the same size. */
    var t0 = performance.now(), token = {};
    S.tween = token;
    (function frame(now) {
      if (S.tween !== token) return;
      var k = Math.min(1, (now - t0) / DUR), e = ease(k);
      S.shown = {
        odds: Math.exp(Math.log(from.odds) + e * (Math.log(to.odds) - Math.log(from.odds))),
        z: from.z + e * (to.z - from.z)
      };
      if (k >= 1) S.shown = to;
      renderShown();
      if (k < 1) requestAnimationFrame(frame);
    })(t0);
    /* rAF pauses in background tabs; make sure the numbers always land. */
    setTimeout(function () {
      if (S.tween !== token) return;
      S.tween = null; S.shown = to; renderShown();
    }, DUR + 80);
  }


  /* ==========================================================================
     PART 4 — CONTROLS
     ====================================================================== */

  var PULSE = { 1: 'tFree', 2: 'tWinner', 3: 'vPrior', 4: 'vFree', 5: 'vWinner', 6: 'verdict' };

  function next() {
    if (S.stage >= LAST) return;
    S.stage++;
    render();
    pulse(PULSE[S.stage]);
    if (S.stage === 3 || S.stage === 6) {
      var panel = $(S.stage === 3 ? 'panel2' : 'panel3');
      var r = panel.getBoundingClientRect();
      if (r.top < 0 || r.bottom > window.innerHeight) panel.scrollIntoView({ behavior: reduceMotion ? 'auto' : 'smooth', block: 'center' });
    }
  }

  function reset() {
    S.stage = 0;
    render();
    window.scrollTo({ top: 0, behavior: reduceMotion ? 'auto' : 'smooth' });
  }

  function boot() {
    $('btnNext').addEventListener('click', next);
    $('btnReset').addEventListener('click', reset);
    [['cbFree', 0], ['cbWinner', 1]].forEach(function (p) {
      var cb = $(p[0]);
      S.x[p[1]] = cb.checked;
      cb.addEventListener('change', function () {
        S.x[p[1]] = cb.checked;
        render();
        pulse(p[1] === 0 ? 'tFree' : 'tWinner');
      });
    });
    renderComparison();
    render();
  }

  boot();

})(typeof globalThis !== 'undefined' ? globalThis : this);
