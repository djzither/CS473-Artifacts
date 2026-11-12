/* ============================================================================
   How Tree Ensembles Learn: the page.
   CS 473 module 14 · Murphy, PML §18.4–18.5.4

   Reads the four runs computed by model.js (window.ENS) and draws them: the
   round player, the new tree with the followed passenger's path, before/after
   for that passenger, accuracy and loss charts, the 500-passenger grid and the
   learning-signal table. Nothing here changes a number; it only displays them.
   ========================================================================= */

(function () {
  'use strict';

  var E = window.ENS;
  var M = E.M;
  var RUNS = { rf: E.runRF(), ls: E.runLS(), ada: E.runAda(), logit: E.runLogit() };
  var SVGNS = 'http://www.w3.org/2000/svg';
  var MINUS = '−';

  var HASH = { rf: 'random-forest', fsam: 'stagewise', ls: 'least-squares', ada: 'adaboost', logit: 'logitboost' };
  var FROM_HASH = {}; Object.keys(HASH).forEach(function (k) { FROM_HASH[HASH[k]] = k; });
  var RUN_NAME = { rf: 'random forest', ls: 'least squares boosting', ada: 'AdaBoost', logit: 'LogitBoost' };
  var LOSS_NAME = { squared: 'squared', exponential: 'exponential', logistic: 'logistic' };

  var DEFAULT_ID = 3;   // Heikkinen, Miss. Laina: 3rd class, survived; the boosted models argue about her
  var defaultK = E.ALL.filter(function (p) { return p.id === DEFAULT_ID; })[0].k;

  var state = { method: 'rf', m: 1, k: defaultK, crowd: 'pred', rows: 'signal', fsamRun: 'ada', timer: null };

  /* ------------------------------------------------------------ helpers -- */
  function $(id) { return document.getElementById(id); }
  function el(tag, attrs, parent) {
    var n = document.createElementNS(SVGNS, tag);
    for (var a in attrs) if (attrs[a] !== undefined && attrs[a] !== null) n.setAttribute(a, attrs[a]);
    if (parent) parent.appendChild(n);
    return n;
  }
  function txt(parent, x, y, s, cls, anchor) {
    var t = el('text', { x: x, y: y, 'class': cls, 'text-anchor': anchor || 'middle' }, parent);
    t.textContent = s;
    return t;
  }
  function esc(s) { return String(s).replace(/[&<>"]/g, function (c) { return { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]; }); }
  function sg(v, d) { d = d === undefined ? 2 : d; var s = Math.abs(v).toFixed(d); return (v < 0 && +s !== 0 ? MINUS : '+') + s; }
  function fx(v, d) { d = d === undefined ? 2 : d; var s = v.toFixed(d); return s.charAt(0) === '-' ? MINUS + s.slice(1) : s; }
  function pct(v) { return (100 * v).toFixed(1) + '%'; }
  function texNum(v, d) { var s = (d === undefined ? v.toFixed(2) : v.toFixed(d)); return s; }
  function texSg(v, d) { var s = Math.abs(v).toFixed(d === undefined ? 2 : d); return (v < 0 ? '-' : '+') + s; }

  /* If KaTeX's stylesheet is blocked (as in some embedded viewers), fall back to
     its MathML output, which the browser lays out without any stylesheet. */
  var TEX_OUT = null;
  function texOutput() {
    if (TEX_OUT) return TEX_OUT;
    var probe = document.createElement('span');
    probe.className = 'katex';
    document.body.appendChild(probe);
    var ff = getComputedStyle(probe).fontFamily || '';
    document.body.removeChild(probe);
    TEX_OUT = /KaTeX/i.test(ff) ? 'htmlAndMathml' : 'mathml';
    return TEX_OUT;
  }
  function renderTexInto(node, src, display) {
    node.setAttribute('data-tex', src);
    if (window.katex) {
      try { window.katex.render(src, node, { throwOnError: false, displayMode: !!display, output: texOutput() }); return; } catch (e) { /* fall through */ }
    }
    node.textContent = src;
    node.classList.add('tex-raw');
  }
  function renderTex(root) {
    root.querySelectorAll('.tex, .tex-block').forEach(function (n) {
      renderTexInto(n, n.getAttribute('data-tex'), n.classList.contains('tex-block'));
    });
  }

  function P(k) { return E.ALL[k]; }
  function shortName(p) { return p.name.replace(/\s*\(.*\)\s*/, ' ').replace(/\s+/g, ' ').trim(); }
  function classTxt(c) { return c === 1 ? '1st' : c === 2 ? '2nd' : '3rd'; }
  function who(p) { return (p.female ? 'F' : 'M') + ' · ' + classTxt(p.pclass) + ' · ' + E.fmtNum(p.age) + ' · £' + p.fare.toFixed(2); }

  function runKey() { return state.method === 'fsam' ? state.fsamRun : state.method; }
  function run() { return RUNS[runKey()]; }
  function rd() { return run().rounds[state.m - 1]; }

  /* A probability-like score in [0, 1] for colouring and meters. */
  function prob(key, f) {
    if (key === 'rf' || key === 'ls') return Math.max(0, Math.min(1, f));
    return E.sigma2(f);   // AdaBoost and LogitBoost: p = 1 / (1 + e^(-2f))
  }

  /* Plain-language description of the leaf a path ends in: "male, fare > £26.27" */
  function condText(node, yes) {
    var f = node.f, t = node.thr;
    if (f === 0) return yes ? 'male' : 'female';
    if (f === 1) return t < 2 ? (yes ? '1st class' : '2nd or 3rd class') : (yes ? '1st or 2nd class' : '3rd class');
    if (f === 2) return 'age ' + (yes ? '≤ ' : '> ') + E.fmtNum(t);
    return 'fare ' + (yes ? '≤ £' : '> £') + t.toFixed(2);
  }
  function leafDesc(tree, leaf) {
    var parts = [];
    (function walk(n, acc) {
      if (n === leaf) { parts = acc; return; }
      if (n.leaf) return;
      walk(n.left, acc.concat([condText(n, true)]));
      walk(n.right, acc.concat([condText(n, false)]));
    })(tree, []);
    return parts.length ? parts.join(', ') : 'everyone';
  }
  function leaves(tree) { var a = []; E.eachLeaf(tree, function (l) { a.push(l); }); return a; }

  /* ------------------------------------------------------ signals ----- */
  /* The quantity that drives tree m for training passenger p (index p.ti). */
  function signalOf(key, r, p) {
    var i = p.ti;
    if (key === 'rf') return r.counts[i];
    if (key === 'ls') return r.resid[i];
    if (key === 'ada') return r.w[i] * E.N_TRAIN;
    return p.y - r.pi[i];                  // LogitBoost: w·z = y − π (before clipping)
  }
  function strength(key, r, p) {
    if (state.method === 'fsam') return E.LOSSES[run().loss](p.y, r.fBefore[p.k]);
    var s = signalOf(key, r, p);
    return key === 'rf' ? s : Math.abs(s);
  }

  /* ============================================================== render == */
  function render() {
    var key = runKey(), R = run(), r = rd();
    document.querySelectorAll('.tab').forEach(function (t) {
      var on = t.getAttribute('data-method') === state.method;
      t.setAttribute('aria-selected', on ? 'true' : 'false');
      t.tabIndex = on ? 0 : -1;
    });
    $('panel').setAttribute('aria-labelledby', 'tab-' + state.method);
    document.querySelectorAll('[data-notes]').forEach(function (n) { n.hidden = n.getAttribute('data-notes') !== state.method; });
    $('fsamPick').hidden = state.method !== 'fsam';
    $('fsamCharts').hidden = state.method !== 'fsam';
    document.querySelectorAll('[data-run]').forEach(function (b) { b.setAttribute('aria-pressed', b.getAttribute('data-run') === state.fsamRun ? 'true' : 'false'); });
    document.querySelectorAll('[data-crowd]').forEach(function (b) { b.setAttribute('aria-pressed', b.getAttribute('data-crowd') === state.crowd ? 'true' : 'false'); });
    document.querySelectorAll('[data-rows]').forEach(function (b) { b.setAttribute('aria-pressed', b.getAttribute('data-rows') === state.rows ? 'true' : 'false'); });

    $('round').value = state.m;
    $('roundNum').textContent = state.m;
    $('back').disabled = state.m <= 1;
    $('next').disabled = state.m >= M;
    $('play').textContent = state.timer ? '❚❚ Pause' : '▶ Play';
    $('play').setAttribute('aria-pressed', state.timer ? 'true' : 'false');
    $('paxSelect').value = String(state.k);

    $('explain').innerHTML = explain(key, R, r);
    drawTree(key, R, r);
    drawPax(key, R, r);
    if (state.method === 'fsam') { drawLineSearch(R, r); drawLossCurve(R); }
    else drawAccuracy(key, R);
    drawCrowd(key, R, r);
    drawTable(key, R, r);
  }

  /* ---------------------------------------------------------- explain -- */
  function biggestLeaf(tree, beta) {
    var best = null;
    leaves(tree).forEach(function (l) { var s = l.n * Math.abs(l.value * beta); if (!best || s > best.s) best = { l: l, s: s }; });
    return best.l;
  }
  function accLine(r) {
    return 'Test accuracy went from <b>' + (r.m === 1 && runKey() === 'rf' ? 'no vote' : pct(r.accTestBefore)) + '</b> to <b>' + pct(r.accTestAfter) + '</b>.';
  }
  function explain(key, R, r) {
    var m = r.m, tr = r.tree;
    if (state.method === 'fsam') {
      var how = { ls: 'a regression tree fitted to the residuals', ada: 'a ±1 classifier fitted to reweighted passengers', logit: 'a weighted regression tree on the working responses' }[key];
      var same = Math.abs(r.beta - r.betaStar) < 0.005;
      return 'Round ' + m + ' of the <b>' + RUN_NAME[key] + '</b> run. Freeze <i>f</i><sub>' + (m - 1) + '</sub>, fit <i>F</i><sub>' + m + '</sub> as ' + how +
        ', then step <b>β<sub>' + m + '</sub> = ' + fx(r.beta, 3) + '</b> along it. ' +
        (same ? 'That is exactly the step an exact line search picks.' : 'An exact line search would pick β* = ' + fx(r.betaStar, 3) + '; LogitBoost takes a fixed Newton step instead.') +
        ' Training ' + LOSS_NAME[R.loss] + ' loss: <b>' + r.lossBefore.toFixed(4) + '</b> → <b>' + r.lossAfter.toFixed(4) + '</b>.';
    }
    if (key === 'rf') {
      var c = tr.cands || [], win = c.filter(function (x) { return !tr.leaf && x.f === tr.f; })[0];
      var other = c.filter(function (x) { return x !== win; })[0];
      var s = 'Tree ' + m + ' was grown on its own bootstrap sample: 500 draws that hit <b>' + r.distinct + '</b> different passengers and left <b>' + (E.N_TRAIN - r.distinct) + '</b> out. ';
      if (win) s += 'At the root it could only choose <b>' + E.FEATS[c[0].f].name.toLowerCase() + '</b> or <b>' + E.FEATS[c[1].f].name.toLowerCase() + '</b>; ' +
        E.FEATS[win.f].name.toLowerCase() + ' split better (Gini ' + win.score.toFixed(3) + (other && other.score !== null ? ' vs ' + other.score.toFixed(3) : '') + '). ';
      s += m === 1 ? 'It is the first vote, so the forest now predicts whatever it says. ' : 'Adding its vote flipped <b>' + r.flips + '</b> training predictions. ';
      return s + accLine(r);
    }
    if (key === 'ls') {
      var lf = biggestLeaf(tr, 1), mn = Infinity, mx = -Infinity;
      r.resid.forEach(function (v) { mn = Math.min(mn, v); mx = Math.max(mx, v); });
      return 'Residuals before this round run from ' + fx(mn) + ' to ' + fx(mx) + '. Tree ' + m + ' fits them, and its biggest correction is <b>' + sg(lf.value, 3) +
        '</b> for the ' + lf.n + ' passengers who are ' + leafDesc(tr, lf) + ', because they ' + (lf.value > 0 ? 'survived more often' : 'died more often') +
        ' than the model predicted. Training squared error: <b>' + r.lossBefore.toFixed(4) + '</b> → <b>' + r.lossAfter.toFixed(4) + '</b>. ' + accLine(r);
    }
    if (key === 'ada') {
      var wrongShare = 0, nWrong = 0, nMiss = 0;
      E.TRAIN.forEach(function (p, i) {
        if (r.predBefore(p.k) !== p.y && m > 1) { wrongShare += r.w[i]; nWrong++; }
        if (r.F[p.k] !== 2 * p.y - 1) nMiss++;
      });
      var lead = m === 1 ? 'Every passenger starts with the same weight. ' :
        'The ' + nWrong + ' passengers the ensemble currently gets wrong hold <b>' + pct(wrongShare) + '</b> of the weight. ';
      return lead + 'Tree ' + m + ' has weighted error <b>' + r.err.toFixed(3) + '</b>, so β = ½ log(' + (1 - r.err).toFixed(3) + ' / ' + r.err.toFixed(3) + ') = <b>' + r.beta.toFixed(3) +
        '</b>. The ' + nMiss + ' passengers it gets wrong will have their weight multiplied by e<sup>β</sup> = ' + Math.exp(r.beta).toFixed(2) + ', and the rest by ' + Math.exp(-r.beta).toFixed(2) + '. ' + accLine(r);
    }
    var lf2 = biggestLeaf(tr, 0.5), wMax = 0, unsure = 0;
    E.TRAIN.forEach(function (p, i) { wMax = Math.max(wMax, r.w[i]); if (r.pi[i] > 0.3 && r.pi[i] < 0.7) unsure++; });
    return (m === 1 ? 'At the start every π = 0.5, so every passenger has the top weight 0.25. ' :
      '<b>' + unsure + '</b> passengers still have π between 0.3 and 0.7, and they carry the most weight. ') +
      'Tree ' + m + '&rsquo;s biggest move is ½F = <b>' + sg(0.5 * lf2.value, 3) + '</b> for the ' + lf2.n + ' passengers who are ' + leafDesc(tr, lf2) +
      '. Training log loss: <b>' + r.lossBefore.toFixed(4) + '</b> → <b>' + r.lossAfter.toFixed(4) + '</b>, ' + r.flips + ' predictions flipped. ' + accLine(r);
  }

  /* ------------------------------------------------------------- tree -- */
  var POS = { r: [380, 30], rL: [190, 0], rR: [570, 0], rLL: [95, 0], rLR: [285, 0], rRL: [475, 0], rRR: [665, 0] };
  function leafLabel(key, lf, r) {
    if (key === 'rf') return { big: lf.value ? 'Survived' : 'Died', cls: lf.value ? 'surv' : 'died', sub: 'p̂ ' + lf.mean.toFixed(2) + ' · ' + lf.W + ' draws' };
    if (key === 'ls') return { big: 'F = ' + sg(lf.value, 3), cls: lf.value >= 0 ? 'surv' : 'died', sub: 'n ' + lf.n + ' · mean residual' };
    if (key === 'ada') return { big: (lf.value > 0 ? '+1 survived' : MINUS + '1 died'), cls: lf.value > 0 ? 'surv' : 'died', sub: 'n ' + lf.n + ' · ' + pct(lf.W) + ' of weight' };
    return { big: 'F = ' + sg(lf.value, 2), cls: lf.value >= 0 ? 'surv' : 'died', sub: 'n ' + lf.n + ' · ½F = ' + sg(lf.value / 2, 2) };
  }
  function drawTree(key, R, r) {
    var svg = $('treeSvg');
    svg.textContent = '';
    var tree = r.tree, p = P(state.k);
    var on = {}; E.route(tree, p.x).forEach(function (n) { on[n.id] = true; });
    var rf = key === 'rf';
    var IH = rf ? 82 : 50, IW = 196, LW = 168, LH = 50;
    var y1 = 30 + IH + 50, y2 = y1 + IH + 46;
    var maxY = 0;
    function place(n) {
      var pos = POS[n.id].slice();
      pos[1] = n.depth === 0 ? 30 : n.depth === 1 ? y1 : y2;
      return pos;
    }
    var gEdges = el('g', {}, svg), gNodes = el('g', {}, svg);
    (function draw(n) {
      var pos = place(n), x = pos[0], y = pos[1];
      var g = el('g', { 'class': on[n.id] ? 'on' : '' }, gNodes);
      if (n.leaf) {
        var L = leafLabel(key, n, r);
        el('rect', { x: x - LW / 2, y: y, width: LW, height: LH, rx: 8, 'class': 'leaf-box',
          style: 'fill:var(--' + L.cls + '-soft);stroke:var(--' + L.cls + ')' }, g);
        var t = txt(g, x, y + 21, L.big, 't-val'); t.setAttribute('style', 'fill:var(--' + L.cls + ')');
        txt(g, x, y + 39, L.sub, 't-sub');
        maxY = Math.max(maxY, y + LH);
        return;
      }
      el('rect', { x: x - IW / 2, y: y, width: IW, height: IH, rx: 6, 'class': 'node-box' }, g);
      txt(g, x, y + 21, E.rule(n.f, n.thr), 't-rule');
      var crit = rf ? 'Gini' : key === 'ada' ? 'w. Gini' : 'w. MSE';
      txt(g, x, y + 38, 'n ' + n.n + ' · ' + crit + ' ' + dec(n.imp) + ' → ' + dec(n.score), 't-sub');
      if (rf && n.cands) {
        n.cands.forEach(function (c, j) {
          var s = E.FEATS[c.f].name.toLowerCase() + ': ' + (c.score === null ? 'no valid split' : shortRule(c) + '  ' + c.score.toFixed(3)) + (c.f === n.f ? '  ✓' : '');
          txt(g, x, y + 56 + j * 14, s, 't-cand' + (c.f === n.f ? ' win' : ''));
        });
      }
      maxY = Math.max(maxY, y + IH);
      [n.left, n.right].forEach(function (ch, j) {
        var cp = place(ch), yes = j === 0;
        var x2 = cp[0], y2c = cp[1];
        el('path', { d: 'M' + x + ',' + (y + IH) + ' C' + x + ',' + (y + IH + 24) + ' ' + x2 + ',' + (y2c - 24) + ' ' + x2 + ',' + y2c,
          'class': 'edge' + (on[n.id] && on[ch.id] ? ' on' : '') }, gEdges);
        txt(gEdges, (x + x2) / 2 + (yes ? -14 : 14), (y + IH + y2c) / 2 + 4, yes ? 'yes' : 'no', 't-yn');
        draw(ch);
      });
    })(tree);
    svg.setAttribute('viewBox', '0 0 760 ' + (maxY + 12));
    $('treeNum').textContent = r.m + (state.method === 'fsam' ? ' · ' + RUN_NAME[key] : '');
    $('treeHint').textContent = rf ? 'each split lists the 2 features it was allowed to try' : 'left branch = yes';
  }
  function dec(v) { return v.toFixed(3).replace(/^0/, ''); }
  function shortRule(c) {
    if (c.f === 0) return 'male?';
    if (c.f === 1) return c.thr < 2 ? '1st?' : '1st/2nd?';
    if (c.f === 2) return '≤ ' + E.fmtNum(c.thr);
    return '≤ £' + c.thr.toFixed(2);
  }

  /* -------------------------------------------------------- passenger -- */
  function drawPax(key, R, r) {
    var p = P(state.k), m = r.m, path = E.route(r.tree, p.x), leaf = path[path.length - 1];
    var steps = [];
    for (var i = 0; i < path.length - 1; i++) {
      var n = path[i], yes = p.x[n.f] <= n.thr;
      steps.push(E.rule(n.f, n.thr) + ' <b>' + (yes ? 'yes' : 'no') + '</b>');
    }
    var L = leafLabel(key, leaf, r);
    $('pathLine').innerHTML = '<b>' + esc(shortName(p)) + '</b>: ' + (steps.length ? steps.join(' → ') + ' → ' : '') +
      'leaf <span class="c-' + (L.cls) + '">' + L.big + '</span>';

    var h = '<p class="pax-name">' + esc(p.name) + '</p><div class="pax-meta">' +
      '<span class="chip">' + (p.female ? 'female' : 'male') + '</span><span class="chip">' + classTxt(p.pclass) + ' class</span>' +
      '<span class="chip">age ' + E.fmtNum(p.age) + '</span><span class="chip">fare £' + p.fare.toFixed(2) + '</span>' +
      '<span class="chip ' + (p.y ? 'surv' : 'died') + '">' + (p.y ? 'survived' : 'died') + '</span>' +
      '<span class="chip' + (p.train ? '' : ' test') + '">' + (p.train ? 'training passenger' : 'test passenger') + '</span></div>';

    var b = r.fBefore[p.k], a = r.fAfter[p.k], F = r.F[p.k];
    var cellB, cellA, arrow, calc = '', sig = '';
    function verdict(v) { return v === null ? '<span class="muted">no prediction yet</span>' : v ? '<span class="c-surv">predicts survived</span>' : '<span class="c-died">predicts died</span>'; }
    function meter(q) { return '<div class="meter" aria-hidden="true"><b style="left:50%"></b><i style="left:' + (100 * Math.max(0, Math.min(1, q))).toFixed(1) + '%"></i></div>'; }
    var pb = r.predBefore(p.k), pa = r.predAfter(p.k);
    if (key === 'rf') {
      var vb = r.votesBefore[p.k], va = r.votesAfter[p.k];
      cellB = cell('Before · ' + (m - 1) + ' trees', m === 1 ? '–' : pct(vb / (m - 1)), m === 1 ? 'no trees yet' : vb + ' of ' + (m - 1) + ' vote survived', m === 1 ? null : vb / (m - 1), pb, false);
      cellA = cell('After · ' + m + ' trees', pct(va / m), va + ' of ' + m + ' vote survived', va / m, pa, true);
      arrow = 'tree ' + m + '<br>votes<br>' + (F ? '<span class="c-surv">survived</span>' : '<span class="c-died">died</span>');
      if (p.train) {
        var cnt = r.counts[p.ti];
        sig = cnt ? 'This passenger was drawn <b>' + cnt + '×</b> into tree ' + m + '&rsquo;s bootstrap sample.' :
          'This passenger is <b>out of bag</b> for tree ' + m + ': the tree never saw them, so its vote on them is a fair test.';
      }
    } else {
      var bt = state.method === 'fsam' || key !== 'rf' ? r.beta : 1;
      var lab = key === 'logit' || key === 'ada' ? 'f' : 'f';
      var subB = key === 'ls' ? 'survival score' : 'p = ' + prob(key, b).toFixed(3);
      var subA = key === 'ls' ? 'survival score' : 'p = ' + prob(key, a).toFixed(3);
      cellB = cell('Before · f' + sub(m - 1), fx(b, 3), subB, prob(key, b), pb, false);
      cellA = cell('After · f' + sub(m), fx(a, 3), subA, prob(key, a), pa, true);
      arrow = '+ βF<br>' + sg(bt * F, 3);
      calc = 'f_{' + m + '}(\\boldsymbol x)=f_{' + (m - 1) + '}(\\boldsymbol x)+\\beta_{' + m + '}F_{' + m + '}(\\boldsymbol x)=' +
        texNum(b, 3) + (bt >= 0 ? '+' : '') + texNum(bt, 3) + '\\times(' + texSg(F, key === 'ada' ? 0 : 3) + ')=' + texNum(a, 3);
      if (p.train) {
        var i2 = p.ti;
        if (key === 'ls') sig = 'Residual before this round: <b>r = ' + sg(r.resid[i2], 3) + '</b>. ' + (Math.abs(r.resid[i2]) > 0.5 ? 'The model is on the wrong side for them, so they pull hard on this tree.' : 'The model already leans the right way for them.');
        if (key === 'ada') sig = 'Weight going into this round: <b>' + (r.w[i2] * E.N_TRAIN).toFixed(2) + '×</b> average. Tree ' + m + ' gets them ' +
          (F === 2 * p.y - 1 ? '<span class="ok">right</span>, so it falls to ' : '<span class="bad">wrong</span>, so it rises to ') + (r.wNext[i2] * E.N_TRAIN).toFixed(2) + '×.';
        if (key === 'logit') sig = 'π = ' + r.pi[i2].toFixed(3) + ', so weight <b>w = ' + r.w[i2].toFixed(3) + '</b> and working response <b>z = ' + sg(r.z[i2], 2) + '</b>' +
          (Math.abs(r.z[i2]) >= E.Z_MAX - 1e-9 ? ' (clipped)' : '') + '.';
        if (state.method === 'fsam') {
          var Lf = E.LOSSES[R.loss];
          sig = 'Their ' + LOSS_NAME[R.loss] + ' loss: <b>' + Lf(p.y, b).toFixed(3) + '</b> → <b>' + Lf(p.y, a).toFixed(3) + '</b>. ' + sig;
        }
      }
    }
    if (!p.train) sig = 'Test passengers are never used to fit a tree. They only show whether a change generalizes.';
    h += '<div class="ba">' + cellB + '<div class="ba-arrow">' + arrow + '</div>' + cellA + '</div>';
    if (sig) h += '<p class="calc">' + sig + '</p>';
    if (calc) h += '<div class="calc"><div class="tex-block" id="paxCalc"></div></div>';
    $('paxBody').innerHTML = h;
    if (calc) renderTexInto($('paxCalc'), calc, true);
    $('paxH').textContent = p.train ? 'Following' : 'Following (test)';

    function cell(label, big, small, q, pred, after) {
      return '<div class="ba-cell' + (after ? ' after' : '') + '"><span class="ba-lab">' + label + '</span><span class="ba-val">' + big + '</span>' +
        '<span class="ba-sub">' + small + '</span>' + (q === null ? '' : meter(q)) + '<span class="ba-sub">' + verdict(pred) + '</span></div>';
    }
  }
  function sub(n) { return String(n).split('').map(function (c) { return '₀₁₂₃₄₅₆₇₈₉'[+c]; }).join(''); }

  /* ----------------------------------------------------------- charts -- */
  var CH = { l: 44, r: 12, t: 16, b: 34 };
  function axes(svg, W, H, x0, x1, y0, y1, yTicks, yFmt, xLab, xTicks) {
    var X = function (v) { return CH.l + (v - x0) / (x1 - x0) * (W - CH.l - CH.r); };
    var Y = function (v) { return H - CH.b - (v - y0) / (y1 - y0) * (H - CH.t - CH.b); };
    yTicks.forEach(function (v) {
      el('line', { x1: CH.l, x2: W - CH.r, y1: Y(v), y2: Y(v), 'class': 'grid-l' }, svg);
      txt(svg, CH.l - 6, Y(v) + 4, yFmt(v), '', 'end');
    });
    el('line', { x1: CH.l, x2: W - CH.r, y1: H - CH.b, y2: H - CH.b, 'class': 'axis' }, svg);
    xTicks.forEach(function (v) { txt(svg, X(v), H - CH.b + 15, yFmt === null ? v : String(+v.toFixed(2)), ''); });
    txt(svg, (CH.l + W - CH.r) / 2, H - 4, xLab, 'lab');
    return { X: X, Y: Y };
  }
  function niceRange(vals, step) {
    var lo = Math.min.apply(null, vals), hi = Math.max.apply(null, vals);
    lo = Math.floor(lo / step) * step; hi = Math.ceil(hi / step) * step;
    if (hi - lo < step) hi = lo + step;
    var t = []; for (var v = lo; v <= hi + 1e-9; v += step) t.push(+v.toFixed(6));
    return { lo: lo, hi: hi, ticks: t };
  }
  function line(points, cls, svg) { el('path', { d: 'M' + points.map(function (q) { return q[0].toFixed(1) + ',' + q[1].toFixed(1); }).join('L'), 'class': cls }, svg); }

  function drawAccuracy(key, R) {
    var svg = $('chartSvg'); svg.textContent = ''; svg.setAttribute('viewBox', '0 0 520 230');
    $('chartH').textContent = 'Accuracy by round';
    var W = 520, H = 230, rs = R.rounds;
    var tr = [], te = [];
    if (key !== 'rf') { tr.push([0, rs[0].accTrainBefore]); te.push([0, rs[0].accTestBefore]); }
    rs.forEach(function (r) { tr.push([r.m, r.accTrainAfter]); te.push([r.m, r.accTestAfter]); });
    var rg = niceRange(tr.concat(te).map(function (q) { return q[1]; }), 0.05);
    var A = axes(svg, W, H, 0, M, rg.lo, rg.hi, rg.ticks, function (v) { return (v * 100).toFixed(0) + '%'; }, 'round m', [0, 10, 20, 30, 40, 50]);
    line(tr.map(function (q) { return [A.X(q[0]), A.Y(q[1])]; }), 'l-train', svg);
    line(te.map(function (q) { return [A.X(q[0]), A.Y(q[1])]; }), 'l-test', svg);
    var r = rs[state.m - 1];
    el('line', { x1: A.X(state.m), x2: A.X(state.m), y1: CH.t, y2: H - CH.b, 'class': 'now' }, svg);
    el('circle', { cx: A.X(state.m), cy: A.Y(r.accTrainAfter), r: 4.5, 'class': 'dot-train' }, svg);
    el('circle', { cx: A.X(state.m), cy: A.Y(r.accTestAfter), r: 4.5, 'class': 'dot-test' }, svg);
    var lx = W - CH.r - 150;
    el('line', { x1: lx, x2: lx + 22, y1: CH.t + 4, y2: CH.t + 4, 'class': 'l-train' }, svg); txt(svg, lx + 28, CH.t + 8, 'training', 'lab', 'start');
    el('line', { x1: lx + 84, x2: lx + 106, y1: CH.t + 4, y2: CH.t + 4, 'class': 'l-test' }, svg); txt(svg, lx + 112, CH.t + 8, 'test', 'lab', 'start');
    $('chartHint').textContent = 'round ' + state.m + ': training ' + pct(r.accTrainAfter) + ', test ' + pct(r.accTestAfter);
  }
  function drawLineSearch(R, r) {
    var svg = $('chartSvg'); svg.textContent = ''; svg.setAttribute('viewBox', '0 0 520 230');
    $('chartH').textContent = 'Line search, round ' + r.m;
    var W = 520, H = 230;
    var hiB = Math.max(1.5, 2.2 * Math.max(r.beta, r.betaStar)), loB = -0.25 * hiB / 1.5;
    var pts = [];
    for (var i = 0; i <= 120; i++) { var b = loB + (hiB - loB) * i / 120; pts.push([b, E.lineLoss(R.loss, r.fBefore, r.F, b)]); }
    var ys = pts.map(function (q) { return q[1]; });
    var lo = Math.min.apply(null, ys), hi = Math.max.apply(null, ys), pad = (hi - lo) * 0.08 || 0.01;
    lo -= pad; hi += pad;
    var yt = []; for (var j = 0; j <= 4; j++) yt.push(lo + (hi - lo) * j / 4);
    var xt = []; var st = hiB > 3 ? 1 : 0.5; for (var v = Math.ceil(loB / st) * st; v <= hiB + 1e-9; v += st) xt.push(+v.toFixed(2));
    var A = axes(svg, W, H, loB, hiB, lo, hi, yt, function (v) { return v.toFixed(3); }, 'step size β along tree ' + r.m, xt);
    el('line', { x1: A.X(0), x2: A.X(0), y1: CH.t, y2: H - CH.b, 'class': 'axis' }, svg);
    line(pts.map(function (q) { return [A.X(q[0]), A.Y(q[1])]; }), 'l-loss', svg);
    el('circle', { cx: A.X(r.betaStar), cy: A.Y(E.lineLoss(R.loss, r.fBefore, r.F, r.betaStar)), r: 8, 'class': 'ring' }, svg);
    el('circle', { cx: A.X(r.beta), cy: A.Y(r.lossAfter), r: 4.5, 'class': 'dot-test' }, svg);
    txt(svg, A.X(0) + 4, CH.t + 10, 'β = 0: keep f' + sub(r.m - 1), '', 'start');
    $('chartHint').textContent = 'dot: β taken = ' + fx(r.beta, 3) + ' · ring: exact minimizer ' + fx(r.betaStar, 3);
  }
  function drawLossCurve(R) {
    var svg = $('lossSvg'); svg.textContent = '';
    var W = 520, H = 200, rs = R.rounds;
    var pts = [[0, rs[0].lossBefore]].concat(rs.map(function (r) { return [r.m, r.lossAfter]; }));
    var ys = pts.map(function (q) { return q[1]; });
    var lo = Math.min.apply(null, ys), hi = Math.max.apply(null, ys), pad = (hi - lo) * 0.06;
    lo = Math.max(0, lo - pad); hi += pad;
    var yt = []; for (var j = 0; j <= 3; j++) yt.push(lo + (hi - lo) * j / 3);
    var A = axes(svg, W, H, 0, M, lo, hi, yt, function (v) { return v.toFixed(3); }, 'round m (' + LOSS_NAME[R.loss] + ' loss, training)', [0, 10, 20, 30, 40, 50]);
    line(pts.map(function (q) { return [A.X(q[0]), A.Y(q[1])]; }), 'l-loss', svg);
    var r = rs[state.m - 1];
    el('line', { x1: A.X(state.m), x2: A.X(state.m), y1: CH.t, y2: H - CH.b, 'class': 'now' }, svg);
    el('circle', { cx: A.X(state.m), cy: A.Y(r.lossAfter), r: 4.5, 'class': 'dot-test' }, svg);
  }

  /* ------------------------------------------------------------ crowd -- */
  var GROUPS = [[1, 1], [1, 2], [1, 3], [0, 1], [0, 2], [0, 3]].map(function (g) {
    var list = E.TRAIN.filter(function (p) { return p.female === g[0] && p.pclass === g[1]; })
      .sort(function (a, b) { return (a.age - b.age) || (a.id - b.id); });
    return { label: (g[0] ? 'F ' : 'M ') + classTxt(g[1]), list: list };
  });
  var COLS = 30, PITCH = 15, CELL = 13, LABW = 54;
  function crowdColour(key, R, r, p) {
    if (state.crowd === 'pred') {
      var q = prob(key, r.fAfter[p.k]), d = Math.abs(q - 0.5) * 2;
      return { c: q > 0.5 ? 'surv' : 'died', o: 0.18 + 0.82 * d };
    }
    var s = signalOf(key, r, p);
    if (state.method === 'fsam') {
      var l = E.LOSSES[R.loss](p.y, r.fBefore[p.k]);
      return { c: 'brass', o: Math.min(1, 0.08 + l / crowdColour.maxLoss) };
    }
    if (key === 'rf') return s === 0 ? { c: 'line-2', o: 0.6 } : { c: 'brass', o: Math.min(1, 0.25 + 0.3 * (s - 1)) };
    if (key === 'ada') return { c: 'brass', o: Math.min(1, 0.08 + 0.92 * Math.sqrt(s / crowdColour.maxW)) };
    return { c: s >= 0 ? 'surv' : 'died', o: 0.08 + 0.92 * Math.min(1, Math.abs(s)) };
  }
  function drawCrowd(key, R, r) {
    var svg = $('crowdSvg'); svg.textContent = '';
    if (state.method === 'fsam') {
      crowdColour.maxLoss = 0; E.TRAIN.forEach(function (p) { crowdColour.maxLoss = Math.max(crowdColour.maxLoss, E.LOSSES[R.loss](p.y, r.fBefore[p.k])); });
    }
    if (key === 'ada') { crowdColour.maxW = 0; E.TRAIN.forEach(function (p) { crowdColour.maxW = Math.max(crowdColour.maxW, r.w[p.ti] * E.N_TRAIN); }); }
    var y = 4, sel = null;
    GROUPS.forEach(function (g) {
      txt(svg, 0, y + 11, g.label, '', 'start');
      g.list.forEach(function (p, i) {
        var cx = LABW + (i % COLS) * PITCH, cy = y + Math.floor(i / COLS) * PITCH;
        var col = crowdColour(key, R, r, p);
        var rect = el('rect', { x: cx, y: cy, width: CELL, height: CELL, rx: 2, 'class': 'cell', 'data-k': p.k,
          style: 'fill:var(--' + col.c + ');fill-opacity:' + col.o.toFixed(3) }, svg);
        var tt = el('title', {}, rect); tt.textContent = shortName(p) + ' (' + who(p) + ', ' + (p.y ? 'survived' : 'died') + ')';
        if (state.crowd === 'pred') {
          var pa = r.predAfter(p.k), pb = r.predBefore(p.k);
          if (pa !== p.y) el('circle', { cx: cx + CELL / 2, cy: cy + CELL / 2, r: 2.2, style: 'fill:var(--ink)', 'pointer-events': 'none' }, svg);
          if (pb !== null && pb !== pa) el('rect', { x: cx - 1, y: cy - 1, width: CELL + 2, height: CELL + 2, rx: 3, 'class': 'flip' }, svg);
        }
        if (p.k === state.k) sel = [cx, cy];
      });
      y += Math.ceil(g.list.length / COLS) * PITCH + 9;
    });
    if (sel) el('rect', { x: sel[0] - 2.5, y: sel[1] - 2.5, width: CELL + 5, height: CELL + 5, rx: 3, 'class': 'sel' }, svg);
    svg.setAttribute('viewBox', '-2 0 ' + (LABW + COLS * PITCH + 4) + ' ' + (y - 4));

    var lg = $('crowdLegend');
    if (state.crowd === 'pred') {
      lg.innerHTML = '<span><i class="sw" style="background:var(--surv)"></i>predicts survived</span><span><i class="sw" style="background:var(--died)"></i>predicts died</span>' +
        '<span>deeper = more confident</span><span>&#9679; wrong prediction</span><span><i class="sw" style="border:2px solid var(--brass);background:transparent"></i>flipped this round</span>' +
        '<span>rows: sex and class, sorted by age</span>';
    } else {
      var t = state.method === 'fsam' ? 'deeper brass = larger ' + LOSS_NAME[R.loss] + ' loss before this round' :
        key === 'rf' ? 'brass shade = times drawn into tree ' + state.m + '&rsquo;s sample (1, 2, 3+); grey = out of bag' :
        key === 'ls' ? 'residual y − f: <span class="c-surv">teal</span> = under-predicted survivors, <span class="c-died">rust</span> = over-predicted' :
        key === 'ada' ? 'deeper brass = more AdaBoost weight going into tree ' + state.m :
        'y − π (the logistic gradient): <span class="c-surv">teal</span> pushes up, <span class="c-died">rust</span> pushes down';
      lg.innerHTML = '<span>' + t + '</span>';
    }
  }

  /* ------------------------------------------------------------ table -- */
  var SAMPLE = (function () {
    var r = E.rng(7), pool = E.TRAIN.slice(), out = [];
    while (out.length < 10) out.push(pool.splice(Math.floor(r() * pool.length), 1)[0]);
    return out;
  })();
  function cols(key, R) {
    var surv = { h: 'Survived', f: function (p) { return p.y ? '<span class="ok">yes</span>' : '<span class="bad">no</span>'; } };
    var base = [
      { h: 'Passenger', cls: 'name', f: function (p) { return esc(shortName(p)); } },
      { h: 'Sex · class · age · fare', cls: 'l', f: function (p) { return who(p); } },
      surv
    ];
    function bar(v, scale, posCls) {
      var w = Math.min(48, Math.abs(v) / scale * 48);
      return '<span class="bar" style="width:' + w.toFixed(1) + 'px;background:var(--' + (posCls || (v >= 0 ? 'surv' : 'died')) + ')"></span>';
    }
    if (state.method === 'fsam') {
      var Lf = E.LOSSES[R.loss];
      return base.concat([
        { h: 'Loss before', f: function (p, r) { return Lf(p.y, r.fBefore[p.k]).toFixed(3); } },
        { h: 'β·F(x)', f: function (p, r) { var v = r.beta * r.F[p.k]; return '<span class="' + (v >= 0 ? 'pos' : 'neg') + '">' + sg(v, 3) + '</span>'; } },
        { h: 'Loss after', f: function (p, r) { return Lf(p.y, r.fAfter[p.k]).toFixed(3); } },
        { h: 'Change', f: function (p, r) { var d = Lf(p.y, r.fAfter[p.k]) - Lf(p.y, r.fBefore[p.k]); return '<span class="' + (d <= 0 ? 'ok' : 'bad') + '">' + sg(d, 3) + '</span>'; } }
      ]);
    }
    if (key === 'rf') return base.concat([
      { h: 'In tree’s sample', f: function (p, r) { var c = r.counts[p.ti]; return c ? '×' + c + bar(c, 4, 'brass') : '<span class="muted">out of bag</span>'; } },
      { h: 'Tree votes', f: function (p, r) { return r.F[p.k] ? '<span class="pos">survived</span>' : '<span class="neg">died</span>'; } },
      { h: 'Votes so far', f: function (p, r) { return r.votesAfter[p.k] + ' / ' + r.m; } },
      { h: 'Forest says', f: function (p, r) { var v = r.predAfter(p.k); return '<span class="' + (v === p.y ? 'ok' : 'bad') + '">' + (v ? 'survived' : 'died') + '</span>'; } }
    ]);
    if (key === 'ls') return base.concat([
      { h: 'f before', f: function (p, r) { return fx(r.fBefore[p.k], 3); } },
      { h: 'Residual r', f: function (p, r) { var v = r.resid[p.ti]; return '<span class="' + (v >= 0 ? 'pos' : 'neg') + '">' + sg(v, 3) + '</span>' + bar(v, 1); } },
      { h: 'Tree F(x)', f: function (p, r) { return sg(r.F[p.k], 3); } },
      { h: 'f after', f: function (p, r) { return fx(r.fAfter[p.k], 3); } },
      { h: 'Residual after', f: function (p, r) { var v = p.y - r.fAfter[p.k]; return '<span class="' + (v >= 0 ? 'pos' : 'neg') + '">' + sg(v, 3) + '</span>'; } }
    ]);
    if (key === 'ada') return base.concat([
      { h: 'ỹ', f: function (p) { return p.y ? '+1' : MINUS + '1'; } },
      { h: 'Weight ×500', f: function (p, r) { var v = r.w[p.ti] * E.N_TRAIN; return v.toFixed(2) + bar(v, 4, 'brass'); } },
      { h: 'Tree F(x)', f: function (p, r) { return r.F[p.k] > 0 ? '+1' : MINUS + '1'; } },
      { h: 'Right?', f: function (p, r) { return r.F[p.k] === 2 * p.y - 1 ? '<span class="ok">yes ×e<sup>−β</sup></span>' : '<span class="bad">no ×e<sup>β</sup></span>'; } },
      { h: 'Next weight', f: function (p, r) { return (r.wNext[p.ti] * E.N_TRAIN).toFixed(2); } }
    ]);
    return base.concat([
      { h: 'π before', f: function (p, r) { return r.pi[p.ti].toFixed(3); } },
      { h: 'z', f: function (p, r) { var v = r.z[p.ti]; return '<span class="' + (v >= 0 ? 'pos' : 'neg') + '">' + sg(v, 2) + '</span>'; } },
      { h: 'Weight w', f: function (p, r) { var v = r.w[p.ti]; return v.toFixed(3) + bar(v, 0.25, 'brass'); } },
      { h: 'Tree F(x)', f: function (p, r) { return sg(r.F[p.k], 2); } },
      { h: 'π after', f: function (p, r) { return E.sigma2(r.fAfter[p.k]).toFixed(3); } }
    ]);
  }
  var NOTES = {
    rf: 'A random forest has no learning signal passed between trees. What makes tree m different is its <b>bootstrap sample</b>: who was drawn, how often, and who was left out.',
    ls: 'Least squares boosting fits each tree to the <b>residuals</b> y − f. Big residuals mean the model is far off for that passenger, so they shape the next split most.',
    ada: 'AdaBoost passes on <b>weights</b>. A passenger the ensemble keeps getting wrong gains weight every round, until some tree is forced to get them right.',
    logit: 'LogitBoost passes on a <b>working response</b> z and a <b>weight</b> w. Their product w·z = y − π is the gradient of the log likelihood.',
    fsam: 'Every boosting method lowers the same thing: the summed training loss. This table shows each passenger&rsquo;s share of it, before and after this round&rsquo;s stage.'
  };
  function drawTable(key, R, r) {
    var C = cols(key, R);
    $('tableNote').innerHTML = NOTES[state.method] + (state.rows === 'signal' ?
      ' Listed: the 10 passengers with the strongest signal this round' + (key === 'rf' && state.method !== 'fsam' ? ' (drawn most often)' : '') + '.' :
      ' Listed: the same 10 randomly chosen passengers every round.');
    var list;
    if (state.rows === 'signal') {
      list = E.TRAIN.slice().sort(function (a, b) { return (strength(key, r, b) - strength(key, r, a)) || (a.id - b.id); }).slice(0, 10);
    } else list = SAMPLE.slice();
    var p0 = P(state.k);
    if (p0.train && list.indexOf(p0) < 0) list.unshift(p0);
    var h = '<thead><tr>' + C.map(function (c) { return '<th class="' + (c.cls === 'name' ? '' : c.cls || '') + '">' + c.h + '</th>'; }).join('') + '</tr></thead><tbody>';
    list.forEach(function (p) {
      h += '<tr data-k="' + p.k + '"' + (p.k === state.k ? ' class="selected"' : '') + '>' +
        C.map(function (c) { return '<td' + (c.cls ? ' class="' + c.cls + '"' : '') + '>' + c.f(p, r) + '</td>'; }).join('') + '</tr>';
    });
    $('sigTable').innerHTML = h + '</tbody>';
  }

  /* ========================================================= controls == */
  function setRound(m) { state.m = Math.max(1, Math.min(M, m)); render(); }
  function stop() { if (state.timer) { clearInterval(state.timer); state.timer = null; } }
  function play() {
    if (state.timer) { stop(); render(); return; }
    if (state.m >= M) state.m = 1;
    state.timer = setInterval(function () {
      if (state.m >= M) { stop(); render(); return; }
      state.m++; render();
    }, 1200);
    render();
  }
  function setMethod(mth, fromHash) {
    if (!HASH[mth]) mth = 'rf';
    state.method = mth;
    if (!fromHash && location.hash !== '#' + HASH[mth]) {
      try { history.replaceState(null, '', '#' + HASH[mth]); } catch (e) { location.hash = HASH[mth]; }
    }
    render();
  }
  function select(k) { if (k >= 0 && k < E.ALL.length) { state.k = k; render(); } }

  function init() {
    renderTex(document);
    var rate = E.TRAIN.filter(function (p) { return p.y; }).length / E.N_TRAIN;
    $('baseRate').textContent = pct(rate) + ' of training passengers survived.';
    $('round').max = M; $('roundMax').textContent = M;

    var sel = $('paxSelect');
    [['Training passengers', E.TRAIN], ['Test passengers', E.TEST]].forEach(function (g) {
      var og = document.createElement('optgroup'); og.label = g[0] + ' (' + g[1].length + ')';
      g[1].slice().sort(function (a, b) { return a.name.localeCompare(b.name); }).forEach(function (p) {
        var o = document.createElement('option'); o.value = p.k;
        o.textContent = shortName(p) + ' (' + who(p) + ')';
        og.appendChild(o);
      });
      sel.appendChild(og);
    });
    sel.addEventListener('change', function () { select(+sel.value); });

    var tabs = Array.prototype.slice.call(document.querySelectorAll('.tab'));
    tabs.forEach(function (t, i) {
      t.addEventListener('click', function () { setMethod(t.getAttribute('data-method')); });
      t.addEventListener('keydown', function (e) {
        var d = e.key === 'ArrowRight' ? 1 : e.key === 'ArrowLeft' ? -1 : 0;
        if (!d) return;
        e.preventDefault();
        var n = tabs[(i + d + tabs.length) % tabs.length];
        n.focus(); setMethod(n.getAttribute('data-method'));
      });
    });
    $('back').addEventListener('click', function () { stop(); setRound(state.m - 1); });
    $('next').addEventListener('click', function () { stop(); setRound(state.m + 1); });
    $('play').addEventListener('click', play);
    $('reset').addEventListener('click', function () { stop(); state.k = defaultK; setRound(1); });
    $('round').addEventListener('input', function () { stop(); setRound(+$('round').value); });
    document.querySelectorAll('[data-run]').forEach(function (b) { b.addEventListener('click', function () { state.fsamRun = b.getAttribute('data-run'); render(); }); });
    document.querySelectorAll('[data-crowd]').forEach(function (b) { b.addEventListener('click', function () { state.crowd = b.getAttribute('data-crowd'); render(); }); });
    document.querySelectorAll('[data-rows]').forEach(function (b) { b.addEventListener('click', function () { state.rows = b.getAttribute('data-rows'); render(); }); });
    $('crowdSvg').addEventListener('click', function (e) { var k = e.target.getAttribute && e.target.getAttribute('data-k'); if (k !== null && k !== undefined) select(+k); });
    $('sigTable').addEventListener('click', function (e) { var tr = e.target.closest('tr[data-k]'); if (tr) select(+tr.getAttribute('data-k')); });
    document.addEventListener('keydown', function (e) {
      var t = e.target, tag = t && t.tagName;
      if (tag === 'INPUT' || tag === 'SELECT' || tag === 'TEXTAREA' || (t && t.getAttribute && t.getAttribute('role') === 'tab')) return;
      if (e.key === 'ArrowRight') { stop(); setRound(state.m + 1); }
      else if (e.key === 'ArrowLeft') { stop(); setRound(state.m - 1); }
    });
    window.addEventListener('hashchange', function () { setMethod(FROM_HASH[location.hash.slice(1)] || 'rf', true); });
    state.method = FROM_HASH[location.hash.slice(1)] || 'rf';
    render();
  }

  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', init); else init();
})();
