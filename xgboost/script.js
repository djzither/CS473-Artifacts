/* ============================================================================
   Boosting Through the Highstorm: the page.
   Reads runs from model.js (window.GBX) and draws them. No number is made here.
   ========================================================================= */

(function () {
  'use strict';

  var G = window.GBX, B = G.BRIDGES, M = G.M;
  var SVGNS = 'http://www.w3.org/2000/svg';
  var MINUS = '−';
  var HASH = { gb: 'gradient-boosting', xgb: 'xgboost' };

  var DEFAULTS = { mode: 'gb', m: 0, eta: 0.3, depth: 2, lambda: 1, gamma: 2, follow: 9, node: 'r' };
  var S = Object.assign({}, DEFAULTS, { timer: null });
  var RUN = {};

  /* ------------------------------------------------------------ helpers -- */
  function $(id) { return document.getElementById(id); }
  function el(tag, attrs, parent) {
    var n = document.createElementNS(SVGNS, tag);
    for (var a in attrs) if (attrs[a] !== undefined && attrs[a] !== null) n.setAttribute(a, attrs[a]);
    if (parent) parent.appendChild(n);
    return n;
  }
  function txt(parent, x, y, s, cls, anchor) { var t = el('text', { x: x, y: y, 'class': cls || '', 'text-anchor': anchor || 'middle' }, parent); t.textContent = s; return t; }
  function fx(v, d) { var s = v.toFixed(d === undefined ? 2 : d); if (+s === 0) s = s.replace('-', ''); return s.charAt(0) === '-' ? MINUS + s.slice(1) : s; }
  function sg(v, d) { var s = fx(v, d); return s.charAt(0) === MINUS ? s : '+' + s; }
  function tn(v, d) { var s = v.toFixed(d === undefined ? 2 : d); return +s === 0 ? s.replace('-', '') : s; }   // TeX number
  function tparen(v, d) { return v < 0 ? '(' + tn(v, d) + ')' : tn(v, d); }

  var TEX_OUT = null;
  function texOutput() {
    if (TEX_OUT) return TEX_OUT;
    var probe = document.createElement('span'); probe.className = 'katex'; document.body.appendChild(probe);
    var ff = getComputedStyle(probe).fontFamily || ''; document.body.removeChild(probe);
    TEX_OUT = /KaTeX/i.test(ff) ? 'htmlAndMathml' : 'mathml';
    return TEX_OUT;
  }
  function renderTex(root) {
    root.querySelectorAll('.tex, .tex-block').forEach(function (n) {
      var src = n.getAttribute('data-tex');
      if (window.katex) {
        try { window.katex.render(src, n, { throwOnError: false, displayMode: n.classList.contains('tex-block'), output: texOutput() }); return; } catch (e) { /* fall through */ }
      }
      n.textContent = src; n.classList.add('tex-raw');
    });
  }
  function T(src, block) { return '<span class="' + (block ? 'tex-block' : 'tex') + '" data-tex="' + src.replace(/"/g, '&quot;') + '"></span>'; }

  function recompute() {
    RUN.gb = G.run('gb', { eta: S.eta, depth: S.depth });
    RUN.xgb = G.run('xgb', { eta: S.eta, depth: S.depth, lambda: S.lambda, gamma: S.gamma });
  }
  function R() { return RUN[S.mode]; }
  function rd() { return S.m ? R().rounds[S.m - 1] : null; }
  function Fnow(run) { return S.m ? run.rounds[S.m - 1].after : B.map(function () { return run.F0; }); }
  function Fprev(run) { return S.m ? run.rounds[S.m - 1].before : B.map(function () { return run.F0; }); }

  function cond(n, yes) {
    var t = G.rule(n.f, n.thr).split(' ≤ ')[1];
    return G.FEATS[n.f].short + (yes ? ' ≤ ' : ' > ') + t;
  }
  function leafDesc(tree, leaf) {
    var out = null;
    (function walk(n, acc) {
      if (n === leaf) { out = acc; return; }
      if (n.leaf) return;
      walk(n.left, acc.concat([cond(n, true)])); walk(n.right, acc.concat([cond(n, false)]));
    })(tree, []);
    return out && out.length ? out.join(' and ') : 'every bridge';
  }
  function findNode(tree, id) { var f = null; G.eachNode(tree, function (n) { if (n.id === id) f = n; }); return f; }
  function meanMiss(leaf, run) {   // the plain average residual in a leaf: what gradient boosting would use
    var r = rd(), s = 0; leaf.rows.forEach(function (i) { s += -r.g[i]; }); return s / leaf.rows.length;
  }

  /* ============================================================= render == */
  function render() {
    document.querySelectorAll('.mode').forEach(function (b) {
      var on = b.getAttribute('data-mode') === S.mode;
      b.setAttribute('aria-selected', on ? 'true' : 'false'); b.tabIndex = on ? 0 : -1;
    });
    $('panel').setAttribute('aria-labelledby', 'tab-' + S.mode);
    document.querySelectorAll('.xgb-only').forEach(function (n) { n.hidden = S.mode !== 'xgb'; });
    document.querySelectorAll('[data-depth]').forEach(function (b) { b.setAttribute('aria-pressed', +b.getAttribute('data-depth') === S.depth ? 'true' : 'false'); });
    $('round').value = S.m; $('roundNum').textContent = S.m;
    $('back').disabled = S.m <= 0; $('next').disabled = S.m >= M;
    $('play').textContent = S.timer ? '❚❚ Pause' : '▶ Play';
    $('play').setAttribute('aria-pressed', S.timer ? 'true' : 'false');
    $('etaVal').textContent = S.eta.toFixed(2);
    $('lambdaVal').textContent = S.lambda.toFixed(1);
    $('gammaVal').textContent = S.gamma.toFixed(0);
    $('eta').value = S.eta; $('lambda').value = S.lambda; $('gamma').value = S.gamma;

    $('explain').innerHTML = explain();
    drawBars();
    drawTree();
    drawSteps();
    drawCands();
    drawLoss();
    drawRoundTable();
  }

  /* ------------------------------------------------------------ explain -- */
  function bigLeaf(tree) {
    var best = null;
    G.leaves(tree).forEach(function (l) { var s = l.rows.length * Math.abs(l.w); if (!best || s > best.s) best = { l: l, s: s }; });
    return best.l;
  }
  function explain() {
    var run = R(), r = rd();
    if (!r) return 'Before any engineer speaks, the forecast is the plain average of all twelve outcomes: <b>' + fx(run.F0) +
      ' highstorms</b> for every bridge. That is the best single number under squared error. Press <b>Next round</b>.';
    var typ = 'Typical miss: <b>' + fx(G.rmse(run.y, r.before)) + '</b> → <b>' + fx(G.rmse(run.y, r.after)) + '</b> highstorms.';
    var l = bigLeaf(r.tree), xg = S.mode === 'xgb';
    if (r.tree.leaf) {
      var why = r.tree.pruned ? 'Its best question, ' + G.rule(r.tree.pruned.f, r.tree.pruned.thr) + '?, gained only ' + fx(r.tree.pruned.best.raw) +
        ', less than the price γ = ' + S.gamma + ', so it was pruned.' : 'No question helped.';
      return 'Engineer ' + S.m + ' asks no questions. ' + why + ' Every bridge gets the same nudge, ' + sg(S.eta * r.tree.w, 3) + '. ' + typ;
    }
    var s = 'Engineer ' + S.m + ' found the biggest misses among bridges with <b>' + leafDesc(r.tree, l) + '</b>: ' +
      (xg ? 'leaf weight <b>w = ' + sg(l.w) + '</b>' : 'an average miss of <b>' + sg(l.w) + '</b>') +
      '. Accepting η = ' + S.eta.toFixed(2) + ' of that moves their forecast by <b>' + sg(S.eta * l.w) + '</b>. ';
    if (xg) {
      var plain = meanMiss(l);
      if (S.lambda > 0) s += 'λ pulled that leaf from the plain average ' + sg(plain) + ' toward zero. ';
      var pr = []; G.eachNode(r.tree, function (n) { if (n.pruned) pr.push(n); });
      if (pr.length) s += 'Pruned: ' + pr.map(function (n) { return G.rule(n.pruned.f, n.pruned.thr) + '? (gain ' + fx(n.pruned.best.gain) + ' after γ)'; }).join('; ') + '. ';
    }
    return s + typ;
  }

  /* --------------------------------------------------------------- bars -- */
  var BAR = null, BW = 640, BH = 330, BL = 40, BR = 8, BT = 14, BB = 46, YMAX = 32;
  function by(v) { return BH - BB - Math.max(0, Math.min(YMAX, v)) / YMAX * (BH - BT - BB); }
  function drawBars() {
    var svg = $('bars'), colW = (BW - BL - BR) / B.length;
    if (!BAR) {
      BAR = { cols: [] };
      for (var v = 0; v <= YMAX; v += 8) {
        el('line', { x1: BL, x2: BW - BR, y1: by(v), y2: by(v), 'class': 'grid-l' }, svg);
        txt(svg, BL - 6, by(v) + 4, String(v), '', 'end');
      }
      el('line', { x1: BL, x2: BW - BR, y1: by(0), y2: by(0), 'class': 'axis' }, svg);
      var tl = txt(svg, 12, (BT + BH - BB) / 2, 'highstorms survived', 'lab');
      tl.setAttribute('transform', 'rotate(-90 12 ' + (BT + BH - BB) / 2 + ')');
      BAR.sel = el('rect', { x: 0, y: BT, width: colW, height: BH - BT - BB, 'class': 'selcol' }, svg);
      BAR.mean = el('line', { x1: BL, x2: BW - BR, 'class': 'mean' }, svg);
      B.forEach(function (b, i) {
        var x = BL + i * colW, c = {};
        c.ghost = el('rect', { x: x + colW * 0.22, width: colW * 0.56, 'class': 'ghost' }, svg);
        c.bar = el('rect', { x: x + colW * 0.22, width: colW * 0.56, 'class': 'bar' }, svg);
        c.miss = el('rect', { x: x + colW / 2 - 1.5, width: 3, rx: 1.5 }, svg);
        var gy = by(b.y);
        c.gem = el('path', { d: 'M' + (x + colW / 2) + ',' + (gy - 7) + 'l7,7l-7,7l-7,-7z', 'class': 'gem' }, svg);
        txt(svg, x + colW / 2, BH - BB + 16, String(i + 1), 'lab');
        c.name = txt(svg, x + colW / 2, BH - BB + 31, '', '');
        var hit = el('rect', { x: x, y: BT, width: colW, height: BH - BT, 'class': 'hit', 'data-i': i }, svg);
        var tt = el('title', {}, hit); tt.textContent = b.name + ': age ' + b.x[0] + ', strength ' + b.x[1] + ', exposure ' + b.x[2] + ', survived ' + b.y;
        BAR.cols.push(c);
      });
      txt(svg, BL - 6, BH - BB + 16, '#', '', 'end');
    }
    var run = R(), Fa = Fnow(run), Fb = Fprev(run);
    BAR.mean.setAttribute('y1', by(run.F0)); BAR.mean.setAttribute('y2', by(run.F0));
    BAR.sel.setAttribute('x', BL + S.follow * colW);
    B.forEach(function (b, i) {
      var c = BAR.cols[i];
      c.bar.style.y = by(Fa[i]) + 'px'; c.bar.style.height = (by(0) - by(Fa[i])) + 'px';
      c.ghost.style.y = by(Fb[i]) + 'px'; c.ghost.style.height = (by(0) - by(Fb[i])) + 'px';
      c.ghost.style.opacity = S.m ? 1 : 0;
      var top = Math.min(by(b.y), by(Fa[i])), bot = Math.max(by(b.y), by(Fa[i]));
      c.miss.style.y = top + 'px'; c.miss.style.height = Math.max(0, bot - top) + 'px';
      c.miss.style.fill = 'var(--' + (b.y >= Fa[i] ? 'emerald' : 'ruby') + ')';
      c.name.textContent = i === S.follow ? '▲' : '';
    });
  }

  /* --------------------------------------------------------------- tree -- */
  var POS = { r: 310, rL: 155, rR: 465, rLL: 78, rLR: 233, rRL: 388, rRR: 543 };
  function drawTree() {
    var svg = $('tree'); svg.textContent = '';
    var r = rd(), run = R(), xg = S.mode === 'xgb';
    if (!r) {
      svg.setAttribute('viewBox', '0 0 620 120');
      el('rect', { x: 210, y: 20, width: 200, height: 64, rx: 10, 'class': 'leaf-box', style: 'fill:var(--sapphire-soft);stroke:var(--sapphire)' }, svg);
      txt(svg, 310, 47, 'F₀ = ' + fx(run.F0), 't-val');
      txt(svg, 310, 68, 'the average of all 12', 't-sub');
      $('path').textContent = 'No engineer yet. Every bridge starts at the average.';
      return;
    }
    var on = {}; G.route(r.tree, B[S.follow].x).forEach(function (n) { on[n.id] = true; });
    var IW = 182, IH = 56, LW = 142, LH = 58, Y = [12, 120, 228];
    var gE = el('g', {}, svg), gN = el('g', {}, svg), maxY = 0;
    (function draw(n) {
      var x = POS[n.id], y = Y[n.depth];
      var g = el('g', { 'class': (n.leaf ? 'leaf' : 'node') + (on[n.id] ? ' on' : '') + (n.id === S.node ? ' picked' : ''), 'data-id': n.id }, gN);
      if (n.leaf) {
        var c = n.w >= 0 ? 'emerald' : 'ruby';
        el('rect', { x: x - LW / 2, y: y, width: LW, height: LH, rx: 9, 'class': 'leaf-box',
          style: 'fill:var(--' + c + '-soft);stroke:var(--' + c + ')' + (n.pruned ? ';stroke-dasharray:5 4' : '') }, g);
        var t = txt(g, x, y + 21, (xg ? 'w = ' : 'avg miss ') + sg(n.w), 't-val'); t.setAttribute('style', 'fill:var(--' + c + ')');
        txt(g, x, y + 37, 'n ' + n.rows.length + ' · ηw ' + sg(S.eta * n.w), 't-sub');
        if (n.pruned) { var pt = txt(g, x, y + 51, 'pruned ' + G.rule(n.pruned.f, n.pruned.thr).replace(' ≤ ', '≤'), 't-sub'); pt.setAttribute('style', 'fill:var(--ruby)'); }
        if (n.pruned) g.setAttribute('class', g.getAttribute('class') + ' node');
        maxY = Math.max(maxY, y + LH);
        return;
      }
      el('rect', { x: x - IW / 2, y: y, width: IW, height: IH, rx: 7, 'class': 'node-box' }, g);
      txt(g, x, y + 22, G.rule(n.f, n.thr) + '?', 't-rule');
      txt(g, x, y + 41, n.rows.length + ' bridges · ' + (xg ? 'gain ' + fx(n.best.gain, 1) : 'drop ' + fx(2 * n.best.raw, 1)), 't-sub');
      maxY = Math.max(maxY, y + IH);
      [n.left, n.right].forEach(function (ch, j) {
        var x2 = POS[ch.id], y2 = Y[ch.depth];
        el('path', { d: 'M' + x + ',' + (y + IH) + ' C' + x + ',' + (y + IH + 22) + ' ' + x2 + ',' + (y2 - 22) + ' ' + x2 + ',' + y2,
          'class': 'edge' + (on[n.id] && on[ch.id] ? ' on' : '') }, gE);
        txt(gE, (x + x2) / 2 + (j ? 14 : -14), (y + IH + y2) / 2 + 4, j ? 'no' : 'yes', 't-yn');
        draw(ch);
      });
    })(r.tree);
    svg.setAttribute('viewBox', '0 0 620 ' + (maxY + 8));
    var steps = [], p = G.route(r.tree, B[S.follow].x);
    for (var i = 0; i < p.length - 1; i++) steps.push(G.rule(p[i].f, p[i].thr) + '? <b>' + (B[S.follow].x[p[i].f] <= p[i].thr ? 'yes' : 'no') + '</b>');
    var lf = p[p.length - 1];
    $('path').innerHTML = '<b>' + B[S.follow].name + '</b>: ' + (steps.length ? steps.join(' → ') + ' → ' : '') +
      'leaf ' + (xg ? 'w = ' : 'average miss ') + '<span class="' + (lf.w >= 0 ? 'pos' : 'neg') + '">' + sg(lf.w) + '</span>';
  }

  /* -------------------------------------------------------------- steps -- */
  function step(n, title, plain, eq, live) {
    return '<article class="step"><span class="step-n">Step ' + n + '</span><h4>' + title + '</h4><p class="plain">' + plain + '</p>' +
      '<div class="eq">' + T(eq, true) + '</div><div class="live">' + live + '</div></article>';
  }
  function drawSteps() {
    var run = R(), r = rd(), i = S.follow, b = B[i], m = S.m, xg = S.mode === 'xgb', h = '';
    var mm = Math.max(m, 1);
    var Fb = r ? r.before[i] : run.F0, Fa = r ? r.after[i] : run.F0;
    var leaf = r ? G.leafOf(r.tree, b.x) : null;
    var nm = '<b>' + b.name + '</b>';
    if (!xg) {
      h += step(1, 'Measure the misses',
        'For every bridge, take what really happened minus the current forecast. A positive miss means the bridge outlasted the forecast.',
        'r_i = y_i - F_{' + (mm - 1) + '}(x_i) = -\\frac{\\partial \\ell}{\\partial F}\\Big|_{F_{' + (mm - 1) + '}}',
        nm + ': ' + T('r = ' + b.y + ' - ' + tn(Fb) + ' = ' + tn(b.y - Fb)) + '<br><span class="muted">With ' + T('\\ell = \\tfrac12 (y-F)^2') + ' the slope is ' + T('F - y') + ', so the miss is minus the slope: the downhill direction.</span>');
      h += step(2, 'Fit a small tree to the misses',
        'Try every yes/no question about age, strength and exposure. Keep the one that sorts the bridges into groups whose misses are most alike. Each leaf&rsquo;s correction is the average miss of the bridges in it.',
        '\\bar r_\\ell = \\frac{1}{n_\\ell}\\sum_{i \\in \\ell} r_i',
        r ? nm + ' lands in the leaf for ' + leafDesc(r.tree, leaf) + ': ' + leaf.rows.length + ' bridges, average miss ' + T(tn(leaf.w)) + '.' : '<span class="muted">Press Next round to fit the first tree.</span>');
      h += step(3, 'Accept only part of it',
        'Taking the whole correction would chase the noise in twelve bridges. Dalinar accepts a fraction &eta; and leaves the rest for later engineers.',
        'F_{' + mm + '}(x) = F_{' + (mm - 1) + '}(x) + \\eta\\,\\bar r_{\\ell(x)}',
        r ? T(tn(Fb) + ' + ' + tn(S.eta) + ' \\times ' + tparen(leaf.w) + ' = ' + tn(Fa)) + ' storms for ' + nm + '.' : '<span class="muted">The forecast is still ' + fx(run.F0) + ' for everyone.</span>');
      h += step(4, 'Check the penalty',
        'Add up the penalty for every bridge. Each round it should shrink, because each tree points downhill.',
        'L = \\sum_{i=1}^{12} \\tfrac12\\big(y_i - F(x_i)\\big)^2',
        r ? T('L: ' + tn(r.lossBefore, 1) + ' \\to ' + tn(r.lossAfter, 1)) + '<br>typical miss ' + fx(G.rmse(run.y, r.before)) + ' → <b>' + fx(G.rmse(run.y, r.after)) + '</b> highstorms' :
          T('L = ' + tn(G.loss(run.y, Fnow(run)), 1)) + '<br>typical miss <b>' + fx(G.rmse(run.y, Fnow(run))) + '</b> highstorms');
    } else {
      var g = r ? r.g[i] : run.F0 - b.y;
      h += step(1, 'Measure slope and curvature',
        'XGBoost writes the miss as a slope <i>g</i>: which way the penalty rises. It also records the curvature <i>h</i>: how fast that slope changes. For squared error the curvature is 1 for every bridge, so a group&rsquo;s <i>H</i> simply counts its bridges.',
        'g_i = \\frac{\\partial \\ell}{\\partial F} = F_{' + (mm - 1) + '}(x_i) - y_i,\\qquad h_i = \\frac{\\partial^2 \\ell}{\\partial F^2} = 1',
        nm + ': ' + T('g = ' + tn(Fb) + ' - ' + b.y + ' = ' + tn(g)) + ', ' + T('h = 1') + '<br><span class="muted">' + T('g') + ' is minus mode 1&rsquo;s miss.</span>');
      var root = r ? r.tree : null, rb = root && (root.best || (root.pruned && root.pruned.best));
      h += step(2, 'Score questions, with brakes',
        'Same search, but a group&rsquo;s score is <i>G</i>&sup2;/(<i>H</i>+&lambda;). &lambda; acts like &lambda; extra bridges with zero miss, so small groups look less certain. A question must also pay its price &gamma;, or it is pruned.',
        '\\text{gain} = \\tfrac12\\Big[\\tfrac{G_L^2}{H_L+\\lambda} + \\tfrac{G_R^2}{H_R+\\lambda} - \\tfrac{G^2}{H+\\lambda}\\Big] - \\gamma',
        rb ? 'Root question ' + G.rule(rb.f, rb.thr) + '?: ' + T('\\tfrac12\\big[\\tfrac{' + tn(rb.GL, 1) + '^2}{' + rb.HL + '+' + S.lambda + '} + \\tfrac{' + tn(rb.GR, 1) + '^2}{' + rb.HR + '+' + S.lambda + '} - \\tfrac{' + tn(rb.GL + rb.GR, 1) + '^2}{' + (rb.HL + rb.HR) + '+' + S.lambda + '}\\big] - ' + S.gamma + ' = ' + tn(rb.gain, 1)) :
          (r ? '<span class="muted">No question could split the root this round.</span>' : '<span class="muted">Press Next round to score the first questions.</span>'));
      h += step(3, 'Set each leaf, then step',
        'XGBoost fits a parabola to each leaf&rsquo;s penalty, using the slopes and curvatures, and jumps to its bottom: &minus;<i>G</i>/(<i>H</i>+&lambda;). That is the average miss, pulled toward zero by &lambda;. For squared error the parabola is exact. Then it accepts a fraction &eta;.',
        'w_\\ell = -\\frac{G_\\ell}{H_\\ell + \\lambda},\\qquad F_{' + mm + '} = F_{' + (mm - 1) + '} + \\eta\\, w_{\\ell(x)}',
        r ? nm + '&rsquo;s leaf: ' + T('w = -\\tfrac{' + tn(leaf.G) + '}{' + leaf.H + ' + ' + S.lambda + '} = ' + tn(leaf.w)) + ' (plain average ' + sg(meanMiss(leaf)) + ')<br>' +
          T(tn(Fb) + ' + ' + tn(S.eta) + '\\times' + tparen(leaf.w) + ' = ' + tn(Fa)) : '<span class="muted">The forecast is still ' + fx(run.F0) + ' for everyone.</span>');
      h += step(4, 'Check the penalty and the price',
        'XGBoost&rsquo;s objective adds a price for complexity: &gamma; for every leaf, and &frac12;&lambda;<i>w</i>&sup2; for every bold leaf weight. A simpler, more cautious tree costs less.',
        '\\text{Obj} = \\sum_i \\ell(y_i, F(x_i)) + \\gamma T + \\tfrac12\\lambda\\sum_{\\ell=1}^{T} w_\\ell^2',
        r ? T('L: ' + tn(r.lossBefore, 1) + ' \\to ' + tn(r.lossAfter, 1)) + ', tree price ' + T('\\Omega = ' + tn(r.omega, 2)) + ' (' + r.leaves + ' leaves)<br>typical miss ' +
          fx(G.rmse(run.y, r.before)) + ' → <b>' + fx(G.rmse(run.y, r.after)) + '</b> highstorms' :
          T('L = ' + tn(G.loss(run.y, Fnow(run)), 1)) + '<br>typical miss <b>' + fx(G.rmse(run.y, Fnow(run))) + '</b> highstorms');
    }
    var box = $('steps'); box.innerHTML = h; renderTex(box);
  }

  /* --------------------------------------------------------- candidates -- */
  function drawCands() {
    var r = rd(), xg = S.mode === 'xgb', tbl = $('cand');
    if (!r) {
      $('candNote').textContent = 'At round 0 there is no tree yet. After you press Next round, this table lists every yes/no question the engineer tried.';
      $('candHint').textContent = ''; tbl.innerHTML = '';
      return;
    }
    var n = findNode(r.tree, S.node);
    if (!n || (n.leaf && !n.pruned)) n = r.tree;
    var cands = n.cands || [];
    var chosen = n.leaf ? null : n.best;
    $('candHint').textContent = (n.id === 'r' ? 'root' : 'node ' + n.id.slice(1).replace(/L/g, 'yes→').replace(/R/g, 'no→').replace(/→$/, '')) + ' · ' + n.rows.length + ' bridges';
    $('candNote').innerHTML = !cands.length ? 'This node was not split: it is at the depth limit or too small to split (every leaf keeps at least 2 bridges).' :
      xg ? 'Every question tried here, best first. <i>G</i> sums the slopes <i>g</i>, <i>H</i> counts bridges. Gain is after paying &gamma; = ' + S.gamma +
        '. ' + (n.pruned ? '<b class="neg">The best gain is negative, so the split was pruned.</b>' : 'The top row became this round&rsquo;s question.') :
      'Every question tried here, best first. &ldquo;Drop&rdquo; is how much the summed squared misses shrink if the bridges are split this way. The top row became this round&rsquo;s question.';
    if (!cands.length) { tbl.innerHTML = ''; return; }
    var head = '<thead><tr><th class="l">Question</th><th>yes / no</th><th>G<sub>yes</sub></th><th>G<sub>no</sub></th>' +
      (xg ? '<th>H<sub>yes</sub></th><th>H<sub>no</sub></th><th>gain</th>' : '<th>drop in squared misses</th>') + '</tr></thead>';
    var rows = cands.slice(0, 9).map(function (c, k) {
      var cls = k === 0 && chosen ? 'best' : (xg && c.gain < 0 ? 'loses' : '');
      return '<tr class="' + cls + '"><td class="l">' + G.rule(c.f, c.thr) + '?</td><td>' + c.nL + ' / ' + c.nR + '</td><td>' + fx(c.GL, 1) + '</td><td>' + fx(c.GR, 1) + '</td>' +
        (xg ? '<td>' + c.HL + '</td><td>' + c.HR + '</td><td class="' + (c.gain >= 0 ? 'pos' : 'neg') + '">' + fx(c.gain, 2) + '</td>' : '<td>' + fx(2 * c.raw, 2) + '</td>') + '</tr>';
    }).join('');
    var more = cands.length > 9 ? '<tr><td class="l muted" colspan="' + (xg ? 7 : 5) + '">… and ' + (cands.length - 9) + ' weaker questions</td></tr>' : '';
    tbl.innerHTML = head + '<tbody>' + rows + more + '</tbody>';
  }

  /* --------------------------------------------------------------- loss -- */
  function drawLoss() {
    var svg = $('lossc'); svg.textContent = '';
    var W = 460, H = 240, L = 40, Rr = 10, Tt = 14, Bb = 34;
    var series = function (run) { return [G.rmse(run.y, run.rounds[0].before)].concat(run.rounds.map(function (r) { return G.rmse(run.y, r.after); })); };
    var main = series(R()), gb = series(RUN.gb);
    var ymax = Math.ceil(Math.max.apply(null, main.concat(gb)));
    var X = function (m) { return L + m / M * (W - L - Rr); }, Y = function (v) { return H - Bb - v / ymax * (H - Tt - Bb); };
    var step = ymax > 6 ? 2 : 1;
    for (var v = 0; v <= ymax; v += step) { el('line', { x1: L, x2: W - Rr, y1: Y(v), y2: Y(v), 'class': 'grid-l' }, svg); txt(svg, L - 6, Y(v) + 4, String(v), '', 'end'); }
    [0, 5, 10, 15, 20].forEach(function (m) { txt(svg, X(m), H - Bb + 15, String(m), ''); });
    txt(svg, (L + W) / 2, H - 4, 'round', 'lab');
    function path(a, cls) { el('path', { d: 'M' + a.map(function (v, m) { return X(m).toFixed(1) + ',' + Y(v).toFixed(1); }).join('L'), 'class': cls }, svg); }
    if (S.mode === 'xgb') path(gb, 'l-gb');
    path(main, 'l-main');
    el('line', { x1: X(S.m), x2: X(S.m), y1: Tt, y2: H - Bb, 'class': 'now' }, svg);
    el('circle', { cx: X(S.m), cy: Y(main[S.m]), r: 5, 'class': 'dot' }, svg);
    if (S.mode === 'xgb') {
      el('line', { x1: W - 170, x2: W - 148, y1: Tt + 4, y2: Tt + 4, 'class': 'l-main' }, svg); txt(svg, W - 143, Tt + 8, 'XGBoost', 'lab', 'start');
      el('line', { x1: W - 84, x2: W - 62, y1: Tt + 4, y2: Tt + 4, 'class': 'l-gb' }, svg); txt(svg, W - 57, Tt + 8, 'mode 1', 'lab', 'start');
    }
    $('lossHint').textContent = 'round ' + S.m + ': ' + fx(main[S.m]) + ' highstorms';
    if (S.mode === 'gb') {
      $('lossNote').innerHTML = 'The typical miss is <span class="tex" data-tex="\\sqrt{\\tfrac{1}{12}\\sum_i (y_i-F(x_i))^2}"></span>, in highstorms. It keeps falling. With only twelve bridges, enough rounds will memorize them, noise included. That is why real projects watch bridges the model never trained on.';
    } else {
      var single = RUN.xgb.rounds.filter(function (r) { return r.tree.leaf; }).length;
      $('lossNote').innerHTML = 'The dashed line is mode 1 with the same &eta; and depth. With &lambda; = ' + S.lambda + ' and &gamma; = ' + S.gamma + ', XGBoost takes smaller steps' +
        (single ? ' and, in ' + single + ' of 20 rounds, no question was worth its price, so the engineer asked nothing' : '') + '. It fits the training bridges less tightly on purpose.';
    }
    renderTex($('lossNote'));
  }

  /* -------------------------------------------------------- round table -- */
  function drawRoundTable() {
    var run = R(), r = rd(), xg = S.mode === 'xgb';
    var Fb = Fprev(run), Fa = Fnow(run);
    var head = '<thead><tr><th class="l">#</th><th class="l">Bridge</th><th>age</th><th>str.</th><th>exp.</th><th>storms y</th><th>F before</th>' +
      (xg ? '<th>slope g</th><th>leaf w</th>' : '<th>miss r</th><th>leaf avg</th>') + '<th>η × leaf</th><th>F after</th></tr></thead>';
    var body = B.map(function (b, i) {
      var lw = r ? G.leafOf(r.tree, b.x).w : null;
      var miss = b.y - Fb[i];
      return '<tr data-i="' + i + '"' + (i === S.follow ? ' class="sel"' : '') + '><td class="l idx">' + (i + 1) + '</td><td class="l name">' + b.name + '</td>' +
        '<td>' + b.x[0] + '</td><td>' + b.x[1] + '</td><td>' + b.x[2] + '</td><td class="c-top">' + b.y + '</td><td>' + fx(Fb[i]) + '</td>' +
        (xg ? '<td class="' + (miss >= 0 ? 'pos' : 'neg') + '">' + (r ? sg(-miss) : '–') + '</td>' : '<td class="' + (miss >= 0 ? 'pos' : 'neg') + '">' + (r ? sg(miss) : '–') + '</td>') +
        '<td>' + (r ? sg(lw) : '–') + '</td><td>' + (r ? sg(S.eta * lw) : '–') + '</td><td class="c-sap">' + fx(Fa[i]) + '</td></tr>';
    }).join('');
    $('rt').innerHTML = head + '<tbody>' + body + '</tbody>';
  }

  /* ------------------------------------------------------------- ledger -- */
  function drawLedger() {
    var h = '<thead><tr><th class="l">#</th><th class="l">Bridge</th><th>Age</th><th>Strength</th><th>Exposure</th><th>Storms</th></tr></thead><tbody>' +
      B.map(function (b, i) {
        return '<tr data-i="' + i + '"><td class="l idx">' + (i + 1) + '</td><td class="l name">' + b.name + '</td><td>' + b.x[0] + '</td><td>' + b.x[1] + '</td><td>' + b.x[2] + '</td><td class="c-top">' + b.y + '</td></tr>';
      }).join('') + '</tbody>';
    $('ledger').innerHTML = h;
  }

  /* ========================================================= controls == */
  function stop() { if (S.timer) { clearInterval(S.timer); S.timer = null; } }
  function setRound(m) { S.m = Math.max(0, Math.min(M, m)); render(); }
  function follow(i) { if (i >= 0 && i < B.length) { S.follow = i; render(); } }
  function setMode(mode, fromHash) {
    S.mode = mode === 'xgb' ? 'xgb' : 'gb';
    if (!fromHash) { try { history.replaceState(null, '', '#' + HASH[S.mode]); } catch (e) { /* ignore */ } }
    render();
  }
  function knob() { recompute(); render(); }

  function init() {
    renderTex(document);
    drawLedger();
    recompute();
    var tabs = Array.prototype.slice.call(document.querySelectorAll('.mode'));
    tabs.forEach(function (t, k) {
      t.addEventListener('click', function () { setMode(t.getAttribute('data-mode')); });
      t.addEventListener('keydown', function (e) {
        if (e.key !== 'ArrowLeft' && e.key !== 'ArrowRight') return;
        e.preventDefault(); var o = tabs[1 - k]; o.focus(); setMode(o.getAttribute('data-mode'));
      });
    });
    $('back').addEventListener('click', function () { stop(); setRound(S.m - 1); });
    $('next').addEventListener('click', function () { stop(); setRound(S.m + 1); });
    $('play').addEventListener('click', function () {
      if (S.timer) { stop(); render(); return; }
      if (S.m >= M) S.m = 0;
      S.timer = setInterval(function () { if (S.m >= M) { stop(); render(); return; } S.m++; render(); }, 1100);
      render();
    });
    $('reset').addEventListener('click', function () {
      stop();
      ['m', 'eta', 'depth', 'lambda', 'gamma', 'follow', 'node'].forEach(function (k) { S[k] = DEFAULTS[k]; });
      knob();
    });
    $('round').addEventListener('input', function () { stop(); setRound(+$('round').value); });
    $('eta').addEventListener('input', function () { S.eta = +$('eta').value; knob(); });
    $('lambda').addEventListener('input', function () { S.lambda = +$('lambda').value; knob(); });
    $('gamma').addEventListener('input', function () { S.gamma = +$('gamma').value; knob(); });
    $('zeroBrakes').addEventListener('click', function () { S.lambda = 0; S.gamma = 0; knob(); });
    document.querySelectorAll('[data-depth]').forEach(function (b) { b.addEventListener('click', function () { S.depth = +b.getAttribute('data-depth'); knob(); }); });
    $('bars').addEventListener('click', function (e) { var i = e.target.getAttribute('data-i'); if (i !== null) follow(+i); });
    ['rt', 'ledger'].forEach(function (id) { $(id).addEventListener('click', function (e) { var tr = e.target.closest('tr[data-i]'); if (tr) follow(+tr.getAttribute('data-i')); }); });
    $('tree').addEventListener('click', function (e) { var g = e.target.closest('g.node'); if (g) { S.node = g.getAttribute('data-id'); render(); } });
    document.addEventListener('keydown', function (e) {
      var t = e.target, tag = t && t.tagName;
      if (tag === 'INPUT' || tag === 'SELECT' || (t.getAttribute && t.getAttribute('role') === 'tab')) return;
      if (e.key === 'ArrowRight') { stop(); setRound(S.m + 1); } else if (e.key === 'ArrowLeft') { stop(); setRound(S.m - 1); }
    });
    window.addEventListener('hashchange', function () { setMode(location.hash === '#xgboost' ? 'xgb' : 'gb', true); });
    S.mode = location.hash === '#xgboost' ? 'xgb' : 'gb';
    render();
  }
  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', init); else init();
})();
