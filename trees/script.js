/* ============================================================================
   Build a Decision Tree with LEGO — CART and ensembles.
   CS 473 module 13 · Murphy, PML §18.1–18.4

   Part 1 is pure math (exact fractions, Gini, MSE, CART, bootstrap, voting) and
   is exported under Node for test-math.js. Part 2 draws the bench, the trees and
   the forest. Every number on the page comes from Part 1 applied to the bricks shown.
   ========================================================================= */

(function (global) {
  'use strict';

  /* ==========================================================================
     PART 1 — MATH (no DOM; exported under Node for test-math.js)
     ====================================================================== */

  /* Exact fractions, so 1/3 on the page is 1/3 in the arithmetic. */
  function gcd(a, b) { a = Math.abs(a); b = Math.abs(b); while (b) { var t = a % b; a = b; b = t; } return a || 1; }
  function F(n, d) {
    if (d === undefined) d = 1;
    if (d < 0) { n = -n; d = -d; }
    var g = gcd(n, d);
    return { n: n / g, d: d / g };
  }
  function fadd(a, b) { return F(a.n * b.d + b.n * a.d, a.d * b.d); }
  function fsub(a, b) { return F(a.n * b.d - b.n * a.d, a.d * b.d); }
  function fmul(a, b) { return F(a.n * b.n, a.d * b.d); }
  function fcmp(a, b) { return a.n * b.d - b.n * a.d; }
  function feq(a, b) { return a.n === b.n && a.d === b.d; }
  function fval(a) { return a.n / a.d; }

  /* ---- data ----
     Scenes 1-2: one feature x (the baseplate position).
     Each row is { id, x: [features...], c: class 0/1, v: numeric target }. */
  var CLS = [0, 0, 1, 1, 0, 0].map(function (c, i) { return { id: i, x: [i + 1], c: c }; });
  var REG = [2, 4, 6, 10, 12, 14].map(function (v, i) { return { id: i, x: [i + 1], v: v }; });

  /* Scene 3: two features. x1 = column on the plate (1-8), x2 = row (1-3). */
  var ENS = [
    ['A', 1, 1, 0, 2], ['B', 2, 3, 0, 4], ['C', 3, 2, 0, 4], ['D', 4, 1, 0, 4], ['E', 4, 3, 1, 6],
    ['F', 5, 2, 1, 8], ['G', 6, 1, 0, 6], ['H', 6, 3, 1, 10], ['I', 7, 2, 1, 10], ['J', 8, 1, 1, 8]
  ].map(function (r, i) { return { id: i, name: r[0], x: [r[1], r[2]], c: r[3], v: r[4] }; });
  var ENS_TEST = [5, 1];
  var ENS_DEPTH = 2;

  function labelOf(mode, r) { return mode === 'cls' ? r.c : r.v; }

  function classCounts(rows) {
    var k = [0, 0];
    rows.forEach(function (r) { k[r.c]++; });
    return k;
  }
  /* Gini impurity G = 1 - sum_c p_c^2 */
  function gini(rows) {
    var n = rows.length, g = F(1);
    if (!n) return F(0);
    classCounts(rows).forEach(function (k) { g = fsub(g, fmul(F(k, n), F(k, n))); });
    return g;
  }
  function mean(rows) {
    var s = 0;
    rows.forEach(function (r) { s += r.v; });
    return F(s, rows.length);
  }
  /* MSE = (1/n) sum (y_i - ybar)^2 */
  function mse(rows) {
    if (!rows.length) return F(0);
    var m = mean(rows), s = F(0);
    rows.forEach(function (r) { var e = fsub(F(r.v), m); s = fadd(s, fmul(e, e)); });
    return fmul(s, F(1, rows.length));
  }
  function impurity(mode, rows) { return mode === 'cls' ? gini(rows) : mse(rows); }

  /* What a leaf predicts: majority class (ties -> 0), or the mean. */
  function leafValue(mode, rows) {
    if (mode === 'reg') return mean(rows);
    var k = classCounts(rows);
    return k[1] > k[0] ? 1 : 0;
  }

  function splitRows(rows, f, t) {
    return {
      L: rows.filter(function (r) { return r.x[f] <= t; }),
      R: rows.filter(function (r) { return r.x[f] > t; })
    };
  }
  /* Split score  (nL/n) G_L + (nR/n) G_R  */
  function scoreSplit(mode, rows, f, t) {
    var s = splitRows(rows, f, t), n = rows.length;
    var GL = impurity(mode, s.L), GR = impurity(mode, s.R);
    return {
      f: f, t: t, L: s.L, R: s.R, n: n, nL: s.L.length, nR: s.R.length, GL: GL, GR: GR,
      score: fadd(fmul(F(s.L.length, n), GL), fmul(F(s.R.length, n), GR))
    };
  }
  /* Thresholds that leave both trays non-empty: every distinct value but the largest. */
  function candidates(rows, f) {
    var vals = [];
    rows.forEach(function (r) { if (vals.indexOf(r.x[f]) < 0) vals.push(r.x[f]); });
    vals.sort(function (a, b) { return a - b; });
    return vals.slice(0, -1);
  }
  /* Every candidate on feature f, plus the indices of the lowest score (ties kept). */
  function compareSplits(mode, rows, f) {
    var all = candidates(rows, f).map(function (t) { return scoreSplit(mode, rows, f, t); });
    var best = [];
    all.forEach(function (s, k) {
      if (!best.length || fcmp(s.score, all[best[0]].score) < 0) best = [k];
      else if (feq(s.score, all[best[0]].score)) best.push(k);
    });
    return { all: all, best: best };
  }

  /* Seeded RNG (mulberry32), so a sample can be replayed. */
  function rng(seed) {
    var a = seed >>> 0;
    return function () {
      a = (a + 0x6D2B79F5) >>> 0;
      var t = a;
      t = Math.imul(t ^ (t >>> 15), t | 1);
      t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }
  /* Bootstrap: n draws with replacement from n rows (indices). */
  function bootstrap(rand, n) {
    var out = [];
    for (var i = 0; i < n; i++) out.push(Math.floor(rand() * n));
    return out;
  }
  function counts(sample, n) {
    var k = [];
    for (var i = 0; i < n; i++) k.push(0);
    sample.forEach(function (i) { k[i]++; });
    return k;
  }

  /* Greedy CART on a (possibly duplicated) list of rows.
     opts: { mode, maxDepth, nFeatures, rf: bool, mtry, rand }
     Each node records which features were eligible and the best split on each. */
  function growTree(rows, opts) {
    var nextId = 0;
    function grow(rows, depth, path) {
      var node = {
        id: nextId++, path: path, depth: depth, rows: rows, n: rows.length,
        imp: impurity(opts.mode, rows), value: leafValue(opts.mode, rows), leaf: true
      };
      var feats = [];
      for (var f = 0; f < opts.nFeatures; f++) feats.push(f);
      if (opts.rf) {
        // draw mtry features without replacement
        var pool = feats.slice(); feats = [];
        while (feats.length < opts.mtry) feats.push(pool.splice(Math.floor(opts.rand() * pool.length), 1)[0]);
        feats.sort();
      }
      node.eligible = feats;
      node.perFeature = {};
      if (depth >= opts.maxDepth || node.imp.n === 0) return node;
      var best = null;
      feats.forEach(function (f) {
        var c = compareSplits(opts.mode, rows, f);
        if (!c.all.length) { node.perFeature[f] = null; return; }
        var b = c.all[c.best[0]];
        node.perFeature[f] = b;
        if (!best || fcmp(b.score, best.score) < 0) best = b;
      });
      if (!best || fcmp(best.score, node.imp) >= 0) return node;   // no improvement: stay a leaf
      node.leaf = false;
      node.f = best.f; node.t = best.t; node.split = best;
      node.left = grow(best.L, depth + 1, path + 'L');
      node.right = grow(best.R, depth + 1, path + 'R');
      return node;
    }
    return grow(rows, 0, '');
  }
  /* Route x down the tree; returns the visited nodes (root ... leaf). */
  function route(node, x) {
    var path = [node];
    while (!node.leaf) { node = x[node.f] <= node.t ? node.left : node.right; path.push(node); }
    return path;
  }

  /* One ensemble member from an explicit bootstrap sample. */
  function ensembleTree(sample, mode, rf, seed) {
    return growTree(sample.map(function (i) { return ENS[i]; }), {
      mode: mode, maxDepth: ENS_DEPTH, nFeatures: 2, rf: rf, mtry: 1, rand: rng(seed)
    });
  }
  function majority(votes) {
    var k = [0, 0];
    votes.forEach(function (v) { k[v]++; });
    return { counts: k, winner: k[1] > k[0] ? 1 : 0 };
  }
  function average(preds) {
    var s = F(0);
    preds.forEach(function (p) { s = fadd(s, p); });
    return fmul(s, F(1, preds.length));
  }
  /* P(majority of m independent voters is wrong) when each is wrong with prob e. */
  function majorityError(e, m) {
    var p = 0, need = Math.floor(m / 2) + 1;
    function choose(n, k) { var c = 1; for (var i = 1; i <= k; i++) c = c * (n - k + i) / i; return c; }
    for (var k = need; k <= m; k++) p += choose(m, k) * Math.pow(e, k) * Math.pow(1 - e, m - k);
    return p;
  }

  /* Preset bootstrap samples (indices into ENS), found by searching seeds and checked
     in test-math.js. With the test brick at (5, 1) and random forest off:
       classify: the three trees vote 1, 0, 1
       regress:  the three trees predict 4, 6, 8 (average 6)                       */
  var PRESETS = {
    cls: [[1, 2, 7, 5, 1, 2, 8, 7, 0, 7], [6, 0, 1, 1, 6, 3, 2, 7, 6, 7], [0, 3, 1, 9, 5, 3, 5, 1, 8, 1]],
    reg: [[4, 4, 3, 8, 0, 4, 0, 3, 4, 5], [6, 0, 1, 1, 6, 3, 2, 7, 6, 7], [1, 2, 7, 5, 1, 2, 8, 7, 0, 7]]
  };

  var MATH = {
    F: F, fadd: fadd, fsub: fsub, fmul: fmul, fcmp: fcmp, feq: feq, fval: fval,
    CLS: CLS, REG: REG, ENS: ENS, ENS_TEST: ENS_TEST, ENS_DEPTH: ENS_DEPTH, PRESETS: PRESETS,
    gini: gini, mean: mean, mse: mse, impurity: impurity, leafValue: leafValue, classCounts: classCounts,
    splitRows: splitRows, scoreSplit: scoreSplit, candidates: candidates, compareSplits: compareSplits,
    rng: rng, bootstrap: bootstrap, counts: counts, growTree: growTree, route: route,
    ensembleTree: ensembleTree, majority: majority, average: average, majorityError: majorityError
  };

  if (typeof module !== 'undefined' && module.exports) module.exports = MATH;
  if (typeof document === 'undefined') return;

  /* ==========================================================================
     PART 2 — DRAWING
     ====================================================================== */

  var NS = 'http://www.w3.org/2000/svg';
  var $ = function (sel, root) { return (root || document).querySelector(sel); };

  function el(tag, attrs, parent) {
    var e = document.createElementNS(NS, tag);
    if (attrs) for (var k in attrs) e.setAttribute(k, attrs[k]);
    if (parent) parent.appendChild(e);
    return e;
  }
  function txt(parent, x, y, cls, s) { var t = el('text', { x: x, y: y, 'class': cls }, parent); t.textContent = s; return t; }
  function clear(g) { while (g.firstChild) g.removeChild(g.firstChild); }
  function setT(node, x, y) { node.style.transform = 'translate(' + x + 'px,' + y + 'px)'; }

  function tex(node, src, display) {
    if (global.katex) {
      try { global.katex.render(src, node, { throwOnError: false, displayMode: !!display }); return; } catch (e) { /* fall through */ }
    }
    node.textContent = src;
    node.classList.add('tex-raw');
  }
  function renderTex(root) {
    root.querySelectorAll('[data-tex]').forEach(function (n) { tex(n, n.getAttribute('data-tex'), n.classList.contains('tex-block')); });
  }
  function texFill(root) {
    root.querySelectorAll('[data-t]').forEach(function (n) { tex(n, n.getAttribute('data-t'), n.hasAttribute('data-d')); });
  }
  function T(src, display) { return '<span data-t="' + src.replace(/"/g, '&quot;') + '"' + (display ? ' data-d' : '') + '></span>'; }

  /* ---- number formatting ---- */
  var BLUE = '#1e5fb4', RED = '#c8281e';
  function shortDec(f) {                       // exact decimal with <= 3 places, else null
    var d = f.d; while (d % 2 === 0) d /= 2; while (d % 5 === 0) d /= 5;
    if (d !== 1) return null;
    var s = fval(f).toFixed(3).replace(/0+$/, '').replace(/\.$/, '');
    return s.length <= 7 ? s : null;
  }
  function fracTex(f) {                          // always a fraction (classification)
    if (f.d === 1) return String(f.n);
    return (f.n < 0 ? '-' : '') + '\\frac{' + Math.abs(f.n) + '}{' + f.d + '}';
  }
  function numTex(f) {                           // decimal when exact and short (regression)
    if (f.d === 1) return String(f.n);
    var s = shortDec(f);
    return s !== null ? s : fracTex(f);
  }
  function approx(f) { return f.d === 1 || shortDec(f) !== null ? '' : '\\approx ' + fval(f).toFixed(2); }
  function fracTxt(f) { return f.d === 1 ? String(f.n) : f.n + '/' + f.d; }
  function numTxt(f) {
    if (f.d === 1) return String(f.n);
    var s = shortDec(f);
    return s !== null ? s : fval(f).toFixed(2);
  }
  function paren(s) { return /^[0-9.]+$/.test(s) ? '(' + s + ')' : '\\left(' + s + '\\right)'; }
  function clsName(c) { return c === 1 ? 'red' : 'blue'; }

  /* Gini arithmetic for one tray. */
  function giniTex(rows) {
    var k = classCounts(rows), n = rows.length;
    return '1-{\\color{' + BLUE + '}\\left(\\frac{' + k[0] + '}{' + n + '}\\right)^2}-{\\color{' + RED + '}\\left(\\frac{' + k[1] + '}{' + n + '}\\right)^2}=' + fracTex(gini(rows));
  }
  function meanTex(rows) {
    return '\\frac{' + rows.map(function (r) { return r.v; }).join('+') + '}{' + rows.length + '}=' + numTex(mean(rows)) + approx(mean(rows));
  }
  function mseTex(rows) {
    var m = numTex(mean(rows));
    var terms = rows.map(function (r) { return '(' + r.v + '-' + m + ')^2'; });
    return '\\frac{' + terms.join('+') + '}{' + rows.length + '}=' + numTex(mse(rows)) + approx(mse(rows));
  }

  /* ---- LEGO bricks ----
     Side view, 76 x 40 body with two studs on top. Local origin = top-left of body. */
  var BW = 76, BH = 40;
  function brickShape(g) {
    el('rect', { x: 11, y: -6, width: 17, height: 7, rx: 2, 'class': 'b-stud' }, g);
    el('rect', { x: 48, y: -6, width: 17, height: 7, rx: 2, 'class': 'b-stud' }, g);
    el('rect', { x: 0, y: 0, width: BW, height: BH, rx: 3, 'class': 'b-body' }, g);
    el('rect', { x: 3, y: 3, width: BW - 6, height: 3, rx: 1.5, 'class': 'b-shine' }, g);
    el('rect', { x: 1, y: BH - 6, width: BW - 2, height: 5, rx: 2, 'class': 'b-shade' }, g);
  }

  /* Top view (for the 2-feature plate and the bootstrap strips). */
  function topBrick(parent, cls, size, main, sub) {
    var g = el('g', { 'class': 'tb ' + cls }, parent), h = size / 2;
    el('rect', { x: -h, y: -h, width: size, height: size, rx: 4, 'class': 'tb-body' }, g);
    el('circle', { cx: 0, cy: 0, r: size * 0.3, 'class': 'tb-stud' }, g);
    if (main != null) txt(g, 0, sub != null ? -1 : size * 0.12, 'tb-text', main).style.fontSize = Math.round(size * 0.3) + 'px';
    if (sub != null) txt(g, 0, size * 0.36, 'tb-sub', sub);
    return g;
  }

  /* ---- generic tree drawing ----
     node API: { kids: [l, r] | null, box: 'split'|'open'|'leaf0'|'leaf1'|'leafv', line1, line2, key, edge: [lText, rText] } */
  var NW = 100, NH = 38, DX = 112, DY = 84;
  function layoutTree(root) {
    var leaves = 0, depth = 0, all = [];
    (function walk(n, d) {
      n.d = d; depth = Math.max(depth, d); all.push(n);
      if (n.kids) { walk(n.kids[0], d + 1); walk(n.kids[1], d + 1); n.x = (n.kids[0].x + n.kids[1].x) / 2; }
      else { n.x = leaves * DX + DX / 2; leaves++; }
      n.y = d * DY + 14;
    })(root, 0);
    return { all: all, W: Math.max(leaves * DX, 3 * DX), H: (depth + 1) * DY - DY + NH + 22, leaves: leaves };
  }
  function drawTree(svg, root, opts) {
    clear(svg);
    var L = layoutTree(root);
    var off = (L.W - L.leaves * DX) / 2;
    L.all.forEach(function (n) { n.x += off; });
    svg.setAttribute('viewBox', '0 0 ' + L.W + ' ' + L.H);
    var gE = el('g', null, svg), gN = el('g', null, svg);
    var onPath = opts.path || {};
    L.all.forEach(function (n) {
      if (!n.kids) return;
      n.kids.forEach(function (k, i) {
        var x1 = n.x, y1 = n.y + NH, x2 = k.x, y2 = k.y - 6;
        el('path', { d: 'M' + x1 + ',' + y1 + ' C' + x1 + ',' + (y1 + 24) + ' ' + x2 + ',' + (y2 - 24) + ' ' + x2 + ',' + y2,
          'class': 't-edge' + (onPath[k.key] ? ' is-path' : '') }, gE);
        txt(gE, (x1 + x2) / 2 + (i ? 22 : -22), (y1 + y2) / 2 + 2, 't-elab', n.edge[i]);
      });
    });
    L.all.forEach(function (n) {
      var g = el('g', { 'class': 't-node t-' + n.box + (opts.fresh && opts.fresh(n.key) ? ' is-new' : '') +
        (opts.pulse && opts.pulse(n) ? ' is-pulse' : '') + (onPath[n.key] ? ' is-visited' : ''),
        transform: 'translate(' + (n.x - NW / 2) + ',' + n.y + ')', tabindex: 0, role: 'button', 'aria-label': n.aria || (n.line1 + ', ' + n.line2) }, gN);
      el('rect', { x: 18, y: -6, width: 18, height: 7, rx: 2, 'class': 't-stud' }, g);
      el('rect', { x: NW - 36, y: -6, width: 18, height: 7, rx: 2, 'class': 't-stud' }, g);
      el('rect', { x: 0, y: 0, width: NW, height: NH, rx: 5, 'class': 't-box' }, g);
      txt(g, NW / 2, 17, 't-text', n.line1);
      txt(g, NW / 2, 31, 't-sub', n.line2);
      if (opts.sel === n.key) el('rect', { x: -5, y: -10, width: NW + 10, height: NH + 15, rx: 8, 'class': 't-sel' }, g);
      if (opts.onClick) {
        g.addEventListener('click', function () { opts.onClick(n); });
        g.addEventListener('keydown', function (e) { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); opts.onClick(n); } });
      }
    });
    var pos = {};
    L.all.forEach(function (n) { pos[n.key] = { x: n.x, y: n.y }; });
    return pos;
  }
  /* A small brick that rides down the tree. */
  function makeToken(svg, cls, label) {
    var g = el('g', { 'class': 'token' }, svg), b = el('g', { 'class': 'brick ' + cls }, g);
    el('rect', { x: 5, y: -4, width: 8, height: 5, rx: 1, 'class': 'b-stud' }, b);
    el('rect', { x: 23, y: -4, width: 8, height: 5, rx: 1, 'class': 'b-stud' }, b);
    el('rect', { x: 0, y: 0, width: 36, height: 20, rx: 2, 'class': 'b-body' }, b);
    var t = txt(b, 18, 14, 'b-top', label); t.setAttribute('style', 'font-size:10.5px;font-weight:700');
    return g;
  }
  function tokenAt(tok, p) { setT(tok, p.x + NW / 2 - 14, p.y - 22); }

  /* ======================================================================
     SCENES 1-2 — the bench builder (one feature, user grows the tree)
     ================================================================== */

  /* Bench geometry (viewBox 640 x 340). */
  var VBW = 640, VBH = 340, PLATE_Y = 96;
  function px(x) { return 70 + (x - 1) * 100; }             // slot centre for position x
  var TRAY = [{ x0: 14 }, { x0: 326 }], TRAY_W = 300, TRAY_Y = 150, TRAY_H = 182;
  function trayPos(side, k) {                               // top-left of brick k in a tray
    return { x: TRAY[side].x0 + 50 + (k % 3) * 100 - BW / 2, y: TRAY_Y + 44 + Math.floor(k / 3) * 62 };
  }

  function Builder(root, cfg) {
    var mode = cfg.mode, ROWS = cfg.rows, ALLT = [1, 2, 3, 4, 5];
    var bench = $('[data-role=bench]', root), treeSvg = $('[data-role=tree]', root);
    var chipsBox = $('[data-role=chips]', root), mathBox = $('[data-role=math]', root);
    var tieNote = $('[data-role=tieNote]', root), treeMsg = $('[data-role=treeMsg]', root);
    var nodeTag = $('[data-role=nodeTag]', root), benchHint = $('[data-role=benchHint]', root);
    var btnCompare = $('[data-role=compare]', root), btnCommit = $('[data-role=commit]', root), btnUndo = $('[data-role=undo]', root);
    var btnTest = $('[data-role=test]', root), testRow = $('[data-role=testRow]', root), btnExample = $('[data-role=example]', root);

    var ops, sel, t, compare, shown, test, timers = [];

    /* ---- static bench parts ---- */
    bench.setAttribute('viewBox', '0 0 ' + VBW + ' ' + VBH);
    var gBand = el('g', null, bench);
    var gPlate = el('g', null, bench);
    el('rect', { x: 10, y: PLATE_Y, width: VBW - 20, height: 14, rx: 2, 'class': 'plate-body' }, gPlate);
    el('rect', { x: 10, y: PLATE_Y + 10, width: VBW - 20, height: 4, rx: 1, 'class': 'plate-edge' }, gPlate);
    for (var sx = 22; sx < VBW - 16; sx += 25) el('rect', { x: sx, y: PLATE_Y - 5, width: 13, height: 5, rx: 1.5, 'class': 'plate-stud' }, gPlate);
    var slotLabs = [];
    for (var i = 1; i <= 6; i++) slotLabs.push(txt(gPlate, px(i), PLATE_Y + 32, 'slot-lab', 'x = ' + i));
    var gTrays = el('g', null, bench), trays = [0, 1].map(function (side) {
      var g = el('g', { 'class': 'tray' }, gTrays), x0 = TRAY[side].x0;
      el('rect', { x: x0, y: TRAY_Y, width: TRAY_W, height: TRAY_H, rx: 10, 'class': 'tray-body' }, g);
      el('rect', { x: x0 + 6, y: TRAY_Y + 28, width: TRAY_W - 12, height: 3, rx: 1.5, 'class': 'tray-lip' }, g);
      return { g: g, head: txt(g, x0 + 12, TRAY_Y + 21, 'tray-head', ''), n: (function () { var t = txt(g, x0 + TRAY_W - 12, TRAY_Y + 21, 'tray-sub', ''); t.setAttribute('text-anchor', 'end'); return t; })(),
               foot: txt(g, x0 + 12, TRAY_Y + TRAY_H - 10, 'tray-sub', '') };
    });
    var gMark = el('g', null, bench);
    /* drop zones between slots: click to set the gate there */
    var gZones = el('g', null, bench);
    ALLT.forEach(function (tt) {
      var z = el('rect', { x: px(tt) + 12, y: 0, width: 76, height: PLATE_Y + 40, 'class': 'drop-zone' }, gZones);
      z.addEventListener('click', function () { choose(tt); });
    });
    var gBricks = el('g', null, bench);
    var bricks = ROWS.map(function (r) {
      var g = el('g', { 'class': 'brick ' + (mode === 'cls' ? 'k' + r.c : 'kv') }, gBricks);
      brickShape(g);
      txt(g, BW / 2, 16, 'b-top', 'x = ' + r.x[0]);
      txt(g, BW / 2, 32, 'b-main', mode === 'cls' ? (r.c + ' ' + clsName(r.c)) : 'y = ' + r.v);
      g.style.pointerEvents = 'none';
      return g;
    });
    /* the gate */
    var gate = el('g', { 'class': 'gate', tabindex: 0, role: 'slider', 'aria-label': 'Gate threshold t', 'aria-valuemin': 1, 'aria-valuemax': 5 }, bench);
    el('rect', { x: -5, y: 30, width: 10, height: PLATE_Y + 16 - 30, rx: 3, 'class': 'gate-post' }, gate);
    el('rect', { x: -30, y: 4, width: 60, height: 28, rx: 6, 'class': 'gate-cap' }, gate);
    el('rect', { x: -22, y: -2, width: 12, height: 6, rx: 1.5, 'class': 'gate-cap' }, gate);
    el('rect', { x: 10, y: -2, width: 12, height: 6, rx: 1.5, 'class': 'gate-cap' }, gate);
    var gateText = txt(gate, 0, 23, 'gate-text', 't = ?');
    el('path', { d: 'M-50,' + (PLATE_Y + 40) + ' L-50,' + (TRAY_Y - 4) + ' M50,' + (PLATE_Y + 40) + ' L50,' + (TRAY_Y - 4), 'class': 'gate-arrow' }, gate);

    /* drag the gate (pointer) and arrow keys */
    var dragging = false;
    function tFromEvent(e) {
      var r = bench.getBoundingClientRect(), x = (e.clientX - r.left) / r.width * VBW;
      return Math.max(1, Math.min(5, Math.round((x - 120) / 100) + 1));
    }
    gate.addEventListener('pointerdown', function (e) { dragging = true; gate.setPointerCapture(e.pointerId); e.preventDefault(); });
    gate.addEventListener('pointermove', function (e) { if (dragging) { var tt = tFromEvent(e); if (tt !== t) choose(tt); } });
    gate.addEventListener('pointerup', function () { dragging = false; });
    gate.addEventListener('keydown', function (e) {
      var node = selNode(), c = canSplit(node) ? candidates(node.rows, 0) : [];
      if (!c.length) return;
      var k = c.indexOf(t);
      if (e.key === 'ArrowLeft' || e.key === 'ArrowDown') { e.preventDefault(); choose(c[Math.max(0, k < 0 ? 0 : k - 1)]); }
      if (e.key === 'ArrowRight' || e.key === 'ArrowUp') { e.preventDefault(); choose(c[Math.min(c.length - 1, k < 0 ? 0 : k + 1)]); }
    });

    /* chips */
    var chips = ALLT.map(function (tt) {
      var b = document.createElement('button');
      b.type = 'button'; b.className = 'chip';
      b.innerHTML = '<span>t = ' + tt + '</span><span class="chip-score"></span>';
      b.addEventListener('click', function () { choose(tt); });
      chipsBox.appendChild(b);
      return b;
    });

    /* ---- the tree model: rebuilt from the list of committed splits ---- */
    function buildModel() {
      var rootN = { path: '', rows: ROWS, lo: -Infinity, hi: Infinity, t: null };
      var byPath = { '': rootN };
      ops.forEach(function (op) {
        var n = byPath[op.path], s = splitRows(n.rows, 0, op.t);
        n.t = op.t;
        n.left = { path: op.path + 'L', rows: s.L, lo: n.lo, hi: op.t, t: null };
        n.right = { path: op.path + 'R', rows: s.R, lo: op.t, hi: n.hi, t: null };
        byPath[n.left.path] = n.left; byPath[n.right.path] = n.right;
      });
      return { root: rootN, byPath: byPath };
    }
    var model;
    function selNode() { return model.byPath[sel]; }
    function isPure(n) { return impurity(mode, n.rows).n === 0; }
    function canSplit(n) { return n.t === null && candidates(n.rows, 0).length > 0 && !(mode === 'cls' && isPure(n)); }
    function openNodes() {
      return Object.keys(model.byPath).map(function (k) { return model.byPath[k]; })
        .filter(function (n) { return n.t === null && mode === 'cls' && !isPure(n); });
    }
    function regionTxt(n) {
      if (n.lo === -Infinity && n.hi === Infinity) return 'all bricks';
      if (n.lo === -Infinity) return 'x ≤ ' + n.hi;
      if (n.hi === Infinity) return 'x > ' + n.lo;
      return n.lo + ' < x ≤ ' + n.hi;
    }
    function regionTex(n) {
      if (n.lo === -Infinity && n.hi === Infinity) return '\\text{any } x';
      if (n.lo === -Infinity) return 'x \\le ' + n.hi;
      if (n.hi === Infinity) return 'x > ' + n.lo;
      return n.lo + ' < x \\le ' + n.hi;
    }
    function countTxt(rows) {
      if (mode === 'reg') return 'y: ' + rows.map(function (r) { return r.v; }).join(', ');
      var k = classCounts(rows);
      return k[0] + ' blue · ' + k[1] + ' red';
    }

    /* ---- actions ---- */
    function stopTest() { timers.forEach(clearTimeout); timers = []; test = null; }
    function choose(tt) {
      var n = selNode();
      if (!canSplit(n) || candidates(n.rows, 0).indexOf(tt) < 0) return;
      if (t === tt) return;
      t = tt; render();
    }
    function select(path) {
      if (!model.byPath[path] || sel === path) return;
      sel = path; t = null; render();
    }
    function commit() {
      var n = selNode();
      if (!canSplit(n) || t === null) return;
      stopTest();
      ops.push({ path: sel, t: t });
      t = null;
      render();
    }
    function undo() {
      if (!ops.length) return;
      stopTest();
      var op = ops.pop();
      sel = op.path; t = op.t;
      render();
    }
    function reset() {
      stopTest();
      ops = []; sel = ''; t = null; compare = false; shown = {};
      render();
    }

    btnCompare.addEventListener('click', function () { compare = !compare; render(); });
    btnCommit.addEventListener('click', commit);
    btnUndo.addEventListener('click', undo);
    if (btnTest) btnTest.addEventListener('click', runTest);
    if (btnExample) btnExample.addEventListener('click', function () {
      stopTest(); ops = [{ path: '', t: 3 }]; sel = 'L'; t = null; render();
    });

    /* ---- the test brick (classification) ---- */
    function runTest() {
      stopTest();
      var x = cfg.testX, path = [model.root];
      while (path[path.length - 1].t !== null) { var n = path[path.length - 1]; path.push(x <= n.t ? n.left : n.right); }
      test = { x: x, path: path, step: 0, log: [] };
      render();
      function next() {
        if (!test) return;
        var n = test.path[test.step];
        if (n.t !== null) {
          var yes = x <= n.t;
          test.log.push('Is ' + x + ' ≤ ' + n.t + '? <b>' + (yes ? 'Yes' : 'No') + '</b> → ' + (yes ? 'left' : 'right') + '.');
          test.step++;
          render();
          timers.push(setTimeout(next, 850));
        } else {
          var v = leafValue(mode, n.rows);
          test.done = true;
          test.log.push('Lands in a <b class="c-' + clsName(v) + '">' + clsName(v) + '</b> leaf (' + countTxt(n.rows) + '): predict <b class="c-' + clsName(v) + '">' + clsName(v) + ' (' + v + ')</b>.');
          sel = n.path; t = null;
          render();
        }
      }
      timers.push(setTimeout(next, 650));
    }

    /* ---- render ---- */
    function render() {
      model = buildModel();
      var node = selNode();
      var splitT = node.t !== null ? node.t : t;      // committed split, or a preview
      var inNode = {};
      node.rows.forEach(function (r) { inNode[r.id] = true; });

      /* bricks */
      var sides = splitT !== null ? splitRows(node.rows, 0, splitT) : null;
      ROWS.forEach(function (r, i) {
        var g = bricks[i], pos;
        g.classList.toggle('is-away', !inNode[r.id]);
        if (sides && inNode[r.id]) {
          var side = r.x[0] <= splitT ? 0 : 1, k = (side ? sides.R : sides.L).indexOf(r);
          pos = trayPos(side, k);
        } else pos = { x: px(r.x[0]) - BW / 2, y: PLATE_Y - BH };
        setT(g, pos.x, pos.y);
      });
      slotLabs.forEach(function (s, k) { s.classList.toggle('is-off', !inNode[k]); });

      /* gate */
      gate.style.display = splitT !== null ? '' : 'none';
      if (splitT !== null) { setT(gate, px(splitT) + 50, 0); gateText.textContent = 't = ' + splitT; gate.setAttribute('aria-valuenow', splitT); }

      /* trays */
      trays.forEach(function (tr, side) {
        if (!sides) {
          tr.g.setAttribute('class', 'tray is-idle');
          tr.head.textContent = side ? 'x > ?' : 'x ≤ ?';
          tr.n.textContent = ''; tr.foot.textContent = '';
          return;
        }
        var rows = side ? sides.R : sides.L, imp = impurity(mode, rows);
        tr.g.setAttribute('class', 'tray' + (imp.n === 0 ? ' is-pure' : ''));
        tr.head.textContent = (side ? 'x > ' : 'x ≤ ') + splitT;
        tr.n.textContent = 'n = ' + rows.length;
        tr.foot.textContent = mode === 'cls'
          ? countTxt(rows) + '  ·  G = ' + fracTxt(imp)
          : 'mean = ' + numTxt(mean(rows)) + '  ·  MSE = ' + numTxt(imp);
      });

      /* regression leaf band: every x in the leaf's region gets the mean */
      clear(gBand);
      if (mode === 'reg' && node.t === null && t === null) {
        var a = node.lo === -Infinity ? 10 : px(node.lo) + 50, b = node.hi === Infinity ? VBW - 10 : px(node.hi) + 50;
        el('rect', { x: a + 2, y: 34, width: b - a - 4, height: PLATE_Y - 20, rx: 6, 'class': 'band' }, gBand);
        txt(gBand, (a + b) / 2, 26, 'band-text', 'every x here → ŷ = ' + numTxt(mean(node.rows)));
      }
      /* test-brick marker under the plate */
      clear(gMark);
      if (test) {
        var mx = px(test.x);
        el('path', { d: 'M' + mx + ',' + (PLATE_Y + 18) + ' l-7,11 h14 z', fill: test.done ? RED : '#6b6e75' }, gMark);
        txt(gMark, mx, PLATE_Y + 44, 'slot-lab', 'x = ' + test.x).setAttribute('style', 'fill:' + (test.done ? RED : '#4a4f5a'));
      }

      /* chips + compare */
      var can = canSplit(node), cands = can ? candidates(node.rows, 0) : [];
      var cmp = can ? compareSplits(mode, node.rows, 0) : null;
      var maxScore = 0;
      if (cmp) cmp.all.forEach(function (s) { maxScore = Math.max(maxScore, fval(s.score)); });
      chips.forEach(function (c, k) {
        var tt = ALLT[k], idx = cands.indexOf(tt), sc = c.querySelector('.chip-score');
        c.disabled = idx < 0;
        c.setAttribute('aria-pressed', splitT === tt ? 'true' : 'false');
        var best = compare && cmp && idx >= 0 && cmp.best.indexOf(idx) >= 0;
        c.classList.toggle('is-best', !!best);
        if (compare && cmp && idx >= 0) {
          var s = cmp.all[idx].score;
          sc.innerHTML = (mode === 'cls' ? fracTxt(s) : numTxt(s)) +
            '<i class="bar" style="width:' + (maxScore ? Math.max(4, 40 * fval(s) / maxScore) : 4) + 'px"></i>';
        } else sc.innerHTML = '';
      });
      btnCompare.setAttribute('aria-pressed', compare ? 'true' : 'false');
      btnCompare.disabled = !can;
      tieNote.innerHTML = '';
      if (compare && cmp) {
        var bestTs = cmp.best.map(function (k) { return cmp.all[k].t; }), bs = cmp.all[cmp.best[0]].score;
        var bsTxt = mode === 'cls' ? fracTxt(bs) : numTxt(bs);
        tieNote.innerHTML = bestTs.length > 1
          ? 'Tie: t = ' + bestTs.join(' and t = ') + ' both score ' + bsTxt + '. Either is a correct CART choice. Pick one.'
          : 'Lowest score: t = ' + bestTs[0] + ' (' + bsTxt + ').';
      }

      btnCommit.disabled = !(can && t !== null);
      btnUndo.disabled = !ops.length;

      /* header + hint */
      nodeTag.textContent = (sel === '' ? 'Root node' : 'Node ' + regionTxt(node)) + ' · ' + node.rows.length + ' brick' + (node.rows.length > 1 ? 's' : '');
      benchHint.textContent = node.t !== null ? 'already split · undo to change'
        : can ? (t === null ? 'tap a threshold or the plate between two slots' : 'drag the gate or use ← →')
        : (mode === 'cls' ? 'pure leaf · nothing to split' : 'one brick · cannot split');

      renderMath(node, splitT, sides);
      renderTree();
    }

    function lmRow(k, sub, v, extra) {
      return '<div class="lm-row' + (extra ? ' ' + extra : '') + '"><span class="lm-k">' + k + (sub ? '<small>' + sub + '</small>' : '') + '</span><span class="lm-v">' + T(v, true) + '</span></div>';
    }
    function renderMath(node, splitT, sides) {
      var h = '', n = node.rows.length, here = impurity(mode, node.rows);
      if (sides) {
        var s = scoreSplit(mode, node.rows, 0, splitT);
        if (mode === 'cls') {
          h += lmRow('Left tray', 'x ≤ ' + splitT + '<br>' + countTxt(s.L), 'G_L=' + giniTex(s.L));
          h += lmRow('Right tray', 'x > ' + splitT + '<br>' + countTxt(s.R), 'G_R=' + giniTex(s.R));
          h += lmRow('Split score', 'weighted', paren('\\frac{' + s.nL + '}{' + n + '}') + paren(fracTex(s.GL)) + '+' + paren('\\frac{' + s.nR + '}{' + n + '}') + paren(fracTex(s.GR)) + '=' + fracTex(s.score), 'lm-score');
          h += '<p class="lm-note">Before this gate the node had <span data-t="G=' + fracTex(here) + '"></span>. After: <b>' + fracTxt(s.score) + '</b>' +
            (s.score.n === 0 ? ', both trays pure.' : '.') + '</p>';
        } else {
          h += lmRow('Left mean', 'x ≤ ' + splitT, '\\bar y_L=' + meanTex(s.L));
          h += lmRow('Left MSE', '', '\\text{MSE}_L=' + mseTex(s.L));
          h += lmRow('Right mean', 'x > ' + splitT, '\\bar y_R=' + meanTex(s.R));
          h += lmRow('Right MSE', '', '\\text{MSE}_R=' + mseTex(s.R));
          h += lmRow('Split score', 'weighted', paren('\\frac{' + s.nL + '}{' + n + '}') + paren(numTex(s.GL)) + '+' + paren('\\frac{' + s.nR + '}{' + n + '}') + paren(numTex(s.GR)) + '=' + numTex(s.score) + approx(s.score), 'lm-score');
          h += '<p class="lm-note">Before this gate the node&rsquo;s MSE was <b>' + numTxt(here) + '</b>. After: <b>' + numTxt(s.score) + '</b>.</p>';
        }
        if (node.t !== null) h += '<p class="lm-note">This node is already split. <b>Undo</b> to try another threshold, or click a branch in the tree.</p>';
      } else if (mode === 'cls') {
        h += lmRow(isPure(node) ? 'Leaf' : 'This node', countTxt(node.rows), 'G=' + giniTex(node.rows));
        if (isPure(node)) {
          var v = leafValue(mode, node.rows);
          h += '<p class="lm-note"><b>Pure</b>: every brick here is ' + clsName(v) + ', so CART stops. Any input that reaches this leaf is predicted <b class="c-' + clsName(v) + '">' + clsName(v) + ' (' + v + ')</b>.</p>';
        } else h += '<p class="lm-note">Mixed tray. Pick a threshold to see how much a gate would tidy it.</p>';
      } else {
        var m = mean(node.rows);
        h += lmRow(node.t === null && sel !== '' ? 'Leaf mean' : 'Node mean', countTxt(node.rows), '\\bar y=' + meanTex(node.rows));
        h += lmRow('Node MSE', '', '\\text{MSE}=' + mseTex(node.rows));
        if (sel !== '' || ops.length) {
          var lo = node.lo === -Infinity ? 0.5 : node.lo, hi = node.hi === Infinity ? 6.5 : node.hi;
          var ex = [0.15, 0.55, 0.95].map(function (f) { return Math.round((lo + (hi - lo) * f) * 10) / 10; });
          h += '<div class="lm-score lm-row"><span class="lm-k">Prediction<small>for this leaf</small></span><span class="lm-v">' +
            T('\\hat y(x)=' + numTex(m) + approx(m) + '\\quad\\text{for every } ' + regionTex(node), true) + '</span></div>' +
            '<p class="lm-note">Inputs <b>x = ' + ex.join(', ') + '</b> all land here, and all get <b>' + numTxt(m) + '</b>. A tree&rsquo;s prediction is flat inside each leaf: it is a staircase, one step per leaf.</p>';
        } else h += '<p class="lm-note">Pick a threshold to split the six bricks.</p>';
      }
      mathBox.innerHTML = h;
      texFill(mathBox);
    }

    function renderTree() {
      function conv(n) {
        var o = { key: n.path, src: n };
        if (n.t !== null) {
          o.box = 'split'; o.kids = [conv(n.left), conv(n.right)];
          o.line1 = 'x ≤ ' + n.t + ' ?'; o.line2 = 'n = ' + n.rows.length;
          o.edge = ['≤ ' + n.t, '> ' + n.t];
        } else if (mode === 'cls') {
          var k = classCounts(n.rows);
          if (isPure(n)) { var v = k[1] ? 1 : 0; o.box = 'leaf' + v; o.line1 = clsName(v) + ' · ' + v; o.line2 = n.rows.length + ' brick' + (n.rows.length > 1 ? 's' : ''); }
          else { o.box = 'open'; o.line1 = k[0] + ' blue · ' + k[1] + ' red'; o.line2 = 'split me'; }
        } else {
          o.box = 'leafv'; o.line1 = 'ŷ = ' + numTxt(mean(n.rows)); o.line2 = 'n = ' + n.rows.length;
        }
        return o;
      }
      var tr = conv(model.root), onPath = {};
      if (test) for (var i = 0; i <= test.step && i < test.path.length; i++) onPath[test.path[i].path] = true;
      var prev = shown; shown = {};
      Object.keys(model.byPath).forEach(function (k) { shown[k] = true; });
      var opens = openNodes();
      var pos = drawTree(treeSvg, tr, {
        sel: sel, path: onPath,
        fresh: function (k) { return prev && !prev[k] && Object.keys(prev).length > 0; },
        pulse: function (n) { return n.box === 'open' && sel !== n.key; },
        onClick: function (n) { select(n.key); }
      });
      if (test) {
        var tok = makeToken(treeSvg, test.done ? 'k' + leafValue(mode, test.path[test.path.length - 1].rows) : 'kq', String(test.x));
        var to = test.path[test.step].path, from = test.done ? to : test.path[Math.max(0, test.step - 1)].path;
        tokenAt(tok, pos[from]);
        tok.getBoundingClientRect();
        requestAnimationFrame(function () { tokenAt(tok, pos[to]); });
      }

      /* message under the tree */
      var msg = '';
      if (mode === 'cls') {
        if (!ops.length) msg = 'One node holding all six bricks. Choose a threshold, then <b>Commit split</b>.';
        else if (opens.length) msg = 'The gate became a node. <b>Click the dashed &ldquo;split me&rdquo; branch</b> to sort just its ' + opens[0].rows.length + ' bricks.';
        else msg = '<b>Tree complete</b>: every leaf is pure (' + ops.length + ' split' + (ops.length > 1 ? 's' : '') + '). Send it a brick it has never seen.';
        if (test) msg = '<p class="ro-title">New brick, x = ' + test.x + '</p><ol>' + test.log.map(function (l) { return '<li>' + l + '</li>'; }).join('') + '</ol>';
        testRow.hidden = !!opens.length;
        btnTest.disabled = !!(test && !test.done);
      } else {
        if (!ops.length) msg = 'One leaf so far: with no split, every input is predicted the overall mean, <b>' + numTxt(mean(ROWS)) + '</b>.';
        else msg = ops.length + ' split' + (ops.length > 1 ? 's' : '') + ', ' + (ops.length + 1) + ' leaves. <b>Click a leaf</b> to see the range of x it covers and the single number it predicts.';
      }
      treeMsg.innerHTML = msg;
    }

    return { reset: reset, init: reset };
  }

  /* ======================================================================
     SCENE 3 — the ensemble
     ================================================================== */

  var ens = (function () {
    var plate = $('#ensPlate'), forest = $('#ensForest'), tally = $('#ensTally'), nodeBox = $('#ensNode');
    var btnStep = $('#ensStep'), btnAgain = $('#ensAgain'), btnRes = $('#ensResample'), btnPre = $('#ensPreset');
    var btnCls = $('#ensCls'), btnReg = $('#ensReg'), rfBox = $('#ensRF');
    var FN = ['x₁', 'x₂'];
    var st, trees, cols = [], busy = false, timers = [];

    /* plate geometry: 8 columns (x1) x 3 rows (x2) */
    var CELL = 54, PX0 = 44, PY0 = 12, PW = PX0 + 8 * CELL + 10, PH = PY0 + 3 * CELL + 44;
    function pcx(x1) { return PX0 + (x1 - 0.5) * CELL; }
    function pcy(x2) { return PY0 + (3 - x2 + 0.5) * CELL; }

    function rowCls(r) { return st.mode === 'cls' ? 'k' + r.c : 'kv'; }
    function stopAnim() { timers.forEach(clearTimeout); timers = []; busy = false; }

    function regrow() {
      trees = st.samples.map(function (s, k) { return ensembleTree(s, st.mode, st.rf, st.seed * 31 + k + 1); });
    }
    function setSamples(samples, seed) { st.samples = samples; st.seed = seed; regrow(); }

    /* ---- plate ---- */
    var gCells, gBr, testG;
    function buildPlate() {
      plate.setAttribute('viewBox', '0 0 ' + PW + ' ' + PH);
      el('rect', { x: PX0 - 6, y: PY0 - 6, width: 8 * CELL + 12, height: 3 * CELL + 12, rx: 8, 'class': 'plate-body' }, plate);
      for (var a = 1; a <= 8; a++) for (var b = 1; b <= 3; b++) el('circle', { cx: pcx(a), cy: pcy(b), r: 9, 'class': 'plate-stud' }, plate);
      for (a = 1; a <= 8; a++) txt(plate, pcx(a), PY0 + 3 * CELL + 22, 'p-axis', a);
      for (b = 1; b <= 3; b++) txt(plate, PX0 - 20, pcy(b) + 4, 'p-axis', b);
      txt(plate, PX0 + 4 * CELL, PH - 2, 'p-axis', 'x₁  (column)');
      var yt = txt(plate, 12, PY0 + 1.5 * CELL, 'p-axis', 'x₂ (row)');
      yt.setAttribute('transform', 'rotate(-90 12 ' + (PY0 + 1.5 * CELL) + ')');
      gCells = el('g', null, plate);
      gBr = el('g', null, plate);
      for (a = 1; a <= 8; a++) for (b = 1; b <= 3; b++) {
        if (ENS.some(function (r) { return r.x[0] === a && r.x[1] === b; })) continue;
        (function (a, b) {
          var c = el('rect', { x: pcx(a) - CELL / 2 + 3, y: pcy(b) - CELL / 2 + 3, width: CELL - 6, height: CELL - 6, rx: 6, 'class': 'p-cell' }, gCells);
          c.addEventListener('click', function () { st.test = [a, b]; resetVotes(); draw(); });
        })(a, b);
      }
    }
    function drawPlate() {
      clear(gBr);
      ENS.forEach(function (r) {
        var g = topBrick(gBr, rowCls(r), 44, st.mode === 'cls' ? r.name : r.v, st.mode === 'cls' ? null : r.name);
        g.setAttribute('transform', 'translate(' + pcx(r.x[0]) + ',' + pcy(r.x[1]) + ')');
      });
      testG = topBrick(gBr, 'kq is-test', 44, '?', '(' + st.test[0] + ',' + st.test[1] + ')');
      setT(testG, pcx(st.test[0]), pcy(st.test[1]));
    }

    /* ---- forest columns ---- */
    function buildForest() {
      for (var k = 0; k < 3; k++) {
        var col = document.createElement('div');
        col.className = 'tree-col';
        col.innerHTML = '<div class="tc-head"><span class="tc-title">Tree ' + (k + 1) + '</span><span class="tc-oob"></span></div>' +
          '<svg class="bag" role="img" aria-label="Bootstrap sample for tree ' + (k + 1) + '"></svg>' +
          '<p class="bag-cap"></p>' +
          '<svg class="mini-tree" role="img" aria-label="Tree ' + (k + 1) + '"></svg>' +
          '<div class="pred">prediction: <span class="pred-big">?</span></div>';
        forest.appendChild(col);
        cols.push({ el: col, oob: $('.tc-oob', col), bag: $('.bag', col), cap: $('.bag-cap', col), tree: $('.mini-tree', col), pred: $('.pred', col) });
      }
    }
    function drawBag(k) {
      var c = cols[k], svg = c.bag, cnt = counts(st.samples[k], ENS.length);
      clear(svg);
      var SW = 38;
      svg.setAttribute('viewBox', '0 0 ' + (SW * ENS.length) + ' 62');
      var out = [];
      ENS.forEach(function (r, i) {
        var cx = SW * i + SW / 2, cy = 30, n = cnt[i];
        if (n === 0) {
          out.push(r.name);
          topBrick(svg, rowCls(r) + ' is-ghost', 28, r.name).setAttribute('transform', 'translate(' + cx + ',' + cy + ')');
          el('path', { d: 'M' + (cx - 9) + ',' + (cy - 9) + ' l18,18 M' + (cx + 9) + ',' + (cy - 9) + ' l-18,18', 'class': 'oob-x' }, svg);
          txt(svg, cx, 58, 'oob-lab', 'out');
          return;
        }
        for (var s = Math.min(n, 3) - 1; s >= 0; s--) {
          topBrick(svg, rowCls(r), 28, s === 0 ? (st.mode === 'cls' ? r.name : r.v) : null).setAttribute('transform', 'translate(' + (cx + s * 3) + ',' + (cy - s * 3) + ')');
        }
        if (n > 1) {
          el('circle', { cx: cx + 12, cy: cy - 16, r: 9, 'class': 'dup-badge' }, svg);
          txt(svg, cx + 12, cy - 12.5, 'dup-text', '×' + n);
        }
        txt(svg, cx, 58, 'p-axis', st.mode === 'cls' ? '' : r.name).setAttribute('style', 'font-size:10px');
      });
      c.oob.innerHTML = out.length ? 'left out: <b>' + out.join(', ') + '</b>' : 'nothing left out';
      var distinct = cnt.filter(function (v) { return v > 0; }).length;
      var dups = ENS.filter(function (r, i) { return cnt[i] > 1; }).map(function (r) { return r.name + ' ×' + cnt[r.id]; });
      c.cap.textContent = '10 draws with replacement → ' + distinct + ' different bricks' + (dups.length ? '; repeated: ' + dups.join(', ') : '');
    }
    function nodeLabel(n) {
      var o = { key: n.path, src: n };
      if (!n.leaf) {
        o.box = 'split'; o.line1 = FN[n.f] + ' ≤ ' + n.t + ' ?'; o.line2 = 'n = ' + n.n;
        o.edge = ['≤ ' + n.t, '> ' + n.t];
        o.kids = [nodeLabel(n.left), nodeLabel(n.right)];
      } else if (st.mode === 'cls') {
        var k = classCounts(n.rows);
        o.box = 'leaf' + n.value; o.line1 = clsName(n.value) + ' · ' + n.value; o.line2 = k[0] + ' blue, ' + k[1] + ' red';
      } else {
        o.box = 'leafv'; o.line1 = 'ŷ = ' + numTxt(n.value); o.line2 = 'n = ' + n.n;
      }
      return o;
    }
    function drawMini(k, animate) {
      var c = cols[k], tr = trees[k], onPath = {}, path = route(tr, st.test);
      var revealed = st.revealed > k;
      if (revealed) path.forEach(function (n) { onPath[n.path] = true; });
      var pos = drawTree(c.tree, nodeLabel(tr), {
        sel: st.sel && st.sel.tree === k ? st.sel.path : null, path: revealed && !animate ? onPath : {},
        onClick: function (n) { if (busy) return; st.sel = { tree: k, path: n.key }; drawInfo(); for (var j = 0; j < 3; j++) drawMini(j); }
      });
      if (revealed) {
        var leaf = path[path.length - 1];
        var tok = makeToken(c.tree, st.mode === 'cls' && !animate ? 'k' + leaf.value : 'kq', '?');
        if (animate) {
          tokenAt(tok, pos[path[0].path]);
          path.forEach(function (n, i) {
            if (!i) return;
            timers.push(setTimeout(function () { tokenAt(tok, pos[n.path]); }, 120 + i * 600));
          });
          timers.push(setTimeout(function () { busy = false; drawMini(k); drawPred(k); drawTally(); syncStep(); }, 220 + path.length * 600));
        } else tokenAt(tok, pos[leaf.path]);
      }
    }
    function predOf(k) { var p = route(trees[k], st.test); return p[p.length - 1].value; }
    function drawPred(k) {
      var c = cols[k], on = st.revealed > k && !(busy && st.revealed === k + 1);
      c.pred.className = 'pred';
      if (!on) { c.pred.innerHTML = 'prediction: <span class="pred-big">?</span>'; return; }
      var v = predOf(k);
      if (st.mode === 'cls') { c.pred.className = 'pred is-on k' + v; c.pred.innerHTML = 'votes <span class="pred-big c-' + clsName(v) + '">' + clsName(v) + ' (' + v + ')</span>'; }
      else { c.pred.className = 'pred is-on kv'; c.pred.innerHTML = 'predicts <span class="pred-big">ŷ = ' + numTxt(v) + '</span>'; }
    }

    /* ---- node info: eligible features ---- */
    function findNode(tr, path) { var n = tr; for (var i = 0; i < path.length; i++) n = path[i] === 'L' ? n.left : n.right; return n; }
    function drawInfo() {
      if (!st.sel) { nodeBox.innerHTML = ''; return; }
      var tr = trees[st.sel.tree], n = findNode(tr, st.sel.path);
      if (!n) { st.sel = { tree: 0, path: '' }; return drawInfo(); }
      var k = classCounts(n.rows), fmt = st.mode === 'cls' ? fracTxt : numTxt;
      var h = '<p class="ro-title">Tree ' + (st.sel.tree + 1) + ', ' + (n.path === '' ? 'root' : 'depth-' + n.depth + ' node') + ' · ' + n.n + ' bricks' +
        (st.mode === 'cls' ? ' (' + k[0] + ' blue, ' + k[1] + ' red)' : '') + '</p>';
      h += '<p>Eligible features: <b>' + n.eligible.map(function (f) { return FN[f]; }).join(' and ') + '</b> ' +
        (st.rf ? '<span class="c-red">(random forest drew 1 of 2 here)</span>' : '(bagging: every feature)') + '</p>';
      var lines = [];
      n.eligible.forEach(function (f) {
        var b = n.perFeature[f];
        if (b === undefined) return;
        lines.push(b ? 'best on ' + FN[f] + ': <b>' + FN[f] + ' ≤ ' + b.t + '</b>, score ' + fmt(b.score) : FN[f] + ': all values equal, no split');
      });
      if (lines.length) h += '<p>' + lines.join(' · ') + '</p>';
      var why = n.leaf ? (n.depth >= ENS_DEPTH ? 'Leaf: depth limit (' + ENS_DEPTH + ') reached.' : n.imp.n === 0 ? 'Leaf: already pure.' : 'Leaf: no eligible split lowers the impurity.') +
        ' Predicts <b>' + (st.mode === 'cls' ? clsName(n.value) + ' (' + n.value + ')' : numTxt(n.value)) + '</b>.'
        : 'Chosen: <b>' + FN[n.f] + ' ≤ ' + n.t + '</b> (impurity ' + fmt(n.imp) + ' → ' + fmt(n.split.score) + ').';
      h += '<p>' + why + '</p>';
      nodeBox.innerHTML = h;
    }

    /* ---- voting ---- */
    function resetVotes() { stopAnim(); st.revealed = 0; }
    function syncStep() {
      var noun = st.mode === 'cls' ? 'Tally the vote' : 'Average the predictions';
      if (st.revealed < 3) btnStep.textContent = 'Send test brick through tree ' + (st.revealed + 1);
      else if (st.revealed === 3) btnStep.textContent = noun;
      else btnStep.textContent = 'Done';
      btnStep.disabled = busy || st.revealed > 3;
      cols.forEach(function (c, k) { c.el.classList.toggle('is-active', busy && st.revealed === k + 1); });
    }
    function step() {
      if (busy || st.revealed > 3) return;
      st.revealed++;
      if (st.revealed <= 3) { busy = true; drawMini(st.revealed - 1, true); drawPred(st.revealed - 1); }
      drawTally(); syncStep();
    }
    function drawTally() {
      var h = '<div class="ballots">';
      var preds = [];
      for (var k = 0; k < 3; k++) {
        var on = st.revealed > k && !(busy && st.revealed === k + 1), v = on ? predOf(k) : null;
        if (on) preds.push(v);
        h += '<div class="ballot' + (on ? (st.mode === 'cls' ? ' k' + v : ' kv') : '') + '"><small>T' + (k + 1) + '</small>' +
          (on ? (st.mode === 'cls' ? v : numTxt(v)) : '?') + '</div>';
      }
      h += '</div>';
      if (st.revealed > 3) {
        if (st.mode === 'cls') {
          var m = majority(preds);
          h += '<div class="verdict">Votes: <b class="c-blue">' + m.counts[0] + ' blue</b>, <b class="c-red">' + m.counts[1] + ' red</b>. Majority → the ensemble predicts <b class="c-' + clsName(m.winner) + '">' + clsName(m.winner) + ' (' + m.winner + ')</b>.' +
            (m.counts[0] && m.counts[1] ? ' One tree was outvoted.' : ' All three agree.') + '</div>';
        } else {
          var avg = average(preds);
          h += '<div class="verdict">' + T('\\hat y=\\frac{' + preds.map(numTex).join('+') + '}{3}=' + numTex(avg) + approx(avg)) + '. The ensemble predicts <b>' + numTxt(avg) + '</b>.</div>';
        }
      } else if (st.revealed === 0) h += '<p class="hint">Test brick at (' + st.test.join(', ') + '). Send it through each tree in turn.</p>';
      tally.innerHTML = h;
      texFill(tally);
    }

    /* ---- top-level draw ---- */
    function draw() {
      btnCls.setAttribute('aria-pressed', st.mode === 'cls' ? 'true' : 'false');
      btnReg.setAttribute('aria-pressed', st.mode === 'reg' ? 'true' : 'false');
      rfBox.checked = st.rf;
      btnPre.textContent = st.mode === 'cls' ? 'Load preset: votes 1, 0, 1' : 'Load preset: predictions 4, 6, 8';
      drawPlate();
      for (var k = 0; k < 3; k++) { drawBag(k); drawMini(k); drawPred(k); }
      drawInfo(); drawTally(); syncStep();
    }
    function loadPreset() {
      resetVotes();
      st.rf = false; st.test = ENS_TEST.slice();
      setSamples(PRESETS[st.mode].map(function (s) { return s.slice(); }), 7);
      st.sel = { tree: 0, path: '' };
      draw();
    }

    function init() {
      buildPlate(); buildForest();
      btnStep.addEventListener('click', step);
      btnAgain.addEventListener('click', function () { resetVotes(); draw(); });
      btnRes.addEventListener('click', function () {
        resetVotes();
        var seed = 1 + Math.floor(Math.random() * 1e9), r = rng(seed);
        setSamples([0, 1, 2].map(function () { return bootstrap(r, ENS.length); }), seed);
        draw();
      });
      btnPre.addEventListener('click', loadPreset);
      rfBox.addEventListener('change', function () { resetVotes(); st.rf = rfBox.checked; regrow(); draw(); });
      function setMode(m) { if (st.mode === m) return; st.mode = m; loadPreset(); }
      btnCls.addEventListener('click', function () { setMode('cls'); });
      btnReg.addEventListener('click', function () { setMode('reg'); });
      var e3 = majorityError(0.3, 3);
      $('#ensMath').innerHTML = '<b>Numbers.</b> Suppose each tree is wrong 30% of the time, independently. A 3-tree majority is wrong only when at least 2 are: ' +
        T('3(0.3)^2(0.7)+(0.3)^3=' + e3.toFixed(3)) + ', i.e. ' + (e3 * 100).toFixed(1) + '%. With 25 such trees it drops to ' + (majorityError(0.3, 25) * 100).toFixed(1) + '%.';
      texFill($('#ensMath'));
      reset();
    }
    function reset() {
      st = { mode: 'cls', rf: false, test: ENS_TEST.slice(), revealed: 0, sel: { tree: 0, path: '' } };
      loadPreset();
    }
    return { init: init, reset: reset };
  })();

  /* ------------------------------------------------------- navigation -- */

  var scene1, scene2;
  var SCENES;
  var sections = Array.from(document.querySelectorAll('.scene'));
  var current = 0;

  function go(i, focus) {
    current = Math.max(0, Math.min(sections.length - 1, i));
    sections.forEach(function (s, k) { s.hidden = k !== current; });
    document.querySelectorAll('#progress button').forEach(function (b, k) {
      if (k === current) b.setAttribute('aria-current', 'step'); else b.removeAttribute('aria-current');
      b.classList.toggle('is-done', k < current);
    });
    $('#prev').disabled = current === 0;
    $('#next').disabled = current === sections.length - 1;
    $('#sceneCount').textContent = 'Scene ' + (current + 1) + ' / ' + sections.length;
    if (location.hash !== '#scene-' + (current + 1)) history.replaceState(null, '', '#scene-' + (current + 1));
    if (focus) $('#console').scrollIntoView({ block: 'start', behavior: 'smooth' });
  }

  function start() {
    renderTex(document);
    var ol = $('#progress');
    sections.forEach(function (s, k) {
      var li = document.createElement('li'), b = document.createElement('button');
      b.type = 'button';
      b.innerHTML = '<span class="p-num">' + (k + 1) + '</span><span class="p-lab">' + s.getAttribute('data-short') + '</span>';
      b.addEventListener('click', function () { go(k, true); });
      li.appendChild(b); ol.appendChild(li);
    });
    scene1 = Builder($('#clsBuilder'), { mode: 'cls', rows: CLS, testX: 3.5 });
    scene2 = Builder($('#regBuilder'), { mode: 'reg', rows: REG });
    SCENES = [scene1, scene2, ens];
    SCENES.forEach(function (s) { s.init(); });
    $('#prev').addEventListener('click', function () { go(current - 1, true); });
    $('#next').addEventListener('click', function () { go(current + 1, true); });
    $('#reset').addEventListener('click', function () { SCENES[current].reset(); });
    var m = /^#scene-(\d+)$/.exec(location.hash);
    go(m ? +m[1] - 1 : 0, false);
  }

  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', start);
  else start();
})(typeof window !== 'undefined' ? window : globalThis);
