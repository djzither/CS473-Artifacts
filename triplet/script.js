/* ============================================================================
   Close Together, Far Apart — contrastive loss and triplet loss, for someone
   who has never seen machine learning. CS 473 module 8 · Murphy, PML §16.2

   Each dot stands for one picture of a person. A good picture-to-dot map puts
   pictures of the same person close together and pictures of different people
   farther apart. Two ways to score a map:

     contrastive   one pair at a time, against a fixed ruler
                     same person       penalty = d²
                     different people  penalty = max(0, m − d)²
     triplet       three pictures at once, only the comparison matters
                     penalty = max(0, d(you, you again) + m − d(you, someone else))

   Distances are plain straight-line distances in "steps" (one grid square).
   Some versions of triplet loss square the distances; the picture is the same.

   Layout of this file
     PART 1  Math          pure, DOM-free, exported for the Node test suite
     PART 2  Drawing       shared SVG helpers, dragging, arrow keys
     PART 3  View 1        two pictures, contrastive loss
     PART 4  View 2        three pictures, triplet loss and the learning step

   PART 1 runs under Node (`require('./script.js')` returns the math only); the
   rest is skipped when there is no document. See test-math.js.
   ========================================================================= */

(function (global) {
  'use strict';

  /* ==========================================================================
     PART 1 — MATH
     ====================================================================== */

  var W = 12, H = 7.6;              // the board, in steps
  var PAD = 0.45;                   // keep dot centres this far from the edge
  var PAIR_MARGIN = 3;              // contrastive: different people need d ≥ 3
  var TRIPLET_MARGIN = 1;           // triplet: the "little extra room"
  var SNAP = 0.35;                  // drop a same-person dot this close and it stacks

  function dist(a, b) { return Math.hypot(a.x - b.x, a.y - b.y); }

  function clampPt(p) {
    return {
      x: Math.min(W - PAD, Math.max(PAD, p.x)),
      y: Math.min(H - PAD, Math.max(PAD, p.y))
    };
  }

  /* Contrastive loss for one pair. Same person: any gap costs, so the only
     penalty-free spot is on top. Different people: only a gap smaller than the
     margin costs, and going farther than the margin earns nothing extra. */
  function contrastivePenalty(a, b, same, m) {
    var d = dist(a, b);
    if (same) return d * d;
    var short = Math.max(0, m - d);
    return short * short;
  }

  /* Which way the penalty pushes dot b (minus its gradient). Dot a gets the
     opposite push. Zero when the penalty is zero. */
  function contrastivePush(a, b, same, m) {
    var dx = b.x - a.x, dy = b.y - a.y, d = Math.hypot(dx, dy);
    if (same) return { x: -2 * dx, y: -2 * dy };
    if (d >= m || d === 0) return { x: 0, y: 0 };
    var k = 2 * (m - d) / d;
    return { x: k * dx, y: k * dy };
  }

  /* Triplet loss. a = you, p = another picture of you, n = someone else.
     Zero as soon as the stranger is at least m farther from you than your
     other picture is. How far apart everything is overall does not matter. */
  function tripletPenalty(a, p, n, m) {
    return Math.max(0, dist(a, p) + m - dist(a, n));
  }

  function unit(from, to) {
    var dx = to.x - from.x, dy = to.y - from.y, d = Math.hypot(dx, dy);
    return d === 0 ? { x: 0, y: 0 } : { x: dx / d, y: dy / d };
  }

  /* Which way the triplet penalty pushes each dot (minus its gradient):
     you and your other picture pull together, the stranger is pushed away
     from you, and you also step away from the stranger. */
  function tripletPush(a, p, n, m) {
    var zero = { x: 0, y: 0 };
    if (tripletPenalty(a, p, n, m) === 0) return { a: zero, p: zero, n: zero };
    var ap = unit(a, p), an = unit(a, n);
    return {
      a: { x: ap.x - an.x, y: ap.y - an.y },
      p: { x: -ap.x, y: -ap.y },
      n: { x: an.x, y: an.y }
    };
  }

  /* One learning step: move each dot a little in its push direction. */
  function tripletStep(a, p, n, m, eta) {
    var g = tripletPush(a, p, n, m);
    function mv(pt, v) { return clampPt({ x: pt.x + eta * v.x, y: pt.y + eta * v.y }); }
    return { a: mv(a, g.a), p: mv(p, g.p), n: mv(n, g.n) };
  }

  var MATH = {
    W: W, H: H, PAD: PAD, PAIR_MARGIN: PAIR_MARGIN, TRIPLET_MARGIN: TRIPLET_MARGIN, SNAP: SNAP,
    dist: dist, clampPt: clampPt,
    contrastivePenalty: contrastivePenalty, contrastivePush: contrastivePush,
    tripletPenalty: tripletPenalty, tripletPush: tripletPush, tripletStep: tripletStep
  };

  if (typeof module !== 'undefined' && module.exports) module.exports = MATH;
  if (typeof document === 'undefined') return;

  /* ==========================================================================
     PART 2 — DRAWING
     ====================================================================== */

  var S = 50;                                   // px per step in the viewBox
  var NS = 'http://www.w3.org/2000/svg';
  var R = 17;                                   // dot radius, px
  var reduceMotion = global.matchMedia && global.matchMedia('(prefers-reduced-motion: reduce)').matches;

  function $(id) { return document.getElementById(id); }

  function el(name, attrs, parent) {
    var e = document.createElementNS(NS, name);
    for (var k in attrs) e.setAttribute(k, attrs[k]);
    if (parent) parent.appendChild(e);
    return e;
  }

  function fmt(v) { return v < 0.05 ? '0' : v.toFixed(1); }

  function grid(svg) {
    var g = el('g', { 'class': 'grid' }, svg);
    for (var x = 1; x < W; x++) el('line', { x1: x * S, y1: 0, x2: x * S, y2: H * S }, g);
    for (var y = 1; y < H; y++) el('line', { x1: 0, y1: y * S, x2: W * S, y2: y * S }, g);
  }

  /* A dot with a label. Label sits above the dot unless that would leave the
     board, and hugs the side walls. */
  function makeDot(svg, cls, label, ariaLabel) {
    var g = el('g', { 'class': 'dot ' + cls, tabindex: 0, role: 'slider', 'aria-label': ariaLabel }, svg);
    el('circle', { r: R + 14, 'class': 'hit' }, g);
    el('circle', { r: R, 'class': 'body' }, g);
    var t = el('text', { 'class': 'dot-label' }, g);
    t.textContent = label;
    return g;
  }

  function placeDot(g, p) {
    var cx = p.x * S, cy = p.y * S;
    g.setAttribute('transform', 'translate(' + cx + ',' + cy + ')');
    var t = g.querySelector('text');
    var above = cy > 52;
    t.setAttribute('y', above ? -(R + 10) : R + 24);
    t.setAttribute('text-anchor', cx < 90 ? 'start' : cx > W * S - 90 ? 'end' : 'middle');
    t.setAttribute('x', cx < 90 ? -R : cx > W * S - 90 ? R : 0);
    g.setAttribute('aria-valuetext', 'x ' + p.x.toFixed(1) + ', y ' + p.y.toFixed(1));
  }

  /* An arrow from point p in direction v (in steps), drawn only if long enough. */
  function setArrow(line, p, v, scale, maxLen) {
    var len = Math.hypot(v.x, v.y) * scale;
    if (len < 0.15) { line.setAttribute('visibility', 'hidden'); return; }
    var k = Math.min(len, maxLen) / Math.hypot(v.x, v.y);
    var ux = v.x / Math.hypot(v.x, v.y), uy = v.y / Math.hypot(v.x, v.y);
    var x1 = p.x * S + ux * (R + 3), y1 = p.y * S + uy * (R + 3);
    line.setAttribute('visibility', 'visible');
    line.setAttribute('x1', x1); line.setAttribute('y1', y1);
    line.setAttribute('x2', x1 + v.x * k * S); line.setAttribute('y2', y1 + v.y * k * S);
  }

  function arrowMarker(svg, id, cls) {
    var defs = svg.querySelector('defs') || el('defs', {}, svg);
    var mk = el('marker', { id: id, viewBox: '0 0 10 10', refX: 6, refY: 5, markerWidth: 5, markerHeight: 5, orient: 'auto-start-reverse' }, defs);
    el('path', { d: 'M0,0 L10,5 L0,10 z', 'class': cls }, mk);
  }

  /* Pointer dragging plus arrow keys. get/set read and write the point. */
  function draggable(svg, g, get, set, onEnd) {
    function toBoard(evt) {
      var pt = svg.createSVGPoint();
      pt.x = evt.clientX; pt.y = evt.clientY;
      var q = pt.matrixTransform(svg.getScreenCTM().inverse());
      return { x: q.x / S, y: q.y / S };
    }
    var grab = null;
    g.addEventListener('pointerdown', function (e) {
      e.preventDefault();
      var q = toBoard(e), p = get();
      grab = { dx: p.x - q.x, dy: p.y - q.y };
      g.setPointerCapture(e.pointerId);
      g.classList.add('is-drag');
    });
    g.addEventListener('pointermove', function (e) {
      if (!grab) return;
      var q = toBoard(e);
      set(clampPt({ x: q.x + grab.dx, y: q.y + grab.dy }));
    });
    function end() {
      if (!grab) return;
      grab = null;
      g.classList.remove('is-drag');
      if (onEnd) onEnd();
    }
    g.addEventListener('pointerup', end);
    g.addEventListener('pointercancel', end);
    g.addEventListener('keydown', function (e) {
      var step = e.shiftKey ? 0.5 : 0.2, p = get(), d = null;
      if (e.key === 'ArrowLeft') d = { x: -step, y: 0 };
      if (e.key === 'ArrowRight') d = { x: step, y: 0 };
      if (e.key === 'ArrowUp') d = { x: 0, y: -step };
      if (e.key === 'ArrowDown') d = { x: 0, y: step };
      if (!d) return;
      e.preventDefault();
      set(clampPt({ x: p.x + d.x, y: p.y + d.y }));
      if (onEnd) onEnd();
    });
  }

  function setMeter(fill, value, full) {
    fill.style.width = (Math.min(1, value / full) * 100) + '%';
  }

  /* ==========================================================================
     PART 3 — VIEW 1 · TWO PICTURES (contrastive)
     ====================================================================== */

  var pair = { a: { x: 3, y: 3.8 }, b: { x: 8.5, y: 3.8 }, same: true };

  var svg1 = $('board1');
  grid(svg1);
  arrowMarker(svg1, 'tip1', 'arrow-tip');
  var ring = el('circle', { 'class': 'margin-ring', r: PAIR_MARGIN * S }, svg1);
  var ringText = el('text', { 'class': 'ring-label' }, svg1);
  ringText.textContent = 'needs to be outside';
  var link1 = el('line', { 'class': 'link' }, svg1);
  var dLabel1 = el('text', { 'class': 'd-label' }, svg1);
  var arA = el('line', { 'class': 'arrow', 'marker-end': 'url(#tip1)' }, svg1);
  var arB = el('line', { 'class': 'arrow', 'marker-end': 'url(#tip1)' }, svg1);
  var dotA = makeDot(svg1, 'p1', 'Picture 1', 'Picture 1. Use arrow keys to move.');
  var dotB = makeDot(svg1, 'p1', 'Picture 2', 'Picture 2. Use arrow keys to move.');

  function render1() {
    var a = pair.a, b = pair.b, d = dist(a, b);
    var pen = contrastivePenalty(a, b, pair.same, PAIR_MARGIN);
    var stacked = pair.same && d === 0;

    dotB.setAttribute('class', 'dot ' + (pair.same ? 'p1' : 'p2') + (stacked ? ' is-stacked' : ''));
    placeDot(dotA, a);
    placeDot(dotB, stacked ? { x: a.x, y: a.y } : b);
    if (stacked) dotB.querySelector('text').setAttribute('y', R + 24);

    ring.setAttribute('cx', a.x * S); ring.setAttribute('cy', a.y * S);
    ring.setAttribute('visibility', pair.same ? 'hidden' : 'visible');
    ring.classList.toggle('is-ok', !pair.same && pen === 0);
    ringText.setAttribute('x', a.x * S);
    ringText.setAttribute('y', Math.max(14, a.y * S - PAIR_MARGIN * S - 8));
    ringText.setAttribute('visibility', pair.same ? 'hidden' : 'visible');
    if (a.y * S - PAIR_MARGIN * S - 8 < 14) ringText.setAttribute('y', Math.min(H * S - 8, a.y * S + PAIR_MARGIN * S + 20));

    link1.setAttribute('x1', a.x * S); link1.setAttribute('y1', a.y * S);
    link1.setAttribute('x2', b.x * S); link1.setAttribute('y2', b.y * S);
    dLabel1.setAttribute('x', (a.x + b.x) / 2 * S);
    dLabel1.setAttribute('y', (a.y + b.y) / 2 * S - 10);
    dLabel1.textContent = d > 0.6 ? d.toFixed(1) + ' steps' : '';

    var push = contrastivePush(a, b, pair.same, PAIR_MARGIN);
    setArrow(arB, b, push, 0.12, 1.6);
    setArrow(arA, a, { x: -push.x, y: -push.y }, 0.12, 1.6);
    arA.setAttribute('class', 'arrow ' + (pair.same ? 'pull' : 'push'));
    arB.setAttribute('class', 'arrow ' + (pair.same ? 'pull' : 'push'));

    $('pen1').textContent = fmt(pen);
    $('dist1').textContent = d.toFixed(1);
    setMeter($('meter1'), pen, 20);
    var good = pen === 0;
    $('status1').className = 'status ' + (good ? 'is-good' : 'is-bad');

    var msg;
    if (pair.same) {
      if (stacked) msg = 'Perfect. Same person, same spot. No penalty.';
      else if (d < 1.5) msg = 'Almost! Drop one dot right on top of the other.';
      else msg = 'Same person, but far apart. Drag them closer!';
    } else {
      if (good) msg = 'Far enough apart. No penalty. Going even farther doesn’t earn anything.';
      else msg = 'Too close for different people. Drag the orange dot outside the circle.';
    }
    $('msg1').textContent = msg;
    $('rule1').textContent = pair.same
      ? 'The rule: the penalty is the distance squared. It only reaches zero when the dots touch.'
      : 'The rule: no penalty outside the circle (3 steps). Inside, the penalty grows the closer they get.';
  }

  function snap1() {
    if (pair.same && dist(pair.a, pair.b) < SNAP) pair.b = { x: pair.a.x, y: pair.a.y };
    render1();
  }

  draggable(svg1, dotA, function () { return pair.a; }, function (p) { pair.a = p; render1(); }, snap1);
  draggable(svg1, dotB, function () { return pair.b; }, function (p) { pair.b = p; render1(); }, snap1);

  function setSame(same) {
    pair.same = same;
    $('btnSame').setAttribute('aria-pressed', same);
    $('btnDiff').setAttribute('aria-pressed', !same);
    if (!same && dist(pair.a, pair.b) === 0) pair.b = clampPt({ x: pair.a.x + 1.2, y: pair.a.y });
    render1();
  }
  $('btnSame').addEventListener('click', function () { setSame(true); });
  $('btnDiff').addEventListener('click', function () { setSame(false); });
  $('reset1').addEventListener('click', function () {
    pair.a = { x: 3, y: 3.8 };
    pair.b = pair.same ? { x: 8.5, y: 3.8 } : { x: 4.8, y: 3.1 };
    render1();
  });

  /* ==========================================================================
     PART 4 — VIEW 2 · THREE PICTURES (triplet)
     ====================================================================== */

  var START = { a: { x: 4.5, y: 3.8 }, p: { x: 6.5, y: 3 }, n: { x: 5, y: 5.5 } };
  var tri = { a: START.a, p: START.p, n: START.n };
  var steps = 0, animating = false;

  var svg2 = $('board2');
  grid(svg2);
  arrowMarker(svg2, 'tip2', 'arrow-tip');
  var band = el('path', { 'class': 'room-band', 'fill-rule': 'evenodd' }, svg2);
  var inner = el('circle', { 'class': 'inner-ring' }, svg2);
  var outer = el('circle', { 'class': 'outer-ring' }, svg2);
  var lineP = el('line', { 'class': 'link link-p' }, svg2);
  var lineN = el('line', { 'class': 'link link-n' }, svg2);
  var labP = el('text', { 'class': 'd-label d-p' }, svg2);
  var labN = el('text', { 'class': 'd-label d-n' }, svg2);
  var ar2 = {
    a: el('line', { 'class': 'arrow', 'marker-end': 'url(#tip2)' }, svg2),
    p: el('line', { 'class': 'arrow', 'marker-end': 'url(#tip2)' }, svg2),
    n: el('line', { 'class': 'arrow', 'marker-end': 'url(#tip2)' }, svg2)
  };
  var dot2 = {
    p: makeDot(svg2, 'p1 is-copy', 'Another picture of you', 'Another picture of you. Use arrow keys to move.'),
    n: makeDot(svg2, 'p2', 'Someone else', 'Someone else. Use arrow keys to move.'),
    a: makeDot(svg2, 'p1', 'You', 'You. Use arrow keys to move.')
  };

  function circlePath(cx, cy, r) {
    return 'M' + (cx - r) + ',' + cy + 'a' + r + ',' + r + ' 0 1,0 ' + 2 * r + ',0a' + r + ',' + r + ' 0 1,0 ' + -2 * r + ',0z';
  }

  function render2() {
    var a = tri.a, p = tri.p, n = tri.n;
    var dp = dist(a, p), dn = dist(a, n);
    var pen = tripletPenalty(a, p, n, TRIPLET_MARGIN);
    var cx = a.x * S, cy = a.y * S, r1 = dp * S, r2 = (dp + TRIPLET_MARGIN) * S;

    inner.setAttribute('cx', cx); inner.setAttribute('cy', cy); inner.setAttribute('r', r1);
    outer.setAttribute('cx', cx); outer.setAttribute('cy', cy); outer.setAttribute('r', r2);
    outer.classList.toggle('is-ok', pen === 0);
    band.setAttribute('d', circlePath(cx, cy, r2) + circlePath(cx, cy, Math.max(0.01, r1)));

    lineP.setAttribute('x1', cx); lineP.setAttribute('y1', cy);
    lineP.setAttribute('x2', p.x * S); lineP.setAttribute('y2', p.y * S);
    lineN.setAttribute('x1', cx); lineN.setAttribute('y1', cy);
    lineN.setAttribute('x2', n.x * S); lineN.setAttribute('y2', n.y * S);
    labP.setAttribute('x', (a.x + p.x) / 2 * S); labP.setAttribute('y', (a.y + p.y) / 2 * S - 10);
    labN.setAttribute('x', (a.x + n.x) / 2 * S); labN.setAttribute('y', (a.y + n.y) / 2 * S - 10);
    labP.textContent = dp > 0.8 ? dp.toFixed(1) : '';
    labN.textContent = dn > 0.8 ? dn.toFixed(1) : '';

    placeDot(dot2.a, a); placeDot(dot2.p, p); placeDot(dot2.n, n);

    var push = tripletPush(a, p, n, TRIPLET_MARGIN);
    ['a', 'p', 'n'].forEach(function (k) {
      setArrow(ar2[k], tri[k], push[k], 0.9, 0.9);
      ar2[k].setAttribute('class', 'arrow ' + (k === 'n' ? 'push' : 'pull'));
    });

    $('dP').textContent = dp.toFixed(1);
    $('dN').textContent = dn.toFixed(1);
    $('dNeed').textContent = (dp + TRIPLET_MARGIN).toFixed(1);
    $('pen2').textContent = fmt(pen);
    setMeter($('meter2'), pen, 5);
    $('status2').className = 'status ' + (pen === 0 ? 'is-good' : 'is-bad');

    var msg;
    if (pen === 0) msg = 'Goal met! The stranger is outside the dashed circle. No penalty.';
    else if (dn > dp) msg = 'Almost. The stranger is farther than your other picture, but without enough extra room.';
    else msg = 'Not yet. The stranger is closer to you than your own other picture!';
    $('msg2').textContent = msg;

    $('step2').disabled = animating;
    $('step2').textContent = pen === 0 ? 'Goal met. Nothing to fix' : 'Take a learning step';
    $('stepCount').textContent = steps ? steps + (steps === 1 ? ' step taken' : ' steps taken') : '';
  }

  function learnStep() {
    if (animating || tripletPenalty(tri.a, tri.p, tri.n, TRIPLET_MARGIN) === 0) return;
    var from = { a: tri.a, p: tri.p, n: tri.n };
    var to = tripletStep(tri.a, tri.p, tri.n, TRIPLET_MARGIN, 0.45);
    steps++;
    if (reduceMotion) { tri = to; render2(); return; }
    animating = true;
    var t0 = null, DUR = 380;
    function lerp(u, v, t) { return { x: u.x + (v.x - u.x) * t, y: u.y + (v.y - u.y) * t }; }
    function frame(ts) {
      if (t0 === null) t0 = ts;
      var t = Math.min(1, (ts - t0) / DUR), e = 1 - Math.pow(1 - t, 3);
      tri = { a: lerp(from.a, to.a, e), p: lerp(from.p, to.p, e), n: lerp(from.n, to.n, e) };
      if (t === 1) { tri = to; animating = false; }
      render2();
      if (animating) requestAnimationFrame(frame);
    }
    requestAnimationFrame(frame);
  }

  ['a', 'p', 'n'].forEach(function (k) {
    draggable(svg2, dot2[k], function () { return tri[k]; }, function (q) {
      if (animating) return;
      tri[k] = q; render2();
    });
  });

  $('step2').addEventListener('click', learnStep);
  $('reset2').addEventListener('click', function () {
    if (animating) return;
    tri = { a: START.a, p: START.p, n: START.n };
    steps = 0;
    render2();
  });

  render1();
  render2();

})(typeof globalThis !== 'undefined' ? globalThis : this);
