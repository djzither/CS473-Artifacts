/* ============================================================================
   Closer Counts More — Gaussian kernel regression, for someone who has never
   seen machine learning. CS 473 module 9 · Murphy, PML §16.3

   Four students, (hours studied, exam score). To predict a new student who
   studied x hours, give each student a share of the say that shrinks with
   how far their study time is from x, then take the weighted average:

     w_n(x) = e^(−(x − x_n)² / 2h²) / Σ_j e^(−(x − x_j)² / 2h²)
     ŷ(x)   = Σ_n w_n(x) · y_n

   h is the bandwidth, in hours. Small h: only the closest student counts.
   Large h: everyone counts the same and ŷ becomes the plain average.

   Layout of this file
     PART 1  Math          pure, DOM-free, exported for the Node test suite
     PART 2  Drawing       the chart: axes, bump, students, line, prediction
     PART 3  Controls      dragging, keyboard, slider, toggles, the walk

   PART 1 runs under Node (`require('./script.js')` returns the math only); the
   rest is skipped when there is no document. See test-math.js.
   ========================================================================= */

(function (global) {
  'use strict';

  /* ==========================================================================
     PART 1 — MATH
     ====================================================================== */

  var DATA = [
    { x: 1,   y: 50 },
    { x: 3.5, y: 75 },
    { x: 4.5, y: 85 },
    { x: 9,   y: 98 }
  ];
  var X_MIN = 0, X_MAX = 10;        // hours on the axis
  var H_MIN = 0.2, H_MAX = 10;      // the bandwidth slider (log scale)
  var START = { x: 4, h: 1 };
  var WALK = { from: 4, to: 9 };

  /* The bell-shaped raw score: 1 at distance 0, shrinking with distance. */
  function kernel(t, x, h) {
    var d = t - x;
    return Math.exp(-d * d / (2 * h * h));
  }

  /* Normalized weights. Exponents are shifted by the largest one before
     exponentiating (it cancels in the ratio), so a tiny h never turns every
     raw score into 0 and the weights into 0/0. */
  function weights(x, h, data) {
    data = data || DATA;
    var e = data.map(function (d) { var u = x - d.x; return -u * u / (2 * h * h); });
    var top = Math.max.apply(null, e);
    var raw = e.map(function (v) { return Math.exp(v - top); });
    var sum = raw.reduce(function (s, v) { return s + v; }, 0);
    return raw.map(function (v) { return v / sum; });
  }

  function predict(x, h, data) {
    data = data || DATA;
    var w = weights(x, h, data);
    return data.reduce(function (s, d, i) { return s + w[i] * d.y; }, 0);
  }

  function plainAverage(data) {
    data = data || DATA;
    return data.reduce(function (s, d) { return s + d.y; }, 0) / data.length;
  }

  var MATH = {
    DATA: DATA, X_MIN: X_MIN, X_MAX: X_MAX, H_MIN: H_MIN, H_MAX: H_MAX, START: START, WALK: WALK,
    kernel: kernel, weights: weights, predict: predict, plainAverage: plainAverage
  };

  if (typeof module !== 'undefined' && module.exports) module.exports = MATH;
  if (typeof document === 'undefined') return;

  /* ==========================================================================
     PART 2 — DRAWING
     ====================================================================== */

  var NS = 'http://www.w3.org/2000/svg';
  var VW = 600;
  var L = 62, R = 582, T = 58, B = 360;          // plot area in the viewBox
  var Y_MIN = 30, Y_MAX = 105;                   // score axis
  var BUMP_H = 70;                               // px height of the bell at its peak
  var R_MIN = 6, R_MAX = 22;                     // student dot radius, px
  var reduceMotion = global.matchMedia && global.matchMedia('(prefers-reduced-motion: reduce)').matches;

  function $(id) { return document.getElementById(id); }

  function el(name, attrs, parent) {
    var e = document.createElementNS(NS, name);
    for (var k in attrs) e.setAttribute(k, attrs[k]);
    if (parent) parent.appendChild(e);
    return e;
  }

  function sx(x) { return L + (x - X_MIN) / (X_MAX - X_MIN) * (R - L); }
  function sy(y) { return B - (y - Y_MIN) / (Y_MAX - Y_MIN) * (B - T); }

  function hours(v) { return v.toFixed(1) + (v === 1 ? ' hour' : ' hours'); }
  function short(v) { return (v % 1 ? v.toFixed(1) : String(v)) + ' h'; }

  /* Whole percents, but never round a real share down to a bare "0%". */
  function pct(w) {
    if (w < 0.0005) return '0%';
    if (w < 0.005) return '<1%';
    return Math.round(w * 100) + '%';
  }

  var state = { x: START.x, h: START.h, compare: false, curve: false };
  var walking = null;

  var svg = $('chart');

  // grid and axes
  var gridG = el('g', { 'class': 'grid' }, svg);
  for (var gx = 1; gx <= X_MAX; gx++) el('line', { x1: sx(gx), y1: T, x2: sx(gx), y2: B }, gridG);
  for (var gy = 40; gy <= 100; gy += 10) el('line', { x1: L, y1: sy(gy), x2: R, y2: sy(gy) }, gridG);
  el('line', { 'class': 'axis', x1: L, y1: B, x2: R, y2: B }, svg);
  el('line', { 'class': 'axis', x1: L, y1: T, x2: L, y2: B }, svg);
  for (var tx = 0; tx <= X_MAX; tx++) {
    var t1 = el('text', { 'class': 'tick', x: sx(tx), y: B + 20, 'text-anchor': 'middle' }, svg);
    t1.textContent = tx;
  }
  for (var ty = 40; ty <= 100; ty += 10) {
    var t2 = el('text', { 'class': 'tick', x: L - 10, y: sy(ty) + 4, 'text-anchor': 'end' }, svg);
    t2.textContent = ty;
  }
  el('text', { 'class': 'axis-title', x: (L + R) / 2, y: B + 50 }, svg).textContent = 'Hours studied';
  el('text', { 'class': 'axis-title', x: 0, y: 0, transform: 'translate(18,' + (T + B) / 2 + ') rotate(-90)' }, svg)
    .textContent = 'Exam score';

  // marks, back to front
  var bump = el('path', { 'class': 'bump' }, svg);
  var bumpLabel = el('text', { 'class': 'bump-label' }, svg);
  bumpLabel.textContent = 'who counts';
  var curve = el('path', { 'class': 'curve' }, svg);
  var avgLine = el('line', { 'class': 'avg-line', x1: L, x2: R }, svg);
  var avgLabel = el('text', { 'class': 'avg-label', x: R - 6 }, svg);
  var threads = DATA.map(function () { return el('line', { 'class': 'thread' }, svg); });

  var study = el('g', { 'class': 'study', tabindex: 0, role: 'slider', 'aria-label': 'Your study time, in hours',
                        'aria-valuemin': X_MIN, 'aria-valuemax': X_MAX }, svg);
  var studyHit = el('rect', { 'class': 'hit', y: 8, height: B - 8, width: 40 }, study);
  var studyLine = el('line', { 'class': 'study-line', y1: 34, y2: B }, study);
  var pill = el('rect', { 'class': 'pill', y: 8, height: 26, rx: 13, width: 96 }, study);
  var pillText = el('text', { 'class': 'pill-text', y: 26 }, study);

  var dots = DATA.map(function (d) {
    return el('circle', { 'class': 'student', cx: sx(d.x), cy: sy(d.y) }, svg);
  });
  var pcts = DATA.map(function () { return el('text', { 'class': 'pct' }, svg); });

  var avgDot = el('circle', { 'class': 'avg-dot', r: 9 }, svg);
  var pred = el('circle', { 'class': 'pred', r: 13 }, svg);
  var predLabel = el('text', { 'class': 'pred-label' }, svg);

  // the contribution table
  var tbody = $('contrib').querySelector('tbody');
  var rows = DATA.map(function (d) {
    var tr = document.createElement('tr');
    tr.innerHTML = '<td>' + short(d.x) + '</td><td class="num">' + d.y + '</td>' +
      '<td class="w-cell"><div class="w-wrap"><div class="w-bar"><div class="w-fill"></div></div>' +
      '<span class="num w-num"></span></div></td>';
    tbody.appendChild(tr);
    return { fill: tr.querySelector('.w-fill'), num: tr.querySelector('.w-num') };
  });

  function bumpPath(x, h) {
    var d = 'M' + L + ',' + B;
    for (var i = 0; i <= 200; i++) {
      var t = X_MIN + (X_MAX - X_MIN) * i / 200;
      d += 'L' + sx(t).toFixed(1) + ',' + (B - BUMP_H * kernel(t, x, h)).toFixed(1);
    }
    return d + 'L' + R + ',' + B + 'Z';
  }

  function curvePath(h) {
    var d = '';
    for (var i = 0; i <= 250; i++) {
      var t = X_MIN + (X_MAX - X_MIN) * i / 250;
      d += (i ? 'L' : 'M') + sx(t).toFixed(1) + ',' + sy(predict(t, h)).toFixed(1);
    }
    return d;
  }

  /* One sentence on who the prediction is listening to. */
  function describe(w) {
    var order = w.map(function (v, i) { return i; }).sort(function (i, j) { return w[j] - w[i]; });
    var spread = w[order[0]] - w[order[w.length - 1]];
    if (spread < 0.1) return 'Everyone counts about the same, so this is close to the plain average.';
    if (w[order[0]] > 0.9) return 'Almost all of it comes from the ' + short(DATA[order[0]].x) + ' student.';
    var big = order.filter(function (i) { return w[i] >= 0.15; })
      .sort(function (i, j) { return DATA[i].x - DATA[j].x; })
      .map(function (i) { return short(DATA[i].x); });
    var list = big.length === 2 ? big.join(' and ')
      : big.slice(0, -1).join(', ') + ' and ' + big[big.length - 1];
    return 'Mostly the ' + list + ' students. The rest barely count.';
  }

  function render() {
    var x = state.x, h = state.h;
    var w = weights(x, h), yhat = predict(x, h), avg = plainAverage();
    var cx = sx(x), cy = sy(yhat);

    bump.setAttribute('d', bumpPath(x, h));
    // Beside the line, not under it: right of the peak unless that runs off the plot.
    var labRight = cx < R - 100;
    bumpLabel.setAttribute('x', labRight ? cx + 10 : cx - 10);
    bumpLabel.setAttribute('text-anchor', labRight ? 'start' : 'end');
    bumpLabel.setAttribute('y', B - BUMP_H - 6);

    curve.setAttribute('d', state.curve ? curvePath(h) : '');

    avgLine.setAttribute('y1', sy(avg)); avgLine.setAttribute('y2', sy(avg));
    avgLabel.setAttribute('y', sy(avg) - 8);
    avgLabel.textContent = 'plain average ' + avg;
    [avgLine, avgLabel, avgDot].forEach(function (e) { e.setAttribute('visibility', state.compare ? 'visible' : 'hidden'); });
    avgDot.setAttribute('cx', cx); avgDot.setAttribute('cy', sy(avg));

    // Area of each dot ∝ its weight, and brightness follows the same share.
    DATA.forEach(function (d, i) {
      var r = R_MIN + (R_MAX - R_MIN) * Math.sqrt(w[i]);
      dots[i].setAttribute('r', r.toFixed(2));
      dots[i].setAttribute('fill-opacity', (0.12 + 0.88 * w[i]).toFixed(3));
      dots[i].setAttribute('stroke-opacity', (0.35 + 0.65 * w[i]).toFixed(3));

      pcts[i].setAttribute('x', sx(d.x));
      pcts[i].setAttribute('y', sy(d.y) - r - 8);
      pcts[i].textContent = pct(w[i]);
      pcts[i].setAttribute('class', 'pct' + (w[i] < 0.005 ? ' is-tiny' : ''));

      var th = threads[i];
      th.setAttribute('x1', sx(d.x)); th.setAttribute('y1', sy(d.y));
      th.setAttribute('x2', cx); th.setAttribute('y2', cy);
      th.setAttribute('stroke-width', (1 + 5 * w[i]).toFixed(2));
      th.setAttribute('stroke-opacity', (0.7 * w[i]).toFixed(3));

      rows[i].fill.style.width = (w[i] * 100) + '%';
      rows[i].num.textContent = pct(w[i]);
    });

    studyLine.setAttribute('x1', cx); studyLine.setAttribute('x2', cx);
    studyHit.setAttribute('x', cx - 20);
    var px = Math.min(R - 48, Math.max(L + 48, cx));
    pill.setAttribute('x', px - 48);
    pillText.setAttribute('x', px);
    pillText.textContent = hours(x);
    study.setAttribute('aria-valuenow', x.toFixed(1));
    study.setAttribute('aria-valuetext', hours(x) + ', predicted score ' + yhat.toFixed(1));

    pred.setAttribute('cx', cx); pred.setAttribute('cy', cy);
    // Try right, left, above, below; keep the first spot clear of every dot.
    var LW = 58, LH = 20, rs = DATA.map(function (d, i) { return R_MIN + (R_MAX - R_MIN) * Math.sqrt(w[i]); });
    var mx = Math.min(R - LW / 2, Math.max(L + LW / 2, cx));
    var spots = [
      { x: cx + 20, y: cy + 7, a: 'start', box: [cx + 18, cy - 12] },
      { x: cx - 20, y: cy + 7, a: 'end', box: [cx - 20 - LW, cy - 12] },
      { x: mx, y: cy - 22, a: 'middle', box: [mx - LW / 2, cy - 40] },
      { x: mx, y: cy + 46, a: 'middle', box: [mx - LW / 2, cy + 28] }
    ];
    function clash(sp) {
      var bx = sp.box[0], by = sp.box[1];
      if (bx < L || bx + LW > R || by < T - 10 || by + LH > B) return Infinity;
      return DATA.reduce(function (n, d, i) {
        var px = Math.max(bx, Math.min(sx(d.x), bx + LW)), py = Math.max(by, Math.min(sy(d.y), by + LH));
        return n + (Math.hypot(px - sx(d.x), py - sy(d.y)) < rs[i] + 2 ? 1 : 0);
      }, 0);
    }
    var best = spots.reduce(function (b, sp) { var c = clash(sp); return c < b.c ? { sp: sp, c: c } : b; }, { sp: spots[0], c: Infinity }).sp;
    predLabel.setAttribute('x', best.x);
    predLabel.setAttribute('y', best.y);
    predLabel.setAttribute('text-anchor', best.a);
    predLabel.textContent = yhat.toFixed(1);

    $('yhat').textContent = yhat.toFixed(1);
    $('xread').textContent = 'if you study ' + hours(x);
    $('hVal').textContent = (h < 1 ? h.toFixed(2) : h.toFixed(1)) + ' h';
    $('avgRow').hidden = !state.compare;
    var diff = yhat - avg;
    $('avgVal').textContent = avg;
    $('avgDiff').textContent = Math.abs(diff) < 0.05 ? 'Same as the kernel guess.'
      : 'The kernel guess is ' + Math.abs(diff).toFixed(1) + (diff > 0 ? ' higher.' : ' lower.');
    $('msg').textContent = describe(w);
  }

  /* ==========================================================================
     PART 3 — CONTROLS
     ====================================================================== */

  function setX(x) {
    state.x = Math.min(X_MAX, Math.max(X_MIN, Math.round(x * 20) / 20));
    render();
  }

  function stopWalk() {
    if (walking) { cancelAnimationFrame(walking); walking = null; }
  }

  function toHours(evt) {
    var pt = svg.createSVGPoint();
    pt.x = evt.clientX; pt.y = evt.clientY;
    var q = pt.matrixTransform(svg.getScreenCTM().inverse());
    return X_MIN + (q.x - L) / (R - L) * (X_MAX - X_MIN);
  }

  // Press anywhere on the chart: the line jumps there and follows the pointer.
  var dragging = false;
  svg.addEventListener('pointerdown', function (e) {
    e.preventDefault();
    stopWalk();
    dragging = true;
    svg.setPointerCapture(e.pointerId);
    study.focus({ preventScroll: true });
    setX(toHours(e));
  });
  svg.addEventListener('pointermove', function (e) { if (dragging) setX(toHours(e)); });
  function end() { dragging = false; }
  svg.addEventListener('pointerup', end);
  svg.addEventListener('pointercancel', end);

  study.addEventListener('keydown', function (e) {
    var step = e.shiftKey ? 0.5 : 0.1, x = state.x;
    if (e.key === 'ArrowLeft' || e.key === 'ArrowDown') x -= step;
    else if (e.key === 'ArrowRight' || e.key === 'ArrowUp') x += step;
    else if (e.key === 'Home') x = X_MIN;
    else if (e.key === 'End') x = X_MAX;
    else return;
    e.preventDefault();
    stopWalk();
    setX(x);
  });

  // The slider runs 0–100 on a log scale, so the small-h end isn't cramped:
  // each notch multiplies h by the same amount.
  var slider = $('hSlider');
  function hToSlider(h) { return 100 * Math.log(h / H_MIN) / Math.log(H_MAX / H_MIN); }
  function sliderToH(v) { return H_MIN * Math.pow(H_MAX / H_MIN, v / 100); }
  slider.value = hToSlider(START.h);
  slider.addEventListener('input', function () { state.h = sliderToH(parseFloat(slider.value)); render(); });

  function toggle(btn, key) {
    btn.addEventListener('click', function () {
      state[key] = !state[key];
      btn.setAttribute('aria-pressed', state[key]);
      render();
    });
  }
  toggle($('cmpBtn'), 'compare');
  toggle($('curveBtn'), 'curve');

  // The walk: put the line at 4 hours and slide it to 9, slowly enough to
  // watch the share move from the 3.5 h and 4.5 h students to the 9 h one.
  $('walk').addEventListener('click', function () {
    stopWalk();
    if (reduceMotion) { setX(WALK.to); return; }
    var t0 = null, DUR = 3200;
    function frame(ts) {
      if (t0 === null) t0 = ts;
      var t = Math.min(1, (ts - t0) / DUR);
      var e = t < 0.5 ? 2 * t * t : 1 - Math.pow(-2 * t + 2, 2) / 2;
      state.x = WALK.from + (WALK.to - WALK.from) * e;
      render();
      walking = t < 1 ? requestAnimationFrame(frame) : null;
    }
    walking = requestAnimationFrame(frame);
  });

  $('reset').addEventListener('click', function () {
    stopWalk();
    state.x = START.x; state.h = START.h;
    slider.value = hToSlider(START.h);
    render();
  });

  render();

})(typeof globalThis !== 'undefined' ? globalThis : this);
