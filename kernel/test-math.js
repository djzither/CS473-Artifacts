/* ============================================================================
   Math checks for the kernel regression module.
   Run:  node test-math.js

   This exercises the SHIPPED script.js, not a copy of its formulas — the file
   exports its pure-math half under Node and skips everything DOM-bound.

   Besides the numbers, it checks the claims the page makes in words: closer
   students count more, the walk from 4 to 9 hours hands the say to the 9-hour
   student, small h copies the nearest student, and large h gives the plain
   average.
   ========================================================================= */

var M = require('./script.js');

var pass = 0, fail = 0;

function ok(name, cond, detail) {
  if (cond) { pass++; console.log('  PASS  ' + name); }
  else { fail++; console.log('  FAIL  ' + name + (detail ? '   -> ' + detail : '')); }
}

function near(a, b, tol) { return Math.abs(a - b) <= (tol || 1e-12); }

function section(t) { console.log('\n' + t); }

function sum(a) { return a.reduce(function (s, v) { return s + v; }, 0); }

/* The formula exactly as written on the page, with no stabilising shift. */
function naiveWeights(x, h) {
  var raw = M.DATA.map(function (d) { return Math.exp(-Math.pow(x - d.x, 2) / (2 * h * h)); });
  var s = sum(raw);
  return raw.map(function (v) { return v / s; });
}

var D = M.DATA;


/* -------------------------------------------------------- 1. the setup -- */
section('1. The four students');
(function () {
  ok('data is (1, 50), (3.5, 75), (4.5, 85), (9, 98)',
     JSON.stringify(D) === JSON.stringify([{ x: 1, y: 50 }, { x: 3.5, y: 75 }, { x: 4.5, y: 85 }, { x: 9, y: 98 }]));
  ok('page starts at x = 4, h = 1', M.START.x === 4 && M.START.h === 1);
  ok('plain average is 77, as the compare card says', M.plainAverage() === 77);
  ok('kernel is 1 at distance 0', M.kernel(3, 3, 0.7) === 1);
})();


/* ------------------------------------------------------- 2. the weights -- */
section('2. Weights');
(function () {
  var xs = [0, 1, 2.3, 4, 6.75, 9, 10], hs = [0.3, 1, 2.5, 5], worst = 0, sums = true;
  xs.forEach(function (x) {
    hs.forEach(function (h) {
      var w = M.weights(x, h), nw = naiveWeights(x, h);
      if (!near(sum(w), 1, 1e-12)) sums = false;
      w.forEach(function (v, i) { worst = Math.max(worst, Math.abs(v - nw[i])); });
    });
  });
  ok('weights always add up to 100%', sums);
  ok('stable weights match the page formula', worst < 1e-12, worst);

  var w = M.weights(4, 1);
  ok('at x = 4, the 3.5 h and 4.5 h students get equal shares', near(w[1], w[2]));
  ok('at x = 4, h = 1, those two share over 99%', w[1] + w[2] > 0.99, (w[1] + w[2]).toFixed(4));

  var closer = true;
  [0.5, 2, 4, 6, 7.5, 10].forEach(function (x) {
    var w = M.weights(x, 1.3);
    for (var i = 0; i < D.length; i++) for (var j = 0; j < D.length; j++) {
      if (Math.abs(x - D[i].x) < Math.abs(x - D[j].x) && !(w[i] > w[j])) closer = false;
    }
  });
  ok('a closer student always gets more weight', closer);

  var tiny = M.weights(6.75, 0.05);
  ok('very small h stays finite (no 0/0)', tiny.every(isFinite) && near(sum(tiny), 1, 1e-12), JSON.stringify(tiny));
})();


/* ---------------------------------------------------- 3. the prediction -- */
section('3. The predicted score');
(function () {
  var w = naiveWeights(4, 1), expect = sum(w.map(function (v, i) { return v * D[i].y; }));
  ok('start: predicted score matches the weighted average (≈ 79.8)', near(M.predict(4, 1), expect, 1e-12) &&
     M.predict(4, 1).toFixed(1) === '79.8', M.predict(4, 1));

  var inside = true;
  for (var x = 0; x <= 10; x += 0.25) [0.2, 1, 5].forEach(function (h) {
    var y = M.predict(x, h);
    if (y < 50 || y > 98) inside = false;
  });
  ok('the prediction always lies between 50 and 98', inside);

  ok('small h copies the nearest student (x = 8, h = 0.2 → 98)', near(M.predict(8, 0.2), 98, 1e-6));
  ok('small h copies the nearest student (x = 1.5, h = 0.2 → 50)', near(M.predict(1.5, 0.2), 50, 1e-6));
  ok('large h gives about the plain average', Math.abs(M.predict(4, 1000) - 77) < 1e-3, M.predict(4, 1000));
  var even = true, lo = 1, hi = 0;
  for (var x2 = 0; x2 <= 10; x2 += 0.25) M.weights(x2, M.H_MAX).forEach(function (v) {
    lo = Math.min(lo, v); hi = Math.max(hi, v);
    if (v < 0.15 || v > 0.35) even = false;
  });
  ok('at the slider max, every student gets about 25% (15–35%) at any study time', even,
     (lo * 100).toFixed(1) + '–' + (hi * 100).toFixed(1) + '%');
})();


/* ------------------------------------------------------ 4. the walk ------ */
section('4. Walk from 4 to 9 hours');
(function () {
  ok('walk goes from 4 to 9', M.WALK.from === 4 && M.WALK.to === 9);
  var up9 = true, down1 = true, upY = true, prev = null;
  for (var x = 4; x <= 9 + 1e-9; x += 0.05) {
    var w = M.weights(x, 1), y = M.predict(x, 1);
    if (prev) {
      if (!(w[3] > prev.w[3])) up9 = false;
      if (!(w[0] < prev.w[0])) down1 = false;
      if (!(y > prev.y)) upY = false;
    }
    prev = { w: w, y: y };
  }
  ok('the 9 h student only gains weight along the way', up9);
  ok('the 1 h student only loses weight along the way', down1);
  ok('the predicted score rises the whole way', upY);
  var end = M.weights(9, 1);
  ok('at 9 hours, the 9 h student has over 99% of the say', end[3] > 0.99, end[3]);
})();


console.log('\n' + pass + ' passed, ' + fail + ' failed');
process.exit(fail ? 1 : 0);
