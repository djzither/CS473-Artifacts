/* ============================================================================
   Math checks for the contrastive / triplet loss module.
   Run:  node test-math.js

   This exercises the SHIPPED script.js, not a copy of its formulas — the file
   exports its pure-math half under Node and skips everything DOM-bound.

   Besides the penalty values, it checks the claims the page makes in words:
   the arrows point downhill, the learning step lowers the penalty and stops
   when the goal is met, and the two losses differ in the ways the comparison
   card says they do.
   ========================================================================= */

var M = require('./script.js');

var pass = 0, fail = 0;

function ok(name, cond, detail) {
  if (cond) { pass++; console.log('  PASS  ' + name); }
  else { fail++; console.log('  FAIL  ' + name + (detail ? '   -> ' + detail : '')); }
}

function near(a, b, tol) { return Math.abs(a - b) <= (tol || 1e-12); }

function section(t) { console.log('\n' + t); }

function P(x, y) { return { x: x, y: y }; }
function shift(p, dx, dy) { return P(p.x + dx, p.y + dy); }

/* Numerical gradient of f at point p, for checking the analytic pushes. */
function numGrad(f, p) {
  var h = 1e-6;
  return P((f(P(p.x + h, p.y)) - f(P(p.x - h, p.y))) / (2 * h),
           (f(P(p.x, p.y + h)) - f(P(p.x, p.y - h))) / (2 * h));
}

var m1 = M.PAIR_MARGIN, m2 = M.TRIPLET_MARGIN;


/* ------------------------------------------------- 1. contrastive values -- */
section('1. Two pictures (contrastive)');
(function () {
  var a = P(3, 3.8);
  ok('same person 5.5 apart: penalty 30.25', near(M.contrastivePenalty(a, P(8.5, 3.8), true, m1), 30.25));
  ok('same person on top of each other: penalty 0', M.contrastivePenalty(a, P(3, 3.8), true, m1) === 0);
  ok('same person 0.1 apart: penalty still above 0', M.contrastivePenalty(a, P(3.1, 3.8), true, m1) > 0);
  ok('different people 1 apart: penalty (3 − 1)² = 4', near(M.contrastivePenalty(a, P(4, 3.8), false, m1), 4));
  ok('different people exactly 3 apart: penalty 0', M.contrastivePenalty(a, P(6, 3.8), false, m1) === 0);
  ok('different people 6 apart: still 0 (farther earns nothing)', M.contrastivePenalty(a, P(9, 3.8), false, m1) === 0);

  var worse = true, prev = -1;
  for (var d = 2.9; d >= 0; d -= 0.1) {
    var v = M.contrastivePenalty(a, P(3 + d, 3.8), false, m1);
    if (v <= prev) worse = false;
    prev = v;
  }
  ok('inside the circle, closer is always worse', worse);
})();


/* -------------------------------------------- 2. contrastive push arrows -- */
section('2. Arrows in view 1 point downhill');
(function () {
  var a = P(3, 3.8), cases = [[P(7, 6), true], [P(4.2, 4.5), false], [P(3.4, 3.1), false]];
  cases.forEach(function (c) {
    var b = c[0], same = c[1];
    var push = M.contrastivePush(a, b, same, m1);
    var g = numGrad(function (q) { return M.contrastivePenalty(a, q, same, m1); }, b);
    ok((same ? 'same' : 'different') + ' at (' + b.x + ', ' + b.y + '): push = −gradient',
       near(push.x, -g.x, 1e-5) && near(push.y, -g.y, 1e-5), JSON.stringify([push, g]));
  });
  var pp = M.contrastivePush(a, P(7, 6), true, m1);
  ok('same person: push on b points toward a', pp.x < 0 && pp.y < 0);
  var pd = M.contrastivePush(a, P(4, 3.8), false, m1);
  ok('different people too close: push on b points away from a', pd.x > 0 && pd.y === 0);
  var pz = M.contrastivePush(a, P(8, 3.8), false, m1);
  ok('different people far enough: no push', pz.x === 0 && pz.y === 0);
})();


/* ---------------------------------------------------- 3. triplet values -- */
section('3. Three pictures (triplet)');
(function () {
  var a = P(0, 0);
  ok('your picture at 2, stranger at 2: penalty 0 + 1 = 1', near(M.tripletPenalty(a, P(2, 0), P(0, 2), m2), 1));
  ok('your picture at 2, stranger at 1: penalty 2', near(M.tripletPenalty(a, P(2, 0), P(1, 0), m2), 2));
  ok('stranger exactly 1 farther: penalty 0', M.tripletPenalty(a, P(2, 0), P(0, 3), m2) === 0);
  ok('stranger farther but not by enough: 0 < penalty < 1',
     M.tripletPenalty(a, P(2, 0), P(0, 2.5), m2) > 0 && M.tripletPenalty(a, P(2, 0), P(0, 2.5), m2) < 1);

  var s = { a: P(4.5, 3.8), p: P(6.5, 3), n: P(5, 5.5) };
  var pen = M.tripletPenalty(s.a, s.p, s.n, m2);
  ok('starting layout on the page has the stranger closer (penalty > 1)', pen > 1, pen);
})();


/* --------------------------------------------------- 4. the learning step -- */
section('4. Take a learning step');
(function () {
  var a = P(4.5, 3.8), p = P(6.5, 3), n = P(5, 5.5);
  var g = M.tripletPush(a, p, n, m2);
  ['a', 'p', 'n'].forEach(function (k) {
    var f = function (q) {
      var t = { a: a, p: p, n: n }; t[k] = q;
      return M.tripletPenalty(t.a, t.p, t.n, m2);
    };
    var ng = numGrad(f, { a: a, p: p, n: n }[k]);
    ok('push on ' + k + ' = −gradient', near(g[k].x, -ng.x, 1e-5) && near(g[k].y, -ng.y, 1e-5));
  });

  var s = { a: a, p: p, n: n }, pens = [M.tripletPenalty(a, p, n, m2)], down = true, k;
  for (k = 0; k < 40 && pens[pens.length - 1] > 0; k++) {
    s = M.tripletStep(s.a, s.p, s.n, m2, 0.45);
    pens.push(M.tripletPenalty(s.a, s.p, s.n, m2));
    if (pens[pens.length - 1] >= pens[pens.length - 2]) down = false;
  }
  ok('every click lowers the penalty', down, pens.map(function (v) { return v.toFixed(2); }).join(' '));
  ok('from the page’s start, the goal is met within 6 clicks', pens[pens.length - 1] === 0 && k <= 6, k + ' clicks');

  var done = M.tripletStep(s.a, s.p, s.n, m2, 0.45);
  ok('once the goal is met, a step moves nothing',
     ['a', 'p', 'n'].every(function (q) { return done[q].x === s[q].x && done[q].y === s[q].y; }));

  var edge = M.tripletStep(P(1, 1), P(3, 1), P(0.5, 0.5), m2, 2);
  ok('steps keep dots on the board',
     ['a', 'p', 'n'].every(function (q) {
       return edge[q].x >= M.PAD && edge[q].x <= M.W - M.PAD && edge[q].y >= M.PAD && edge[q].y <= M.H - M.PAD;
     }));
})();


/* ---------------------------------------------- 5. how the two differ ---- */
section('5. The comparison card');
(function () {
  var a = P(3, 3), p = P(6, 3), n = P(3, 7);
  ok('triplet satisfied while your two pictures are still 3 apart', M.tripletPenalty(a, p, n, m2) === 0);
  ok('...but contrastive still charges that same-person pair 9', near(M.contrastivePenalty(a, p, true, m1), 9));

  var moved = [a, p, n].map(function (q) { return shift(q, 2.5, -1.2); });
  ok('slide all three together: triplet penalty unchanged',
     near(M.tripletPenalty(moved[0], moved[1], moved[2], m2), M.tripletPenalty(a, p, n, m2)));
  var t = { a: P(4.5, 3.8), p: P(6.5, 3), n: P(5, 5.5) };
  var tm = [t.a, t.p, t.n].map(function (q) { return shift(q, -1, 0.5); });
  ok('slide all three together (from the start layout): same penalty',
     near(M.tripletPenalty(tm[0], tm[1], tm[2], m2), M.tripletPenalty(t.a, t.p, t.n, m2), 1e-12));

  var close = M.tripletPush(P(0, 0), P(1, 0), P(0, 1.5), m2);
  ok('triplet push pulls your picture in and pushes the stranger out',
     close.p.x < 0 && close.n.y > 0);
})();


console.log('\n' + pass + ' passed, ' + fail + ' failed');
process.exit(fail ? 1 : 0);
