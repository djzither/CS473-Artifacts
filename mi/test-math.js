/* ============================================================================
   Math checks for the mutual information module.   Run:  node test-math.js

   This exercises the SHIPPED script.js, not a copy of its formulas — the file
   exports its pure-math half under Node and skips everything DOM-bound.

   The closed forms on the page are checked against a second, independent
   route: build the full joint p(x, y, z) over all 8 outcomes and compute every
   mutual information from plain entropies, H(·) of marginals.
   ========================================================================= */

var M = require('./script.js');

var pass = 0, fail = 0;

function ok(name, cond, detail) {
  if (cond) { pass++; console.log('  PASS  ' + name); }
  else { fail++; console.log('  FAIL  ' + name + (detail ? '   -> ' + detail : '')); }
}

function near(a, b, tol) { return Math.abs(a - b) <= (tol || 1e-9); }

function section(t) { console.log('\n' + t); }


/* ---------------------------------------------- brute force from the joint -- */

function joint(p, q) {
  var J = {};
  for (var x = 0; x < 2; x++)
    for (var y = 0; y < 2; y++)
      for (var z = 0; z < 2; z++)
        J['' + x + y + z] = 0.5 * (y !== x ? p : 1 - p) * (z !== y ? q : 1 - q);
  return J;
}

/* Entropy of the marginal over the listed positions (0 = X, 1 = Y, 2 = Z). */
function H(J, idx) {
  var marg = {}, k, h = 0;
  for (k in J) {
    var key = idx.map(function (i) { return k[i]; }).join('');
    marg[key] = (marg[key] || 0) + J[k];
  }
  for (k in marg) if (marg[k] > 0) h -= marg[k] * Math.log2(marg[k]);
  return h;
}

/* I(A;B|C) = H(A,C) + H(B,C) − H(A,B,C) − H(C). With C empty, H(C) = 0. */
function I(J, a, b, c) {
  c = c || [];
  return H(J, a.concat(c)) + H(J, b.concat(c)) - H(J, a.concat(b, c)) - (c.length ? H(J, c) : 0);
}

var X = [0], Y = [1], Z = [2];

var GRID = [];
for (var i = 0; i <= 50; i++) for (var j = 0; j <= 50; j++) GRID.push([i / 100, j / 100]);


/* ------------------------------------------------------ 1. known values ---- */
section('1. Known values');
(function () {
  ok('h2(0) = 0', M.h2(0) === 0);
  ok('h2(1) = 0', M.h2(1) === 0);
  ok('h2(0.5) = 1', near(M.h2(0.5), 1));
  ok('h2(0.1) = 0.4690', near(M.h2(0.1), 0.468996, 1e-6));
  ok('h2 is symmetric', near(M.h2(0.3), M.h2(0.7)));
  ok('r(0.1, 0.2) = 0.26', near(M.compose(0.1, 0.2), 0.26));
  ok('r(p, 0) = p', near(M.compose(0.17, 0), 0.17));
  ok('r(p, 0.5) = 0.5 for any p', near(M.compose(0.1, 0.5), 0.5) && near(M.compose(0, 0.5), 0.5));
  var m = M.measures(0.1, 0);
  ok('default: I(X;Y) = 0.531', near(m.Ixy, 0.531004, 1e-6));
  ok('default: I(X;Z) = I(X;Y), I(X;Y|Z) = 0', near(m.Ixz, m.Ixy) && near(m.IxyGz, 0));
  m = M.measures(0, 0);
  ok('noiseless: Y = Z = X carries exactly 1 bit', near(m.Ixy, 1) && near(m.Ixz, 1));
})();


/* --------------------------------------------- 2. presets, as advertised ---- */
section('2. Presets');
(function () {
  var a = M.measures(0.1, 0), b = M.measures(0.1, 0.2), c = M.measures(0.1, 0.5);
  ok('no information lost: I(X;Y|Z) = 0', near(a.IxyGz, 0));
  ok('some lost: 0 < I(X;Z) < I(X;Y)', b.Ixz > 0.1 && b.Ixz < b.Ixy - 0.1,
     b.Ixz.toFixed(4) + ' vs ' + b.Ixy.toFixed(4));
  ok('all lost: I(X;Z) = 0', near(c.Ixz, 0));
  ok('all lost: I(X;Y|Z) = I(X;Y)', near(c.IxyGz, c.Ixy));
  var J = joint(0.1, 0.5), indep = true;
  for (var x = 0; x < 2; x++) for (var z = 0; z < 2; z++) {
    var pxz = J[x + '0' + z] + J[x + '1' + z];
    if (!near(pxz, 0.25)) indep = false;
  }
  ok('q = 0.5: p(x, z) = p(x) p(z) exactly', indep);
})();


/* -------------------------------- 3. closed forms vs. the full joint ---- */
section('3. Closed forms agree with brute force on a 51 x 51 grid of (p, q)');
(function () {
  var bad = { xy: 0, xz: 0, xyGz: 0, xzGy: 0, xYZ: 0 }, worst = 0;
  GRID.forEach(function (pq) {
    var J = joint(pq[0], pq[1]), m = M.measures(pq[0], pq[1]);
    var d = [
      ['xy', I(J, X, Y), m.Ixy],
      ['xz', I(J, X, Z), m.Ixz],
      ['xyGz', I(J, X, Y, Z), m.IxyGz],
      ['xzGy', I(J, X, Z, Y), m.IxzGy],
      ['xYZ', I(J, X, Y.concat(Z)), m.IxYZ]
    ];
    d.forEach(function (t) {
      var e = Math.abs(t[1] - t[2]);
      worst = Math.max(worst, e);
      if (e > 1e-9) bad[t[0]]++;
    });
  });
  ok('I(X;Y) = 1 - h2(p)', bad.xy === 0, bad.xy + ' bad');
  ok('I(X;Z) = 1 - h2(r)', bad.xz === 0, bad.xz + ' bad');
  ok('I(X;Y|Z) = h2(r) - h2(p)', bad.xyGz === 0, bad.xyGz + ' bad');
  ok('I(X;Z|Y) = 0 (Markov chain)', bad.xzGy === 0, bad.xzGy + ' bad');
  ok('I(X;Y,Z) = I(X;Y)', bad.xYZ === 0, bad.xYZ + ' bad');
  console.log('        worst disagreement: ' + worst.toExponential(2) + ' bits');
})();


/* ------------------------------------------ 4. chain rule and the DPI ---- */
section('4. Chain rule and data processing inequality');
(function () {
  var badChainA = 0, badChainB = 0, badBars = 0, badDPI = 0, badNonneg = 0, badRange = 0;
  GRID.forEach(function (pq) {
    var J = joint(pq[0], pq[1]), m = M.measures(pq[0], pq[1]);
    var IxYZ = I(J, X, Y.concat(Z));
    if (!near(IxYZ, I(J, X, Y) + I(J, X, Z, Y))) badChainA++;
    if (!near(IxYZ, I(J, X, Z) + I(J, X, Y, Z))) badChainB++;
    if (!near(m.Ixy, m.Ixz + m.IxyGz)) badBars++;
    if (m.Ixz > m.Ixy + 1e-12) badDPI++;
    if (m.IxyGz < -1e-12 || m.Ixz < -1e-12) badNonneg++;
    if (m.r < pq[0] - 1e-12 || m.r > 0.5 + 1e-12) badRange++;
  });
  ok('I(X;Y,Z) = I(X;Y) + I(X;Z|Y)', badChainA === 0, badChainA + ' bad');
  ok('I(X;Y,Z) = I(X;Z) + I(X;Y|Z)', badChainB === 0, badChainB + ' bad');
  ok('top bar = middle bar + bottom bar', badBars === 0, badBars + ' bad');
  ok('I(X;Z) <= I(X;Y) everywhere', badDPI === 0, badDPI + ' bad');
  ok('no bar is ever negative', badNonneg === 0, badNonneg + ' bad');
  ok('p <= r <= 0.5 on the slider range', badRange === 0, badRange + ' bad');
})();


/* ------------------------------------------------------ 5. the sample tape ---- */
section('5. Sample tape');
(function () {
  var ds = M.draws(200000, 7), p = 0.1, q = 0.2;
  var rows = M.transmit(ds, p, q), ones = 0, dY = 0, dZ = 0;
  rows.forEach(function (r) { ones += r.x; if (r.y !== r.x) dY++; if (r.z !== r.x) dZ++; });
  var n = rows.length;
  ok('X is a fair bit', near(ones / n, 0.5, 0.005), (ones / n).toFixed(4));
  ok('Y disagrees with X at rate p', near(dY / n, p, 0.004), (dY / n).toFixed(4));
  ok('Z disagrees with X at rate r', near(dZ / n, M.compose(p, q), 0.004), (dZ / n).toFixed(4));

  var small = M.draws(40, 473005), monotone = true, prev = -1;
  for (var k = 0; k <= 50; k++) {
    var flips = M.transmit(small, k / 100, 0).filter(function (r) { return r.f1; }).length;
    if (flips < prev) monotone = false;
    prev = flips;
  }
  ok('raising p only adds flips, never reshuffles', monotone);
  ok('q = 0 gives Z = Y on every draw', M.transmit(small, 0.3, 0).every(function (r) { return r.z === r.y; }));
})();


console.log('\n' + pass + ' passed, ' + fail + ' failed');
process.exit(fail ? 1 : 0);
