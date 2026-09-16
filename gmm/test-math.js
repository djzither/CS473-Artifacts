/* ============================================================================
   Math checks for the GMM module.   Run:  node test-math.js

   This exercises the SHIPPED script.js, not a copy of its formulas — the file
   exports its pure-math half under Node and skips everything DOM-bound, so a
   green run here is a statement about the code the browser actually executes.
   ========================================================================= */

var M = require('./script.js');

var pass = 0, fail = 0;

function ok(name, cond, detail) {
  if (cond) { pass++; console.log('  PASS  ' + name); }
  else { fail++; console.log('  FAIL  ' + name + (detail ? '   -> ' + detail : '')); }
}

function near(a, b, tol) { return Math.abs(a - b) <= tol; }

function section(t) { console.log('\n' + t); }

var rnd = M.mulberry32(20240516);
function u(lo, hi) { return lo + (hi - lo) * rnd(); }

/* A pool of random but legal parameter sets to hammer every invariant with. */
function randomParams() {
  return {
    comps: [
      { mx: u(-4, 4), my: u(-3, 3), sx: u(0.25, 2.6), sy: u(0.25, 2.6), rot: u(-Math.PI / 2, Math.PI / 2) },
      { mx: u(-4, 4), my: u(-3, 3), sx: u(0.25, 2.6), sy: u(0.25, 2.6), rot: u(-Math.PI / 2, Math.PI / 2) }
    ],
    pi1: u(0.05, 0.95)
  };
}

/* ---------------------------------------------------------------- 1. weights */
section('1. Mixture weights sum to 1');
(function () {
  var worst = 0;
  for (var i = 0; i < 400; i++) {
    var p = randomParams();
    for (var t = 0; t <= 1.0001; t += 0.05) {
      var m = M.modelAt(p, t);
      worst = Math.max(worst, Math.abs(m.pi[0] + m.pi[1] - 1));
      if (m.pi[0] < 0 || m.pi[1] < 0) { worst = 99; break; }
    }
  }
  ok('pi1 + pi2 = 1 across 400 parameter sets x 21 blend steps', worst < 1e-12,
     'worst deviation ' + worst.toExponential(2));
})();

/* -------------------------------------------------------- 2. responsibilities */
section('2. Responsibilities sum to 1');
(function () {
  var worst = 0, worstFar = 0;
  for (var i = 0; i < 300; i++) {
    var p = randomParams();
    for (var t = 0; t <= 1.0001; t += 0.25) {
      var m = M.modelAt(p, t);
      for (var j = 0; j < 20; j++) {
        var r = M.responsibilities(u(-7, 7), u(-5, 5), m);
        worst = Math.max(worst, Math.abs(r[0] + r[1] - 1));
        if (r[0] < 0 || r[0] > 1) worst = 99;
      }
      // Deliberately absurd coordinates: this is where a naive
      // pi*p / (pi*p + pi*p) implementation returns 0/0 = NaN.
      var far = M.responsibilities(1e6, -1e6, m);
      worstFar = Math.max(worstFar, Math.abs(far[0] + far[1] - 1));
      if (!isFinite(far[0])) worstFar = 99;
    }
  }
  ok('r1 + r2 = 1 for 30000 random points', worst < 1e-12, 'worst ' + worst.toExponential(2));
  ok('still exactly 1 at y = (1e6, -1e6) where raw densities underflow to 0',
     worstFar < 1e-12, 'worst ' + worstFar.toExponential(2));
})();

section('2b. sigmoid(log-odds) agrees with the textbook ratio');
(function () {
  // Murphy Eq. 3.99 written out literally, for points where it is safe to do so.
  var worst = 0;
  for (var i = 0; i < 2000; i++) {
    var p = randomParams(), m = M.modelAt(p, 0);
    var x = u(-5, 5), y = u(-4, 4);
    var a = m.pi[0] * Math.exp(M.logGauss2(x, y, m.comps[0]));
    var b = m.pi[1] * Math.exp(M.logGauss2(x, y, m.comps[1]));
    if (a + b < 1e-280) continue;
    worst = Math.max(worst, Math.abs(M.responsibilities(x, y, m)[0] - a / (a + b)));
  }
  ok('matches pi_k p_k / sum_j pi_j p_j to 1e-12', worst < 1e-12, 'worst ' + worst.toExponential(2));
})();

/* ------------------------------------------------------------- 3. covariance */
section('3. Covariance matrices stay valid');
(function () {
  var symOK = true, pdOK = true, detOK = true, blendOK = true;
  var worstDet = 0;
  for (var i = 0; i < 2000; i++) {
    var sx = u(0.25, 2.6), sy = u(0.25, 2.6), th = u(-Math.PI, Math.PI);
    var S = M.covFrom(sx, sy, th);
    var e = M.eig2(S);
    if (!isFinite(S.a + S.b + S.d)) symOK = false;            // stored symmetric by construction
    if (e.l1 <= 0 || e.l2 <= 0) pdOK = false;                 // positive definite
    var err = Math.abs(M.covDet(S) - sx * sx * sy * sy);
    worstDet = Math.max(worstDet, err);
    if (err > 1e-12) detOK = false;
  }
  ok('symmetric and finite for every (sx, sy, theta)', symOK);
  ok('both eigenvalues > 0 (positive definite)', pdOK);
  ok('det(Sigma) = sx^2 * sy^2 exactly (rotation is volume-preserving)', detOK,
     'worst ' + worstDet.toExponential(2));

  // Every intermediate matrix on the K-means path must also be legal.
  for (i = 0; i < 400; i++) {
    var p = randomParams();
    for (var t = 0; t <= 1.0001; t += 0.02) {
      var m = M.modelAt(p, t);
      for (var k = 0; k < 2; k++) {
        var ee = M.eig2(m.comps[k].S);
        if (!(ee.l1 > 0 && ee.l2 > 0 && isFinite(ee.l1) && isFinite(ee.l2))) blendOK = false;
        if (M.covDet(m.comps[k].S) <= 0) blendOK = false;
      }
    }
  }
  ok('stays positive definite at every step of the GMM -> K-means blend', blendOK);
})();

/* ---------------------------------------------------------------- 4. ellipse */
section('4. Drawn ellipses really are the Mahalanobis contours');
(function () {
  // Walk the exact ellipse the renderer draws (semi-axes t*sqrt(lambda) along
  // the eigenvectors) and confirm every point on it is at Mahalanobis
  // distance t from the centre. If the drawing and the matrix ever disagreed,
  // this is where it would show.
  var worst = 0;
  for (var i = 0; i < 500; i++) {
    var sx = u(0.25, 2.6), sy = u(0.25, 2.6), th = u(-Math.PI, Math.PI);
    var S = M.covFrom(sx, sy, th);
    var e = M.eig2(S);
    var ax = Math.cos(e.angle), ay = Math.sin(e.angle);       // first eigenvector
    var bx = -Math.sin(e.angle), by = Math.cos(e.angle);      // second
    for (var t = 1; t <= 2; t++) {
      for (var s = 0; s < 64; s++) {
        var ph = s / 64 * Math.PI * 2;
        var rx = t * Math.sqrt(e.l1) * Math.cos(ph);
        var ry = t * Math.sqrt(e.l2) * Math.sin(ph);
        var dx = rx * ax + ry * bx;
        var dy = rx * ay + ry * by;
        worst = Math.max(worst, Math.abs(Math.sqrt(M.maha2(S, dx, dy)) - t));
      }
    }
  }
  ok('64000 sampled ellipse points sit at Mahalanobis distance t', worst < 1e-9,
     'worst error ' + worst.toExponential(2));
})();

/* ----------------------------------------------------------- 5. normalisation */
section('5. Densities integrate to 1');
(function () {
  var S = { mx: 0.4, my: -0.3, S: M.covFrom(1.3, 0.7, 0.5) };
  var lo = -12, hi = 12, n = 1400, h = (hi - lo) / n, sum = 0;
  for (var i = 0; i < n; i++) {
    for (var j = 0; j < n; j++) {
      sum += Math.exp(M.logGauss2(lo + (i + 0.5) * h, lo + (j + 0.5) * h, S));
    }
  }
  sum *= h * h;
  ok('2D Gaussian integrates to 1', near(sum, 1, 2e-4), 'got ' + sum.toFixed(7));

  var p = { comps: [
      { mx: -1.5, my: 0.4, sx: 1.2, sy: 0.6, rot: 0.4 },
      { mx: 1.8, my: -0.6, sx: 0.8, sy: 1.4, rot: -0.3 }], pi1: 0.35 };
  var m = M.modelAt(p, 0), s2 = 0;
  for (i = 0; i < n; i++) {
    for (j = 0; j < n; j++) {
      s2 += Math.exp(M.logMixture(lo + (i + 0.5) * h, lo + (j + 0.5) * h, m));
    }
  }
  s2 *= h * h;
  ok('the 2-component mixture also integrates to 1', near(s2, 1, 2e-4), 'got ' + s2.toFixed(7));

  var s1 = 0, n1 = 400000, h1 = 24 / n1;
  for (i = 0; i < n1; i++) s1 += M.gauss1(-12 + (i + 0.5) * h1, 0.7, 1.45) * h1;
  ok('1D Gaussian integrates to 1', near(s1, 1, 1e-6), 'got ' + s1.toFixed(9));
})();

/* --------------------------------------------------------------- 6. sampling */
section('6. Sampler reproduces the covariance it claims');
(function () {
  var c = { mx: 1.0, my: -0.5, sx: 1.6, sy: 0.55, rot: 0.7 };
  var want = M.covFrom(c.sx, c.sy, c.rot);
  var r = M.mulberry32(99), N = 400000;
  var sx = 0, sy = 0, sxx = 0, syy = 0, sxy = 0;
  for (var i = 0; i < N; i++) {
    var g = M.gaussPair(r);
    var p = M.samplePoint(c, g[0], g[1]);
    sx += p.x; sy += p.y; sxx += p.x * p.x; syy += p.y * p.y; sxy += p.x * p.y;
  }
  var mx = sx / N, my = sy / N;
  var ca = sxx / N - mx * mx, cb = sxy / N - mx * my, cd = syy / N - my * my;
  ok('empirical mean matches mu', near(mx, c.mx, 0.02) && near(my, c.my, 0.02),
     mx.toFixed(3) + ', ' + my.toFixed(3));
  ok('empirical covariance matches Sigma (400k draws, tol 0.02)',
     near(ca, want.a, 0.02) && near(cb, want.b, 0.02) && near(cd, want.d, 0.02),
     '[' + ca.toFixed(3) + ' ' + cb.toFixed(3) + ' ' + cd.toFixed(3) + '] vs [' +
     want.a.toFixed(3) + ' ' + want.b.toFixed(3) + ' ' + want.d.toFixed(3) + ']');
})();

/* ---------------------------------------------------------------- 7. K-means */
section('7. The K-means limit matches the written explanation');
(function () {
  var eqW = true, spherical = true, equalS = true, smaller = true, nearest = true;
  var maxOff = 0, maxDiff = 0, disagreements = 0, tested = 0;

  for (var i = 0; i < 300; i++) {
    var p = randomParams();
    var m0 = M.modelAt(p, 0);
    var m1 = M.modelAt(p, 1);

    if (Math.abs(m1.pi[0] - 0.5) > 1e-12) eqW = false;

    for (var k = 0; k < 2; k++) {
      maxOff = Math.max(maxOff, Math.abs(m1.comps[k].S.b));
      if (Math.abs(m1.comps[k].S.b) > 1e-12) spherical = false;
      if (Math.abs(m1.comps[k].S.a - m1.comps[k].S.d) > 1e-12) spherical = false;
      // "small": the shared variance must be well below the originals
      if (!(m1.comps[k].S.a < 0.2 * Math.max(m0.comps[k].S.a, m0.comps[k].S.d))) smaller = false;
    }
    maxDiff = Math.max(maxDiff,
      Math.abs(m1.comps[0].S.a - m1.comps[1].S.a),
      Math.abs(m1.comps[0].S.d - m1.comps[1].S.d));
    if (maxDiff > 1e-12) equalS = false;

    // The claim on the page: at the K-means end, assignment is nearest-centre
    // and nothing else. Murphy Eq. 3.101.
    for (var j = 0; j < 60; j++) {
      var x = u(-7, 7), y = u(-5, 5);
      var r = M.responsibilities(x, y, m1);
      var byResp = r[0] >= r[1] ? 0 : 1;
      var d0 = Math.hypot(x - m1.comps[0].mx, y - m1.comps[0].my);
      var d1 = Math.hypot(x - m1.comps[1].mx, y - m1.comps[1].my);
      var byDist = d0 <= d1 ? 0 : 1;
      tested++;
      if (byResp !== byDist && Math.abs(d0 - d1) > 1e-6) { nearest = false; disagreements++; }
    }
  }
  ok('t=1 forces equal mixture weights (0.50 / 0.50)', eqW);
  ok('t=1 forces spherical covariance (off-diagonal 0, a = d)', spherical,
     'max |off-diagonal| ' + maxOff.toExponential(2));
  ok('t=1 gives both components the SAME matrix', equalS, 'max diff ' + maxDiff.toExponential(2));
  ok('t=1 shrinks that matrix well below the originals', smaller);
  ok('t=1 assignment = nearest centre, over ' + tested + ' points', nearest,
     disagreements + ' disagreements');

  // And responsibilities must actually be hard, not merely tied.
  var soft = 0, total = 0;
  for (i = 0; i < 200; i++) {
    var pp = randomParams(), mm = M.modelAt(pp, 1);
    for (var q = 0; q < 40; q++) {
      var xx = u(-6, 6), yy = u(-4, 4);
      var rr = M.responsibilities(xx, yy, mm);
      var dd = Math.abs(Math.hypot(xx - mm.comps[0].mx, yy - mm.comps[0].my) -
                        Math.hypot(xx - mm.comps[1].mx, yy - mm.comps[1].my));
      if (dd > 0.3) { total++; if (Math.max(rr[0], rr[1]) < 0.99) soft++; }
    }
  }
  ok('t=1 responsibilities are hard (>0.99) away from the boundary', soft === 0,
     soft + ' of ' + total + ' still soft');
})();

section('7b. The transition is monotone, as the slider copy claims');
(function () {
  var monoPi = true, monoOff = true;
  for (var i = 0; i < 200; i++) {
    var p = randomParams();
    var prevPi = Math.abs(M.modelAt(p, 0).pi[0] - 0.5);
    var prevOff = Math.abs(M.modelAt(p, 0).comps[0].S.b) / M.modelAt(p, 0).comps[0].S.a;
    for (var t = 0.02; t <= 1.0001; t += 0.02) {
      var m = M.modelAt(p, t);
      var dPi = Math.abs(m.pi[0] - 0.5);
      var off = Math.abs(m.comps[0].S.b) / m.comps[0].S.a;
      if (dPi > prevPi + 1e-12) monoPi = false;
      if (off > prevOff + 1e-12) monoOff = false;
      prevPi = dPi; prevOff = off;
    }
  }
  ok('distance of pi1 from 0.5 decreases monotonically', monoPi);
  ok('relative off-diagonal (the correlation) decreases monotonically', monoOff);
})();

/* -------------------------------------------------------------------- report */
console.log('\n' + (fail === 0 ? 'ALL GREEN' : 'FAILURES') + ' — ' + pass + ' passed, ' + fail + ' failed\n');
process.exit(fail === 0 ? 0 : 1);
