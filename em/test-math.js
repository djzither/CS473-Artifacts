/* ============================================================================
   Math checks for the EM module.   Run:  node test-math.js

   This exercises the SHIPPED script.js, not a copy of its formulas — the file
   exports its pure-math half under Node and skips everything DOM-bound.

   The headline identity, ℓ(θ) − 𝓛(θ, q) = Σ_n KL(q_n ‖ p(z_n | y_n, θ)), is
   checked by computing both sides independently: the left from the log
   likelihood and the ELBO, the right from log posteriors.
   ========================================================================= */

var M = require('./script.js');

var pass = 0, fail = 0;

function ok(name, cond, detail) {
  if (cond) { pass++; console.log('  PASS  ' + name); }
  else { fail++; console.log('  FAIL  ' + name + (detail ? '   -> ' + detail : '')); }
}

function near(a, b, tol) { return Math.abs(a - b) <= (tol || 1e-9); }

function section(t) { console.log('\n' + t); }

var rnd = M.mulberry32(473006);
function u(lo, hi) { return lo + (hi - lo) * rnd(); }

function randomQ() {
  return M.DATA.map(function () { var a = rnd(); return [a, 1 - a]; });
}

function sum(v) { return v.reduce(function (s, x) { return s + x; }, 0); }


/* ---------------------------------------------- 1. stable building blocks -- */
section('1. Stable building blocks');
(function () {
  ok('logNormal(0 | 0, 1) = −½ log 2π', near(M.logNormal(0, 0, 1), -0.5 * Math.log(2 * Math.PI)));
  ok('logSumExp matches naive on moderate values', near(M.logSumExp([1, 2, 3]), Math.log(Math.exp(1) + Math.exp(2) + Math.exp(3))));
  ok('logSumExp([−1000, −1000]) = −1000 + log 2 (naive gives −∞)', near(M.logSumExp([-1000, -1000]), -1000 + Math.log(2)));
  ok('logSumExp([800, 0]) = 800 (naive gives ∞)', near(M.logSumExp([800, 0]), 800));

  var worst = 0, finite = true;
  for (var i = 0; i < 2000; i++) {
    var post = M.posterior(u(-60, 60), u(-60, 60));
    post.forEach(function (r) {
      worst = Math.max(worst, Math.abs(r.p[0] + r.p[1] - 1));
      if (!isFinite(r.logp[0]) || !isFinite(r.logp[1])) finite = false;
    });
  }
  ok('responsibilities sum to 1 for 2000 random centre pairs in ±60', worst < 1e-12, 'worst ' + worst);
  ok('log responsibilities stay finite even 60σ from the data', finite);

  var naiveAgree = true;
  for (i = 0; i < 200; i++) {
    var m1 = u(-5, 6), m2 = u(-5, 6);
    var post2 = M.posterior(m1, m2);
    M.DATA.forEach(function (y, n) {
      var a = 0.5 * Math.exp(M.logNormal(y, m1, 1)), b = 0.5 * Math.exp(M.logNormal(y, m2, 1));
      if (!near(post2[n].p[0], a / (a + b), 1e-12)) naiveAgree = false;
    });
  }
  ok('posterior matches the naive ratio where the naive ratio is safe', naiveAgree);
})();


/* ------------------------------------------------- 2. the KL identity ----- */
section('2. ℓ − 𝓛 = Σ KL, for any q');
(function () {
  var worst = 0, minKL = Infinity;
  for (var i = 0; i < 3000; i++) {
    var m1 = u(-6, 7), m2 = u(-6, 7), q = randomQ();
    var lhs = M.logLik(m1, m2) - M.elbo(m1, m2, q);
    var kl = M.klTerms(m1, m2, q);
    worst = Math.max(worst, Math.abs(lhs - sum(kl)));
    minKL = Math.min(minKL, Math.min.apply(null, kl));
  }
  ok('identity holds for 3000 random (μ₁, μ₂, q) to 1e-9', worst < 1e-9, 'worst ' + worst);
  ok('every KL term is ≥ 0', minKL >= 0);

  var q0 = M.uniformQ();
  ok('identity holds for the 50/50 starting q', near(M.logLik(1, -2) - M.elbo(1, -2, q0), sum(M.klTerms(1, -2, q0)), 1e-9));

  var below = true;
  for (i = 0; i < 500; i++) {
    var qq = randomQ(), b2 = u(-6, 7);
    for (var x = -6; x <= 7; x += 0.25) if (M.elbo(x, b2, qq) > M.logLik(x, b2) + 1e-9) below = false;
  }
  ok('the bound never rises above the true score, at any μ₁, for any q', below);
})();


/* ---------------------------------------------------- 3. the E step ------- */
section('3. E step');
(function () {
  var gapWorst = 0, touchWorst = 0;
  for (var i = 0; i < 1000; i++) {
    var m1 = u(-5.5, 6.5), m2 = u(-5.5, 6.5);
    var q = M.qFromPosterior(M.posterior(m1, m2));
    var m = M.measure(m1, m2, q);
    gapWorst = Math.max(gapWorst, Math.abs(m.gap), m.klSum);
    touchWorst = Math.max(touchWorst, Math.abs(M.elbo(m1, m2, q) - M.logLik(m1, m2)));
  }
  ok('after an E step the KL gap is 0 (1000 random states)', gapWorst < 1e-9, 'worst ' + gapWorst);
  ok('after an E step the bound touches ℓ at the current μ₁', touchWorst < 1e-9);

  // the E step only touches q: μ₁ and ℓ are untouched by construction
  var run = M.runEM(-0.5, -2, 9);
  var same = true;
  for (var k = 1; k < run.length; k++) if (run[k].kind === 'E') {
    if (run[k].mu1 !== run[k - 1].mu1 || run[k].m.ll !== run[k - 1].m.ll) same = false;
  }
  ok('E steps leave μ₁ and the true score exactly unchanged', same);

  // tangency: the bound and ℓ have the same slope where they touch
  var h = 1e-5, m1t = 0.7, m2t = -2, qt = M.qFromPosterior(M.posterior(m1t, m2t));
  var dl = (M.logLik(m1t + h, m2t) - M.logLik(m1t - h, m2t)) / (2 * h);
  var db = (M.elbo(m1t + h, m2t, qt) - M.elbo(m1t - h, m2t, qt)) / (2 * h);
  ok('at the touch point the bound is tangent to ℓ (equal slopes)', near(dl, db, 1e-6), dl + ' vs ' + db);
})();


/* ---------------------------------------------------- 4. the M step ------- */
section('4. M step');
(function () {
  var isMax = true, llUp = true, lbUp = true;
  for (var i = 0; i < 1000; i++) {
    var m1 = u(-5.5, 6.5), m2 = u(-5.5, 6.5);
    var q = M.qFromPosterior(M.posterior(m1, m2));
    var mNew = M.mStep(q, m1);
    var peak = M.elbo(mNew, m2, q);
    for (var x = -6; x <= 7; x += 0.05) if (M.elbo(x, m2, q) > peak + 1e-9) isMax = false;
    if (peak < M.elbo(m1, m2, q) - 1e-9) lbUp = false;
    if (M.logLik(mNew, m2) < M.logLik(m1, m2) - 1e-9) llUp = false;
  }
  ok('the M step lands on the bound\'s maximum (checked on a grid)', isMax);
  ok('the M step never lowers the bound', lbUp);
  ok('the M step never lowers the true score', llUp);

  ok('the M step is the q-weighted mean', near(M.mStep([[1, 0], [1, 0], [0, 1]], 0, [1, 3, 100]), 2));
  ok('with zero weight on cluster 1 the M step stays put', M.mStep([[0, 1], [0, 1]], 1.5, [1, 3]) === 1.5);
})();


/* ------------------------------------------- 5. what EM does NOT promise -- */
section('5. Runs and the presets');
(function () {
  function settle(m1, m2) {
    var run = M.runEM(m1, m2, 80), last = run[run.length - 1];
    return { run: run, mu: last.mu1, ll: last.m.ll };
  }
  function monotone(run) {
    for (var k = 1; k < run.length; k++) if (run[k].m.ll < run[k - 1].m.ll - 1e-9) return false;
    return true;
  }

  var a = settle(-0.5, -2);
  ok('smooth climb settles near μ₁ = 3.30', near(a.mu, 3.299, 2e-3), a.mu);
  ok('smooth climb: ℓ never decreases', monotone(a.run));

  var b = settle(5.5, 3.3);
  var mGaps = b.run.filter(function (e) { return e.kind === 'M'; }).map(function (e) { return e.m.gap; });
  ok('crossing run: the second M-step gap is larger than the first', mGaps[1] > 3 * mGaps[0], mGaps.slice(0, 3).join(', '));
  ok('crossing run: ℓ still never decreases', monotone(b.run));
  ok('crossing run: every E step closes the gap', b.run.every(function (e) { return e.kind !== 'E' || e.m.gap < 1e-9; }));

  var c = settle(4, 0.6), d = settle(-4, 0.6);
  ok('stuck run settles on the right group (μ₁ ≈ 3.35)', near(c.mu, 3.35, 5e-3), c.mu);
  ok('the other start settles on the left group (μ₁ ≈ −2.05)', near(d.mu, -2.05, 5e-3), d.mu);
  ok('the left peak scores higher, so EM\'s answer depends on the start', d.ll > c.ll + 5, d.ll + ' vs ' + c.ll);

  var all = true;
  for (var i = 0; i < 300; i++) if (!monotone(M.runEM(u(-5.5, 6.5), u(-5.5, 6.5), 40))) all = false;
  ok('ℓ never decreases over 300 random runs of 40 half-steps', all);
})();


console.log('\n' + pass + ' passed, ' + fail + ' failed');
process.exit(fail ? 1 : 0);
