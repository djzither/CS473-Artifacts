/* Math checks for "Two Kinds of 50/50".   Run:  node test-math.js */

var B = require('./model.js');
var pass = 0, fail = 0;
function ok(name, cond, detail) {
  if (cond) { pass++; console.log('  PASS  ' + name); }
  else { fail++; console.log('  FAIL  ' + name + (detail !== undefined ? '   -> ' + detail : '')); }
}
function close(a, b, tol) { return Math.abs(a - b) <= (tol || 1e-9); }
var W = B.PRIOR;

console.log('\nEntropy');
ok('H(0) = H(1) = 0 (0 log 0 = 0)', B.H2(0) === 0 && B.H2(1) === 0);
ok('H(0.5) = 1 bit', close(B.H2(0.5), 1));
ok('H(0.99) = H(0.01) ≈ 0.0808', close(B.H2(0.99), 0.080793, 1e-6) && close(B.H2(0.01), B.H2(0.99)));
ok('equal weights over two models = 1 bit', close(B.H(W), 1));

console.log('\nPresets');
var d = B.bald(B.PRESETS.disagree, W), s = B.bald(B.PRESETS.shared, W);
ok('both presets average to 50% interested', close(d.pbar, 0.5) && close(s.pbar, 0.5));
ok('confident disagreement: BALD ≈ 0.919 bits', close(d.score, 0.919, 5e-4), d.score);
ok('shared uncertainty: BALD = 0 exactly', s.score === 0, s.score);
ok('both have H(average) = 1 bit', close(d.Hbar, 1) && close(s.Hbar, 1));

console.log('\nReveal');
var gi = B.realizedGain(B.PRESETS.disagree, W, 1), gn = B.realizedGain(B.PRESETS.disagree, W, 0);
ok('"interested": A gave 0.99, B gave 0.01', close(gi.lik[0], 0.99) && close(gi.lik[1], 0.01));
ok('posterior weights 0.99 / 0.01', close(gi.post[0], 0.99) && close(gi.post[1], 0.01));
ok('realized drop for either answer ≈ 0.919 bits', close(gi.gain, 0.919, 5e-4) && close(gn.gain, gi.gain));
var si = B.realizedGain(B.PRESETS.shared, W, 1);
ok('shared uncertainty: weights stay 50/50, drop 0', close(si.post[0], 0.5) && close(si.gain, 0));

console.log('\nBALD = expected drop in model-weight entropy');
var good = true, worst = 0;
for (var a = 0; a <= 100; a += 5) for (var b = 0; b <= 100; b += 5) {
  var ps = [a / 100, b / 100], e = B.expectedGain(ps, W), k = B.bald(ps, W).score;
  worst = Math.max(worst, Math.abs(e - k));
  if (!close(e, k, 1e-9)) good = false;
}
ok('holds on a 21 × 21 grid of slider settings', good, worst);
ok('BALD is never negative and never above 1 bit',
  [[0, 1], [1, 0], [0.3, 0.8], [0.2, 0.2]].every(function (p) { var v = B.bald(p, W).score; return v >= 0 && v <= 1 + 1e-12; }));
ok('opposite certainties give exactly 1 bit', close(B.bald([1, 0], W).score, 1));

console.log('\nZero-probability answers');
ok('both at 0%: "interested" impossible', B.posterior([0, 0], W, 1) === null && B.posterior([0, 0], W, 0) !== null);
ok('both at 100%: "not interested" impossible', B.posterior([1, 1], W, 0) === null);
ok('one at 0%, one at 100%: both answers possible', B.posterior([0, 1], W, 1) && B.posterior([0, 1], W, 0));

console.log('\n' + pass + ' passed, ' + fail + ' failed');
process.exitCode = fail ? 1 : 0;
