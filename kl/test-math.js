/* ============================================================================
   Math checks for the KL module.   Run:  node test-math.js

   This exercises the SHIPPED script.js, not a copy of its formulas — the file
   exports its pure-math half under Node and skips everything DOM-bound.
   ========================================================================= */

var M = require('./script.js');

var pass = 0, fail = 0;

function ok(name, cond, detail) {
  if (cond) { pass++; console.log('  PASS  ' + name); }
  else { fail++; console.log('  FAIL  ' + name + (detail ? '   -> ' + detail : '')); }
}

function near(a, b, tol) { return Math.abs(a - b) <= (tol || 1e-9); }

function section(t) { console.log('\n' + t); }

var rnd = M.mulberry32(473004);

function randomDist(n, floor) {
  var a = [], s = 0, i;
  for (i = 0; i < n; i++) { a.push(Math.pow(rnd(), 2) + 1e-3); s += a[i]; }
  a = a.map(function (x) { return x / s; });
  return floor ? M.withFloor(a, floor) : a;
}

function sum(a) { return a.reduce(function (s, x) { return s + x; }, 0); }


/* ------------------------------------------------------ 1. known values ---- */
section('1. Known values');
(function () {
  var p = [0.5, 0.25, 0.125, 0.125];
  ok('H(1/2, 1/4, 1/8, 1/8) = 1.75 bits', near(M.entropy(p), 1.75));
  ok('Uniform over 4 has H = 2 bits', near(M.entropy([0.25, 0.25, 0.25, 0.25]), 2));
  ok('A sure thing has H = 0', near(M.entropy([1, 0, 0, 0]), 0));
  ok('Uniform code on p costs 2.00 bits', near(M.crossEntropy(p, [0.25, 0.25, 0.25, 0.25]), 2));
  ok('D(p || uniform) = log K - H(p) = 0.25', near(M.kl(p, [0.25, 0.25, 0.25, 0.25]), 0.25));
  ok('Reversed code costs 2.625 bits', near(M.crossEntropy(p, [0.125, 0.125, 0.25, 0.5]), 2.625));
  ok('Reversed code wastes 0.875 bits', near(M.kl(p, [0.125, 0.125, 0.25, 0.5]), 0.875));
  ok('Fair coin vs 99% heads: D(p||q) = 2.33', near(M.kl([0.5, 0.5], [0.99, 0.01]), 2.329, 1e-3));
  ok('99% heads vs fair coin: D(q||p) = 0.92', near(M.kl([0.99, 0.01], [0.5, 0.5]), 0.919, 1e-3));
  ok('q = 0 where p > 0 gives infinite KL', M.kl([0.5, 0.5], [1, 0]) === Infinity);
  ok('p = 0 terms contribute nothing', near(M.kl([1, 0], [0.5, 0.5]), 1));
})();


/* --------------------------------------------------- 2. identities ---- */
section('2. Identities over 500 random pairs');
(function () {
  var badSplit = 0, badPos = 0, badArea = 0, badMin = 0;
  for (var t = 0; t < 500; t++) {
    var n = 2 + Math.floor(rnd() * 5);
    var p = randomDist(n), q = randomDist(n, 0.01);
    var H = M.entropy(p), Hce = M.crossEntropy(p, q), D = M.kl(p, q);
    if (!near(Hce, H + D, 1e-9)) badSplit++;
    if (D < -1e-12) badPos++;
    if (Hce < H - 1e-12) badMin++;
    var cols = M.columns(p, q), w = 0, s = 0;
    cols.forEach(function (c) { w += c.wasted; s += c.saved; });
    if (!near(w - s, D, 1e-9)) badArea++;
  }
  ok('H(p,q) = H(p) + D(p||q)', badSplit === 0, badSplit + ' bad');
  ok('D(p||q) >= 0 (Gibbs)', badPos === 0, badPos + ' bad');
  ok('H(p,q) >= H(p) (source coding)', badMin === 0, badMin + ' bad');
  ok('wasted area - saved area = D(p||q)', badArea === 0, badArea + ' bad');
  ok('D(p||p) = 0', near(M.kl([0.2, 0.3, 0.5], [0.2, 0.3, 0.5]), 0));
})();


/* ------------------------------------------------------ 3. columns ---- */
section('3. Area chart geometry');
(function () {
  var p = [0.5, 0.25, 0.125, 0.125], q = [0.25, 0.25, 0.25, 0.25];
  var cols = M.columns(p, q);
  ok('columns tile [0, 1]', near(cols[3].x0 + cols[3].w, 1));
  ok('Sun: spent 2, ideal 1', near(cols[0].spent, 2) && near(cols[0].ideal, 1));
  ok('Sun wastes 0.5 bit-area', near(cols[0].wasted, 0.5));
  ok('Rain saves 0.125 bit-area', near(cols[2].saved, 0.125));
  var area = cols.reduce(function (s, c) { return s + c.w * c.spent; }, 0);
  ok('total area = cross entropy', near(area, M.crossEntropy(p, q)));
  var zero = M.columns([0.5, 0.5, 0], [0.4, 0.4, 0.2]);
  ok('a p = 0 column has no width and no waste', zero[2].w === 0 && zero[2].wasted === 0);
})();


/* ---------------------------------------------- 4. editing the bars ---- */
section('4. setShare / withFloor');
(function () {
  var badSum = 0, badFloor = 0, badSet = 0, badRatio = 0;
  for (var t = 0; t < 500; t++) {
    var floor = rnd() < 0.5 ? 0 : 0.01;
    var a = randomDist(4, floor);
    var i = Math.floor(rnd() * 4), v = rnd() * 1.2 - 0.1;
    var b = M.setShare(a, i, v, floor);
    if (!near(sum(b), 1, 1e-9)) badSum++;
    if (b.some(function (x) { return x < floor - 1e-12; })) badFloor++;
    var want = Math.min(1 - floor * 3, Math.max(floor, v));
    if (!near(b[i], want, 1e-12)) badSet++;
    if (floor === 0) {
      var j = (i + 1) % 4, k = (i + 2) % 4;
      if (a[k] > 1e-6 && b[k] > 1e-6 && !near(a[j] / a[k], b[j] / b[k], 1e-6)) badRatio++;
    }
  }
  ok('always sums to 1', badSum === 0, badSum + ' bad');
  ok('never drops below the floor', badFloor === 0, badFloor + ' bad');
  ok('the dragged bar lands where asked (clamped)', badSet === 0, badSet + ' bad');
  ok('the other bars keep their ratios', badRatio === 0, badRatio + ' bad');

  var f = M.withFloor([0.6, 0.4, 0, 0], 0.01);
  ok('withFloor sums to 1', near(sum(f), 1));
  ok('withFloor lifts zeros to the floor', near(f[2], 0.01) && near(f[3], 0.01));
  ok('withFloor leaves a valid dist alone', near(M.withFloor([0.25, 0.25, 0.25, 0.25], 0.01)[0], 0.25));
  var all = M.setShare([0, 1, 0, 0], 1, 0.4, 0);
  ok('dragging the only nonzero bar down shares the rest evenly', near(all[0], 0.2) && near(all[3], 0.2));
})();


/* ------------------------------------------------------ 5. huffman ---- */
section('5. Huffman codebook');
(function () {
  var c = M.huffman([0.5, 0.25, 0.125, 0.125]);
  ok('dyadic case gives 0, 10, 110, 111', c.join(',') === '0,10,110,111', c.join(','));
  var badPrefix = 0, badKraft = 0, badBound = 0;
  for (var t = 0; t < 300; t++) {
    var n = 2 + Math.floor(rnd() * 6), q = randomDist(n, 0.01);
    var code = M.huffman(q);
    for (var a = 0; a < n; a++) for (var b = 0; b < n; b++) {
      if (a !== b && code[b].indexOf(code[a]) === 0) badPrefix++;
    }
    var kraft = code.reduce(function (s, w) { return s + Math.pow(2, -w.length); }, 0);
    if (!near(kraft, 1, 1e-12)) badKraft++;
    var L = q.reduce(function (s, x, i) { return s + x * code[i].length; }, 0);
    var H = M.entropy(q);
    if (L < H - 1e-9 || L >= H + 1) badBound++;
  }
  ok('prefix-free', badPrefix === 0, badPrefix + ' bad');
  ok('Kraft sum = 1 (a full tree)', badKraft === 0, badKraft + ' bad');
  ok('H(q) <= average length < H(q) + 1', badBound === 0, badBound + ' bad');
})();


/* ------------------------------------------------------ 6. sampling ---- */
section('6. Sampling the message');
(function () {
  var p = [0.4, 0.3, 0.1, 0.2], r = M.mulberry32(9), counts = [0, 0, 0, 0], N = 200000;
  for (var k = 0; k < N; k++) counts[M.sampleIndex(r(), p)]++;
  var maxErr = Math.max.apply(null, counts.map(function (c, i) { return Math.abs(c / N - p[i]); }));
  ok('empirical frequencies match p', maxErr < 0.005, 'max error ' + maxErr.toFixed(4));
  var z = [0.5, 0, 0.5, 0], hitZero = false;
  for (k = 0; k < 10000; k++) { var i = M.sampleIndex(r(), z); if (z[i] === 0) hitZero = true; }
  ok('never emits a symbol with p = 0', !hitZero);
  ok('u just below 1 still lands on a real symbol', M.sampleIndex(0.9999999999, [0.3, 0.7, 0]) === 1);

  var q = [0.25, 0.25, 0.25, 0.25], cols = M.columns(p, q), extra = 0;
  r = M.mulberry32(11);
  for (k = 0; k < N; k++) extra += cols[M.sampleIndex(r(), p)].extra;
  ok('extra bits per symbol converge to D(p||q)', near(extra / N, M.kl(p, q), 0.01),
     (extra / N).toFixed(4) + ' vs ' + M.kl(p, q).toFixed(4));
})();


console.log('\n' + pass + ' passed, ' + fail + ' failed');
process.exit(fail ? 1 : 0);
