/* ============================================================================
   Math checks for the naive Bayes / logistic regression module.
   Run:  node test-math.js

   This exercises the SHIPPED script.js, not a copy of its formulas — the file
   exports its pure-math half under Node and skips everything DOM-bound.

   The headline claim, that multiplying odds (step 1) and adding log votes
   (step 2) give the same P(spam), is checked against a third route: Bayes'
   rule applied directly, with no odds and no logs.
   ========================================================================= */

var M = require('./script.js');

var pass = 0, fail = 0;

function ok(name, cond, detail) {
  if (cond) { pass++; console.log('  PASS  ' + name); }
  else { fail++; console.log('  FAIL  ' + name + (detail ? '   -> ' + detail : '')); }
}

function near(a, b, tol) { return Math.abs(a - b) <= (tol || 1e-12); }

function section(t) { console.log('\n' + t); }

var ALL = [[true, true], [true, false], [false, true], [false, false]];
function label(x) { return '(free ' + (x[0] ? 'yes' : 'no') + ', winner ' + (x[1] ? 'yes' : 'no') + ')'; }


/* ---------------------------------------------------------- 1. the numbers -- */
section('1. The example numbers');
(function () {
  ok('prior odds = 0.2 / 0.8 = 0.25', near(M.priorOdds(), 0.25));
  ok('"free" present: ×6', near(M.likelihoodRatio(M.WORDS[0], true), 6));
  ok('"free" absent: ×4/9 (uses 1 − p, not omitted)', near(M.likelihoodRatio(M.WORDS[0], false), 4 / 9));
  ok('"winner" present: ×10', near(M.likelihoodRatio(M.WORDS[1], true), 10));
  ok('"winner" absent: ×10/19', near(M.likelihoodRatio(M.WORDS[1], false), 10 / 19));
  ok('both words: final odds 15, P = 15/16', near(M.multiplyRoute([true, true]).odds, 15) && near(M.multiplyRoute([true, true]).p, 0.9375));
  ok('absent words still move the answer (neither ≠ prior)', Math.abs(M.multiplyRoute([false, false]).p - 0.2) > 0.1);
})();


/* ------------------------------------------- 2. three routes, one answer -- */
section('2. Multiply, add, and Bayes\' rule agree');
(function () {
  ALL.forEach(function (x) {
    var a = M.multiplyRoute(x).p, b = M.logRoute(x).p, c = M.bayesDirect(x);
    ok('P(spam) ' + label(x) + ': odds route = log route = Bayes\' rule', near(a, b) && near(b, c), a + ' / ' + b + ' / ' + c);
  });
  ALL.forEach(function (x) {
    var m = M.multiplyRoute(x), l = M.logRoute(x);
    var each = m.chain.every(function (o, k) { return near(Math.log(o), l.running[k]); });
    ok('every running total is the log of the odds so far ' + label(x), each);
  });
  ok('the page\'s "differ by less than 0.000001%" holds for every email',
     ALL.every(function (x) { return Math.abs(M.multiplyRoute(x).p - M.logRoute(x).p) < 1e-8; }));
  ok('votes are positive exactly when the multiplier is above 1',
     ALL.every(function (x) {
       var m = M.multiplyRoute(x), l = M.logRoute(x);
       return m.ratios.every(function (r, j) { return (r > 1) === (l.votes[j + 1] > 0); });
     }));
})();


/* ---------------------------------------------------------- 3. sigmoid -- */
section('3. The squashing curve');
(function () {
  ok('σ(0) = ½', near(M.sigmoid(0), 0.5));
  ok('σ(z) + σ(−z) = 1', [0.3, 1.7, 4, 9].every(function (z) { return near(M.sigmoid(z) + M.sigmoid(-z), 1); }));
  ok('σ stays finite and in [0, 1] at ±800', M.sigmoid(800) === 1 && M.sigmoid(-800) >= 0 && isFinite(M.sigmoid(-800)));
  ok('σ(log o) = o / (1 + o)', [0.01, 0.25, 1, 15, 400].every(function (o) { return near(M.sigmoid(Math.log(o)), o / (1 + o)); }));
})();


/* --------------------------------------- 4. naive Bayes as a linear rule -- */
section('4. Naive Bayes in logistic regression\'s form');
(function () {
  var w = M.nbWeights();
  ALL.forEach(function (x) {
    ok('z = w0 + w·x reproduces naive Bayes ' + label(x), near(M.predictLinear(w, x), M.bayesDirect(x)));
  });
  ok('w0 = log 0.25 + log(4/9) + log(10/19)', near(w.w0, Math.log(0.25 * 4 / 9 * 10 / 19)));
  ok('w_free = log(6 ÷ 4/9) = log 13.5', near(w.w[0], Math.log(13.5)));
  ok('w_winner = log(10 ÷ 10/19) = log 19', near(w.w[1], Math.log(19)));
})();


/* ------------------------------------- 5. same form, different weights -- */
section('5. The correlated mailbox');
(function () {
  var est = M.nbFromCounts(M.MAILBOX);
  ok('mailbox has 1,000 emails', M.MAILBOX.reduce(function (s, c) { return s + c.spam + c.ham; }, 0) === 1000);
  ok('mailbox prior is exactly 20%', near(est.prior, 0.2));
  ok('mailbox word rates match the page exactly',
     M.WORDS.every(function (w, j) { return near(est.pSpam[j], w.pSpam) && near(est.pHam[j], w.pHam); }));
  ok('"winner" never appears without "free"', M.MAILBOX.every(function (c) { return !(c.x[1] && !c.x[0]) || c.spam + c.ham === 0; }));

  var lr = M.fitLogistic(M.MAILBOX), nb = M.nbWeights();
  /* With three non-empty cells and three weights, the best fit reproduces each
     cell's spam share exactly, so the weights have closed forms to check. */
  ok('logistic regression w0 = log(80/720)', near(lr.w0, Math.log(80 / 720), 1e-9), lr.w0);
  ok('logistic regression w_free = log(20/40) − w0', near(lr.w[0], Math.log(20 / 40) - Math.log(80 / 720), 1e-9), lr.w[0]);
  ok('logistic regression w_winner = log(100/40) − log(20/40)', near(lr.w[1], Math.log(100 / 40) - Math.log(20 / 40), 1e-9), lr.w[1]);
  ok('logistic regression matches the data: 100 / 140 with both words', near(M.predictLinear(lr, [true, true]), 100 / 140, 1e-9));
  ok('naive Bayes is overconfident there (93.75% vs 71.4%)', near(M.predictLinear(nb, [true, true]), 0.9375) && M.predictLinear(nb, [true, true]) - 100 / 140 > 0.2);
  ok('same form, different weights: both word weights smaller under logistic regression', lr.w[0] < nb.w[0] - 0.5 && lr.w[1] < nb.w[1] - 0.5);

  /* Control: if the words ARE independent within each class, logistic
     regression's best fit lands on naive Bayes's weights. */
  var indep = [[1, 1], [1, 0], [0, 1], [0, 0]].map(function (x) {
    function share(cls) {
      return M.WORDS.reduce(function (p, w, j) {
        var q = cls ? w.pSpam : w.pHam;
        return p * (x[j] ? q : 1 - q);
      }, 1);
    }
    return { x: x, spam: 1e6 * 0.2 * share(1), ham: 1e6 * 0.8 * share(0) };
  });
  var lr2 = M.fitLogistic(indep);
  ok('control: with truly independent words, the fits agree',
     near(lr2.w0, nb.w0, 1e-8) && near(lr2.w[0], nb.w[0], 1e-8) && near(lr2.w[1], nb.w[1], 1e-8),
     [lr2.w0, lr2.w[0], lr2.w[1]].join(', '));
})();


console.log('\n' + pass + ' passed, ' + fail + ' failed');
process.exit(fail ? 1 : 0);
