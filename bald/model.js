/* ============================================================================
   Two Kinds of 50/50 — BALD math.
   No DOM. Loaded by index.html and by test-math.js under Node.
   All entropies in bits (log base 2), with 0 log 0 = 0.
   ========================================================================= */

(function (global) {
  'use strict';

  function xlog2x(p) { return p <= 0 ? 0 : p * Math.log2(p); }

  /* Binary entropy H(p) = -p log2 p - (1-p) log2(1-p). */
  function H2(p) { return -(xlog2x(p) + xlog2x(1 - p)); }

  /* Entropy of a discrete distribution (e.g. model weights). */
  function H(ws) { return -ws.reduce(function (s, w) { return s + xlog2x(w); }, 0); }

  /* Everything BALD needs for one reply.
     ps = each model's P(interested), ws = model weights (sum to 1). */
  function bald(ps, ws) {
    var pbar = 0, avgH = 0;
    ps.forEach(function (p, m) { pbar += ws[m] * p; avgH += ws[m] * H2(p); });
    var Hbar = H2(pbar);
    return { pbar: pbar, Hbar: Hbar, avgH: avgH, ents: ps.map(H2), score: Math.max(0, Hbar - avgH) };
  }

  /* Probability each model gave the observed label y (1 = interested). */
  function likelihoods(ps, y) { return ps.map(function (p) { return y === 1 ? p : 1 - p; }); }

  /* Bayes: posterior weight of model m ∝ prior weight × P(y | model m).
     Returns null when the models give y zero total probability. */
  function posterior(ps, ws, y) {
    var lik = likelihoods(ps, y), z = 0;
    lik.forEach(function (l, m) { z += ws[m] * l; });
    if (z <= 0) return null;
    return { lik: lik, evidence: z, post: lik.map(function (l, m) { return ws[m] * l / z; }) };
  }

  /* Realized drop in the entropy of the model weights after seeing y. */
  function realizedGain(ps, ws, y) {
    var r = posterior(ps, ws, y);
    if (!r) return null;
    return { before: H(ws), after: H(r.post), gain: H(ws) - H(r.post), post: r.post, lik: r.lik, evidence: r.evidence };
  }

  /* Expected drop over both labels, weighted by P(y). Equals BALD. */
  function expectedGain(ps, ws) {
    var s = 0;
    [1, 0].forEach(function (y) { var g = realizedGain(ps, ws, y); if (g) s += g.evidence * g.gain; });
    return s;
  }

  var PRIOR = [0.5, 0.5];
  var PRESETS = {
    disagree: [0.99, 0.01],
    shared: [0.5, 0.5]
  };

  var API = { H2: H2, H: H, bald: bald, likelihoods: likelihoods, posterior: posterior,
    realizedGain: realizedGain, expectedGain: expectedGain, PRIOR: PRIOR, PRESETS: PRESETS };

  if (typeof module !== 'undefined' && module.exports) module.exports = API;
  else global.BALD = API;
})(typeof window !== 'undefined' ? window : globalThis);
