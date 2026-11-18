# Two Kinds of 50/50

BALD (Bayesian Active Learning by Disagreement) for an advanced ML class. At
Byld Blue, two models predict whether a customer reply means "interested." Two
replies both average to 50% interested, but only one of them is worth a human
label.

**The one idea:** total uncertainty `H(p̄)` is the same for both replies. BALD
subtracts the uncertainty each model *already expects* to keep, `E_m H(p_m)`.
What remains is disagreement between the models, and a label can resolve that.

Covers Murphy, *Probabilistic Machine Learning: An Introduction* §19.4 (active
learning, information-theoretic acquisition), and Houlsby et al. (2011).

## The page

The page is one screen with two scenario cards side by side. Both models have
equal prior weights of ½.

| | Reply 1 · Confident disagreement | Reply 2 · Shared uncertainty |
|---|---|---|
| Preset | A = 99%, B = 1% | A = B = 50% |
| Averaged | 50% | 50% |
| BALD | ≈ 0.919 bits | exactly 0 bits |

Each card has the following parts:

- **Sliders and bars** for each model's P(interested), plus the averaged prediction.
- **A binary entropy curve.** It shows A, B and the average on the curve, the
  chord between A and B, and the vertical gap between `H(average)` and the
  average of the `H` values. That gap is BALD, which is Jensen's inequality in
  picture form.
- **The live calculation in bits**, with log base 2 and `0 log 0 = 0`.
- **Reveal interested / not interested buttons.**
  - Each reveal shows how much probability each model gave the answer, then
    animates the model weights from the prior to the posterior.
  - It shows the realized drop in the entropy of the weights.
  - It then averages that drop over both labels, weighted by `P(y)`, and the
    result reproduces BALD.
  - The original BALD score stays visible.
  - A button is disabled when the models give that answer zero total probability.

The mutual-information equations sit behind **Show the mutual-information
equations**. A one-question check at the end lets you answer before you see the
explanation.

## Files

`model.js` holds the math (no DOM). `script.js` does the drawing. Like the other
modules, it uses no framework and no build step.

## Check the math

```sh
node test-math.js
```

The 18 checks all pass. They cover:

- the entropy edge cases
- both presets (0.919 bits and exactly 0)
- the reveal likelihoods and posteriors
- BALD = expected realized drop, on a 21 × 21 grid of slider settings
- the bounds 0 ≤ BALD ≤ 1 bit
- disabling an answer that has zero probability
