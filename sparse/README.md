# Which Ships Count?

SVMs in practice, as a Rebel sensor mission. It follows module 11
([The Widest Corridor](../svm/)), which built the SVM. This module uses it: read
its scores, see which training ships it relies on, tune it, turn scores into
probabilities, and predict fuel use. Labels are Rebel `y = −1` and Imperial
`y = +1`. There are six short scenes with Back/Next, a progress bar and a
Reset button.

**The one idea:** a flat region in the loss means many training examples end up
with weight exactly zero, so the prediction depends on only a few of them.

Covers Murphy, *Probabilistic Machine Learning: An Introduction* §17.3.5–17.4
(SVM probabilities, the hinge loss, choosing C and the kernel, SVM regression,
sparse vector machines).

## The scenes

| | Scene | Interaction |
|---|---|---|
| 1 | **Classify a ship**: `f(x) = wᵀx + b`, a score is not a probability | drag an unknown ship; live score, prediction and margin zone |
| 2 | **Support vectors**: `max(0, 1 − y f(x))` | hide non-support ships (the line doesn't move); click a ship for its hinge loss step by step; optional logistic-loss curve |
| 3 | **C and γ**: RBF kernel `exp(−γ‖x − x′‖²)` | C and γ sliders and Underfit / Balanced / Overfit presets; training vs. held-out accuracy and support-vector count |
| 4 | **Probabilities**: `p(Imperial | x) = σ(a f(x) + b)` | a and b sliders, "plain sigmoid" vs. "fit on validation" (Platt); calibration plot on test ships |
| 5 | **SVR**: `max(0, |y − f(x)| − ε)` | ε slider; click a reading for its error and loss step by step and its signed coefficient `θ = α − α*` |
| 6 | **Comparison**: KRR, SVM/SVR, L1VM, RVM, GP | summary table, plus two common misstatements flagged |

Each scene puts the equation beside the plot it explains, defines every symbol,
and keeps the derivations behind a **Show the math** button.

## What is computed

The ships and fuel readings are synthetic, generated from a fixed seed. Every
model on the page is fitted in the browser by one SMO solver (LIBSVM's
maximal-violating-pair rule) on the dual

```
min_β  ½ βᵀQβ + pᵀβ   s.t.  zᵀβ = 0,  0 ≤ β ≤ C,   Q_st = z_s z_t K_st
```

which covers both the classifier (`z = y`, `p = −1`) and ε-SVR
(`β = (α, α*)`, `z = (+1, −1)`, `p = (ε − y, ε + y)`). Scene 3 refits for each
C/γ setting and scene 5 for each ε, so nothing is precomputed or illustrative.
Platt's `a, b` are fitted by Newton's method on 300 validation ships with
Platt's smoothed targets; calibration is checked on 800 separate test ships.
Only the scene 6 table is a summary rather than a computation.

## Accuracy notes

- A correctly classified point **inside the margin** (`0 < yf < 1`) has positive
  hinge loss and `α = C` (Y-wing Gray in scene 2). Points exactly on a margin
  edge have zero loss but can still be support vectors.
- SVR coefficients `θₙ = αₙ − αₙ*` take **either sign**: positive on or above
  the tube, negative on or below. When ε > 0, `αₙ` and `αₙ*` are never both positive.
- A sigmoid of the score is always in [0, 1] but is not automatically
  calibrated. With C = 0.03 here, plain `σ(f)` is underconfident: test ships
  given about 78% are Imperial 95% of the time. Platt scaling fixes it.

## Run it locally

No toolchain. KaTeX and fonts load from CDNs; without them equations fall back
to their TeX source.

```sh
start index.html          # Windows
```

A scene can be linked directly, e.g. `#scene-4`.

## Check the math

```sh
node test-math.js
```

32 checks against the shipped `script.js`, all passing: KKT conditions and
strong duality, the inside-margin and on-edge claims, SV-only predictions and
retraining, convergence and KKT across all 42 C × γ settings, that the presets
under/overfit as labelled, that the Platt fit is a minimum and fixes the
70–90% bin, and the SVR tube/sign conditions and sparsity as ε grows.
