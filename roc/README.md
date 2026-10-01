# Where Do You Draw the Line?

An interactive ROC curve. Drag the decision threshold and watch the confusion
matrix, the operating point and the precision-recall curve all move together.

**The one idea:** a classifier hands you a *score*, not an answer. Somebody still
has to decide how high is high enough, and moving that line trades one kind of
mistake for the other. **The ROC curve is the picture of every line you could
have drawn**; the dot riding on it is the one you actually did.

Covers Murphy, *Probabilistic Machine Learning: An Introduction* §5.1.2–5.1.3 —
zero-one loss and the MAP rule (Eq. 5.4–5.7), cost-sensitive classification
(Eq. 5.9–5.10), the confusion matrix (Tables 5.3–5.5), the thresholded decision
rule (Eq. 5.16–5.17), TPR/FPR (Eq. 5.18–5.19), AUC and EER, and the class
imbalance problem of §5.1.3.3.

## What's in it

| | |
|---|---|
| **Score chart** | The two class distributions with a draggable threshold through them. The four regions shade into TN / FP / FN / TP, labelled in the same 2×2 arrangement as the matrix card, with live counts. Hue is the true class; **hatching** is a mistake. Actual samples sit in two lanes under the axis, with errors drawn darker. |
| **Confusion matrix** | Updates on every pixel of drag. A dropdown switches between Murphy's three tables: raw counts (5.3), row-normalised `p(ŷ\|y)` (5.4), and column-normalised `p(y\|ŷ)` (5.5) — so you can watch the same four numbers become TPR/FPR or precision/FDR depending only on which way you divide. |
| **ROC curve** | Drag the big dot; the threshold line above follows, and vice versa. Shows the chance diagonal, shaded AUC, the EER point, corner annotations, and a gold iso-cost ruler. |
| **"The parts of it"** | Ten labelled parts — axes, both forced endpoints, the curve, the diagonal, the perfect corner, the operating point, AUC, EER, the cost line. Hover or keyboard-focus any one and the chart dims everything else. |
| **Cost panel** | A miss : false-alarm ratio slider. Shows the optimal cut-off `1/(1+c)`, where it lands on the score axis, your cost per sample versus the best available, and a button that snaps the line to the optimum. |
| **Imbalance panel** | A precision-recall curve for the same model and the same dot. Drag the base rate down and watch PR sag while the ROC sits perfectly still. |
| **Walkthrough** | Five moves: refuse everything, call everything, ruin the classifier, make misses expensive, make positives rare. |

## Run it locally

No toolchain, no dependencies, no build step. Open the file:

```sh
start index.html          # Windows
open index.html           # macOS
```

It works straight off the filesystem. If you'd rather serve it:

```sh
python -m http.server 8000     # then visit localhost:8000/roc/
```

The only network request is the Google Fonts stylesheet for IBM Plex; offline,
the page falls back to system fonts and everything still works.

## Check the math

```sh
node test-math.js
```

`script.js` exports its pure-math half under Node and skips everything
DOM-bound, so the suite exercises **the shipped file**, not a copy of its
formulas. 30 checks, currently all green:

- the four cells always sum to `n`, and the row totals are the class counts
- threshold at −∞ calls everything positive, at +∞ everything negative
- FNR = 1−TPR, FPR = 1−TNR, FDR = 1−precision, FOR = 1−NPV — and a rate is
  `NaN` *exactly* when its column is empty, never otherwise
- the curve never steps left or down, and starts and ends exactly on the corners
- **every vertex reproduces its own confusion matrix** — 10 025 of them checked,
  so the dot you drag cannot misrepresent which matrix it stands for
- AUC by the Mann–Whitney rank formula equals the trapezoidal area under the
  drawn curve to 1e−12, though the two are computed independently
- empirical AUC matches the binormal truth `Φ(sep/√2)`, and EER matches `Φ(−sep/2)`
- **tripling the negatives leaves every ROC vertex and the AUC bit-for-bit
  identical while precision collapses** — the page's central claim, checked
  exactly rather than statistically
- `posterior` and `scoreForProb` are exact inverses, and the posterior is
  *calibrated* against the sampler over 60 bins at n = 400 000
- the optimal threshold sits exactly at `p = 1/(1+c)`; `c = 2` gives 1/3 and
  `c = 1` gives 0.5, reproducing Murphy's worked examples
- a brute-force search over the population cost finds the same threshold the
  closed form does
- **the ROC curve is tangent to the iso-cost line exactly at the optimum** — the
  "slide the ruler until it touches" claim, proven rather than asserted
- the fast per-frame cost scan matches a full rescan at every vertex
- `normCdf` agrees with an independent series expansion to 2e−7

## Publish on GitHub Pages

This folder is part of the **CS 473 Artifacts** repo, whose
`.github/workflows/pages.yml` already publishes the repository root on every
push to `main`. So there is nothing to configure — commit and push:

```sh
git add roc
git commit -m "Add ROC curve module"
git push
```

The module lands at `https://<user>.github.io/<repo>/roc/`.

**One-time repo setup** (only if Pages has never been enabled): **Settings →
Pages → Build and deployment → Source → GitHub Actions**.

## Files

```
index.html     structure and copy
styles.css     design tokens, light + dark themes, responsive layout
script.js      math (exported for tests) + rendering + interaction
test-math.js   30 checks against the shipped script.js
```

## Notes on a few choices

**The score axis is primary; the probability axis is derived.** Negatives are
drawn from `N(−sep/2, 1)` and positives from `N(+sep/2, 1)`. Equal unit
variances are not laziness — they make the posterior an *exact* logistic of the
score, `p(y=1|s) = σ(sep·s + logit π)`. That one fact is what lets the page talk
about a score cut-off and a probability cut-off as the same object, and lets
Murphy's `1/(1+c)` rule land on the score axis in closed form instead of being
searched for. The suite checks both the inverse relationship and that the
posterior is genuinely calibrated against the sampler.

**The curve is empirical, not a smooth fit.** It is built by the sweep in
Eq. 5.17, counting real samples, so every vertex is a threshold you could
actually choose and the staircase you see at low `n` is honest. Ties are
consumed as a group, which is what keeps the drawn curve consistent with the
rank-based AUC when duplicate scores exist.

**AUC is computed twice, two different ways.** The headline number comes from
the Mann–Whitney rank identity; the shaded region comes from the polyline. They
are mathematically the same quantity and the suite pins them together to 1e−12,
so the number and the picture cannot drift apart.

**Imbalance is demonstrated, not just asserted.** The test suite triples the
negatives and shows the ROC curve is unchanged *to the last bit* while precision
falls — the exact-arithmetic version of the argument the panel makes in prose.

**Moving the base rate re-rolls which samples are positive.** Each sample carries
a fixed uniform draw compared against the prevalence, so changing it genuinely
migrates samples between classes rather than resampling the whole set. Changing
separation slides the existing scores. Both deform the picture continuously
instead of making it flicker.

**Right and wrong is a separate channel from which class.** Hue carries the true
class (blue negative, red positive — the same pair as module 2). Fill carries
correctness: solid when the model was right, hatched when it was wrong. Neither
reading depends on the other, so the four regions stay distinguishable without
colour discrimination. Gold is reserved for cost and appears nowhere else, so a
gold mark always means "this is what your loss function wants" rather than
"this is what the data says".

**The cost ruler really is tangent.** Its slope is `N/(c·P)`, the slope of an
iso-cost line; the ROC curve's own slope at threshold `t` is `exp(t·sep)`. The
two are equal precisely at `t*`, which the suite verifies numerically. So
"slide it up-left until it touches" is a construction, not a metaphor.
