# Build a Decision Tree with LEGO

CART and ensembles, taught with studded bricks. Each brick is one training
example: its position on the baseplate is the feature `x`, and its colour and
print are the label `y`. A movable gate sorts the bricks into two trays, and
committed gates become nodes in a tree that grows one split at a time. There are
three scenes with Back/Next, a progress bar and a Reset button.

**The one idea:** at each node CART scores every candidate gate by how tidy the
two trays directly beneath it are, keeps the lowest score, and moves on. It is
greedy. It never looks ahead.

Covers Murphy, *Probabilistic Machine Learning: An Introduction* §18.1–18.4
(classification and regression trees, ensemble learning, bagging, random forests).

## The scenes

| | Scene | Interaction |
|---|---|---|
| 1 | **Classification tree**: `G = 1 − Σ p_c²`, score `(n_L/n)G_L + (n_R/n)G_R` | thresholds t = 1…5 (chips, drag the gate, click the plate, or ← →); bricks animate into trays `x ≤ t` / `x > t` with the arithmetic beside them; **Compare splits** shows 2/5, 1/3, 4/9, 1/3, 2/5 and flags the t = 2 / t = 4 tie; commit, click the open branch, split again; undo; send a new brick x = 3.5 down to the red leaf |
| 2 | **Regression tree**: leaf mean `ȳ` and MSE in the same weighted formula | same bench with numeric targets 2, 4, 6, 10, 12, 14; click any leaf to see the x-range it covers and the single mean it predicts; a button jumps to the 2, 4, 6 → 4 leaf |
| 3 | **Ensemble**: three depth-2 trees on bootstrap samples, two features | **Resample training bricks** (duplicates shown stacked with ×n badges, omitted bricks crossed out); **Random forest** toggle (each node sees 1 of 2 features; click a node to see which); send the same test brick through each tree, then tally the vote or average; presets give votes 1, 0, 1 and predictions 4, 6, 8 → 6 |

## What is computed

Nothing is precomputed. Scores use exact fractions, so the page shows 1/3 rather
than 0.333. Scenes 1–2 rebuild the tree from the list of committed splits on
every change. Scene 3 grows each tree with greedy CART (Gini or MSE, depth ≤ 2,
ties to the first feature and smallest threshold, majority ties to class 0). Its
bootstrap draws use a seeded RNG. The two preset samples were found by searching
seeds and are checked in `test-math.js`.

## Run it locally

No toolchain. KaTeX and fonts load from CDNs. Without them, equations fall back
to their TeX source.

```sh
start index.html          # Windows
```

A scene can be linked directly, e.g. `#scene-3`.

## Check the math

```sh
node test-math.js
```

39 checks against the shipped `script.js`, all passing. They cover:

- every number quoted on the page
- the tie, plus both the t = 2 and t = 4 paths
- that x = 3.5 reaches a red leaf in **every** tree a user can build
- the regression leaf (2, 4, 6 → 4, MSE 8/3)
- both presets
- that bagging nodes see both features and random-forest nodes see exactly one
- that each grown node takes the lowest-scoring eligible split
- the 21.6% majority-error figure
