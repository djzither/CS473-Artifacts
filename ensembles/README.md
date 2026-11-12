# How Tree Ensembles Learn

Random forests and boosting on the real Titanic passenger list. Five tabs share one
training/test split, one followed passenger and one round counter. Step through 50
rounds with Back, Next, Play/Pause, Reset or the round slider. Each round shows the
new depth-2 tree with its split rules and leaf outputs, the followed passenger's path
through it, their ensemble prediction before and after, a one-line explanation of
what changed, accuracy (or loss) by round, all 500 training passengers as a grid,
and a table that makes the method's learning signal visible.

Covers Murphy, *Probabilistic Machine Learning: An Introduction* §18.4 (random
forests) and §18.5–18.5.4 (boosting, forward stagewise additive modeling, least
squares boosting, AdaBoost, LogitBoost).

## The tabs

| Tab | Each round shows |
|---|---|
| **Random Forest** | a tree grown on its own bootstrap sample, with the 2 random feature candidates at every split and the score each one offered; the running vote before and after; who was drawn how often and who was out of bag. Tree *m* uses its own seed, so the order is only playback. |
| **Forward Stagewise** | the general step `f_m = f_{m-1} + β_m F_m` on a chosen run (least squares, AdaBoost or LogitBoost): the training loss along the new tree as a function of β, the step the method took vs. the exact minimizer, the loss curve over rounds, and each passenger's loss before and after |
| **Least Squares Boosting** | a regression tree fitted to residuals `y − f_{m−1}`, β = 1, f₀ = training survival rate |
| **AdaBoost** | weights `ω ∝ exp(−ỹ f_{m−1})`, weighted error, `β = ½ log((1−err)/err)`, and each passenger's weight going up by `e^β` or down by `e^{−β}` |
| **LogitBoost** | `π`, the working response `z` (clipped to ±4) and weight `π(1−π)`, a weighted least-squares tree, and the fixed Newton step ½ |

Tabs can be linked directly: `#random-forest`, `#stagewise`, `#least-squares`,
`#adaboost`, `#logitboost`.

## Data

`data.js` holds the 714 passengers with a recorded age from the Kaggle *Titanic:
Machine Learning from Disaster* `train.csv` (891 rows, compiled from Encyclopedia
Titanica), taken from the
[datasciencedojo/datasets](https://github.com/datasciencedojo/datasets/blob/master/titanic.csv)
mirror. Rows with a missing age are dropped. Nothing else is changed. Features:
sex, passenger class, age, fare. A seeded shuffle (seed 473) puts 500 passengers in
training and 214 in test.

## What is computed

`model.js` grows every tree in the browser: CART with case weights, depth ≤ 2, at
least 5 distinct passengers per leaf, Gini for classification trees and weighted MSE
for regression trees. Ties go to the earlier feature and the smaller threshold.
`script.js` only displays the results.

## Run it locally

No toolchain. KaTeX and fonts load from CDNs. Without them, equations fall back to
their TeX source.

```sh
start index.html          # Windows
```

## Check the math

```sh
node test-math.js
```

49 checks against the shipped `model.js`, all passing. They include:

- the split is reproducible
- every random forest split saw exactly 2 features and took the best one
- growing the 50 trees alone or in reverse order gives the same forest
- least squares β = 1 exactly
- AdaBoost's closed-form β matches a numeric line search
- after reweighting, the passengers AdaBoost just got wrong hold exactly half the weight
- LogitBoost's `w·z = y − π`, and its ½ step differs from the exact line search
- `f_m = f_{m−1} + β_m F_m` for every passenger, round and run
- training loss never rises when the line search is exact
