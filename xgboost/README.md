# Boosting Through the Highstorm

Gradient boosting and XGBoost, round by round, told through *The Way of Kings*
(no later spoilers). Dalinar's engineers forecast how many highstorms each of twelve
bridges will survive. Each round, a new engineer (one small tree) looks only at where
the forecast so far went wrong and proposes a correction. Every idea is explained in
plain English first, then as an equation, then with live numbers for the bridge you
are following.

**The data is invented for teaching.** The bridges, their names and their numbers do
not come from the books.

## Two connected modes

Both modes share the bridges, the round counter, the followed bridge, the learning
rate η and the tree depth.

| Mode | Each round shows |
|---|---|
| **1 · Gradient boosting** | squared-error loss, F₀ = average target; misses r = y − F (minus the slope of the loss); a depth-1 or depth-2 tree fit to the misses with leaf = average miss; F ← F + η·leaf; the loss before and after |
| **2 · XGBoost** | slopes g = F − y and curvatures h = 1; split gain ½[G_L²/(H_L+λ) + G_R²/(H_R+λ) − G²/(H+λ)] − γ with every candidate listed; leaf weight −G/(H+λ) beside the plain average; bottom-up pruning of splits whose gain is negative after γ; the objective's complexity price γT + ½λΣw²; the loss curve over mode 1's |

**Set λ = γ = 0** makes XGBoost reproduce mode 1 exactly. This is checked in
`test-math.js`.

Controls: Back, Next round, Play/Pause, Reset, round slider (0–20), η, depth, λ, γ.
Click a bridge in the chart or a table to follow it, and click a question in the
tree to see every split it beat. Link a mode with `#gradient-boosting` or
`#xgboost`.

## Run it locally

```sh
start index.html
```

## Check the math

```sh
node test-math.js
```

29 checks against the shipped `model.js`, all passing. They include:

- F₀ is the best constant
- leaves are the average misses
- the "drop" column equals the fall in summed squared misses
- w = −G/(H+λ) is the parabola's minimum and is pulled toward zero
- the gain formula
- pruning
- Ω
- λ = γ = 0 matches mode 1 for four settings
