# The Widest Corridor

An interactive introduction to support vector machines, told as a Rebel sensor
mission. Leia's crew (Luke, Han, R2-D2 and C-3PO) has to classify arriving
ships as Rebel (`ỹ = −1`) or Imperial (`ỹ = +1`) from two readings: engine
output and Imperial transponder strength. The lesson runs as nine short scenes
with Back/Next buttons and a progress bar.

**The one idea:** of all the lines that separate the two sides, pick the one
with the widest empty corridor around it. Only the few ships touching the
corridor's edges (the support vectors) decide where it goes.

Covers Murphy, *Probabilistic Machine Learning: An Introduction* §17.3
(large-margin classifiers, the dual problem, soft margins, the kernel trick,
and briefly the ν-SVM).

## The scenes

| | Scene | Interaction |
|---|---|---|
| 1 | **Ships on the scope**: `xₙ` and `ỹₙ`, why features are standardized | click a ship for its raw log (MW, dB) and standardized position |
| 2 | **A score for every ship**: `f(x) = wᵀx + w₀`, `ŷ = sign f(x)`, score ≠ probability | sliders for `w₁`, `w₂`, `w₀`; boundary, `w` arrow, mistakes |
| 3 | **Which line?**: the margin | compare hand-drawn lines A and B with trained line C |
| 4 | **Score is not distance**: `f(x)/‖w‖` | scale `w` and `w₀` by `s`: scores change, the boundary and distances don't, the `f = ±1` lines do |
| 5 | **Training**: `min ½‖w‖²` s.t. `ỹₙf(xₙ) ≥ 1`, margin `1/‖w‖` | multiply the trained solution by `t` to see constraints break (t < 1) or the corridor narrow (t > 1) |
| 6 | **Support vectors**: `w = Σ αₙỹₙxₙ`, `αₙ[ỹₙf(xₙ) − 1] = 0` | drag ships; the solver retrains live and the α list updates |
| 7 | **Kernels**: the dual, `f(x) = Σ αₙỹₙK(x, xₙ) + w₀` | linear, quadratic and Gaussian kernels on ring-shaped data; γ slider |
| 8 | **Soft margins**: `½‖w‖² + CΣξₙ`, slack | C slider (0.01 to 100); slack drawn as red segments with ξ values |
| 9 | **Final mission** | click (or type raw readings) to place a new ship; score, sign, distance and zone, then a recap table |

Each scene has a short character exchange, the plot, the equation (KaTeX)
with every new symbol defined beside it, a plain-language translation, and
optional **Show why** sections (perpendicular `w`, the distance derivation,
the Lagrangian, the dual, kernels as feature-space dot products, the dual box
constraint and hinge loss, the ν-SVM).

## What is computed

Every boundary labelled *trained* is solved on the page. The solver is
sequential minimal optimization on the dual

```
max_α  Σ αₙ − ½ Σₙ Σₘ αₙ αₘ ỹₙ ỹₘ K(xₙ, xₘ)
s.t.   0 ≤ αₙ ≤ C,  Σ αₙ ỹₙ = 0
```

using the maximal-violating-pair rule (as in LIBSVM), stopping at a KKT gap of
10⁻⁷. `w₀` is read from the KKT conditions: averaged over free support
vectors, or the midpoint of the feasible interval if there are none. The hard
margin uses `C = 10⁶`; if some ship still ends up with `ỹf < 1` the data are
reported as not linearly separable (scene 6 then refuses the drag).

Hand-set lines are tagged as such: the scene 2 sliders and lines A and B in
scene 3. Scenes 4 and 5 rescale the trained solution and say so.

The plot uses equal scales on both axes, so distances on screen are the
distances the SVM measures.

## Run it locally

No toolchain and no build step. KaTeX and the fonts load from CDNs; without
them the equations fall back to their TeX source.

```sh
start index.html          # Windows
open index.html           # macOS
```

Or serve the repo root with `python -m http.server 8000` and visit
`localhost:8000/svm/`. A scene can be linked directly, e.g. `#scene-6`.

## Check the math

```sh
node test-math.js
```

`script.js` exports its pure-math half under Node, so the suite exercises the
shipped file. It runs 75 checks, all currently passing. Highlights:

- the hard-margin solution satisfies every constraint and the KKT conditions,
  and its margin (0.746) matches a brute-force search over 200,000 directions
- strong duality: primal and dual objectives agree, hard and soft
- exactly three support vectors, all on the margin edges; Blue Three sits on
  an edge with α = 0, as scene 6 says
- lines A and B separate the data with smaller margins (0.27, 0.31)
- scaling `w, w₀` by `s` scales scores and leaves distances unchanged
- moving a non-support ship leaves the line unchanged; moving a support
  vector changes it; an inseparable layout is detected
- the shuttle makes the data inseparable; across nine values of C the KKT
  conditions hold, every ξ > 0 ship has α = C, and larger C never widens the
  corridor or increases total slack
- a ship can be inside the corridor yet correctly classified (slack ≠ mistake)
- linear fails on the ring data; quadratic and Gaussian kernels converge and
  satisfy KKT; `(xᵀz + 1)² = φ(x)ᵀφ(z)` for every pair
