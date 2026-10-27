# Many Functions, Then Fewer

An interactive introduction to Gaussian process regression. It starts with no
data, so you see the GP prior: a flat mean, a constant band, and a handful of
smooth possible functions. Observations arrive one at a time, and the
posterior mean, band, and possible functions update smoothly with each.

**The one idea:** a GP is a distribution over whole functions. Each observation
rules out the functions that don't pass near it, and how far that reaches is
set by the length scale ℓ.

Covers Murphy, *Probabilistic Machine Learning: An Introduction* §17.2
(Gaussian processes: GP regression with noise-free and noisy observations).

## The model

Zero prior mean, RBF kernel with unit signal variance, and
`y = f(x) + ε`, `ε ~ N(0, σ²)`. The hidden true function is `sin(x)` on
`[−5, 5]`.

```
K(x, x')            = exp(−(x − x')² / 2ℓ²)
posterior mean       = k*ᵀ (K + σ²I)⁻¹ y
posterior covariance = K** − K*ᵀ (K + σ²I)⁻¹ K*
```

The page factors `K + σ²I` by Cholesky (never inverting it), always with a
10⁻⁸ jitter on the diagonal that grows tenfold if the factorization fails.
Possible functions are joint draws, `mean + L z` with `L Lᵀ` the full
posterior covariance over 201 grid points. Each `z` is fixed until you press
**Redraw**, so the samples morph rather than jump as data arrive. The band is
`mean ± 2 sd` for the latent `f`, never including σ².

## What's in it

| | |
|---|---|
| **The graph** | Posterior mean (blue), ±2 sd band (shaded), possible functions (faint violet, labelled), observations (orange), and the hidden `sin(x)` (dashed, off by default). |
| **Adding data** | **Add next observation** steps through 10 preset inputs. **Back** removes the last point, **Reset** returns to the prior with the starting ℓ and σ. Click the graph to add your own point: its `y` is either the height you clicked or `sin(x)` plus fresh noise, depending on the mode you pick. A click within 0.1 of an existing input reuses that x exactly, so repeats are handled cleanly. |
| **Explanation panel** | Prior vs. posterior, what noise-free vs. noisy means at the observed points, the pointwise-band caveat, and after each point, whether the mean actually got less accurate anywhere (checked against the hidden function, when revealed). |
| **Model settings** | Length scale ℓ (log slider, 0.2 to 3), observation noise σ (0 = noise-free, up to 1), number of possible functions (1 to 10), and toggles for the true function and possible functions. |
| **Math** | The two formulas with every symbol defined in plain English, and how the page computes them. |

## Run it locally

No toolchain, no dependencies, no build step. Open the file:

```sh
start index.html          # Windows
open index.html           # macOS
```

Or serve the repo root with `python -m http.server 8000` and visit
`localhost:8000/gp/`.

## Check the math

```sh
node test-math.js
```

`script.js` exports its pure-math half under Node, so the suite exercises the
shipped file. It runs 47 checks, all currently passing. Highlights:

- the prior has mean 0 and sd 1 everywhere, and noise does not widen it
- one observation matches the closed form; several match a direct
  Gauss–Jordan inverse to 1e-6
- noise-free, the mean passes through all 10 points and the sd there is about
  10⁻⁴, at ℓ = 0.3, 1 and 3
- noisy, the mean misses the points and `Var f(x₁) = 1 − 1/(1 + σ²)`, not plus σ²
- adding a point never widens the band, yet some steps make the mean less
  accurate somewhere
- exact repeats (conflicting y averages), inputs 10⁻¹² apart, and snapping
- 4000 joint draws reproduce the posterior covariance, neighbours correlate at
  0.98, and noise-free samples pass through the data
