# Who Made This Point?

An interactive two-component Gaussian mixture model. Drag the clouds around and
watch responsibilities change.

**The one idea:** every point came from *one* hidden probability cloud. We never
get to see which. So instead of guessing, the model computes how **responsible**
each cloud is for explaining that point.

Covers Murphy, *Probabilistic Machine Learning: An Introduction* §3.5 —
Eq. 3.94 (the mixture), 3.95–3.97 (the latent-variable rewrite), 3.99
(responsibilities), and 3.101 (the K-means limit).

## What's in it

| | |
|---|---|
| **2D plot** | Two clouds, covariance ellipses at Mahalanobis 1 and 2, a background responsibility field, and the exact `r₁ = r₂` decision boundary. Points are coloured by responsibility: blue, red, or washed-out violet when the model genuinely can't tell. |
| **Draggable means** | Grab either numbered centre. Ellipses, colours, boundary and heatmap all follow immediately. Keyboard: focus the plot, press <kbd>1</kbd>/<kbd>2</kbd>, then arrow keys. |
| **Point inspector** | Click any point for its coordinates, density under each component, both responsibilities, and both distances — plus a plain-language sentence explaining *why* it came out that way. |
| **"Watch one point get made"** | Runs the generative story live: roll the hidden coin, draw from the chosen cloud, throw the label away, infer it back. |
| **GMM → K-means slider** | Flattens the weights, rounds and shrinks both covariances, and hardens the assignment until only nearest-centre distance is left. A counter shows how many points change cluster. |
| **1D panel** | Two bell curves, their weighted contributions, and the mixture. Hover to split `p(y)` into the two numbers that add to it — the point being that we add *densities*, not samples. |

## Run it locally

No toolchain, no dependencies, no build step. Open the file:

```sh
start index.html          # Windows
open index.html           # macOS
```

It works straight off the filesystem. If you'd rather serve it:

```sh
python -m http.server 8000     # then visit localhost:8000/gmm/
```

The only network request is the Google Fonts stylesheet for IBM Plex; offline,
the page falls back to system fonts and everything still works.

## Check the math

```sh
node test-math.js
```

`script.js` exports its pure-math half under Node and skips everything
DOM-bound, so the suite exercises **the shipped file**, not a copy of its
formulas. 22 checks, currently all green:

- mixture weights sum to 1 at every point of the K-means blend
- responsibilities sum to 1 across 30 000 random points — including at
  `y = (1e6, -1e6)`, where computing `πp / Σπp` directly would return `NaN`
- `sigmoid(log-odds)` agrees with Murphy's ratio form to 1e-12
- covariance matrices stay symmetric and positive definite everywhere, with
  `det Σ = σx²σy²` exactly
- **the drawn ellipses really are the Mahalanobis contours** — the suite walks
  64 000 points along the ellipse the renderer draws and confirms each sits at
  distance *t*
- 2D Gaussian, the mixture, and the 1D Gaussian all integrate to 1
- the sampler's empirical covariance over 400 000 draws matches the Σ it claims
- at the K-means end: weights exactly equal, covariances spherical *and*
  identical *and* shrunk, assignment provably nearest-centre over 18 000 points,
  and responsibilities hard (> 0.99) away from the boundary
- the transition is monotone in both the prior and the correlation

## Publish on GitHub Pages

This folder is part of the **CS 473 Artifacts** repo, whose
`.github/workflows/pages.yml` already publishes the repository root on every
push to `main`. So there is nothing to configure — commit and push:

```sh
git add gmm
git commit -m "Add GMM responsibility module"
git push
```

The module lands at `https://<user>.github.io/<repo>/gmm/`.

**One-time repo setup** (only if Pages has never been enabled): **Settings →
Pages → Build and deployment → Source → GitHub Actions**.

### Publishing it standalone

To put just this folder in its own repo instead, the same three files are all
you need — Pages can serve them with no workflow at all:

1. Create a repo and push `index.html`, `styles.css`, `script.js`.
2. **Settings → Pages → Source → Deploy from a branch**, branch `main`, folder
   `/ (root)`.
3. It appears at `https://<user>.github.io/<repo>/` within a minute or so.

Paths in `index.html` are relative, so the page works from any subdirectory.

## Files

```
index.html     structure and copy
styles.css     design tokens, light + dark themes, responsive layout
script.js      math (exported for tests) + rendering + interaction
test-math.js   22 checks against the shipped script.js
```

## Notes on a few choices

**Covariance is built from spread + rotation**, `Σ = R diag(σx², σy²) Rᵀ`, rather
than from raw matrix entries. That makes an invalid covariance unreachable — no
validation needed, because non-positive-definite input can't be expressed.

**Ellipses are derived from Σ by eigendecomposition**, not from the slider
values. The picture therefore cannot drift out of agreement with the matrix the
model is actually using, and the test suite verifies exactly that.

**One scalar field drives three layers.** For two components, `r₁` is exactly
`sigmoid(log π₁p₁ − log π₂p₂)`. Sampling that log-odds on a grid gives the
background tint, the confidence shading, and — as its zero contour, via marching
squares — the decision boundary. All three agree by construction, and marching
squares draws the boundary correctly whether it's a line, ellipse, parabola or a
two-branch hyperbola.

**Everything stays in log space.** At the K-means end the shared variance falls
to ~1e-6, where raw densities overflow a float; log-sum-exp and an overflow-safe
logistic keep the numbers exact.

**Uncertainty is encoded three times** — hue (blue↔red), saturation (the
midpoint violet is deliberately low-chroma, so "unsure" looks washed out), and a
ring drawn around ambiguous points. Nothing depends on colour discrimination
alone. The blue/red pair was checked for colour-vision-deficiency separation in
both themes (worst case ΔE 23.1 protanopia, 32.6 normal vision).

**Moving π₁ re-rolls the hidden coin.** Each point carries a fixed uniform draw
compared against π₁, so changing the mixture weight genuinely migrates points
between clouds — which is what a prior over the latent `z` actually means.
Sliding the K-means blend, by contrast, leaves the data untouched and changes
only the model.
