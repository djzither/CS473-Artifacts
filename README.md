# CS 473 Artifacts

Interactive notes for CS 473. Each module is a single self-contained HTML file —
no framework, no build step, no dependencies. Open the file in a browser, or view
the deployed site on GitHub Pages.

## Modules

| | Module | Covers |
|---|---|---|
| ✅ | [`index.html`](index.html) — **The Covariance Ellipse** | Murphy §3.2: covariance matrix, Mahalanobis contours, eigendecomposition, conditioning a 2D Gaussian |
| ✅ | [`gmm/`](gmm/) — **Who Made This Point?** | Murphy §3.5: mixture models, the latent variable z, responsibilities, the GMM → K-means limit |
| ☐ | Uncorrelated ≠ independent | ρ ≈ 0 for a Gaussian blob vs. Y = X², side by side |
| ☐ | Simpson's paradox | Iris sepal data with a "merge species" toggle |
| ☐ | Mahalanobis vs. Euclidean | drop a test point, compare both distances |

## Module 1 — The Covariance Ellipse

Three sliders (σ₁, σ₂, ρ) drive four linked views:

- **Joint plot** — sample cloud with Mahalanobis contours at Δ = 1, 2, 3
- **Eigenvector overlay** — arrows along the ellipse axes, length √λ
- **Conditional slice** — drag the `y₂ = c` line; `p(y₁|y₂)` slides but never changes width
- **Live readout** — Σ, its eigenvalues, tilt, |Σ|, and the conditional mean/variance

The sample cloud is drawn from a *fixed* set of standard-normal draws pushed through
the Cholesky factor of Σ, so moving a slider morphs the existing points instead of
resampling. The cloud deforms continuously, which is the whole point.

Presets reproduce Murphy's Figure 3.5: `full` (any tilt), `diagonal` (axis-aligned),
`spherical` (circular), plus `ρ → 1` to watch Σ go singular.

## Local development

No toolchain. Open `index.html` directly:

```sh
start index.html          # Windows
```

Or serve the folder if you prefer a real origin:

```sh
python -m http.server 8000
```

## Deployment

`.github/workflows/pages.yml` publishes the repository root to GitHub Pages on every
push to `main`.

**One-time setup:** in the repo, go to **Settings → Pages → Build and deployment →
Source** and choose **GitHub Actions**. After that the workflow handles every deploy.

## Source

Kevin P. Murphy, *Probabilistic Machine Learning: An Introduction*, §3.2 —
the multivariate Gaussian. CC-BY-NC-ND.
