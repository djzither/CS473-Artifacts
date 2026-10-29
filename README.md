# CS 473 Artifacts

Interactive notes for CS 473. Each module is a single self-contained HTML file —
no framework, no build step, no dependencies. Open the file in a browser, or view
the deployed site on GitHub Pages.

## Modules

| | Module | Covers |
|---|---|---|
| ✅ | [`index.html`](index.html) — **The Covariance Ellipse** | Murphy §3.2: covariance matrix, Mahalanobis contours, eigendecomposition, conditioning a 2D Gaussian |
| ✅ | [`gmm/`](gmm/) — **Who Made This Point?** | Murphy §3.5: mixture models, the latent variable z, responsibilities, the GMM → K-means limit |
| ✅ | [`roc/`](roc/) — **Where Do You Draw the Line?** | Murphy §5.1.2–5.1.3: the decision threshold, cost-sensitive classification, the confusion matrix, ROC and AUC, class imbalance |
| ✅ | [`kl/`](kl/) — **The Wrong Codebook** | Murphy §6.1–6.2: entropy, cross entropy, KL divergence as wasted bits, Shannon code lengths, KL's asymmetry |
| ✅ | [`mi/`](mi/) — **Copies of Copies** | Murphy §6.3: mutual information, conditional MI and the chain rule, the data processing inequality |
| ✅ | [`em/`](em/) — **Touch, Then Climb** | Murphy §8.7.2–8.7.3: EM for a GMM, the lower bound (ELBO), the KL gap, why the true score never decreases |
| ✅ | [`nb/`](nb/) — **Same Votes, Different Teachers** | Murphy §9.3–9.4, §10.2: naive Bayes as multiplied odds and added log votes, the shared logistic form, why matching forms don't mean matching weights |
| ✅ | [`triplet/`](triplet/) — **Close Together, Far Apart** | Murphy §16.2: contrastive and triplet loss for beginners, same person close and different people apart, a fixed ruler vs. a comparison, gradient steps on draggable dots |
| ✅ | [`kernel/`](kernel/) — **Closer Counts More** | Murphy §16.3: Gaussian kernel regression for beginners, weights that shrink with distance, a draggable study-time line, the bandwidth h from copy-the-nearest to plain average |
| ✅ | [`gp/`](gp/) — **Many Functions, Then Fewer** | Murphy §17.2: Gaussian process regression, a prior over functions becoming a posterior one point at a time, noise-free vs. noisy observations, the length scale, joint posterior samples |
| ✅ | [`svm/`](svm/) — **The Widest Corridor** | Murphy §17.3: support vector machines as a Star Wars mission, scores vs. distances, the hard margin, support vectors, the dual and kernels, soft margins and C, with an SMO solver running on the page |
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

Kevin P. Murphy, *Probabilistic Machine Learning: An Introduction*. CC-BY-NC-ND.
Each module names the sections it covers in its own README.
