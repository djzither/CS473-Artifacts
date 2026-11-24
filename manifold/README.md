# Manifold Learning Playground

Manifold learning for an advanced ML class. You build intuition first, with
surfaces, a digit and geodesics. Then six real algorithms run side by side on
the same points, and finally each objective is written out.

**The one idea:** data with thousands of coordinates often has only a few real
degrees of freedom, and it lies near a curved low-dimensional surface. Each
method writes "keep the shape of that surface" as a different objective, and
those objectives succeed and fail in different places.

Covers Murphy, *Probabilistic Machine Learning: An Introduction* §20.4
(manifolds, the manifold hypothesis, MDS, Isomap, kernel PCA, LLE, Laplacian
eigenmaps, t-SNE).

## The page

| § | Section | What you do |
|---|---|---|
| 1 | Understanding manifolds | Rotate and zoom a sphere, Swiss roll, saddle or torus. Click to move a point and see its tangent plane, basis vectors and normal. A radius slider shows δ/r → 0, meaning "locally Euclidean." Readouts give the metric tensor *g* and the Gaussian curvature *K*: the roll has K = 0 and can be unrolled, the sphere cannot. |
| 2 | The manifold hypothesis | A drawn digit as a point in ℝ⁷⁸⁴, set by 5 sliders. Pick which variables vary, and 300 family members are projected onto two principal components. Rotation alone traces a closed loop. A cumulative-variance curve shows that linear PCA needs far more components than the intrinsic dimension. A random point of ℝ⁷⁸⁴ is static, not a digit. |
| 3 | Euclidean vs. geodesic | Click two points on a Swiss roll in 3-D or on its exact unrolled chart (s, h). The page draws the chord, the kNN-graph shortest path and the true geodesic, and reports all three distances. Raising *k* past ≈ 20 or adding noise produces red short-circuit edges. |
| 4 | Playground | Swiss roll, S-curve, circle or linear plane. Ground truth plus PCA, kernel PCA, Isomap, LLE, Laplacian eigenmaps and t-SNE, with colors kept across plots, hover-linked neighborhoods, hyperparameter sliders, eigenvalue spectra, a live t-SNE KL curve and two scores: 10-NN kept (local) and distance correlation (global). |
| 5 | The math | One tab per method: problem, what is preserved, objective, steps, parametric or not, success and failure. Buttons load a success or failure preset into the playground. A summary table follows. |
| 6 | Quiz | 16 questions on intrinsic vs. ambient dimension, the hypothesis, tangent spaces and Riemannian metrics, geodesics, linear vs. nonlinear, parametric vs. nonparametric, and method differences. Every option is explained once you answer. |

## What is real and what is simplified

Every algorithm is the real one, run exactly in the browser at N ≤ 600:

- one dense symmetric eigensolver (Householder + implicit QL, after JAMA's `tred2`/`tql2`)
- Dijkstra with a binary heap for graph geodesics
- t-SNE with the exact O(N²) gradient, early exaggeration, momentum and gains (no Barnes–Hut)

Simplifications, also labeled on the page:

- kernel PCA uses only the RBF kernel
- t-SNE runs a fixed number of iterations from one seeded start
- embeddings are Procrustes-aligned to the true coordinates for display, which never changes their shape

The Swiss roll is 10 tall instead of sklearn's 21, so a kNN graph at N = 400 stays on the sheet for k ≲ 12.
Its true coordinates use exact arc length, s(t) = ½[t√(1+t²) + asinh t], so (s, h) is an isometric chart and true geodesics are straight lines in it.

## Files

| File | Role |
|---|---|
| `model.js` | All math, with no DOM. Loaded by the page and by the tests. |
| `plot.js` | The canvas 3-D viewer (drag, wheel or pinch, painter's sort) and the 2-D scatter with animated transitions. |
| `explore.js` | Sections 1–3. |
| `playground.js` | Section 4, plus the section 5 tabs and "see it" buttons. |
| `quiz.js` | Section 6. |

No framework and no build step, like the other modules.

## Check the math

```sh
node test-math.js
```

The 48 checks all pass. They cover:

- the eigensolver (residual, orthonormality, ordering)
- the arc-length chart being isometric
- the curvature of all four surfaces and δ ≈ r²/2R on the sphere
- MDS on Euclidean distances = PCA, and wide-RBF kernel PCA → PCA
- every success and failure preset the page offers (Isomap k = 8 vs. 22 and noise 0.6, LLE k = 12 vs. 4, the Laplacian arch, t-SNE perplexity 30 vs. 2)
- the section 3 numbers: the layer pair's ≈ 8× geodesic/Euclidean ratio, first short circuit near k = 21
- t-SNE perplexity calibration and KL descent
- Procrustes alignment
