# PCA Interactive Learning Lab

Principal components analysis for an advanced ML class. Seven linked labs and a
cheat sheet, built for exam prep on the three assigned subsections. The
optimization derivation in §20.1.2 isn't assigned and is skipped; the page uses
its result and explains why it makes sense.

**The one idea:** total variance `tr Σ̂ = Σ λ_j` is fixed. PCA splits it into the
part the first `L` directions keep (`Σ_{j≤L} λ_j`) and the part reconstruction
loses (`Σ_{j>L} λ_j`). Every module is a different view of that split.

Covers Murphy, *Probabilistic Machine Learning: An Introduction* §20.1.1
(examples), §20.1.3 (computational issues) and §20.1.4 (choosing the number of
latent dimensions), plus Zhu & Ghodsi (2006) for profile likelihood.

## The page

Tabs across the top; each module initializes the first time you open it.

| | Module | What you can do |
|---|---|---|
| 1 | Understanding PCA (§20.1.1) | Rotate `w` by dragging the arrow or the curve, and drag data points. Watch projected variance and reconstruction error add up to `tr Σ̂`. Compress a 64×64 image, or your own, with `L` components. A synthetic-"3"s demo shows the mean image, eigen-images and a 2-D score map in the style of the MNIST and eigenfaces figures, computed with the Gram trick. |
| 2 | Covariance vs. correlation (§20.1.3.1) | Rescale height or weight (m/cm/mm/in, kg/g/lb/t) and switch between the raw, centered and standardized views. Shows both 2×2 matrices with closed-form eigenpairs, PC arrows, the other method's PC1 as a dashed line, and the uncentered "PC1" pointing at the mean. |
| 3 | High-dimensional data (§20.1.3.2) | Log sliders for N and D, with presets such as Olivetti faces and MNIST. Compares the shapes of `X`, `XᵀX` and `XXᵀ`, their entries, memory and O(·) costs, and recommends one. Includes an editable 4×6 example that steps through Eq. 20.22, with checks against the 6×6 eigenproblem. |
| 4 | PCA via SVD (§20.1.3.3) | Edit or generate `X` (N ≤ 8, D ≤ 5). Shows every intermediate (`x̄`, `X_c`, `U`, `S`, `Vᵀ`, `V`, `W = V_L`, `Z = U_L S_L`, `X̂_c`, `X̂`, residual) with its dimensions and the kept parts highlighted. Checks `λ_j = s_j²/N` against a direct eigendecomposition and `‖X_c − X̂_c‖²_F = Σ_{j>L} s_j²`. |
| 5 | Choosing L (§20.1.4) | Edit the eigenvalues as chips or by dragging bars. The scree plot colors retained and discarded components, with live Eq. 20.26 / 20.27 and a minimum `L` for 90/95/99% or a custom target. A train/test simulation draws from your spectrum; both curves fall to 0, with no U shape. |
| 6 | Profile likelihood (§20.1.4.3) | Drag the change point. Shows the group means, the shared-σ bands, Eqs. 20.28–20.31 with your numbers, ℓ(L) for every valid split with the argmax, and a comparison with an elbow heuristic and variance thresholds. |
| 7 | Exam practice | Randomized questions across 11 topics, numeric, dimension and multiple choice. You answer first, with an optional hint and a step-by-step solution. A score, streak and per-topic record is saved in `localStorage` and lists topics under 70%, with an optional "focus on weak topics" mode. |
| Σ | Cheat sheet | The formulas with one-line reasons, plus common traps. |

All data are synthetic and labeled as such on the page. Equation numbers follow
the 2022 MIT Press edition.

## Files

| File | Role |
|---|---|
| `model.js` | The math, with no DOM: Jacobi eigendecomposition, SVD, PCA fit and reconstruction, the Gram trick, Eqs. 20.26–20.31, data generators |
| `practice.js` | The question generators and the grader (no DOM) |
| `ui.js` | Shared plotting, matrix and editor helpers, tooltips, tabs |
| `m1.js` … `m7.js` | One file per module |

Like the other modules, it uses no framework and no build step. KaTeX is loaded from cdnjs.

## Check the math

```sh
node test-math.js
```

The 48 checks all pass. They cover:

- the eigendecomposition and SVD identities, including wide rank-deficient matrices
- PCA = truncated SVD for every L
- training error = Σ of the discarded eigenvalues
- correlation-matrix invariance to units
- the Gram trick: shared eigenvalues, `‖Xᵀu‖² = λ`, unit columns
- Eqs. 20.26–20.27 and the thresholds
- monotone train and test error
- Eqs. 20.28–20.31 and the closed form of ℓ(L)
- 440 random practice questions graded against their own worked answers
