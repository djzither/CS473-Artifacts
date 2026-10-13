# Touch, Then Climb

An interactive, step-by-step look at EM for a Gaussian mixture: the lower bound
it builds, and the KL gap between that bound and the true score. Step through
the algorithm one half-step at a time and watch three linked views update.

**The one idea:** we know the true score's formula, but its peak is hard to find
directly. EM builds an easier curve that sits underneath and touches it, climbs
that, and repeats. The bound is not an arbitrary simplification: it differs from
the true score by exactly the KL gap.

```
ℓ(θ) − 𝓛(θ, q) = Σₙ D_KL( qₙ(zₙ) ‖ p(zₙ | yₙ, θ) )
```

Covers Murphy, *Probabilistic Machine Learning: An Introduction* §8.7.2–8.7.3:
the EM algorithm, its lower bound, and EM for a GMM.

## The model, simplified on purpose

- **14 fixed points** on a line, in a larger left group and a smaller right one.
- **Two Gaussian clusters**, both with fixed variance σ² = 1 and fixed weight π = ½.
- **Cluster 2's centre μ₂ is fixed** wherever you put it.
- **EM learns one number:** cluster 1's centre μ₁.

Because μ₁ is the only parameter, plotting the true score against μ₁ shows the
whole landscape, not a slice of it. The bound picture is exactly what the
optimizer is working with.

## What's in it

| | |
|---|---|
| **Controls** | Separate **E step** and **M step** buttons, enforced to alternate, plus Auto-play and Reset. Three phase cards light up in turn: *Update our beliefs about membership*, *Improve the model with those beliefs fixed*, *Rebuild the bound at the new model*. |
| **1 · Clusters** | The data, both weighted Gaussian bumps, and two rows of dots per point. One row shows the beliefs q currently in use. The other shows the posterior (the responsibilities) under the centres as drawn now. After an M step the two rows disagree, and the points whose beliefs are now out of date get amber rings. Drag either centre to choose where a run starts. |
| **2 · Score and bound** | The true score ℓ(μ₁) and the bound 𝓛(μ₁, q). The E step morphs the bound until it touches ℓ at the current μ₁, which stays put. The M step freezes the bound and slides μ₁ to the bound's peak. The shaded band is the gap at every μ₁, and a bracket labels it at the current one. |
| **3 · Progress** | ℓ and 𝓛 recorded after every half-step, and the KL gap again as bars on their own scale. A live readout checks the identity two independent ways. |
| **Side notes** | Each view has a note beside it that quotes the numbers the last half-step just produced. |
| **Presets** | A smooth climb. A run where the second M step opens a much bigger gap than the first. A run stuck on a lower peak, and the same setup started from the other side. |

## What it is careful *not* to claim

- The KL gap does **not** shrink steadily. It is exactly 0 after every E step and
  can open wider after a later M step than after an earlier one.
- μ₁ does **not** necessarily head for the highest peak. Once a run settles below
  the best score, the page marks the higher peak EM never visited.
- The only guarantee is that **the true score never decreases**:
  `ℓ(θ_new) ≥ 𝓛(θ_new, q) ≥ 𝓛(θ_old, q) = ℓ(θ_old)`.

## Run it locally

No toolchain, no dependencies, no build step. Open the file:

```sh
start index.html          # Windows
open index.html           # macOS
```

Or serve the repo root with `python -m http.server 8000` and visit
`localhost:8000/em/`.

## Check the math

```sh
node test-math.js
```

`script.js` exports its pure-math half under Node, so the suite exercises the
shipped file. It runs 29 checks, all currently passing. Highlights:

- log-sum-exp and the responsibilities stay finite and sum to 1 with the
  centres up to 60σ away from the data, where the naive ratio returns `NaN`
- the identity `ℓ − 𝓛 = Σ KL` holds to 1e-9 for 3000 random `(μ₁, μ₂, q)`, with
  the two sides computed by separate routes
- the bound never rises above ℓ, for any q, at any μ₁
- after an E step the gap is 0, the bound is tangent to ℓ (equal slopes), and μ₁
  and ℓ are unchanged
- the M step lands on the bound's maximum and never lowers either curve
- ℓ never decreases over 300 random runs, and each preset behaves as its
  description says
