# Closer Counts More

An interactive introduction to Gaussian kernel regression, written for someone
with no machine learning background (pitched at a high school student). Four
students report hours studied and exam score. A new student drags a line to
their own study time and gets a predicted score.

**The one idea:** the prediction is a weighted average of the four scores, and
students who studied about as long as you get the biggest share. The bandwidth
`h` sets how quickly that share falls off with distance.

Covers Murphy, *Probabilistic Machine Learning: An Introduction* §16.3 (kernel
density estimation and kernel regression, the Nadaraya–Watson model).

## The example

| studied (h) | 1 | 3.5 | 4.5 | 9 |
|---|---|---|---|---|
| score | 50 | 75 | 85 | 98 |

```
w_n(x) = e^(−(x − x_n)² / 2h²) / Σ_j e^(−(x − x_j)² / 2h²)
ŷ(x)   = Σ_n w_n(x) · y_n
```

The page starts at `x = 4`, `h = 1`: the 3.5 h and 4.5 h students split about
99.4% of the weight evenly, and `ŷ = 79.8`. The plain average of all four is 77.

## What's in it

| | |
|---|---|
| **The graph** | Hours studied against exam score. Drag anywhere (or use the arrow keys) to move the vertical study-time line. Each student's dot grows (area ∝ weight) and brightens with its weight, and its percentage is printed above it. Lines run from each student to the prediction, thicker for more weight. A shaded bell along the bottom shows the kernel itself. The prediction is the large orange dot. |
| **Readout** | Predicted score, a one-sentence summary of who it's listening to, the bandwidth slider (log scale, 0.2 to 10 hours), and a table of each student's percentage contribution. |
| **Buttons** | **Walk from 4 to 9 hours** animates the line so you can watch the share move to the 9 h student. **Compare with plain average** adds the 77 line, a hollow dot, and the difference. **Show every prediction** traces ŷ for every study time, so the slider visibly takes the curve from steps (small h) to flat (large h). |
| **Below** | A small-h vs. large-h card, and the formulas behind **Show the math**. |

## Run it locally

No toolchain, no dependencies, no build step. Open the file:

```sh
start index.html          # Windows
open index.html           # macOS
```

Or serve the repo root with `python -m http.server 8000` and visit
`localhost:8000/kernel/`.

## Check the math

```sh
node test-math.js
```

`script.js` exports its pure-math half under Node, so the suite exercises the
shipped file. It runs 21 checks, all currently passing. Highlights:

- the stabilized weights match the formula as printed, always sum to 100%, and
  stay finite at very small h
- a closer student always gets more weight, and the prediction always stays
  between 50 and 98
- the starting prediction is 79.8, small h copies the nearest student, and at
  the slider's maximum every student gets 15–35% at any study time
- along the 4 → 9 walk, the 9 h student only gains weight, the 1 h student
  only loses it, and the predicted score rises the whole way
