# Copies of Copies

An interactive look at mutual information, the chain rule and the data
processing inequality. A fair bit goes through two noisy channels; drag the
noise levels and watch where the information about the original bit goes.

**The one idea:** processing can keep information or destroy it, never create
it. The chain rule says exactly where the lost bits went:

```
I(X;Y) = I(X;Z) + I(X;Y|Z)
```

The top bar on the page *is* the other two laid end to end.

Covers Murphy, *Probabilistic Machine Learning: An Introduction* §6.3: mutual
information, conditional mutual information and its chain rule, and the data
processing inequality.

## The model

- **X** is a fair random bit.
- **Y** is X, flipped with probability **p** (independent noise).
- **Z** is Y, flipped again with probability **q** (independent noise).

So X → Y → Z is a Markov chain of two binary symmetric channels, and every
quantity has an exact closed form in bits:

| | |
|---|---|
| `h₂(t)` | `−t log₂ t − (1−t) log₂(1−t)`, with `h₂(0) = h₂(1) = 0` |
| `r` | `p + q − 2pq`, the chance X and Z disagree (two flips cancel) |
| `I(X;Y)` | `1 − h₂(p)` |
| `I(X;Z)` | `1 − h₂(r)` |
| `I(X;Y\|Z)` | `h₂(r) − h₂(p)` |
| `I(X;Z\|Y)` | `0`, because Z is generated from Y alone |

## What's in it

| | |
|---|---|
| **Pipeline** | X → Y → Z drawn as two binary symmetric channels. Straight lines keep the bit, crossing red lines flip it; thickness is the probability. |
| **Sample tape** | 40 fixed random draws pushed through both channels, with each flip marked. Raising p or q only adds flips, so the tape changes smoothly instead of reshuffling. Double flips that cancel out are called out, which is where the `−2pq` in `r` comes from. |
| **Bars** | I(X;Y), I(X;Z) and I(X;Y\|Z) on a shared 0–1 bit scale. Blue is what survives to Z; hatched red is what processing lost. The I(X;Y\|Z) bar starts where I(X;Z) ends, so the stack lines up with the top bar. |
| **Sliders** | p and q, each from 0 to 0.5. Native range inputs, so arrow keys work. |
| **Presets** | No information lost (q = 0), some lost (q = 0.2), all lost (q = 0.5). |
| **Notation, chain rule, DPI** | Plain-language readings of `;`, `,` and `\|`, the two-way chain rule derivation, and why `I(X;Z) ≤ I(X;Y)`. |

## Run it locally

No toolchain, no dependencies, no build step. Open the file:

```sh
start index.html          # Windows
open index.html           # macOS
```

Or serve the repo root with `python -m http.server 8000` and visit
`localhost:8000/mi/`.

## Check the math

```sh
node test-math.js
```

`script.js` exports its pure-math half under Node, so the suite exercises the
shipped file. It runs 32 checks, all currently passing. They include an
**independent route**: the suite builds the full joint `p(x, y, z)` over all 8
outcomes and computes every mutual information from plain entropies. Across a
51 × 51 grid of (p, q) the two routes agree to within about 1e-15 bits. The
suite also checks both orderings of the chain rule, that `I(X;Z|Y) = 0`
everywhere, that the DPI holds everywhere, and that `q = 0.5` makes X and Z
exactly independent.
