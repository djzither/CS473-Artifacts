# Close Together, Far Apart

An interactive introduction to contrastive loss and triplet loss, written for
someone with no machine learning background (pitched at a high school student).
Each colored dot stands for one picture of a person, and the same color means
the same person.

**The one idea:** we want pictures of the same person close together, and
pictures of different people farther apart. A *penalty* scores how far a
layout is from that goal. Contrastive loss checks one pair at a time against a
fixed ruler. Triplet loss only compares: your other picture has to be closer to
you than a stranger's picture, with a little extra room.

Covers Murphy, *Probabilistic Machine Learning: An Introduction* §16.2
(learning distance metrics: the pairwise contrastive and triplet ranking
losses). Mahalanobis distance and network diagrams are left out on purpose.

## The two rules

| | penalty | zero when |
|---|---|---|
| two pictures, same person | `d²` | the dots touch |
| two pictures, different people | `max(0, m − d)²`, m = 3 | at least 3 steps apart |
| three pictures | `max(0, d(you, you again) + m − d(you, someone else))`, m = 1 | the stranger is at least 1 step farther than your other picture |

Distances are plain straight-line distances in grid steps. Formulas are hidden
behind a **Show the math** button on each view.

## What's in it

| | |
|---|---|
| **1 · Comparing two pictures** | Two draggable dots and a toggle: *the same person* or *different people*. Same person: arrows pull the dots together, and dropping one on the other stacks them for a penalty of 0. Different people: a red dashed circle marks how far apart they need to be, turning green once the second dot is outside it. A big penalty number and bar update live. |
| **2 · Comparing three pictures** | Draggable dots labelled *You*, *Another picture of you* and *Someone else*, with lines from You to the other two. A solid circle passes through your other picture and a dashed circle sits one step farther out. The shaded ring between them is the "extra room". **Take a learning step** runs one animated gradient step on all three dots. From the starting layout, the goal is met in a few clicks, and after that the button has nothing to fix. |
| **Below** | A side-by-side card: a fixed ruler vs. a comparison, why the same-person pull never stops under contrastive loss but triplet loss goes quiet once the goal is met, and why both work. |

Dots can be moved with the arrow keys (Shift for bigger steps) as well as by
dragging.

## Run it locally

No toolchain, no dependencies, no build step. Open the file:

```sh
start index.html          # Windows
open index.html           # macOS
```

Or serve the repo root with `python -m http.server 8000` and visit
`localhost:8000/triplet/`.

## Check the math

```sh
node test-math.js
```

`script.js` exports its pure-math half under Node, so the suite exercises the
shipped file. It runs 30 checks, all currently passing. Highlights:

- every arrow on the page is the negative gradient of its penalty, checked
  against a numerical gradient
- for different people, closer is always worse inside the circle, and farther
  than the margin earns nothing
- from the page's starting triplet, every learning step lowers the penalty and
  the goal is met within 6 clicks, and once it is met a step moves nothing
- the comparison card's claims: a layout where triplet loss is satisfied while
  contrastive still charges the same-person pair, and sliding all three dots
  together leaves the triplet penalty unchanged
