# Same Votes, Different Teachers

An interactive look at why naive Bayes and logistic regression make predictions
in the same form even though they learn in different ways. A two-word spam
filter is worked by hand in three steps, one **Next step** click at a time.

**The one idea:** naive Bayes multiplies the prior odds by one likelihood ratio
per feature. Take logs and those factors become votes that add up to a total
`z`, and `P(spam) = 1 / (1 + e^−z)`. That is exactly logistic regression's form.
The two models differ in *how they get the votes*, not in how they use them.

Covers Murphy, *Probabilistic Machine Learning: An Introduction* §9.3 (naive
Bayes classifiers), §9.4 (generative vs. discriminative classifiers) and §10.2
(binary logistic regression).

## The example

| | P(· \| spam) | P(· \| not spam) | multiplier if present | if absent |
|---|---|---|---|---|
| prior | 20% | 80% | odds 0.25 | |
| "free" | 60% | 10% | ×6 | ×0.44 (40 ÷ 90) |
| "winner" | 50% | 5% | ×10 | ×0.53 (50 ÷ 95) |

Naive Bayes assumes the two words are conditionally independent given the class.
An absent word is not skipped: its multiplier uses `1 − p`.

## What's in it

| | |
|---|---|
| **Controls** | Two checkboxes (the email contains "free" / "winner"), a **Next step** button, Start over, and three step chips. A caption narrates each sub-step with its real numbers. |
| **1 · Multiply evidence** | Tiles for the starting odds and each word's multiplier, revealed one at a time, with the running odds and a spam vs. not-spam balance bar that animates. |
| **2 · Add votes** | The same calculation in log space as a diverging horizontal bar chart. Red bars to the right vote for spam, blue bars to the left vote for not spam, and a total bar gives `z`. |
| **3 · Predict** | The S-curve `1 / (1 + e^−z)` with the current `z` marked, and both routes worked side by side: `odds ÷ (1 + odds)` and `1 ÷ (1 + e^−z)`. They agree. |
| **Below** | Naive Bayes learns feature probabilities per class and derives the votes. Logistic regression learns the votes directly by fitting the labels, with no independence assumption. A 1,000-email mailbox with the same per-class rates, but where "winner" never appears without "free", shows the fitted weights differ. Naive Bayes says 93.8% for both words, logistic regression says 71.4%, and the data says 71.4%. |

## What it is careful *not* to claim

- Logistic regression does **not** need conditional independence. Only naive
  Bayes assumes it.
- The same prediction form does **not** mean the same weights. The weights
  agree when the words really are independent within each class and there is
  enough data, which the test suite checks as a control.

## Run it locally

No toolchain, no dependencies, no build step. Open the file:

```sh
start index.html          # Windows
open index.html           # macOS
```

Or serve the repo root with `python -m http.server 8000` and visit
`localhost:8000/nb/`.

## Check the math

```sh
node test-math.js
```

`script.js` exports its pure-math half under Node, so the suite exercises the
shipped file. It runs 39 checks, all currently passing. Highlights:

- for all four emails, the odds route, the log route and Bayes' rule applied
  directly (no odds, no logs) give the same P(spam) to 1e-12
- every running vote total is the log of the odds so far
- naive Bayes rewritten as `w0 + w·x` reproduces it exactly, with
  `w_free = log 13.5` and `w_winner = log 19`
- the mailbox's per-class rates match the page exactly, and the Newton fit of
  logistic regression lands on the closed-form weights to 1e-9
- control: on a mailbox where the words *are* independent, logistic
  regression's fit equals naive Bayes's weights
