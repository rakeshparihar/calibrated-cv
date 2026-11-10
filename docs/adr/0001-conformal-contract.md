# ADR 1: Conformal sets as the decision contract, temperature scaling as calibration

## Context

Two designs were debated by two advocates:
- A: temperature scaling, ECE and reliability tables, and abstain below a probability threshold chosen for a target precision.
- B: split conformal prediction (APS) returning a set of labels with a coverage guarantee; auto-tag single-label sets, send larger sets to review.

Both advocates ended up recommending the two methods together and disagreed only on which one the API should expose as its decision rule, so no judge round was needed.

## Decision

- Temperature scaling is fitted first; ECE before and after is reported.
- The API's decision is conformal: a one-label set is auto-tagged; a larger set goes to review, and the reviewer chooses from the set.
- The probability-threshold rule is reported alongside as a comparison.
- Coverage, auto-tag accuracy and set size are reported per slice (gender, season, master category), since the guarantee is only marginal.

## Reasons

- Conformal coverage holds for any model if calibration and new data are exchangeable; a threshold only hits its precision target on average.
- A set tells reviewers what to choose between, which cuts review effort.
- Neither method guarantees coverage per slice, so slices are measured, not assumed.
