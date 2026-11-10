# ADR 2: Serve LAC conformal sets, not APS

## Status

Accepted, after the first full eval.

## Context

ADR 1 chose conformal sets as the decision contract and named APS as the score function. The first eval, on 600 calibration and 600 test images, showed APS was unusable with this model:

| Method | Threshold q | Coverage | Mean set size | Sets of one |
| --- | --- | --- | --- | --- |
| APS (no randomised tie-break) | 0.9999 | 100.0% | 14.57 of 25 classes | 3.2% |
| LAC | 0.7103 | 92.8% | 1.04 | 96.3% |

The head is confident (95% of test images are above 0.93 after temperature scaling), so almost every APS calibration score sits near 1. The 95% quantile is then 0.9999, and every set has to collect nearly all probability mass. A set of 16 labels saves a reviewer nothing.

## Decision

Serve LAC (score = 1 minus the probability of the true class). Keep APS in the code and in the eval comparison table.

## Consequences

- Coverage measured at 92.8% against a 95% target on 600 test images. Sampling noise at that size is about plus or minus 1.8 points (95% interval), so this run is below target by slightly more than noise alone explains. Worth rechecking with a larger calibration split before trusting the guarantee.
- Two slices are under-covered: women's products (89.4%) and footwear (86.1%). The marginal guarantee does not protect them. Slice-conditional (Mondrian) calibration is the planned fix (v0.2).
- Randomised APS, or RAPS with a size penalty, would be the fairer comparison for adaptive sets and is a v0.2 item.
