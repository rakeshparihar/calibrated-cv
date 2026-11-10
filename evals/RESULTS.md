# Eval results

Encoder `google/siglip-base-patch16-224` (frozen) + logistic-regression head, 25 article types. Splits: 1800 train, 600 calibration, 600 test. Target coverage 95%.

| Metric | Value |
| --- | --- |
| Test accuracy (top-1) | 91.2% |
| Temperature | 0.46 |
| ECE before / after temperature scaling | 0.223 / 0.041 |
| Conformal method served | LAC |
| Conformal set coverage on test | 92.8% (target 95%) |
| Mean set size | 1.04 |
| Auto-tagged (set of one) | 96.3% of images, 92.7% correct |
| Sent to review | 3.7% of images, true label in the set 95.5% of the time |
| Probability threshold for 99% precision on calibration (0.989) | auto-tags 51.0% of test, 97.1% correct |

## Conformal score functions compared (same calibration split, test set)

| Method | Threshold q | Coverage | Mean set size | Median set size | Sets of one |
| --- | --- | --- | --- | --- | --- |
| APS | 0.9999 | 100.0% | 14.57 | 16 | 3.2% |
| LAC | 0.7103 | 92.8% | 1.04 | 1 | 96.3% |

## Per slice (test set)

| Slice | Images | Accuracy | Set coverage | Auto-tag rate | Auto-tag accuracy |
| --- | --- | --- | --- | --- | --- |
| gender = Men | 334 | 95.5% | 96.7% | 97.6% | 96.6% |
| gender = Women | 216 | 86.6% | 89.4% | 93.5% | 89.1% |
| season = Summer | 302 | 90.1% | 92.4% | 95.7% | 92.0% |
| season = Fall | 170 | 91.8% | 93.5% | 96.5% | 93.3% |
| season = Winter | 109 | 93.6% | 93.6% | 98.2% | 94.4% |
| masterCategory = Apparel | 306 | 91.8% | 93.8% | 96.4% | 93.6% |
| masterCategory = Footwear | 165 | 83.6% | 86.1% | 94.5% | 85.9% |
| masterCategory = Accessories | 119 | 100.0% | 100.0% | 99.2% | 100.0% |

Slices more than 3 points under the coverage target: gender = Women (89.4%), masterCategory = Footwear (86.1%).

## Reliability after temperature scaling (test set)

| Confidence bin | Images | Mean confidence | Accuracy |
| --- | --- | --- | --- |
| 0.13-0.20 | 1 | 0.199 | 0.000 |
| 0.20-0.27 | 1 | 0.258 | 1.000 |
| 0.27-0.33 | 1 | 0.297 | 0.000 |
| 0.33-0.40 | 2 | 0.393 | 0.500 |
| 0.40-0.47 | 3 | 0.432 | 0.667 |
| 0.47-0.53 | 9 | 0.510 | 0.222 |
| 0.53-0.60 | 2 | 0.581 | 0.000 |
| 0.60-0.67 | 17 | 0.635 | 0.647 |
| 0.67-0.73 | 16 | 0.705 | 0.875 |
| 0.73-0.80 | 23 | 0.770 | 0.783 |
| 0.80-0.87 | 25 | 0.838 | 0.880 |
| 0.87-0.93 | 40 | 0.899 | 0.875 |
| 0.93-1.00 | 460 | 0.987 | 0.959 |
