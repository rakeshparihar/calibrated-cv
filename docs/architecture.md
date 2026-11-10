# Architecture

## Goals and non-goals

Goals:
- Classify product photos with probabilities that mean what they say.
- Abstain with a reason: send uncertain images to a person, with a short list of likely labels.
- Show where the guarantees hold and where they do not, by slice.

Non-goals:
- Top accuracy; the classifier is deliberately simple (linear head on frozen SigLIP).
- Robustness to distribution shift, which breaks both calibration and coverage.

## Current architecture (v0.1)

```mermaid
flowchart LR
    D[(HF dataset)] --> SP[Split: train / calibration / test]
    SP --> E[SigLIP encoder<br/>frozen, cached]
    E --> H[Logistic-regression head<br/>train split]
    H --> T[Temperature scaling<br/>calibration split]
    T --> Q[Conformal threshold q<br/>calibration split]
    Q --> M[(Calibrated model:<br/>head, T, q)]
    M --> API[FastAPI /classify]
    M --> EV[Test: accuracy, ECE, coverage,<br/>set size, per-slice]
```

## Target architecture

```mermaid
flowchart LR
    UP[Catalog upload] --> API[Classifier API]
    API --> DEC{Set size}
    DEC -- 1 --> AUTO[Auto-tag]
    DEC -- more than 1 --> Q[Review queue<br/>shows the set]
    Q --> FIX[(Reviewer labels)]
    FIX --> MON[Coverage monitor<br/>per slice, weekly]
    MON -- coverage drops --> RECAL[Recalibrate T and q<br/>on fresh labels]
    RECAL --> API
```

## Request flow

```mermaid
sequenceDiagram
    autonumber
    participant C as Client
    participant A as API
    participant E as Encoder
    participant M as Calibrated model
    C->>A: POST /classify (image)
    A->>E: embed
    E-->>A: vector
    A->>M: logits / T, softmax, conformal set with q
    M-->>A: label, probability, set, decision
    A-->>C: auto_tag or review, coverage target
```

## Code map

| Path | Responsibility |
| --- | --- |
| `src/calibcv/calibration.py` | Softmax, temperature fitting, ECE with reliability table |
| `src/calibcv/conformal.py` | APS scores, calibration quantile, prediction sets |
| `src/calibcv/model.py` | Calibrated model with decision rule; save and load |
| `src/calibcv/data.py` | Sample and three-way split with slice columns |
| `src/calibcv/encoder.py` | SigLIP embeddings with cache |
| `src/calibcv/api.py` | `/classify` endpoint |
| `evals/run_eval.py` | Fits everything and reports overall and per-slice results |

## Failure modes

| Failure | Effect | Mitigation |
| --- | --- | --- |
| Distribution shift (new product types, new photo style) | Coverage and calibration silently degrade | Monitor coverage on reviewed items; recalibrate |
| Small slice under-covered | Guarantee fails for that group | Per-slice report; class- or slice-conditional conformal (v0.2) |
| Calibration split too small | Noisy q and T | Calibration split size reported; at least ~1,000 images |
| Over-large sets | Review load grows | Mean set size tracked as a cost metric |

## Scaling path

1. **v0.1 (now):** temperature scaling, LAC conformal sets (APS measured and rejected, see ADR 2), per-slice report, API.
2. **v0.2:** slice-conditional (Mondrian) conformal for under-covered slices.
3. **v0.3:** coverage monitoring on reviewed items and scheduled recalibration.
