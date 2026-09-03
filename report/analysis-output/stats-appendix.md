# Statistical Appendix

## Unit of Analysis
- Training metrics: one validation pass per epoch from one deterministic run with seed 42.
- Frame diagnostic: annotated object instances across 87 test frames.
- Physical-object acceptance unit: not yet collected; it must be a distinct real object, not a video frame.

## Descriptive Results

| Quantity | Value |
|---|---:|
| Training images | 421 |
| Validation images | 87 |
| Test images | 87 |
| Training annotations | 506 |
| Validation annotations | 104 |
| Test annotations | 104 |
| Best validation mAP@0.5:0.95 | 0.98571 (epoch 76) |
| Final validation precision | 0.99417 |
| Final validation recall | 0.99405 |
| Frame diagnostic matched objects | 103/104 |
| Frame diagnostic precision | 100.00% |
| Frame diagnostic recall | 99.04% |
| False negatives / false positives | 1 / 0 |

## Per-Class Frame Diagnostic

| Class | Ground truth | Correct | False negative | Recall |
|---|---:|---:|---:|---:|
| mouse | 28 | 28 | 0 | 100.00% |
| laptop | 17 | 17 | 0 | 100.00% |
| cup | 42 | 41 | 1 | 97.62% |
| phone | 17 | 17 | 0 | 100.00% |

## Inferential Statistics
No inferential test is reported. There is only one training run, and test frames are correlated within source videos. Treating frames as independent samples would underestimate uncertainty.

## Acceptance Calculation to Complete
For at least 20 distinct physical objects:

`accuracy = correctly recognized objects / tested objects * 100%`

The acceptance threshold is 80%, meaning at least 16 correct results out of 20 tests.
