# v2 diagnostic cases

These images come from `results/v2_test_evaluation/errors/` and are kept as
typical cases for the report.  They should not be described as pure model
failures without discussing the source annotations:

- `cup_cup4_000086.jpg`: the source JSON contains two nearly identical `cup`
  boxes (IoU about 0.993) for one visible cup, so the model's one detection is
  counted as a duplicate-label miss.
- `laptop_laptop_000050.jpg`: a mouse is visible at the right edge but is not
  annotated in the source JSON; the model's low-confidence mouse detection is
  therefore counted as a false positive by the strict evaluator.

For the final 20-object acceptance test, review all annotations before
computing the score.
