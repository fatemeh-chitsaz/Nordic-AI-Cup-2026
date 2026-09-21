# Improved solution

The supplied `weights/best.pt` is kept, but the end-to-end strategy has been
changed substantially.

## What changed

- Replaced static bbox memory with motion-compensated tracking.
- Estimates source-frame motion with ORB matches and affine RANSAC.
- Associates detections spatially even when the predicted class changes, then
  accumulates class evidence over time.
- Expires weak single-frame tracks quickly and confirmed tracks after a short
  camera excursion. Old boxes are not emitted forever.
- Replaced the 4x4 Level-2 sweep with:
  1. a six-view Level-1 full-frame bootstrap;
  2. a fast Level-1 scan of the top entry band;
  3. one-frame Level-2 confirmation for uncertain candidates.
- Raised negative-crop retention during training and matched crop sampling to
  the new camera policy.
- Uses a chronological validation tail and prevents balancing crops from ever
  using validation frames.

## Run

```bash
pip install -r requirements.txt
python api.py
```

In a second terminal, when the official `src/` scene folder is available:

```bash
python local_evaluator.py
```

Run the included deterministic tests:

```bash
python -m unittest -v test_solution.py
```

Rebuild the dataset and fine-tune:

```bash
python train_yolo.py --crops-per-frame 24 --epochs 80 --model-size n
```

For final competition training, compare changes by submitting one at a time.
The 25 Helsinki frames cannot provide a trustworthy cross-scene score; the
competition validation sequence is the meaningful A/B test.
