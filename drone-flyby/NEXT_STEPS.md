# What changed from the original baseline

Two files were added or rewritten on top of the original kit. Everything
else (`api.py`, `dtos.py`, `utils.py`, `local_evaluator.py`, `visualize.py`)
is untouched.

## `example.py` — memory + a systematic sweep + optional fine-tuned model

The original baseline was stateless: it only ever reported what was visible
in the current crop, even though a response has to cover the whole source
frame. It also swept sideways at a fixed height and never actually moved
between rows (the comment said it would; the code didn't).

This version:

- Keeps a `SequenceMemory` per `sequence_id`. Every detection is merged into
  an existing track by class + IoU, or starts a new one. Every response
  reports the *entire* current set of tracks, not just this frame's crop —
  so something seen once keeps being reported even while the camera is
  looking elsewhere.
- Tiles the whole 3840x2160 frame into a 4x4 grid of non-overlapping
  Level-2 (960x540) windows and sweeps them in a boustrophedon route,
  respecting `camera_constraints` at every step (verified against
  `local_evaluator.Camera` with zero rejected moves over an 80-frame
  synthetic run). The route loops, so a second pass re-confirms weak
  detections instead of losing coverage.
- `detect()` looks for fine-tuned weights at `weights/best.pt` and uses them
  if present; otherwise it falls back to the original placeholder edge
  detector, unchanged. The project runs out of the box either way.

## `train_yolo.py` — new

Converts the supplied `src/<scene>/` frames into a YOLO-format dataset by
sampling random *legal* camera views (matching `camera_constraints`'
resolution levels and center bounds) and cropping/downsampling exactly the
way `local_evaluator.render_view` does, then keeping only the ground-truth
boxes still meaningfully visible in each crop, re-expressed in that crop's
own normalized coordinates. This matters because training on the raw 4K
frames would teach a different problem than the one the model is actually
asked to solve.

It then fine-tunes a COCO-pretrained YOLOv8 (`ultralytics`) on that dataset
and copies the result to `weights/best.pt`, where `example.py` picks it up
automatically.

**This has not been run against the real 25-frame Helsinki scene** — only
against a small synthetic scene built for testing, because the actual
`src/helsinki/images` and `src/helsinki/annotations` were never available
in this session (only the `.py` source files were shared). The coordinate
math (crop generation, visibility thresholding, YOLO label writing) was
unit-tested directly and is correct; the training run itself needs to
happen wherever the real data lives — your own machine or server.

### To actually train

```cmd
pip install -r requirements.txt
python train_yolo.py                       # builds yolo_dataset/, then trains
python local_evaluator.py                  # compare against the baseline
python local_evaluator.py --realtime       # check it still fits the time budget
```

Useful flags: `--crops-per-frame` (default 16), `--epochs` (default 60),
`--model-size n|s|m` (default `n`, the smallest/fastest), `--skip-training`
to only inspect the generated dataset first.

## Actual results (this was run for real)

The real `src/helsinki/` data was supplied and used. `weights/best.pt` in
this package is a real fine-tuned YOLOv8n checkpoint, not a placeholder.

**Per-crop validation during training** (406 train / 106 val crops built by
`train_yolo.py`, matching the real camera protocol) reached, at its best
checkpoint (epoch 16 of a partial run):

| Metric | Value |
|---|---|
| mAP@0.50 | 0.753 |
| mAP@0.50-95 | 0.445 |
| Precision | 0.636 |
| Recall | 0.738 |

This run only got to ~16 epochs, done as repeated 1-epoch continuations
(explained below) rather than one continuous run — the loss curve was still
trending upward when it stopped. More epochs would very likely help further.

**Full-sequence `local_evaluator.py` score on the 25-frame Helsinki scene**
is much lower — **COCO mAP@0.50 ≈ 0.04** (offline and `--realtime` alike) —
despite the strong per-crop numbers above. This is not a contradiction, and
not the number to extrapolate from:

- The sweep (`example.py`) tiles the frame into 16 Level-2 windows and takes
  roughly 2 frames to move between adjacent tiles. A full lap takes ~32
  frames. **The Helsinki scene is only 25 frames long** — the sweep never
  completes even one lap before the sequence ends, so most of the frame is
  never seen in sharp detail and those objects are simply never reported.
- The real validation and evaluation sequences are ~250 frames — 10x longer
  — which is roughly 7-8 full sweep laps. The memory-plus-sweep design was
  built for that regime, not a 25-frame clip. Expect the real score on the
  actual validation sequence to look nothing like this number.
- Confirmed no timing problems: `--realtime` only dropped 2 of 25 frames,
  with a ~87ms mean round trip — nowhere near the 3333ms budget.

**One real bug this run caught and fixed:** the first request timed out
(no answer within 3333ms) before this fix, because the fine-tuned model was
loaded lazily on the first request. `example.py` now loads and runs the
model once at import time (`_warm_up()`), so `api.py` pays that cost during
startup instead of on frame 0. Confirmed fixed: 0 timeouts after the fix.

### Why training happened in 1-epoch steps, and what that cost

This sandbox has 1 CPU core and no GPU. A background training process also
does not survive idle time between conversation turns here, so a single
long `model.train(epochs=25, ...)` call wasn't reliable. Training instead
proceeded as ~16 separate single-epoch continuations, each reloading the
previous checkpoint. Each restart resets the learning-rate schedule, which
is why the metrics above bounced around between rounds (e.g. epoch 13 dipped
before epoch 14 and 16 set new highs) instead of improving smoothly. A
single continuous run on real hardware (a GPU, or even just an
uninterrupted CPU run) with `python train_yolo.py --epochs 60` should do
noticeably better than this checkpoint, with a smooth cosine learning-rate
decay across the whole run instead of ~16 resets.


### License note

`ultralytics` (YOLOv8 code and pretrained weights) is AGPL-3.0. Confirm
that's acceptable for your entry before depending on it; if not, the same
`train_yolo.py` dataset (`data.yaml` + YOLO-format labels) works with any
other detector that accepts that format.
