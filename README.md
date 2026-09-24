# Nordic AI Cup 2026 — Competition Solutions

My working repository for the **Nordic AI Cup 2026**, with competition baselines plus the solutions I developed during the event.



## What I worked on

### 1. Drone Flyby — detection + active camera control

The drone challenge is not only object detection: the model receives a 960×540 camera crop from a 3840×2160 source frame and must also decide where the camera should look next.

My solution adds:

- **sequence memory** so detections persist across frames;
- a **systematic Level-2 camera sweep** over a 4×4 grid;
- IoU-based track merging;
- a **YOLOv8n fine-tuning pipeline** that creates legal camera crops matching the evaluation protocol;
- model warm-up before the first request to avoid inference-time startup latency;
- local tests and Docker deployment support.

### Drone validation results

The included fine-tuned checkpoint reached the following best per-crop validation metrics during the recorded run:

| Metric | Value |
|---|---:|
| mAP@0.50 | **0.753** |
| mAP@0.50–0.95 | **0.445** |
| Precision | **0.636** |
| Recall | **0.738** |

On the supplied 25-frame Helsinki sequence, the full-sequence score was much lower (~0.04 mAP@0.50) because the 4×4 sweep requires roughly 32 frames to complete one lap. The repository documents this limitation instead of treating the short-sequence number as representative of the longer evaluation setting.

See:

- `drone-flyby/SOLUTION_NOTES.md`
- `drone-flyby/NEXT_STEPS.md`
- `drone-flyby/train_yolo.py`
- `drone-flyby/test_solution.py`

---

### 2. Medical Appointment — ASR + QA + learned evidence ranking

The medical task requires both:

1. answering yes/no questions about a conversation; and
2. returning the **precise audio evidence span** supporting positive answers.

My later solution separates these objectives instead of asking one language model to do everything.

```text
MP3
 ├─ Whisper base.en
 └─ Whisper medium.en
          ↓
candidate evidence windows
          ↓
lexical high-recall retrieval
          ↓
FLAN-T5-base + FLAN-T5-large
          ↓
5-fold learned evidence ranker
          ↓
yes/no calibration + hard-negative rules
          ↓
boolean answer + audio evidence span
```

The learned ranker is trained from features derived from the supplied labeled conversations using **conversation-level five-fold splits**.

Measured earlier baselines documented in the project:

| Version | Accuracy | Mean temporal IoU | Score |
|---|---:|---:|---:|
| v1 | 0.867 | 0.440 | 0.611 |
| v2 | 0.887 | 0.425 | 0.610 |

The v3 design focuses specifically on recovering evidence-localization quality while preserving classification improvements.

See `medical-appointment/solution/README.md` for training, evaluation, CUDA, and deployment instructions.

---

### 3. Survival Simulator

The original challenge materials are retained in `survival-simulator/`. This repository does **not** present a completed custom survival-simulator solution as a finished result.

## Repository structure

```text
Nordic-AI-Cup-2026/
├── drone-flyby/
│   ├── example.py
│   ├── train_yolo.py
│   ├── test_solution.py
│   ├── SOLUTION_NOTES.md
│   └── weights/
├── medical-appointment/
│   ├── src/
│   └── solution/
│       ├── asr.py
│       ├── retrieval.py
│       ├── reasoning.py
│       ├── ranker.py
│       ├── train_ranker.py
│       └── solution.py
└── survival-simulator/
```


