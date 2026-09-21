# Medical appointment v3 — learned evidence selection

This version is intentionally different from v2. The two measured runs showed that the larger pipeline improved yes/no accuracy but hurt evidence localization. Since the official score weights temporal IoU more heavily than classification, v3 separates the two jobs and learns evidence selection from the supplied training spans.

## What changed

The pipeline now uses two complementary ASR transcripts and two complementary QA models:

```text
MP3
 ├─ Whisper base.en  ─┐
 └─ Whisper medium.en ├─ candidate pool
                      │   ├─ punctuation/neighbor windows
                      │   └─ query-anchored 1.8–8 s word windows
                      ↓
             lexical high-recall retrieval
                      ↓
       FLAN-T5-base + FLAN-T5-large scores
                      ↓
        learned evidence ranker (5-fold ensemble)
                      ↓
       learned yes/no calibrator + hard-negative rules
                      ↓
          boolean + precise audio evidence span
```

The learned models are tiny. They are trained only on features derived from the supplied 39 labeled conversations, with conversation-level 5-fold splits. No training conversation is used to make its own out-of-fold prediction when the training script chooses the final answer threshold and evidence padding.

The evidence model is trained on the actual temporal-IoU target. This is the main change: v1/v2 asked a generic language model to both understand the claim and implicitly pick the timestamp. v3 lets the language models judge support, while a separate ranker learns which candidate shape/source tends to align with the gold annotation.

## Colab setup

Run from the repository root, where `src/`, `data/`, and `solution/` are siblings.

```bash
python --version
```

Use the Python 3.12 environment you already created on Colab. Then:

```bash
/content/medical-env/bin/python -m pip install -r solution/requirements-cuda.txt
/content/medical-env/bin/python -m solution.download_models
```

The download step fetches:

- Systran/faster-whisper-base.en
- Systran/faster-whisper-medium.en
- google/flan-t5-base
- google/flan-t5-large

## One-time training step

This is new and important. Before serving v3, train the tiny evidence/answer ensemble from the supplied labeled data:

```bash
DEVICE=cuda:0 /content/medical-env/bin/python -m solution.train_ranker all
```

`all` has two stages:

1. `prepare`: transcribe the 39 training conversations, generate candidates, run both QA models, and cache label-free features plus the known training targets.
2. `fit`: train five small fold models and choose the answer threshold/padding by out-of-fold official score.

The expensive prepare stage is resumable. If Colab stops after 20 conversations, rerun the same command and it will continue from the cached files.

You can also run the stages separately:

```bash
DEVICE=cuda:0 /content/medical-env/bin/python -m solution.train_ranker prepare
DEVICE=cuda:0 /content/medical-env/bin/python -m solution.train_ranker fit
```

At the end you should see something like:

```text
5-fold out-of-fold estimate
  Accuracy:  ...
  Mean tIoU: ...
  Score:     ...
  threshold: ...
  padding:   ...
Saved ensemble to .../solution/models/ranker_ensemble.pt
```

Do not judge v3 before this file exists. Without it the server deliberately falls back to a heuristic so it can still run, but that is not the intended v3 system.

Check it:

```bash
ls -lh solution/models/ranker_ensemble.pt
```

## Start the server

```bash
pkill -f "uvicorn.*9054" || true

nohup env \
  PATH="/content/medical-env/bin:$PATH" \
  DEVICE=cuda:0 \
  REQUIRE_RANKER=1 \
  bash solution/run_cuda.sh \
  -m uvicorn solution.solution:app \
  --host 0.0.0.0 \
  --port 9054 \
  > /tmp/medical_v3.log 2>&1 &
```

Wait for startup and check:

```bash
tail -100 /tmp/medical_v3.log
curl http://127.0.0.1:9054/
```

Expected health response:

```json
{"ready":true,"ranker_loaded":true}
```

If `ranker_loaded` is false, do not run the competition evaluation yet; run the training step above.

## Evaluate locally before Cloudflare

Use the official evaluator against localhost:

```bash
/content/medical-env/bin/python -m src.evaluate \
  --url http://127.0.0.1:9054/predict \
  --verbose
```

You can also get richer v3 diagnostics directly:

```bash
DEVICE=cuda:0 /content/medical-env/bin/python -m solution.evaluate run --failures
```

Compare the final three values with the measured baselines:

```text
v1: Accuracy 0.867, mean tIoU 0.440, score 0.611
v2: Accuracy 0.887, mean tIoU 0.425, score 0.610
```

The goal of v3 is specifically to keep the classification gains while recovering and improving tIoU. The 5-fold out-of-fold score printed by `train_ranker fit` is the useful signal before you submit; it is much more informative than tuning directly on all 390 labels and quoting the same training score.

## Cloudflare

Only after localhost works:

```bash
cloudflared tunnel --url http://localhost:9054
```

Submit:

```text
https://<generated-name>.trycloudflare.com/predict
```

Keep Colab, Uvicorn, and cloudflared alive during evaluation.

## Useful switches

For ablation/testing:

```bash
DUAL_ASR=0          # base Whisper only
DUAL_QA=0           # FLAN-T5-base only
TOP_K=12            # fewer candidates / faster
REQUIRE_RANKER=1    # fail startup if trained ranker is missing
```

The default is the high-recall competition path: dual ASR + dual QA + learned ranker.
