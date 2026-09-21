"""Prepare labeled candidate features and train a 5-fold evidence/answer ensemble.

Run from the medical-appointment repository root:

    python -m solution.train_ranker all

The expensive `prepare` stage is resumable. `fit` is fast and can be repeated.
"""

import argparse
import base64
import hashlib
import json
import math
import random
from pathlib import Path

import torch
import torch.nn.functional as F

from src.utils import gold_evidence, group_questions_by_conversation, load_sample_audio, temporal_iou

from .asr import Transcriber
from .config import Config, ROOT
from .ranker import (
    CANDIDATE_FEATURE_NAMES,
    QUESTION_FEATURE_NAMES,
    CandidateNet,
    QuestionNet,
    candidate_features,
    question_features,
)
from .reasoning import Reasoner
from .retrieval import retrieve

CACHE_DIR = ROOT / "training_cache_v3"
MODEL_PATH = ROOT / "models" / "ranker_ensemble.pt"


def _fold(name, folds=5):
    return int(hashlib.sha256(name.encode()).hexdigest(), 16) % folds


def _safe_json(value):
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return 0.0
    return value


def prepare(config, cache_dir, force=False, limit=None):
    cache_dir.mkdir(parents=True, exist_ok=True)
    conversations = group_questions_by_conversation()
    if limit:
        conversations = conversations[:limit]

    transcriber = Transcriber(config)
    reasoner = Reasoner(config)
    transcriber.warmup()
    reasoner.warmup()

    for index, (name, rows) in enumerate(conversations, 1):
        path = cache_dir / f"{Path(name).stem}.json"
        if path.exists() and not force:
            print(f"[{index}/{len(conversations)}] cached {name}", flush=True)
            continue

        audio = load_sample_audio(name)
        transcript = transcriber.transcribe(audio)
        questions = [row["question"] for row in rows]
        retrieved = [
            retrieve(q, transcript, top_k=config.top_k, max_window=config.max_window)
            for q in questions
        ]
        contexts = reasoner.score_context(questions, transcript)
        reasoner.score_candidates(questions, retrieved)

        record = {"name": name, "questions": []}
        for row, candidates, context in zip(rows, retrieved, contexts):
            gold = gold_evidence(row)
            candidate_rows = []
            for candidate in candidates:
                span = (candidate["start"], candidate["end"])
                target = temporal_iou(gold, span) if gold is not None else 0.0
                candidate_rows.append(
                    {
                        "features": candidate_features(row["question"], candidate, context),
                        "target": float(target),
                        "start": float(candidate["start"]),
                        "end": float(candidate["end"]),
                        "quantity_conflict": bool(candidate.get("quantity_conflict", False)),
                        "polarity_conflict": bool(candidate.get("polarity_conflict", False)),
                        "source": candidate.get("source"),
                        "kind": candidate.get("kind"),
                    }
                )
            record["questions"].append(
                {
                    "question": row["question"],
                    "question_id": row.get("question_id"),
                    "question_type": row.get("question_type"),
                    "label": int(row["label"]),
                    "gold": list(gold) if gold is not None else None,
                    "question_features": question_features(row["question"], candidates, context),
                    "candidates": candidate_rows,
                }
            )

        path.write_text(json.dumps(record, default=_safe_json))
        print(
            f"[{index}/{len(conversations)}] prepared {name} "
            f"({sum(len(q['candidates']) for q in record['questions'])} candidates)",
            flush=True,
        )


def load_cache(cache_dir):
    records = []
    for path in sorted(cache_dir.glob("conversation_sample_*.json")):
        records.append(json.loads(path.read_text()))
    if not records:
        raise FileNotFoundError(
            f"No prepared feature files in {cache_dir}. Run `python -m solution.train_ranker prepare`."
        )
    return records


def _stats(x):
    mean = x.mean(0)
    std = x.std(0, unbiased=False).clamp_min(1e-4)
    return mean, std


def _candidate_dataset(records):
    xs, ys, groups = [], [], []
    gid = 0
    for record in records:
        for q in record["questions"]:
            indices = []
            for c in q["candidates"]:
                xs.append(c["features"])
                ys.append(c["target"])
                indices.append(len(xs) - 1)
            if indices:
                groups.append(indices)
            gid += 1
    return (
        torch.tensor(xs, dtype=torch.float32),
        torch.tensor(ys, dtype=torch.float32),
        groups,
    )


def _question_dataset(records):
    xs, ys = [], []
    for record in records:
        for q in record["questions"]:
            xs.append(q["question_features"])
            ys.append(q["label"])
    return torch.tensor(xs, dtype=torch.float32), torch.tensor(ys, dtype=torch.float32)


def train_candidate_model(records, seed):
    random.seed(seed)
    torch.manual_seed(seed)
    x, y, groups = _candidate_dataset(records)
    mean, std = _stats(x)
    xn = (x - mean) / std
    model = CandidateNet(x.shape[1])
    optimizer = torch.optim.AdamW(model.parameters(), lr=2.5e-3, weight_decay=2e-3)

    for epoch in range(280):
        model.train()
        logits = model(xn)
        weights = 0.35 + 3.5 * y
        point = F.binary_cross_entropy_with_logits(logits, y, weight=weights)

        pair_losses = []
        for indices in groups:
            if len(indices) < 2:
                continue
            target_values = y[indices]
            best_local = int(torch.argmax(target_values))
            best_index = indices[best_local]
            best_target = y[best_index]
            if best_target < 0.05:
                continue
            # Compare the best span with up to four materially worse spans.
            ordered = sorted(indices, key=lambda i: float(y[i]))
            for other in ordered[:4]:
                gap = best_target - y[other]
                if gap <= 0.05:
                    continue
                pair_losses.append(F.softplus(-(logits[best_index] - logits[other])) * gap)
        pair = torch.stack(pair_losses).mean() if pair_losses else torch.tensor(0.0)
        loss = point + 0.55 * pair
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    model.eval()
    return model, mean, std


def train_question_model(records, seed):
    random.seed(seed)
    torch.manual_seed(seed)
    x, y = _question_dataset(records)
    mean, std = _stats(x)
    xn = (x - mean) / std
    model = QuestionNet(x.shape[1])
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.5e-2, weight_decay=2.5e-2)
    for _ in range(450):
        logits = model(xn)
        loss = F.binary_cross_entropy_with_logits(logits, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    model.eval()
    return model, mean, std


def _predict_candidate(model, mean, std, rows):
    if not rows:
        return []
    x = torch.tensor([r["features"] for r in rows], dtype=torch.float32)
    with torch.inference_mode():
        return torch.sigmoid(model((x - mean) / std)).tolist()


def _predict_question(model, mean, std, features):
    x = torch.tensor([features], dtype=torch.float32)
    with torch.inference_mode():
        return float(torch.sigmoid(model((x - mean) / std))[0])


def _evaluate_oof(records, fold_models):
    predictions = []
    for record in records:
        fold = _fold(record["name"])
        fm = fold_models[fold]
        for q in record["questions"]:
            c_scores = _predict_candidate(
                fm["candidate"], fm["candidate_mean"], fm["candidate_std"], q["candidates"]
            )
            safe = [
                (score, c)
                for score, c in zip(c_scores, q["candidates"])
                if not c["quantity_conflict"] and not c["polarity_conflict"]
            ]
            best = max(safe, key=lambda x: x[0])[1] if safe else None
            p = _predict_question(
                fm["question"], fm["question_mean"], fm["question_std"], q["question_features"]
            )
            predictions.append((q, p, best))

    best_result = None
    for threshold_i in range(38, 73):
        threshold = threshold_i / 100.0
        for padding_i in range(0, 7):
            padding = padding_i * 0.05
            correct = 0
            tiou_sum = 0.0
            positives = 0
            for q, p, best in predictions:
                answer = bool(best is not None and p >= threshold)
                label = bool(q["label"])
                correct += int(answer == label)
                if label:
                    positives += 1
                    if answer and best is not None:
                        gs, ge = q["gold"]
                        ps = max(0.0, best["start"] - padding)
                        pe = best["end"] + padding
                        inter = max(0.0, min(ge, pe) - max(gs, ps))
                        union = max(ge, pe) - min(gs, ps)
                        tiou_sum += inter / max(union, 1e-9)
            accuracy = correct / len(predictions)
            mean_tiou = tiou_sum / max(1, positives)
            score = 0.4 * accuracy + 0.6 * mean_tiou
            current = (score, accuracy, mean_tiou, threshold, padding)
            if best_result is None or current > best_result:
                best_result = current
    return best_result


def fit(cache_dir, output_path):
    records = load_cache(cache_dir)
    folds = []
    fold_models = {}
    for fold in range(5):
        train_records = [r for r in records if _fold(r["name"]) != fold]
        held = [r for r in records if _fold(r["name"]) == fold]
        if not held:
            raise RuntimeError(f"Fold {fold} is empty; expected all 39 conversations in the cache")
        candidate, cm, cs = train_candidate_model(train_records, seed=1000 + fold)
        question, qm, qs = train_question_model(train_records, seed=2000 + fold)
        fold_models[fold] = {
            "candidate": candidate,
            "candidate_mean": cm,
            "candidate_std": cs,
            "question": question,
            "question_mean": qm,
            "question_std": qs,
        }
        folds.append(
            {
                "candidate_state": {k: v.detach().cpu() for k, v in candidate.state_dict().items()},
                "candidate_mean": cm.cpu(),
                "candidate_std": cs.cpu(),
                "question_state": {k: v.detach().cpu() for k, v in question.state_dict().items()},
                "question_mean": qm.cpu(),
                "question_std": qs.cpu(),
            }
        )
        print(f"trained fold {fold}: {len(train_records)} train / {len(held)} held-out conversations")

    score, accuracy, mean_tiou, threshold, padding = _evaluate_oof(records, fold_models)
    payload = {
        "version": 3,
        "candidate_feature_names": CANDIDATE_FEATURE_NAMES,
        "question_feature_names": QUESTION_FEATURE_NAMES,
        "answer_threshold": threshold,
        "padding": padding,
        "oof_score": score,
        "oof_accuracy": accuracy,
        "oof_mean_tiou": mean_tiou,
        "folds": folds,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output_path)
    print("\n5-fold out-of-fold estimate")
    print(f"  Accuracy:  {accuracy:.3f}")
    print(f"  Mean tIoU: {mean_tiou:.3f}")
    print(f"  Score:     {score:.3f}")
    print(f"  threshold: {threshold:.2f}")
    print(f"  padding:   {padding:.2f}s")
    print(f"Saved ensemble to {output_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["prepare", "fit", "all"])
    parser.add_argument("--cache", type=Path, default=CACHE_DIR)
    parser.add_argument("--output", type=Path, default=MODEL_PATH)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    config = Config.from_env()
    if args.command in {"prepare", "all"}:
        prepare(config, args.cache, force=args.force, limit=args.limit)
    if args.command in {"fit", "all"}:
        fit(args.cache, args.output)


if __name__ == "__main__":
    main()
