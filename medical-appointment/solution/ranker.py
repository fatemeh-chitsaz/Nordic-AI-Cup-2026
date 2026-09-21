"""Learned candidate ranker and question calibrator trained only on supplied training labels."""

from pathlib import Path
import math
import re

import torch
import torch.nn as nn

from .text_utils import content_tokens, quantities

CANDIDATE_FEATURE_NAMES = [
    "qa_base",
    "qa_large",
    "retrieval_log",
    "token_recall",
    "token_precision",
    "token_jaccard",
    "exact_quantity_match",
    "quantity_conflict",
    "polarity_conflict",
    "duration",
    "duration_delta_3",
    "word_count_log",
    "source_base",
    "source_medium",
    "kind_clause",
    "kind_anchor",
    "source_agreement",
    "retrieval_rank_inv",
    "context_base",
    "context_large",
    "question_has_quantity",
    "question_has_negation",
]

QUESTION_FEATURE_NAMES = [
    "max_qa_base",
    "max_qa_large",
    "second_qa_base",
    "second_qa_large",
    "margin_qa_base",
    "margin_qa_large",
    "max_retrieval_log",
    "max_token_recall",
    "max_source_agreement",
    "context_base",
    "context_large",
    "safe_fraction",
    "strong_base_count",
    "strong_large_count",
    "exact_quantity_any",
    "question_has_quantity",
    "question_has_negation",
    "question_token_count_log",
]


def _question_flags(question):
    lower = question.lower()
    has_quantity = float(bool(quantities(question)))
    has_negation = float(bool(re.search(r"\b(no|not|never|without|didn.t|isn.t|wasn.t|weren.t)\b", lower)))
    return has_quantity, has_negation


def candidate_features(question, candidate, context):
    duration = max(0.01, candidate["end"] - candidate["start"])
    q_has_quantity, q_has_negation = _question_flags(question)
    word_count = max(1, len(candidate.get("words", [])) or len(candidate["text"].split()))
    return [
        float(candidate.get("qa_base", 0.0)),
        float(candidate.get("qa_large", 0.0)),
        math.log1p(max(0.0, candidate.get("retrieval_score", 0.0))),
        float(candidate.get("token_recall", 0.0)),
        float(candidate.get("token_precision", 0.0)),
        float(candidate.get("token_jaccard", 0.0)),
        float(candidate.get("exact_quantity_match", 0.0)),
        float(candidate.get("quantity_conflict", False)),
        float(candidate.get("polarity_conflict", False)),
        float(min(duration, 20.0) / 10.0),
        float(min(abs(duration - 3.0), 12.0) / 10.0),
        math.log1p(word_count) / 5.0,
        float(candidate.get("source") == "base"),
        float(candidate.get("source") == "medium"),
        float(candidate.get("kind") == "clause"),
        float(candidate.get("kind") == "anchor"),
        float(candidate.get("source_agreement", 0.0)),
        1.0 / (1.0 + float(candidate.get("retrieval_rank", 0))),
        float(context.get("base", 0.0)),
        float(context.get("large", 0.0)),
        q_has_quantity,
        q_has_negation,
    ]


def question_features(question, candidates, context):
    safe = [c for c in candidates if not c.get("quantity_conflict") and not c.get("polarity_conflict")]
    source = safe or candidates
    pb = sorted((float(c.get("qa_base", 0.0)) for c in source), reverse=True)
    pl = sorted((float(c.get("qa_large", 0.0)) for c in source), reverse=True)
    p1b, p2b = (pb + [0.0, 0.0])[:2]
    p1l, p2l = (pl + [0.0, 0.0])[:2]
    q_has_quantity, q_has_negation = _question_flags(question)
    return [
        p1b,
        p1l,
        p2b,
        p2l,
        p1b - p2b,
        p1l - p2l,
        max([math.log1p(max(0.0, c.get("retrieval_score", 0.0))) for c in source] or [0.0]),
        max([float(c.get("token_recall", 0.0)) for c in source] or [0.0]),
        max([float(c.get("source_agreement", 0.0)) for c in source] or [0.0]),
        float(context.get("base", 0.0)),
        float(context.get("large", 0.0)),
        len(safe) / max(1, len(candidates)),
        min(6, sum(x >= 0.55 for x in pb)) / 6.0,
        min(6, sum(x >= 0.55 for x in pl)) / 6.0,
        float(any(c.get("exact_quantity_match", 0.0) for c in source)),
        q_has_quantity,
        q_has_negation,
        math.log1p(len(content_tokens(question))) / 4.0,
    ]


class CandidateNet(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, 24),
            nn.ReLU(),
            nn.Dropout(0.08),
            nn.Linear(24, 12),
            nn.ReLU(),
            nn.Linear(12, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


class QuestionNet(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.linear = nn.Linear(dim, 1)

    def forward(self, x):
        return self.linear(x).squeeze(-1)


def _normalize(x, mean, std):
    return (x - mean) / std.clamp_min(1e-6)


class LearnedEnsemble:
    def __init__(self, payload, device="cpu"):
        self.device = torch.device(device)
        self.answer_threshold = float(payload.get("answer_threshold", 0.58))
        self.padding = float(payload.get("padding", 0.05))
        self.candidate_models = []
        self.question_models = []
        for fold in payload["folds"]:
            c = CandidateNet(len(CANDIDATE_FEATURE_NAMES)).to(self.device)
            c.load_state_dict(fold["candidate_state"])
            c.eval()
            q = QuestionNet(len(QUESTION_FEATURE_NAMES)).to(self.device)
            q.load_state_dict(fold["question_state"])
            q.eval()
            self.candidate_models.append(
                (c, fold["candidate_mean"].to(self.device), fold["candidate_std"].to(self.device))
            )
            self.question_models.append(
                (q, fold["question_mean"].to(self.device), fold["question_std"].to(self.device))
            )

    @classmethod
    def load(cls, path, device="cpu"):
        path = Path(path)
        if not path.exists():
            return None
        payload = torch.load(path, map_location="cpu", weights_only=False)
        return cls(payload, device=device)

    def candidate_scores(self, feature_rows):
        if not feature_rows:
            return []
        x = torch.tensor(feature_rows, dtype=torch.float32, device=self.device)
        outputs = []
        with torch.inference_mode():
            for model, mean, std in self.candidate_models:
                outputs.append(torch.sigmoid(model(_normalize(x, mean, std))))
        return torch.stack(outputs).mean(0).cpu().tolist()

    def answer_probability(self, features):
        x = torch.tensor([features], dtype=torch.float32, device=self.device)
        outputs = []
        with torch.inference_mode():
            for model, mean, std in self.question_models:
                outputs.append(torch.sigmoid(model(_normalize(x, mean, std)))[0])
        return float(torch.stack(outputs).mean().cpu())


def fallback_candidate_score(candidate):
    if candidate.get("quantity_conflict") or candidate.get("polarity_conflict"):
        return 0.0
    pb = float(candidate.get("qa_base", 0.0))
    pl = float(candidate.get("qa_large", 0.0))
    agreement = float(candidate.get("source_agreement", 0.0))
    recall = float(candidate.get("token_recall", 0.0))
    duration = candidate["end"] - candidate["start"]
    prior = max(0.0, 1.0 - abs(duration - 3.0) / 8.0)
    return 0.34 * pb + 0.46 * pl + 0.08 * agreement + 0.08 * recall + 0.04 * prior


def fallback_answer_probability(question, candidates, context):
    safe = [c for c in candidates if not c.get("quantity_conflict") and not c.get("polarity_conflict")]
    if not safe:
        return 0.0
    local = max(fallback_candidate_score(c) for c in safe)
    ctx = 0.4 * context.get("base", 0.0) + 0.6 * context.get("large", 0.0)
    return 0.78 * local + 0.22 * ctx
