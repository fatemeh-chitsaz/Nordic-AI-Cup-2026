"""High-recall candidate generation with clause windows and query-anchored word windows."""

import math
import re
from collections import Counter
from difflib import get_close_matches

from .text_utils import content_tokens, quantities


def flatten_words(segments):
    return [
        dict(word)
        for segment in segments
        for word in segment.get("words", [])
        if word.get("start") is not None and word.get("end") is not None
    ]


def evidence_units(segments):
    units, pending = [], []

    def finish():
        if not pending:
            return
        units.append(
            {
                "start": pending[0]["start"],
                "end": pending[-1]["end"],
                "text": " ".join(w["text"].strip() for w in pending).strip(),
                "words": [dict(w) for w in pending],
            }
        )
        pending.clear()

    for segment in segments:
        words = segment.get("words", [])
        if not words:
            finish()
            units.append(
                {
                    "start": segment["start"],
                    "end": segment["end"],
                    "text": segment["text"],
                    "words": [],
                }
            )
            continue
        for word in words:
            if pending and word["start"] - pending[-1]["end"] > 0.8:
                finish()
            pending.append(word)
            token = word["text"].strip()
            if re.search(r"[.!?;:,]$", token):
                finish()
            elif pending[-1]["end"] - pending[0]["start"] >= 8.0:
                finish()
    finish()
    return [u for u in units if u["end"] > u["start"] and u["text"]]


def make_clause_candidates(units, source, max_window=3):
    candidates = []
    for start in range(len(units)):
        for width in range(1, max_window + 1):
            group = units[start : start + width]
            if len(group) != width:
                break
            duration = group[-1]["end"] - group[0]["start"]
            if duration > 20 and width > 1:
                break
            candidates.append(
                {
                    "start": group[0]["start"],
                    "end": group[-1]["end"],
                    "text": " ".join(u["text"] for u in group),
                    "words": [w for u in group for w in u.get("words", [])],
                    "source": source,
                    "kind": "clause",
                    "width": width,
                }
            )
    return candidates


def _word_matches_question(question_tokens, word_text):
    tokens = content_tokens(word_text)
    if any(t in question_tokens for t in tokens):
        return True
    for token in tokens:
        if len(token) >= 6 and get_close_matches(token, question_tokens, n=1, cutoff=0.84):
            return True
    return False


def make_anchor_candidates(question, words, source):
    if not words:
        return []
    q_tokens = set(content_tokens(question))
    if not q_tokens:
        return []

    hits = [i for i, word in enumerate(words) if _word_matches_question(q_tokens, word["text"])]
    if not hits:
        return []

    clusters = []
    for index in hits:
        if not clusters or index - clusters[-1][-1] > 5:
            clusters.append([index])
        else:
            clusters[-1].append(index)

    durations = (1.8, 2.6, 3.4, 4.5, 6.0, 8.0)
    skews = (0.50, 0.38, 0.62)
    candidates = []
    for cluster in clusters:
        anchor_start = words[cluster[0]]["start"]
        anchor_end = words[cluster[-1]]["end"]
        center = (anchor_start + anchor_end) / 2.0
        for target_duration in durations:
            for left_fraction in skews:
                wanted_start = center - target_duration * left_fraction
                wanted_end = wanted_start + target_duration
                selected = [w for w in words if w["end"] > wanted_start and w["start"] < wanted_end]
                if not selected:
                    continue
                start, end = selected[0]["start"], selected[-1]["end"]
                if end <= start:
                    continue
                candidates.append(
                    {
                        "start": start,
                        "end": end,
                        "text": " ".join(w["text"].strip() for w in selected).strip(),
                        "words": [dict(w) for w in selected],
                        "source": source,
                        "kind": "anchor",
                        "width": len(selected),
                    }
                )
    return candidates


def _deduplicate(candidates):
    seen, result = set(), []
    for c in candidates:
        key = (c["source"], round(c["start"], 1), round(c["end"], 1), c["text"].lower())
        if key in seen:
            continue
        seen.add(key)
        result.append(c)
    return result


def build_candidates(question, transcript_bundle, max_window=3):
    all_candidates = []
    for source, transcript in transcript_bundle["sources"].items():
        segments = transcript["segments"]
        units = evidence_units(segments)
        words = flatten_words(segments)
        all_candidates.extend(make_clause_candidates(units, source, max_window=max_window))
        all_candidates.extend(make_anchor_candidates(question, words, source))
    return _deduplicate(all_candidates)


def _temporal_iou(a, b):
    inter = max(0.0, min(a["end"], b["end"]) - max(a["start"], b["start"]))
    union = max(a["end"], b["end"]) - min(a["start"], b["start"])
    return inter / max(union, 1e-6)


def _coverage(query, document):
    q, d = set(query), set(document)
    if not q:
        return 0.0, 0.0, 0.0
    inter = len(q & d)
    recall = inter / len(q)
    precision = inter / max(1, len(d))
    jaccard = inter / max(1, len(q | d))
    return recall, precision, jaccard


def score_candidates(question, candidates):
    query = content_tokens(question)
    documents = [content_tokens(c["text"]) for c in candidates]
    frequency = Counter(token for doc in documents for token in set(doc))
    expanded = set(query)
    for token in list(expanded):
        if token not in frequency and token.isalpha() and len(token) >= 6:
            match = get_close_matches(token, frequency.keys(), n=1, cutoff=0.80)
            if match:
                expanded.add(match[0])
    query = list(expanded)
    avg_len = sum(map(len, documents)) / max(1, len(documents))
    q_quantities = quantities(question)

    scored = []
    for candidate, document in zip(candidates, documents):
        counts = Counter(document)
        bm25 = 0.0
        for token in query:
            count = counts[token]
            if not count:
                continue
            idf = math.log(1 + (len(documents) - frequency[token] + 0.5) / (frequency[token] + 0.5))
            denominator = count + 1.2 * (0.25 + 0.75 * len(document) / max(1, avg_len))
            weight = 1.25 if re.fullmatch(r"\d+(?:\.\d+)?", token) else 1.0
            bm25 += weight * idf * count * 2.2 / denominator
        recall, precision, jaccard = _coverage(query, document)
        observed = quantities(candidate["text"])
        exact_quantity = len(q_quantities & observed)
        score = bm25 + 1.35 * recall + 1.5 * exact_quantity
        item = dict(candidate)
        item.update(
            retrieval_score=float(score),
            token_recall=float(recall),
            token_precision=float(precision),
            token_jaccard=float(jaccard),
            exact_quantity_match=float(exact_quantity > 0),
        )
        scored.append(item)

    # Cross-ASR temporal agreement is a strong reliability signal and is label-free.
    for item in scored:
        agreement = 0.0
        for other in scored:
            if item["source"] == other["source"]:
                continue
            agreement = max(agreement, _temporal_iou(item, other))
        item["source_agreement"] = float(agreement)

    scored.sort(key=lambda x: x["retrieval_score"], reverse=True)
    for rank, item in enumerate(scored):
        item["retrieval_rank"] = rank
    return scored


def retrieve(question, transcript_bundle, top_k=16, max_window=3):
    candidates = score_candidates(question, build_candidates(question, transcript_bundle, max_window))
    candidates = [c for c in candidates if c["retrieval_score"] > 0]
    if not candidates:
        return []

    selected, seen = [], set()

    def add(c):
        key = (c["source"], round(c["start"], 2), round(c["end"], 2))
        if key not in seen and len(selected) < top_k:
            selected.append(c)
            seen.add(key)

    # Preserve several different time regions, then preserve short/long variants within them.
    anchors = []
    for c in candidates:
        if all(_temporal_iou(c, a) < 0.55 for a in anchors):
            anchors.append(c)
        if len(anchors) >= 6:
            break

    for anchor in anchors:
        family = [c for c in candidates if _temporal_iou(c, anchor) >= 0.35]
        family.sort(key=lambda c: (-c["retrieval_score"], c["end"] - c["start"]))
        add(anchor)
        if family:
            add(min(family, key=lambda c: (c["end"] - c["start"], -c["retrieval_score"])))
            add(max(family, key=lambda c: (c["source_agreement"], c["token_recall"])))

    for c in candidates:
        add(c)
        if len(selected) >= top_k:
            break
    return selected
