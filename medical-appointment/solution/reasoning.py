"""Dual local entailment scorers plus conservative contradiction rules."""

import logging
import time

from .text_utils import explicit_question_conflict, polarity_conflict, quantity_conflict

logger = logging.getLogger(__name__)

PROMPT = (
    "Decide whether the medical consultation passage directly supports the entire claim. "
    "Answer YES only when every important detail is supported. A near-miss is NO: a different "
    "dose, duration, date, body side, medicine, test result, timing, frequency, or plan makes "
    "the claim false. Same topic is not enough.\n\n"
)


class _QAModel:
    def __init__(self, model_path, config):
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        kwargs = dict(local_files_only=True, use_safetensors=True)
        if config.device_type == "cuda":
            kwargs["dtype"] = torch.float16
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_path, **kwargs)
        self.model = self.model.to(config.torch_device).eval()
        self.answer_ids = []
        for answer in ("no", "yes"):
            ids = self.tokenizer.encode(answer, add_special_tokens=False)
            if len(ids) != 1:
                raise ValueError(f"Tokenizer for {model_path} does not encode {answer!r} as one token")
            self.answer_ids.append(ids[0])

    def score_batch(self, pairs):
        import torch

        if not pairs:
            return []
        prompts = [PROMPT + f"Passage: {p}\nQuestion: {q}\nAnswer:" for q, p in pairs]
        inputs = self.tokenizer(
            prompts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        ).to(self.config.torch_device)
        decoder_ids = torch.full(
            (len(pairs), 1),
            self.model.config.decoder_start_token_id,
            dtype=torch.long,
            device=self.config.torch_device,
        )
        with torch.inference_mode():
            logits = self.model(**inputs, decoder_input_ids=decoder_ids).logits[:, 0]
            probs = logits[:, self.answer_ids].softmax(dim=-1)[:, 1]
        return probs.float().cpu().tolist()


class Reasoner:
    def __init__(self, config):
        config.validate_device()
        import torch

        torch.set_num_threads(config.cpu_threads)
        self.config = config
        self.base = _QAModel(config.qa_base_model, config)
        self.large = _QAModel(config.qa_large_model, config) if config.dual_qa else None

    def warmup(self):
        pair = [("Is the dose 100 mg?", "Take 100 mg daily.")]
        self.base.score_batch(pair)
        if self.large is not None:
            self.large.score_batch(pair)

    def _score_model(self, model, pairs, deadline=None):
        results = [0.0] * len(pairs)
        if model is None:
            return results
        bs = self.config.qa_batch_size
        for offset in range(0, len(pairs), bs):
            if deadline is not None and time.perf_counter() >= deadline:
                break
            batch = pairs[offset : offset + bs]
            values = model.score_batch(batch)
            results[offset : offset + len(values)] = values
        return results

    def score_candidates(self, questions, retrieved, deadline=None):
        jobs, owners = [], []
        for qi, (question, candidates) in enumerate(zip(questions, retrieved)):
            for ci, candidate in enumerate(candidates):
                candidate["quantity_conflict"] = quantity_conflict(question, candidate["text"])
                candidate["polarity_conflict"] = polarity_conflict(question, candidate["text"])
                jobs.append((question, candidate["text"]))
                owners.append((qi, ci))

        base_scores = self._score_model(self.base, jobs, deadline)
        large_scores = self._score_model(self.large, jobs, deadline) if self.large else base_scores
        for (qi, ci), pb, pl in zip(owners, base_scores, large_scores):
            retrieved[qi][ci]["qa_base"] = float(pb)
            retrieved[qi][ci]["qa_large"] = float(pl)

    def score_context(self, questions, transcript_bundle, deadline=None):
        # Context is intentionally lightweight: only 10 questions x <=2 transcripts.
        pairs, owners = [], []
        for source, transcript in transcript_bundle["sources"].items():
            words = " ".join(seg["text"] for seg in transcript["segments"]).split()
            chunks = []
            size, stride = 250, 210
            for start in range(0, len(words), stride):
                chunk = " ".join(words[start : start + size])
                if chunk:
                    chunks.append(chunk)
                if start + size >= len(words):
                    break
            for qi, question in enumerate(questions):
                for chunk in chunks or [""]:
                    pairs.append((question, chunk))
                    owners.append((qi, source))

        base_scores = self._score_model(self.base, pairs, deadline)
        large_scores = self._score_model(self.large, pairs, deadline) if self.large else base_scores
        result = [dict(base=0.0, large=0.0) for _ in questions]
        for (qi, _source), pb, pl in zip(owners, base_scores, large_scores):
            result[qi]["base"] = max(result[qi]["base"], float(pb))
            result[qi]["large"] = max(result[qi]["large"], float(pl))
        return result


def resolve_explicit_conflicts(questions, response, diagnostics):
    for i in range(len(questions)):
        if not response.answers[i]:
            continue
        for j in range(i + 1, len(questions)):
            if not response.answers[j]:
                continue
            if not explicit_question_conflict(questions[i], questions[j]):
                continue
            pi = diagnostics[i].get("answer_probability", 0.0)
            pj = diagnostics[j].get("answer_probability", 0.0)
            loser = j if pi >= pj else i
            response.answers[loser] = False
            response.evidence_start[loser] = None
            response.evidence_end[loser] = None
            diagnostics[loser]["conflict_demoted"] = True
