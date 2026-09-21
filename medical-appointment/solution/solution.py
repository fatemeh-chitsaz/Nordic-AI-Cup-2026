"""Competition API using dual ASR, dual QA, and a learned evidence-selection ensemble."""

import base64
import logging
import threading
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI

from src.dtos import ASRQuestionRequestDto, ASRQuestionResponseDto
from src.utils import validate_response

from .asr import Transcriber
from .config import Config
from .ranker import (
    LearnedEnsemble,
    candidate_features,
    fallback_answer_probability,
    fallback_candidate_score,
    question_features,
)
from .reasoning import Reasoner, resolve_explicit_conflicts
from .retrieval import retrieve

logger = logging.getLogger(__name__)


def empty_response(count):
    return ASRQuestionResponseDto(
        answers=[False] * count,
        evidence_start=[None] * count,
        evidence_end=[None] * count,
    )


class Solution:
    def __init__(self, config=None, load_asr=True):
        self.config = config or Config.from_env()
        self.transcriber = Transcriber(self.config) if load_asr else None
        self.reasoner = Reasoner(self.config)
        self.ensemble = LearnedEnsemble.load(self.config.ranker_model, device="cpu")
        if self.config.require_ranker and self.ensemble is None:
            raise FileNotFoundError(
                f"Missing learned ranker at {self.config.ranker_model}. "
                "Run `python -m solution.train_ranker all` first."
            )
        if self.ensemble is None:
            logger.warning(
                "No learned ranker found at %s; using heuristic fallback. "
                "For the v3 path, run `python -m solution.train_ranker all`.",
                self.config.ranker_model,
            )
        self.lock = threading.Lock()

    def warmup(self):
        if self.transcriber is not None:
            self.transcriber.warmup()
        self.reasoner.warmup()

    def answer_transcript(self, questions, transcript_bundle, deadline=None):
        started = time.perf_counter()
        retrieved = []
        for question in questions:
            try:
                retrieved.append(
                    retrieve(
                        question,
                        transcript_bundle,
                        top_k=self.config.top_k,
                        max_window=self.config.max_window,
                    )
                )
            except Exception:
                logger.exception("Retrieval failed for %s", question)
                retrieved.append([])
        retrieval_time = time.perf_counter() - started

        qa_started = time.perf_counter()
        contexts = self.reasoner.score_context(questions, transcript_bundle, deadline)
        self.reasoner.score_candidates(questions, retrieved, deadline)

        response = empty_response(len(questions))
        diagnostics = []
        duration = transcript_bundle["duration"]

        for index, (question, candidates, context) in enumerate(zip(questions, retrieved, contexts)):
            try:
                safe = [
                    c
                    for c in candidates
                    if not c.get("quantity_conflict", False)
                    and not c.get("polarity_conflict", False)
                ]

                if self.ensemble is not None:
                    feature_rows = [candidate_features(question, c, context) for c in candidates]
                    learned_scores = self.ensemble.candidate_scores(feature_rows)
                    for candidate, score in zip(candidates, learned_scores):
                        candidate["learned_score"] = float(score)
                    best = max(safe, key=lambda c: c.get("learned_score", 0.0), default=None)
                    answer_probability = self.ensemble.answer_probability(
                        question_features(question, candidates, context)
                    )
                    threshold = self.ensemble.answer_threshold
                    padding = self.ensemble.padding
                else:
                    for candidate in candidates:
                        candidate["learned_score"] = fallback_candidate_score(candidate)
                    best = max(safe, key=lambda c: c.get("learned_score", 0.0), default=None)
                    answer_probability = fallback_answer_probability(question, candidates, context)
                    threshold = self.config.fallback_answer_threshold
                    padding = self.config.fallback_padding

                answer = bool(best is not None and answer_probability >= threshold)
                response.answers[index] = answer
                if answer:
                    start = max(0.0, best["start"] - padding)
                    end = min(duration, best["end"] + padding)
                    if end > start:
                        response.evidence_start[index] = round(start, 3)
                        response.evidence_end[index] = round(end, 3)

                diagnostics.append(
                    {
                        "selected": best,
                        "candidates": candidates,
                        "context": context,
                        "answer_probability": answer_probability,
                        "answer_threshold": threshold,
                    }
                )
            except Exception:
                logger.exception("Answer selection failed for question %d", index)
                diagnostics.append(
                    {
                        "selected": None,
                        "candidates": candidates,
                        "context": context,
                        "answer_probability": 0.0,
                    }
                )

        resolve_explicit_conflicts(questions, response, diagnostics)
        timings = {"retrieval": retrieval_time, "qa": time.perf_counter() - qa_started}
        validate_response(response, len(questions))
        return response, diagnostics, timings

    def predict_with_details(self, request):
        started = time.perf_counter()
        response = empty_response(len(request.questions))
        details = {"questions": [], "timings": {}, "error": None}
        if not request.questions:
            return response, details
        try:
            audio_bytes = base64.b64decode(request.audio_base64, validate=True)
            base64_time = time.perf_counter() - started
            acquired = self.lock.acquire(timeout=max(0.0, 55.0 - (time.perf_counter() - started)))
            if not acquired:
                raise TimeoutError("Another request is still using the models")
            try:
                transcript_bundle = self.transcriber.transcribe(audio_bytes)
                response, diagnostics, timings = self.answer_transcript(
                    request.questions,
                    transcript_bundle,
                    deadline=started + 55.0,
                )
            finally:
                self.lock.release()
            details.update(transcript=transcript_bundle, questions=diagnostics)
            details["timings"] = dict(transcript_bundle["timings"], **timings, base64=base64_time)
        except Exception as error:
            logger.exception("Conversation failed; returning correctly sized fallback")
            details["error"] = f"{type(error).__name__}: {error}"
        details["timings"]["total"] = time.perf_counter() - started
        logger.info("%s timing=%s", request.audio_filename, details["timings"])
        return response, details


_solution = None


def load_solution():
    global _solution
    if _solution is None:
        model = Solution()
        model.warmup()
        _solution = model
    return _solution


def predict(request):
    if _solution is None:
        raise RuntimeError("Call load_solution() at server startup, before accepting requests")
    return _solution.predict_with_details(request)[0]


@asynccontextmanager
async def lifespan(app):
    load_solution()
    yield


app = FastAPI(lifespan=lifespan)


@app.get("/")
def health():
    return {
        "ready": _solution is not None,
        "ranker_loaded": bool(_solution is not None and _solution.ensemble is not None),
    }


@app.post("/predict", response_model=ASRQuestionResponseDto)
def predict_endpoint(request: ASRQuestionRequestDto):
    return predict(request)


if __name__ == "__main__":
    import uvicorn

    logging.basicConfig(level=logging.INFO)
    uvicorn.run(app, host="0.0.0.0", port=9054)
