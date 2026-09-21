"""Fast regression tests for v3; no model download required."""

import tempfile
import unittest
from pathlib import Path

import torch

from .config import Config
from .ranker import (
    CANDIDATE_FEATURE_NAMES,
    QUESTION_FEATURE_NAMES,
    CandidateNet,
    LearnedEnsemble,
    QuestionNet,
    candidate_features,
    question_features,
)
from .retrieval import evidence_units, retrieve
from .text_utils import explicit_question_conflict, normalize_numbers, polarity_conflict, quantity_conflict, quantities


class TextTests(unittest.TestCase):
    def test_number_normalization(self):
        self.assertIn("120", normalize_numbers("one hundred and twenty mg"))

    def test_quantity_equivalence(self):
        self.assertEqual(quantities("two weeks"), quantities("14 days"))
        self.assertEqual(quantities("0.1 g"), quantities("100 mg"))

    def test_quantity_mismatch(self):
        self.assertTrue(quantity_conflict("Was the dose 200 mg?", "Take 100 mg daily."))
        self.assertFalse(quantity_conflict("Was the dose 100 mg?", "Take 100 mg daily."))

    def test_explicit_question_conflict(self):
        self.assertTrue(explicit_question_conflict("Was the dose 100 mg?", "Was the dose 200 mg?"))


class RetrievalTests(unittest.TestCase):
    def bundle(self):
        base_words = [
            {"start": 0.0, "end": 0.4, "text": "Take"},
            {"start": 0.5, "end": 0.9, "text": "100"},
            {"start": 1.0, "end": 1.3, "text": "mg"},
            {"start": 1.4, "end": 2.0, "text": "daily"},
            {"start": 2.1, "end": 2.4, "text": "for"},
            {"start": 2.5, "end": 2.9, "text": "two"},
            {"start": 3.0, "end": 3.5, "text": "weeks."},
        ]
        med_words = [dict(w) for w in base_words]
        return {
            "duration": 8.0,
            "sources": {
                "base": {"segments": [{"start": 0.0, "end": 3.5, "text": "Take 100 mg daily for two weeks.", "words": base_words}]},
                "medium": {"segments": [{"start": 0.0, "end": 3.5, "text": "Take 100 mg daily for two weeks.", "words": med_words}]},
            },
        }

    def test_units_preserve_words(self):
        units = evidence_units(self.bundle()["sources"]["base"]["segments"])
        self.assertTrue(units)
        self.assertTrue(units[0]["words"])

    def test_retrieval_finds_dose_region(self):
        result = retrieve("Was the dose 100 mg daily?", self.bundle(), top_k=8)
        self.assertTrue(result)
        self.assertTrue(any("100" in item["text"] for item in result))

    def test_dual_source_agreement_feature(self):
        result = retrieve("Was the dose 100 mg daily?", self.bundle(), top_k=8)
        self.assertGreater(max(item["source_agreement"] for item in result), 0.9)


class FeatureTests(unittest.TestCase):
    def candidate(self):
        return {
            "start": 0.5,
            "end": 3.5,
            "text": "Take 100 mg daily for two weeks.",
            "words": [{"start": 0.5, "end": 1.0, "text": "Take"}],
            "source": "base",
            "kind": "clause",
            "qa_base": 0.8,
            "qa_large": 0.9,
            "retrieval_score": 5.0,
            "retrieval_rank": 0,
            "token_recall": 0.8,
            "token_precision": 0.5,
            "token_jaccard": 0.45,
            "exact_quantity_match": 1.0,
            "quantity_conflict": False,
            "polarity_conflict": False,
            "source_agreement": 0.9,
        }

    def test_feature_dimensions(self):
        context = {"base": 0.8, "large": 0.9}
        candidate = self.candidate()
        self.assertEqual(len(candidate_features("Was the dose 100 mg?", candidate, context)), len(CANDIDATE_FEATURE_NAMES))
        self.assertEqual(len(question_features("Was the dose 100 mg?", [candidate], context)), len(QUESTION_FEATURE_NAMES))

    def test_ranker_serialization(self):
        c = CandidateNet(len(CANDIDATE_FEATURE_NAMES))
        q = QuestionNet(len(QUESTION_FEATURE_NAMES))
        payload = {
            "answer_threshold": 0.55,
            "padding": 0.1,
            "folds": [
                {
                    "candidate_state": c.state_dict(),
                    "candidate_mean": torch.zeros(len(CANDIDATE_FEATURE_NAMES)),
                    "candidate_std": torch.ones(len(CANDIDATE_FEATURE_NAMES)),
                    "question_state": q.state_dict(),
                    "question_mean": torch.zeros(len(QUESTION_FEATURE_NAMES)),
                    "question_std": torch.ones(len(QUESTION_FEATURE_NAMES)),
                }
            ],
        }
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "ranker.pt"
            torch.save(payload, path)
            ensemble = LearnedEnsemble.load(path)
            self.assertIsNotNone(ensemble)
            self.assertAlmostEqual(ensemble.answer_threshold, 0.55)


class ConfigTests(unittest.TestCase):
    def test_default_colab_gpu(self):
        self.assertEqual(Config().device, "cuda:0")

    def test_invalid_device_rejected(self):
        with self.assertRaises(ValueError):
            Config(device="mps")


if __name__ == "__main__":
    unittest.main()
