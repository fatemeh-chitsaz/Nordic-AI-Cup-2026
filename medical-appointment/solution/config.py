"""Configuration for the dual-ASR + dual-QA + learned evidence ranker pipeline."""

import os
import re
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def _flag(name: str, default: bool) -> bool:
    return os.getenv(name, "1" if default else "0") == "1"


@dataclass(frozen=True)
class Config:
    asr_base_model: str = str(ROOT / "models" / "whisper-base.en")
    asr_medium_model: str = str(ROOT / "models" / "whisper-medium.en")
    qa_base_model: str = str(ROOT / "models" / "flan-t5-base")
    qa_large_model: str = str(ROOT / "models" / "flan-t5-large")
    ranker_model: str = str(ROOT / "models" / "ranker_ensemble.pt")

    device: str = "cuda:0"
    cpu_threads: int = 4
    base_beam_size: int = 3
    medium_beam_size: int = 5
    qa_batch_size: int = 18
    top_k: int = 16
    max_window: int = 3

    # Fallback values used only when the learned ensemble is absent.
    fallback_answer_threshold: float = 0.58
    fallback_padding: float = 0.05

    dual_asr: bool = True
    dual_qa: bool = True
    require_ranker: bool = False

    def __post_init__(self):
        if not re.fullmatch(r"cpu|cuda(?::[0-9]+)?", self.device):
            raise ValueError("DEVICE must be cpu, cuda, or cuda:<nonnegative index>")

    @property
    def device_type(self):
        return self.device.split(":")[0]

    @property
    def device_index(self):
        return int(self.device.split(":")[1]) if ":" in self.device else 0

    @property
    def torch_device(self):
        return f"cuda:{self.device_index}" if self.device_type == "cuda" else "cpu"

    @property
    def asr_compute_type(self):
        return "float16" if self.device_type == "cuda" else "int8"

    def validate_device(self):
        if self.device_type == "cpu":
            return
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError(
                f"DEVICE={self.device} requested, but CUDA is unavailable. "
                "Install CUDA-enabled PyTorch or set DEVICE=cpu."
            )
        count = torch.cuda.device_count()
        if self.device_index >= count:
            raise RuntimeError(
                f"DEVICE={self.device} requested, but only {count} CUDA device(s) are visible. "
                "On Colab, use DEVICE=cuda:0."
            )

    @classmethod
    def from_env(cls):
        d = cls()
        return cls(
            asr_base_model=os.getenv("ASR_BASE_MODEL", d.asr_base_model),
            asr_medium_model=os.getenv("ASR_MEDIUM_MODEL", d.asr_medium_model),
            qa_base_model=os.getenv("QA_BASE_MODEL", d.qa_base_model),
            qa_large_model=os.getenv("QA_LARGE_MODEL", d.qa_large_model),
            ranker_model=os.getenv("RANKER_MODEL", d.ranker_model),
            device=os.getenv("DEVICE", d.device),
            cpu_threads=int(os.getenv("CPU_THREADS", d.cpu_threads)),
            base_beam_size=int(os.getenv("BASE_BEAM_SIZE", d.base_beam_size)),
            medium_beam_size=int(os.getenv("MEDIUM_BEAM_SIZE", d.medium_beam_size)),
            qa_batch_size=int(os.getenv("QA_BATCH_SIZE", d.qa_batch_size)),
            top_k=int(os.getenv("TOP_K", d.top_k)),
            max_window=int(os.getenv("MAX_WINDOW", d.max_window)),
            fallback_answer_threshold=float(
                os.getenv("FALLBACK_ANSWER_THRESHOLD", d.fallback_answer_threshold)
            ),
            fallback_padding=float(os.getenv("FALLBACK_PADDING", d.fallback_padding)),
            dual_asr=_flag("DUAL_ASR", d.dual_asr),
            dual_qa=_flag("DUAL_QA", d.dual_qa),
            require_ranker=_flag("REQUIRE_RANKER", d.require_ranker),
        )
