"""Download all local models used by v3. Serving itself never downloads weights."""

import argparse
from pathlib import Path

from huggingface_hub import snapshot_download

from .config import ROOT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-large", action="store_true")
    args = parser.parse_args()

    specs = [
        (
            "Systran/faster-whisper-base.en",
            ROOT / "models/whisper-base.en",
            ["*.json", "*.bin", "vocabulary.*", "*.txt"],
        ),
        (
            "Systran/faster-whisper-medium.en",
            ROOT / "models/whisper-medium.en",
            ["*.json", "*.bin", "vocabulary.*", "*.txt"],
        ),
        (
            "google/flan-t5-base",
            ROOT / "models/flan-t5-base",
            ["*.json", "*.safetensors", "spiece.model", "*.txt"],
        ),
    ]
    if not args.skip_large:
        specs.append(
            (
                "google/flan-t5-large",
                ROOT / "models/flan-t5-large",
                ["*.json", "*.safetensors", "spiece.model", "*.txt"],
            )
        )

    for model_id, directory, patterns in specs:
        snapshot_download(model_id, local_dir=str(directory), allow_patterns=patterns)
        print(f"Downloaded {model_id} to {directory}", flush=True)


if __name__ == "__main__":
    main()
