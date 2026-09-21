"""Direct local evaluation with detailed failures for the v3 solution."""

import argparse
import base64
import json
import statistics
import time
from pathlib import Path

from src.dtos import ASRQuestionRequestDto
from src.evaluate import Statistics, oracle
from src.utils import gold_evidence, group_questions_by_conversation, load_sample_audio

from .config import ROOT
from .solution import Solution


def run(limit=None, failures=False, output=None):
    conversations = group_questions_by_conversation()
    if limit:
        conversations = conversations[:limit]
    model = Solution()
    model.warmup()
    stats = Statistics()
    records = []
    for name, rows in conversations:
        request = ASRQuestionRequestDto(
            audio_base64=base64.b64encode(load_sample_audio(name)).decode(),
            audio_filename=name,
            questions=[r["question"] for r in rows],
        )
        started = time.perf_counter()
        response, details = model.predict_with_details(request)
        elapsed = time.perf_counter() - started
        late = elapsed >= 60
        stats.record_request(len(rows), elapsed * 1000, late, late)
        for i, row in enumerate(rows):
            answer = response.answers[i]
            span = None
            if answer and response.evidence_start[i] is not None and response.evidence_end[i] is not None:
                span = (response.evidence_start[i], response.evidence_end[i])
            prediction = -1 if late else int(answer)
            iou = stats.record(
                row["question_type"], int(row["label"]), prediction, gold_evidence(row), None if late else span
            )
            diag = details["questions"][i] if i < len(details["questions"]) else {}
            records.append({**row, "prediction": prediction, "predicted_span": span, "tiou": iou})
            if failures and (prediction != int(row["label"]) or (row["label"] == "1" and iou < 0.3)):
                print(f"\n{name} {row['question_id']} {row['question_type']}")
                print(f"Q: {row['question']}")
                print(f"gold={row['answer']} pred={answer} gold_span={gold_evidence(row)} pred_span={span} tIoU={iou:.3f}")
                print(f"answer_probability={diag.get('answer_probability', 0.0):.3f}")
                selected = diag.get("selected")
                if selected:
                    print(
                        f"selected source={selected.get('source')} kind={selected.get('kind')} "
                        f"rank={selected.get('learned_score', 0.0):.3f} "
                        f"qa=({selected.get('qa_base', 0):.3f},{selected.get('qa_large', 0):.3f}) "
                        f"[{selected['start']:.2f},{selected['end']:.2f}] {selected['text']}"
                    )
        print(f"{name}: {elapsed:.2f}s running score={stats.final_score:.3f}", flush=True)
    print(stats.report())
    if output:
        Path(output).write_text(json.dumps({"records": records}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["oracle", "run"], default="run")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--failures", action="store_true")
    parser.add_argument("--output", default=str(ROOT / "results/v3-evaluation.json"))
    args = parser.parse_args()
    if args.command == "oracle":
        print(oracle().report())
    else:
        run(args.limit, args.failures, args.output)


if __name__ == "__main__":
    main()
