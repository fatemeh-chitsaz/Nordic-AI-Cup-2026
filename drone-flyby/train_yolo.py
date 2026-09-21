"""Turn the supplied frames into a YOLO dataset and fine-tune a detector on it.

The 25 supplied frames are 4K, full-frame images with source-pixel ground
truth. What a fine-tuned model actually has to work on at inference time is
never that: it is always a 960x540 crop taken at one of the three resolution
levels, exactly the way ``local_evaluator.render_view`` produces it. Training
on the raw 4K frames (or on naive resizes of them) teaches the model a
different problem than the one it will be asked to solve.

So this script does not train on the 25 frames directly. For every frame it
samples many random legal camera views — the same resolution levels and the
same center bounds ``camera_constraints`` would offer during an attempt —
crops and downsamples exactly as the protocol does, and keeps only the
ground-truth boxes that are still meaningfully visible inside that crop, in
that crop's own normalized coordinates. That synthetic-but-faithful dataset
is what gets fine-tuned on.

Usage:

    pip install ultralytics
    python train_yolo.py                        # build the dataset and train
    python train_yolo.py --skip-training         # only (re)build the dataset
    python train_yolo.py --crops-per-frame 24 --epochs 80 --model-size s

Only 25 source frames means 16 object instances, seen from many angles as
the drone moves. That is enough to fine-tune a small pretrained model (start
from COCO weights, do not train from scratch) but not enough to expect it to
generalize perfectly to the validation and evaluation scenes, which show a
different flight over different terrain. Treat the result as a stronger
starting point than the placeholder edge detector, not a finished model —
validate it (``local_evaluator.py``) before trusting it.

Ultralytics' YOLO weights and library are distributed under AGPL-3.0. Check
that this is acceptable for your entry before relying on it in a submission.
"""

import argparse
import random
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from dtos import OBJECT_CLASSES, SOURCE_REGION_SIZES, TRANSMITTED_VIEW_SIZE
from utils import (
    DEFAULT_SCENE,
    center_bounds_for_level,
    frame_numbers,
    load_annotations,
    load_frame,
    source_bbox_to_view,
    source_region_for_view,
)

CATEGORY_INDEX = {name: index for index, name in enumerate(OBJECT_CLASSES)}

# How much of an object's own area must still fall inside a crop for it to
# count as visible there. Too low and the model is trained to draw full boxes
# around slivers it can barely see; too high and heavily-cropped-but-still-
# recognisable objects near a view's edge are thrown away as negatives.
MINIMUM_VISIBLE_FRACTION = 0.35

# Match the active policy in example.py: Level 1 discovers objects, Level 2 is
# used for short confirmation visits, and Level 0 is mainly the initial frame.
LEVEL_WEIGHTS = {0: 1, 1: 7, 2: 2}

# Hold out one contiguous tail of the flight. Adjacent frames show almost the
# same physical object and background, so a modulo split gives a misleadingly
# easy validation set. This is still not cross-scene validation, but it avoids
# direct future/neighbor mixing and is useful for comparing experiments.
DEFAULT_VALIDATION_FRACTION = 0.20

# With one instance per class in the whole scene, purely random camera views
# "catch" some classes far more often than others by sheer chance — a small
# object near the edge of the flight path might end up in 8 crops while a
# large, central one ends up in 200. Below this many training crops for a
# class, extra crops are deliberately aimed at that class's known object
# location (see _object_centered_view) rather than left to chance. This only
# tops up the TRAIN split; VAL is left as the untouched random sample so the
# validation score stays an honest read of real performance.
TARGET_MIN_TRAIN_CROPS_PER_CLASS = 120
# A cap so a class that only appears very briefly (or not at all) can't spin
# forever trying to reach the target.
MAX_BALANCING_CROPS_PER_CLASS = TARGET_MIN_TRAIN_CROPS_PER_CLASS * 4
# Object-centered crops always use Level 1 or 2 (there is only one possible
# Level-0 view, the full frame, so "centering" on it is meaningless, and it
# downsamples small/medium objects too much to be a useful close-up example).
OBJECT_CENTERED_LEVEL_WEIGHTS = {1: 0.4, 2: 0.6}
# How far an object-centered crop's center may drift from the object's own
# center, as a fraction of that level's half-width/height. Some jitter gives
# positional variety instead of every extra crop looking identically centered;
# too much and the crop stops reliably containing the object at all.
OBJECT_CENTER_JITTER_FRACTION = 0.30


def sample_view(level: int) -> Tuple[int, int, int]:
    """A random legal camera center for one resolution level."""
    minimum_x, maximum_x, minimum_y, maximum_y = center_bounds_for_level(level)
    center_x = random.randint(minimum_x, maximum_x)
    center_y = random.randint(minimum_y, maximum_y)
    return level, center_x, center_y


def render_training_view(image: np.ndarray, level: int, center_x: int, center_y: int):
    """Crop and downsample exactly as ``local_evaluator.render_view`` does."""
    x1, y1, x2, y2 = source_region_for_view(level, center_x, center_y)
    view = image[y1:y2, x1:x2]
    if (view.shape[1], view.shape[0]) != TRANSMITTED_VIEW_SIZE:
        view = cv2.resize(view, TRANSMITTED_VIEW_SIZE, interpolation=cv2.INTER_AREA)
    return view, (x1, y1, x2, y2)


def visible_boxes_for_view(
    annotations: List[Dict],
    source_region_xyxy: Tuple[int, int, int, int],
) -> List[Tuple[str, Tuple[float, float, float, float]]]:
    """Ground-truth boxes still meaningfully inside one crop, in view coordinates."""
    region_x1, region_y1, region_x2, region_y2 = source_region_xyxy
    visible: List[Tuple[str, Tuple[float, float, float, float]]] = []
    for annotation in annotations:
        x1, y1, x2, y2 = (float(c) for c in annotation["bbox"])
        area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        if area <= 0:
            continue
        inter_x1, inter_y1 = max(x1, region_x1), max(y1, region_y1)
        inter_x2, inter_y2 = min(x2, region_x2), min(y2, region_y2)
        intersection = max(0.0, inter_x2 - inter_x1) * max(0.0, inter_y2 - inter_y1)
        if intersection / area < MINIMUM_VISIBLE_FRACTION:
            continue
        view_x1, view_y1, view_x2, view_y2 = source_bbox_to_view(
            (x1, y1, x2, y2), source_region_xyxy
        )
        # Clip to the crop: a box that extends past the edge is still a real,
        # partially-visible object, but the label has to stay inside [0, 1].
        view_x1, view_x2 = max(0.0, view_x1), min(1.0, view_x2)
        view_y1, view_y2 = max(0.0, view_y1), min(1.0, view_y2)
        if view_x2 - view_x1 <= 1e-4 or view_y2 - view_y1 <= 1e-4:
            continue
        visible.append((annotation["object_id"], (view_x1, view_y1, view_x2, view_y2)))
    return visible


def write_yolo_label(
    path: Path, boxes: List[Tuple[str, Tuple[float, float, float, float]]]
) -> None:
    lines = []
    for object_id, (x1, y1, x2, y2) in boxes:
        center_x, center_y = (x1 + x2) / 2, (y1 + y2) / 2
        width, height = x2 - x1, y2 - y1
        lines.append(
            f"{CATEGORY_INDEX[object_id]} {center_x:.6f} {center_y:.6f} "
            f"{width:.6f} {height:.6f}"
        )
    text = "\n".join(lines)
    path.write_text(text + "\n" if text else "")


def weighted_level_choice() -> int:
    levels, weights = zip(*LEVEL_WEIGHTS.items())
    return random.choices(levels, weights=weights, k=1)[0]


def _balancing_level_choice() -> int:
    levels, weights = zip(*OBJECT_CENTERED_LEVEL_WEIGHTS.items())
    return random.choices(levels, weights=weights, k=1)[0]


def object_centered_view(
    bbox: Tuple[float, float, float, float], level: int
) -> Tuple[int, int, int]:
    """A legal camera view for ``level``, aimed at (not just hoping to hit) this object.

    Centers on the object's own bounding-box center, jittered a little for
    positional variety, then clipped into the level's legal bounds exactly
    the way a real requested_view would be. This is still a view the camera
    could legitimately take — it just isn't left to random chance whether it
    happens to contain the object.
    """
    x1, y1, x2, y2 = bbox
    object_center_x, object_center_y = (x1 + x2) / 2, (y1 + y2) / 2
    minimum_x, maximum_x, minimum_y, maximum_y = center_bounds_for_level(level)
    width, height = SOURCE_REGION_SIZES[level]

    jitter_x = random.uniform(-1, 1) * OBJECT_CENTER_JITTER_FRACTION * (width / 2)
    jitter_y = random.uniform(-1, 1) * OBJECT_CENTER_JITTER_FRACTION * (height / 2)

    center_x = int(min(max(object_center_x + jitter_x, minimum_x), maximum_x))
    center_y = int(min(max(object_center_y + jitter_y, minimum_y), maximum_y))
    return level, center_x, center_y


def balance_rare_classes(
    output_dir: Path,
    object_instances: Dict[str, List[Tuple[int, Dict]]],
    frame_image_cache: Dict[int, np.ndarray],
    scene: str,
    train_class_counts: Dict[str, int],
    target_min: int,
) -> Dict[str, int]:
    """Top up any class below ``target_min`` TRAIN crops with object-centered ones.

    Only classes that actually appear somewhere in the scene can be topped
    up — a class with zero instances has nothing to center a camera on, and
    is reported rather than silently skipped.
    """
    images_dir = output_dir / "images" / "train"
    labels_dir = output_dir / "labels" / "train"
    added: Dict[str, int] = {name: 0 for name in OBJECT_CLASSES}

    for object_id in OBJECT_CLASSES:
        current = train_class_counts.get(object_id, 0)
        if current >= target_min:
            continue
        instances = object_instances.get(object_id, [])
        if not instances:
            print(
                f"  {object_id}: 0 training crops and no instances in this "
                "scene at all — cannot balance it."
            )
            continue

        attempts = 0
        index = 0
        while (
            train_class_counts.get(object_id, 0) < target_min
            and attempts < MAX_BALANCING_CROPS_PER_CLASS
        ):
            frame, annotation = instances[index % len(instances)]
            index += 1
            attempts += 1

            if frame not in frame_image_cache:
                frame_image_cache[frame] = load_frame(frame, scene)
            image = frame_image_cache[frame]
            annotations = load_annotations(frame, scene)

            level = _balancing_level_choice()
            level, center_x, center_y = object_centered_view(
                tuple(float(c) for c in annotation["bbox"]), level
            )
            view, source_region_xyxy = render_training_view(
                image, level, center_x, center_y
            )
            boxes = visible_boxes_for_view(annotations, source_region_xyxy)
            if not any(box_object_id == object_id for box_object_id, _ in boxes):
                # Jitter pushed the target itself out of frame; try again.
                continue

            stem = f"balance_{object_id}_{attempts:04d}_L{level}"
            cv2.imwrite(
                str(images_dir / f"{stem}.jpg"), view, [cv2.IMWRITE_JPEG_QUALITY, 92]
            )
            write_yolo_label(labels_dir / f"{stem}.txt", boxes)
            for box_object_id, _ in boxes:
                train_class_counts[box_object_id] = (
                    train_class_counts.get(box_object_id, 0) + 1
                )
                if box_object_id == object_id:
                    added[object_id] += 1

        if train_class_counts.get(object_id, 0) < target_min:
            actual_count = train_class_counts.get(object_id, 0)
            print(
                f"  {object_id}: reached {actual_count}/{target_min} "
                f"after {attempts} attempts (object may rarely be cleanly "
                "visible at these zoom levels)."
            )

    return added


def build_dataset(
    output_dir: Path,
    scene: str = DEFAULT_SCENE,
    crops_per_frame: int = 16,
    negative_keep_probability: float = 0.50,
    seed: int = 0,
    target_min_train_crops_per_class: int = TARGET_MIN_TRAIN_CROPS_PER_CLASS,
    balance_classes: bool = True,
    validation_fraction: float = DEFAULT_VALIDATION_FRACTION,
) -> Path:
    """Write a YOLO dataset and return the path to its data.yaml file."""
    random.seed(seed)
    images_dir = output_dir / "images"
    labels_dir = output_dir / "labels"
    for split in ("train", "val"):
        (images_dir / split).mkdir(parents=True, exist_ok=True)
        (labels_dir / split).mkdir(parents=True, exist_ok=True)

    frames = frame_numbers(scene)
    if not frames:
        raise RuntimeError(f"No frames found for scene {scene!r}.")
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be between 0 and 1.")

    validation_count = max(1, int(round(len(frames) * validation_fraction)))
    validation_frames = set(frames[-validation_count:])
    print(
        f"Chronological split: {len(frames) - validation_count} training frames, "
        f"{validation_count} validation frames "
        f"({min(validation_frames)}..{max(validation_frames)})."
    )

    written = {"train": 0, "val": 0}
    train_class_counts: Dict[str, int] = {name: 0 for name in OBJECT_CLASSES}
    # Every (frame, annotation) an object appears in, so the balancing pass
    # below knows where to aim a camera for each under-represented class.
    object_instances: Dict[str, List[Tuple[int, Dict]]] = {
        name: [] for name in OBJECT_CLASSES
    }
    frame_image_cache: Dict[int, np.ndarray] = {}

    for frame in frames:
        image = load_frame(frame, scene)
        annotations = load_annotations(frame, scene)
        split = "val" if frame in validation_frames else "train"

        # The balancing pass is a TRAIN augmentation. Never give it validation
        # frames or their annotations; doing so leaks held-out imagery into the
        # training directory.
        if split == "train":
            frame_image_cache[frame] = image
            for annotation in annotations:
                object_instances.setdefault(annotation["object_id"], []).append(
                    (frame, annotation)
                )

        for crop_index in range(crops_per_frame):
            level = weighted_level_choice()
            level, center_x, center_y = sample_view(level)
            view, source_region_xyxy = render_training_view(
                image, level, center_x, center_y
            )
            boxes = visible_boxes_for_view(annotations, source_region_xyxy)

            if not boxes and random.random() > negative_keep_probability:
                continue  # skip most empty crops; keep the dataset positive-heavy

            stem = f"frame{frame:06d}_crop{crop_index:03d}_L{level}"
            cv2.imwrite(
                str(images_dir / split / f"{stem}.jpg"),
                view,
                [cv2.IMWRITE_JPEG_QUALITY, 92],
            )
            write_yolo_label(labels_dir / split / f"{stem}.txt", boxes)
            written[split] += 1
            if split == "train":
                for object_id, _ in boxes:
                    train_class_counts[object_id] = (
                        train_class_counts.get(object_id, 0) + 1
                    )

    print(
        f"Wrote {written['train']} training and {written['val']} validation "
        f"crops to {output_dir}"
    )

    if balance_classes:
        under_target = {
            name: count
            for name, count in train_class_counts.items()
            if count < target_min_train_crops_per_class
        }
        if under_target:
            print(
                f"Topping up {len(under_target)} under-represented class(es) toward "
                f"{target_min_train_crops_per_class} training crops each: "
                f"{sorted(under_target)}"
            )
            added = balance_rare_classes(
                output_dir,
                object_instances,
                frame_image_cache,
                scene,
                train_class_counts,
                target_min_train_crops_per_class,
            )
            total_added = sum(added.values())
            written["train"] += total_added
            print(f"Added {total_added} object-centered training crops. New counts:")
            for name in sorted(under_target):
                print(f"  {name}: {train_class_counts[name]}")

    data_yaml = output_dir / "data.yaml"
    data_yaml.write_text(
        "path: {path}\ntrain: images/train\nval: images/val\nnames:\n{names}\n".format(
            path=output_dir.resolve(),
            names="\n".join(
                f"  {index}: {name}" for index, name in enumerate(OBJECT_CLASSES)
            ),
        )
    )
    return data_yaml


def train(
    data_yaml: Path, epochs: int, model_size: str, image_size: int, batch: int
) -> Optional[Path]:
    """Fine-tune a COCO-pretrained YOLO on the generated dataset."""
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise SystemExit(
            "ultralytics is not installed. Run `pip install ultralytics` first, "
            "or pass --skip-training to only build the dataset."
        ) from exc

    model = YOLO(f"yolov8{model_size}.pt")  # start from COCO weights, not from scratch
    results = model.train(
        data=str(data_yaml),
        epochs=epochs,
        imgsz=image_size,
        batch=batch,
        patience=max(10, epochs // 4),
        project=str(data_yaml.parent / "runs"),
        name="drone_flyby",
        exist_ok=True,
    )
    best_weights = Path(results.save_dir) / "weights" / "best.pt"
    if not best_weights.exists():
        print(
            f"Training finished but {best_weights} was not produced; "
            "check the run above."
        )
        return None

    destination = Path(__file__).resolve().parent / "weights" / "best.pt"
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best_weights, destination)
    print(
        f"Copied fine-tuned weights to {destination} — example.py will pick "
        "these up automatically."
    )
    return destination


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", default=DEFAULT_SCENE, help="Scene under src/.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("yolo_dataset"),
        help="Dataset output directory.",
    )
    parser.add_argument("--crops-per-frame", type=int, default=16)
    parser.add_argument(
        "--negative-keep-probability",
        type=float,
        default=0.50,
        help="Fraction of empty terrain crops to keep (default: 0.50).",
    )
    parser.add_argument(
        "--validation-fraction",
        type=float,
        default=DEFAULT_VALIDATION_FRACTION,
        help="Contiguous tail fraction held out for validation (default: 0.20).",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--target-min-crops-per-class",
        type=int,
        default=TARGET_MIN_TRAIN_CROPS_PER_CLASS,
        help=(
            "Top up any class below this many TRAIN crops with object-centered extras."
        ),
    )
    parser.add_argument(
        "--disable-class-balancing",
        action="store_true",
        help="Skip the object-centered top-up pass; use only the raw random crops.",
    )
    parser.add_argument(
        "--skip-training", action="store_true", help="Only build the dataset."
    )
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument(
        "--model-size", choices=["n", "s", "m"], default="n", help="YOLOv8 size: n/s/m."
    )
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--batch", type=int, default=8)
    arguments = parser.parse_args()

    data_yaml = build_dataset(
        arguments.output,
        scene=arguments.scene,
        crops_per_frame=arguments.crops_per_frame,
        negative_keep_probability=arguments.negative_keep_probability,
        seed=arguments.seed,
        target_min_train_crops_per_class=arguments.target_min_crops_per_class,
        balance_classes=not arguments.disable_class_balancing,
        validation_fraction=arguments.validation_fraction,
    )

    if arguments.skip_training:
        print("--skip-training set: dataset built, no model trained.")
        return 0

    train(
        data_yaml,
        arguments.epochs,
        arguments.model_size,
        arguments.imgsz,
        arguments.batch,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
