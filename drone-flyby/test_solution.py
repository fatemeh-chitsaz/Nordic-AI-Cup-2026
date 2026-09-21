"""Deterministic tests for tracking, motion, and camera legality.

These tests do not need the Helsinki images and do not run detector inference:

    python -m unittest -v test_solution.py
"""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import cv2
import numpy as np

from dtos import (
    CameraConstraintsDto,
    CameraLevelBoundsDto,
    DroneFlybyPredictRequestDto,
    DroneFlybyViewDto,
)
from example import (
    BOOTSTRAP_ROUTE,
    SequenceMemory,
    _identity_with_translation,
    choose_next_view,
    estimate_source_motion,
)
from local_evaluator import Camera
import train_yolo


def make_request(frame: int, camera: Camera, sequence_id: str = "test-sequence"):
    constraints = camera.constraints()
    return DroneFlybyPredictRequestDto(
        sequence_id=sequence_id,
        frame=frame,
        frame_index=frame,
        request_id=f"{sequence_id}:{frame}",
        frame_interval_ms=333,
        response_timeout_ms=3333,
        original_width=3840,
        original_height=2160,
        view=DroneFlybyViewDto(
            resolution_level=camera.resolution_level,
            center_x=camera.center_x,
            center_y=camera.center_y,
            view_id=f"view:{frame}",
            image="",
            image_media_type="image/png",
            width=960,
            height=540,
            source_region_xyxy=list(camera.source_region),
        ),
        camera_constraints=CameraConstraintsDto(
            maximum_center_delta=constraints["maximum_center_delta"],
            allowed_resolution_levels=constraints["allowed_resolution_levels"],
            center_bounds=[
                CameraLevelBoundsDto(**bounds)
                for bounds in constraints["center_bounds"]
            ],
            full_view_reset_exempt_from_delta=True,
        ),
    )


class TrackingTests(unittest.TestCase):
    def test_motion_moves_a_track_instead_of_duplicating_it(self):
        memory = SequenceMemory()
        original = (0.20, 0.20, 0.23, 0.24)
        memory.integrate_many([("tank", original, 0.75)], frame=0)

        transform = _identity_with_translation(24.0, 68.0)
        memory._move_tracks(transform, gap=1)
        moved = (
            0.20 + 24 / 3840,
            0.20 + 68 / 2160,
            0.23 + 24 / 3840,
            0.24 + 68 / 2160,
        )
        memory.integrate_many([("tank", moved, 0.80)], frame=1)

        self.assertEqual(len(memory.tracks), 1)
        self.assertEqual(memory.tracks[0].hits, 2)
        self.assertAlmostEqual(memory.tracks[0].bbox[1], moved[1], places=3)

    def test_single_hit_track_expires(self):
        memory = SequenceMemory()
        memory.integrate_many([("jammer", (0.1, 0.1, 0.12, 0.13), 0.7)], frame=0)
        self.assertEqual(len(memory.as_predictions(0)), 1)
        self.assertEqual(len(memory.as_predictions(3)), 0)


class MotionTests(unittest.TestCase):
    def test_source_motion_survives_camera_pan(self):
        rng = np.random.default_rng(7)
        base = rng.integers(0, 256, size=(2160, 3840), dtype=np.uint8)
        base = cv2.GaussianBlur(base, (3, 3), 0)
        previous_full = cv2.cvtColor(base, cv2.COLOR_GRAY2BGR)
        expected_tx, expected_ty = 22.0, 64.0
        current_full = cv2.warpAffine(
            previous_full,
            np.float32([[1, 0, expected_tx], [0, 1, expected_ty]]),
            (3840, 2160),
            borderMode=cv2.BORDER_REFLECT,
        )

        previous_region = (0, 0, 1920, 1080)
        current_region = (960, 0, 2880, 1080)
        previous_view = cv2.resize(
            previous_full[0:1080, 0:1920], (960, 540), interpolation=cv2.INTER_AREA
        )
        current_view = cv2.resize(
            current_full[0:1080, 960:2880], (960, 540), interpolation=cv2.INTER_AREA
        )
        transform = estimate_source_motion(
            previous_view,
            previous_region,
            current_view,
            current_region,
        )

        self.assertIsNotNone(transform)
        self.assertAlmostEqual(float(transform[0, 2]), expected_tx, delta=5.0)
        self.assertAlmostEqual(float(transform[1, 2]), expected_ty, delta=5.0)


class CameraPolicyTests(unittest.TestCase):
    def test_commands_are_legal_and_bootstrap_finishes(self):
        camera = Camera()
        seen_level_one_centers = []
        for frame in range(12):
            request = make_request(frame, camera, sequence_id="camera-legality")
            command = choose_next_view(request, [])
            self.assertIsNotNone(command)
            camera.apply(command.resolution_level, command.center_x, command.center_y)
            if camera.resolution_level == 1:
                seen_level_one_centers.append((camera.center_x, camera.center_y))

        for target in BOOTSTRAP_ROUTE:
            self.assertIn(target, seen_level_one_centers)


class TrainingSplitTests(unittest.TestCase):
    def test_balancing_receives_training_frames_only(self):
        fake_image = np.zeros((2160, 3840, 3), dtype=np.uint8)
        seen_instance_frames = []

        def capture_balancing(
            _output_dir,
            object_instances,
            _frame_image_cache,
            _scene,
            _train_class_counts,
            _target_min,
        ):
            for instances in object_instances.values():
                seen_instance_frames.extend(frame for frame, _ in instances)
            return {name: 0 for name in train_yolo.OBJECT_CLASSES}

        annotation = {"object_id": "tank", "bbox": [100, 100, 180, 180]}
        with (
            TemporaryDirectory() as directory,
            patch.object(train_yolo, "frame_numbers", return_value=[0, 1, 2, 3, 4]),
            patch.object(train_yolo, "load_frame", return_value=fake_image),
            patch.object(train_yolo, "load_annotations", return_value=[annotation]),
            patch.object(
                train_yolo, "balance_rare_classes", side_effect=capture_balancing
            ),
        ):
            train_yolo.build_dataset(
                Path(directory),
                crops_per_frame=0,
                target_min_train_crops_per_class=1,
                validation_fraction=0.20,
            )

        self.assertTrue(seen_instance_frames)
        self.assertNotIn(4, seen_instance_frames)


if __name__ == "__main__":
    unittest.main()
