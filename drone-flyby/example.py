"""Motion-aware detector, tracker, and active camera policy.

The detector sees only the current 960x540 camera view, while the evaluator
scores a whole 3840x2160 frame. The important part of this solution is
therefore the sequence logic:

* YOLO detections are lifted from view coordinates to full-frame coordinates.
* ORB feature matches estimate apparent ground motion between frames, even
  when the camera center or zoom changes.
* Tracks are moved by that transform before fresh detections are associated.
  Old boxes are never returned forever at their original pixel positions.
* A short Level-1 bootstrap covers the frame, then the camera scans the top
  entry band. Level 2 is used briefly to confirm uncertain candidates.

The code uses only OpenCV plus the existing Ultralytics model. If the model
cannot be loaded, an empty prediction is safer than flooding the score with
low-quality contour proposals.
"""

import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np

from dtos import (
    IMAGE_HEIGHT,
    IMAGE_WIDTH,
    OBJECT_CLASSES,
    TRANSMITTED_VIEW_SIZE,
    DroneFlybyPredictionDto,
    DroneFlybyPredictRequestDto,
    DroneFlybyPredictResponseDto,
    RequestedViewDto,
)
from utils import clip_bbox_to_frame, decode_view, view_bbox_to_global

logger = logging.getLogger(__name__)

BBox = Tuple[float, float, float, float]
Detection = Tuple[str, BBox, float]
SourceRegion = Tuple[int, int, int, int]


# ---------------------------------------------------------------------------
# Public prediction entrypoint
# ---------------------------------------------------------------------------


def predict(request: DroneFlybyPredictRequestDto) -> DroneFlybyPredictResponseDto:
    """Process one frame and return whole-frame predictions plus a camera move."""
    if request.camera_command_feedback is not None:
        feedback = request.camera_command_feedback
        logger.warning(
            "Camera command from frame %s was ignored: %s",
            feedback.frame,
            feedback.reason,
        )

    image = decode_view(request.view)
    memory = _memory_for(request.sequence_id)

    try:
        memory.advance_to_frame(image, request)
    except Exception:
        # Tracking is useful, but a feature-matching failure must never lose a
        # request. Existing tracks simply age without a new transform.
        logger.exception("Motion estimation failed on frame %s", request.frame)
        memory.advance_without_image(request.frame)

    try:
        fresh_detections = detect(image, request)
    except Exception:
        logger.exception("Detector failed on frame %s", request.frame)
        fresh_detections = []

    memory.integrate_many(fresh_detections, request.frame)

    return DroneFlybyPredictResponseDto(
        request_id=request.request_id,
        frame=request.frame,
        annotations=memory.as_predictions(request.frame),
        requested_view=choose_next_view(request, fresh_detections),
    )


# ---------------------------------------------------------------------------
# Detector
# ---------------------------------------------------------------------------


WEIGHTS_PATH = Path(__file__).resolve().parent / "weights" / "best.pt"
MODEL_CONFIDENCE = 0.18
MODEL_IOU = 0.50
MAXIMUM_DETECTIONS_PER_VIEW = 60

_model = None
_model_load_attempted = False


def _load_model():
    """Load the fine-tuned detector once; return None when unavailable."""
    global _model, _model_load_attempted
    if _model_load_attempted:
        return _model
    _model_load_attempted = True

    if not WEIGHTS_PATH.exists():
        logger.error("No detector weights found at %s", WEIGHTS_PATH)
        return None
    try:
        from ultralytics import YOLO

        _model = YOLO(str(WEIGHTS_PATH))
        logger.info("Loaded fine-tuned weights from %s", WEIGHTS_PATH)
    except Exception:
        logger.exception("Could not load detector weights at %s", WEIGHTS_PATH)
        _model = None
    return _model


def _warm_up() -> None:
    """Pay model-loading and first-inference costs before frame zero."""
    model = _load_model()
    if model is None:
        return
    dummy = np.zeros(
        (TRANSMITTED_VIEW_SIZE[1], TRANSMITTED_VIEW_SIZE[0], 3), dtype=np.uint8
    )
    try:
        model.predict(
            dummy,
            conf=MODEL_CONFIDENCE,
            iou=MODEL_IOU,
            imgsz=TRANSMITTED_VIEW_SIZE[0],
            max_det=MAXIMUM_DETECTIONS_PER_VIEW,
            agnostic_nms=True,
            verbose=False,
        )
    except Exception:
        logger.exception("Model warm-up failed")


def detect(image: np.ndarray, request: DroneFlybyPredictRequestDto) -> List[Detection]:
    """Detect objects in the current view and lift boxes to full-frame space."""
    model = _load_model()
    if model is None:
        return []

    height, width = image.shape[:2]
    result = model.predict(
        image,
        conf=MODEL_CONFIDENCE,
        iou=MODEL_IOU,
        imgsz=TRANSMITTED_VIEW_SIZE[0],
        max_det=MAXIMUM_DETECTIONS_PER_VIEW,
        agnostic_nms=True,
        verbose=False,
    )[0]

    detections: List[Detection] = []
    for box in result.boxes:
        class_index = int(box.cls[0])
        if not 0 <= class_index < len(OBJECT_CLASSES):
            continue
        x1, y1, x2, y2 = (float(value) for value in box.xyxy[0])
        view_bbox = (x1 / width, y1 / height, x2 / width, y2 / height)
        global_bbox = view_bbox_to_global(
            view_bbox,
            request.view.source_region_xyxy,
            request.original_width,
            request.original_height,
        )
        clipped = clip_bbox_to_frame(global_bbox)
        if clipped is not None:
            detections.append(
                (OBJECT_CLASSES[class_index], clipped, float(box.conf[0]))
            )
    return _deduplicate_detections(detections)


def _deduplicate_detections(detections: Iterable[Detection]) -> List[Detection]:
    """Remove near-identical boxes, including duplicates with different labels."""
    kept: List[Detection] = []
    for candidate in sorted(detections, key=lambda item: item[2], reverse=True):
        if any(_iou(candidate[1], existing[1]) >= 0.72 for existing in kept):
            continue
        kept.append(candidate)
    return kept


_warm_up()


# ---------------------------------------------------------------------------
# Motion estimation
# ---------------------------------------------------------------------------


def _view_points_to_source(
    points: np.ndarray, region: SourceRegion, view_shape: Tuple[int, int]
) -> np.ndarray:
    """Convert Nx2 view-pixel points into full source-frame pixels."""
    height, width = view_shape
    x1, y1, x2, y2 = region
    converted = points.astype(np.float32).copy()
    converted[:, 0] = x1 + converted[:, 0] * (x2 - x1) / float(width)
    converted[:, 1] = y1 + converted[:, 1] * (y2 - y1) / float(height)
    return converted


def estimate_source_motion(
    previous_image: np.ndarray,
    previous_region: SourceRegion,
    current_image: np.ndarray,
    current_region: SourceRegion,
    frame_gap: int = 1,
) -> Optional[np.ndarray]:
    """Estimate an affine map from previous to current source coordinates.

    Feature descriptors are matched in the transmitted images, but both point
    sets are converted through their own source regions before RANSAC. That
    cancels intentional camera pan/zoom and leaves the apparent drone motion.
    """
    previous_gray = cv2.cvtColor(previous_image, cv2.COLOR_BGR2GRAY)
    current_gray = cv2.cvtColor(current_image, cv2.COLOR_BGR2GRAY)
    orb = cv2.ORB_create(nfeatures=1400, fastThreshold=10)
    previous_keypoints, previous_descriptors = orb.detectAndCompute(previous_gray, None)
    current_keypoints, current_descriptors = orb.detectAndCompute(current_gray, None)
    if (
        previous_descriptors is None
        or current_descriptors is None
        or len(previous_keypoints) < 12
        or len(current_keypoints) < 12
    ):
        return None

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    pairs = matcher.knnMatch(previous_descriptors, current_descriptors, k=2)
    good = [
        pair[0]
        for pair in pairs
        if len(pair) == 2 and pair[0].distance < 0.76 * pair[1].distance
    ]
    if len(good) < 10:
        return None

    previous_points = np.float32(
        [previous_keypoints[match.queryIdx].pt for match in good]
    )
    current_points = np.float32(
        [current_keypoints[match.trainIdx].pt for match in good]
    )
    previous_source = _view_points_to_source(
        previous_points, previous_region, previous_gray.shape
    )
    current_source = _view_points_to_source(
        current_points, current_region, current_gray.shape
    )

    transform, inlier_mask = cv2.estimateAffinePartial2D(
        previous_source,
        current_source,
        method=cv2.RANSAC,
        ransacReprojThreshold=14.0,
        maxIters=2000,
        confidence=0.995,
        refineIters=10,
    )
    if transform is None or inlier_mask is None:
        return None

    inliers = int(inlier_mask.sum())
    if inliers < 8 or inliers / len(good) < 0.28:
        return None

    a, b, tx = (float(value) for value in transform[0])
    c, d, ty = (float(value) for value in transform[1])
    scale_x, scale_y = math.hypot(a, c), math.hypot(b, d)
    if not 0.92 <= scale_x <= 1.08 or not 0.92 <= scale_y <= 1.08:
        return None
    if math.hypot(tx, ty) > 320.0 * max(1, frame_gap):
        return None
    return transform.astype(np.float32)


def _transform_bbox(bbox: BBox, transform: np.ndarray) -> Optional[BBox]:
    """Move one normalized bbox with a source-pixel affine transform."""
    x1, y1, x2, y2 = bbox
    corners = np.float32(
        [
            [x1 * IMAGE_WIDTH, y1 * IMAGE_HEIGHT],
            [x2 * IMAGE_WIDTH, y1 * IMAGE_HEIGHT],
            [x2 * IMAGE_WIDTH, y2 * IMAGE_HEIGHT],
            [x1 * IMAGE_WIDTH, y2 * IMAGE_HEIGHT],
        ]
    ).reshape(-1, 1, 2)
    moved = cv2.transform(corners, transform).reshape(-1, 2)
    moved_bbox = (
        float(moved[:, 0].min() / IMAGE_WIDTH),
        float(moved[:, 1].min() / IMAGE_HEIGHT),
        float(moved[:, 0].max() / IMAGE_WIDTH),
        float(moved[:, 1].max() / IMAGE_HEIGHT),
    )
    return clip_bbox_to_frame(moved_bbox)


def _identity_with_translation(tx: float, ty: float) -> np.ndarray:
    return np.asarray([[1.0, 0.0, tx], [0.0, 1.0, ty]], dtype=np.float32)


# ---------------------------------------------------------------------------
# Motion-aware tracking
# ---------------------------------------------------------------------------


MAXIMUM_TRACKS = 80
MAXIMUM_CONFIRMED_AGE = 9
MAXIMUM_SINGLE_HIT_AGE = 2
TRACK_CONFIDENCE_DECAY = 0.965


def _iou(a: BBox, b: BBox) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    intersection = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if intersection <= 0.0:
        return 0.0
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - intersection
    return intersection / union if union > 0.0 else 0.0


def _center_distance_pixels(a: BBox, b: BBox) -> float:
    ax = (a[0] + a[2]) * IMAGE_WIDTH / 2.0
    ay = (a[1] + a[3]) * IMAGE_HEIGHT / 2.0
    bx = (b[0] + b[2]) * IMAGE_WIDTH / 2.0
    by = (b[1] + b[3]) * IMAGE_HEIGHT / 2.0
    return math.hypot(ax - bx, ay - by)


def _bbox_diagonal_pixels(bbox: BBox) -> float:
    return math.hypot(
        (bbox[2] - bbox[0]) * IMAGE_WIDTH,
        (bbox[3] - bbox[1]) * IMAGE_HEIGHT,
    )


@dataclass
class Track:
    bbox: BBox
    confidence: float
    class_scores: Dict[str, float]
    hits: int
    last_seen_frame: int

    @property
    def object_id(self) -> str:
        return max(self.class_scores, key=self.class_scores.get)


@dataclass
class SequenceMemory:
    tracks: List[Track] = field(default_factory=list)
    previous_image: Optional[np.ndarray] = None
    previous_region: Optional[SourceRegion] = None
    previous_frame: Optional[int] = None
    motion_per_frame: np.ndarray = field(
        default_factory=lambda: _identity_with_translation(0.0, 0.0)
    )

    def advance_to_frame(
        self, image: np.ndarray, request: DroneFlybyPredictRequestDto
    ) -> None:
        current_region = tuple(int(value) for value in request.view.source_region_xyxy)
        if (
            self.previous_frame is None
            or self.previous_image is None
            or self.previous_region is None
        ):
            self._remember_image(image, current_region, request.frame)
            return

        gap = max(1, request.frame - self.previous_frame)
        transform = estimate_source_motion(
            self.previous_image,
            self.previous_region,
            image,
            current_region,
            gap,
        )
        if transform is not None:
            per_frame_tx = float(transform[0, 2]) / gap
            per_frame_ty = float(transform[1, 2]) / gap
            self.motion_per_frame = _identity_with_translation(
                0.65 * per_frame_tx + 0.35 * float(self.motion_per_frame[0, 2]),
                0.65 * per_frame_ty + 0.35 * float(self.motion_per_frame[1, 2]),
            )
            self._move_tracks(transform, gap)
        else:
            fallback = _identity_with_translation(
                float(self.motion_per_frame[0, 2]) * gap,
                float(self.motion_per_frame[1, 2]) * gap,
            )
            self._move_tracks(fallback, gap)
        self._remember_image(image, current_region, request.frame)

    def advance_without_image(self, frame: int) -> None:
        if self.previous_frame is None:
            self.previous_frame = frame
            return
        gap = max(1, frame - self.previous_frame)
        fallback = _identity_with_translation(
            float(self.motion_per_frame[0, 2]) * gap,
            float(self.motion_per_frame[1, 2]) * gap,
        )
        self._move_tracks(fallback, gap)
        self.previous_frame = frame

    def _remember_image(
        self, image: np.ndarray, region: SourceRegion, frame: int
    ) -> None:
        self.previous_image = image.copy()
        self.previous_region = region
        self.previous_frame = frame

    def _move_tracks(self, transform: np.ndarray, gap: int) -> None:
        moved: List[Track] = []
        for track in self.tracks:
            new_bbox = _transform_bbox(track.bbox, transform)
            if new_bbox is None:
                continue
            track.bbox = new_bbox
            track.confidence *= TRACK_CONFIDENCE_DECAY**gap
            moved.append(track)
        self.tracks = moved

    def integrate_many(self, detections: List[Detection], frame: int) -> None:
        """Greedily associate detections to predicted tracks, independent of class."""
        pairs: List[Tuple[float, int, int]] = []
        for track_index, track in enumerate(self.tracks):
            track_diagonal = _bbox_diagonal_pixels(track.bbox)
            for detection_index, (object_id, bbox, _confidence) in enumerate(
                detections
            ):
                detection_diagonal = _bbox_diagonal_pixels(bbox)
                size_ratio = detection_diagonal / max(track_diagonal, 1.0)
                if not 0.35 <= size_ratio <= 2.85:
                    continue
                distance = _center_distance_pixels(track.bbox, bbox)
                gate = max(85.0, 2.4 * max(track_diagonal, detection_diagonal))
                overlap = _iou(track.bbox, bbox)
                if distance > gate and overlap < 0.08:
                    continue
                class_penalty = 0.12 if object_id != track.object_id else 0.0
                cost = distance / gate - 0.55 * overlap + class_penalty
                pairs.append((cost, track_index, detection_index))

        used_tracks = set()
        used_detections = set()
        for _cost, track_index, detection_index in sorted(pairs):
            if track_index in used_tracks or detection_index in used_detections:
                continue
            object_id, bbox, confidence = detections[detection_index]
            self._update_track(
                self.tracks[track_index], object_id, bbox, confidence, frame
            )
            used_tracks.add(track_index)
            used_detections.add(detection_index)

        for detection_index, (object_id, bbox, confidence) in enumerate(detections):
            if detection_index in used_detections:
                continue
            self.tracks.append(
                Track(
                    bbox=bbox,
                    confidence=confidence,
                    class_scores={object_id: confidence},
                    hits=1,
                    last_seen_frame=frame,
                )
            )

        self._prune(frame)

    @staticmethod
    def _update_track(
        track: Track,
        object_id: str,
        bbox: BBox,
        confidence: float,
        frame: int,
    ) -> None:
        # The detector is the tight-box authority; motion prediction only keeps
        # the association close enough to find it again.
        track.bbox = tuple(
            0.20 * predicted + 0.80 * observed
            for predicted, observed in zip(track.bbox, bbox)
        )
        for name in list(track.class_scores):
            track.class_scores[name] *= 0.90
        track.class_scores[object_id] = (
            track.class_scores.get(object_id, 0.0) + confidence
        )
        track.confidence = 0.55 * track.confidence + 0.45 * confidence
        track.hits += 1
        track.last_seen_frame = frame

    def _prune(self, frame: int) -> None:
        self.tracks = [
            track
            for track in self.tracks
            if frame - track.last_seen_frame
            <= (MAXIMUM_CONFIRMED_AGE if track.hits >= 2 else MAXIMUM_SINGLE_HIT_AGE)
            and track.confidence >= 0.08
        ]
        self.tracks.sort(
            key=lambda track: (track.hits >= 2, track.confidence), reverse=True
        )
        del self.tracks[MAXIMUM_TRACKS:]

    def as_predictions(self, frame: int) -> List[DroneFlybyPredictionDto]:
        self._prune(frame)
        predictions: List[DroneFlybyPredictionDto] = []
        for track in self.tracks:
            age = max(0, frame - track.last_seen_frame)
            confidence = track.confidence * (0.97**age)
            if confidence < MODEL_CONFIDENCE:
                continue
            clipped = clip_bbox_to_frame(track.bbox)
            if clipped is None:
                continue
            predictions.append(
                DroneFlybyPredictionDto(
                    object_id=track.object_id,
                    bbox=[float(value) for value in clipped],
                    confidence=round(min(0.99, confidence), 4),
                )
            )
        return predictions


_memory: Dict[str, SequenceMemory] = {}


def _memory_for(sequence_id: str) -> SequenceMemory:
    return _memory.setdefault(sequence_id, SequenceMemory())


# ---------------------------------------------------------------------------
# Active camera policy
# ---------------------------------------------------------------------------


BOOTSTRAP_ROUTE = (
    (960, 540),
    (1920, 540),
    (2880, 540),
    (2880, 1620),
    (1920, 1620),
    (960, 1620),
)
ENTRY_SCAN_ROUTE = ((960, 540), (1920, 540), (2880, 540), (1920, 540))

CONFIRMATION_MIN_CONFIDENCE = 0.20
CONFIRMATION_MAX_CONFIDENCE = 0.68
CONFIRMATION_COOLDOWN_FRAMES = 3
# Camera centers are integers, so no percentage safety margin is needed.
# A tiny absolute epsilon only protects against floating-point round-off.
CAMERA_DISTANCE_EPSILON = 1e-6


@dataclass
class CameraState:
    bootstrap_index: int = 0
    scan_index: int = 0
    pending_route_target: Optional[Tuple[int, int]] = None
    confirming: bool = False
    last_confirmation_frame: int = -10_000


_camera_states: Dict[str, CameraState] = {}


def _camera_state_for(sequence_id: str) -> CameraState:
    return _camera_states.setdefault(sequence_id, CameraState())


def _near_view(request: DroneFlybyPredictRequestDto, target: Tuple[int, int]) -> bool:
    return (
        math.hypot(request.view.center_x - target[0], request.view.center_y - target[1])
        <= 8.0
    )


def _advance_route_if_reached(
    request: DroneFlybyPredictRequestDto, state: CameraState
) -> None:
    if state.pending_route_target is None or request.view.resolution_level != 1:
        return
    if not _near_view(request, state.pending_route_target):
        return
    if state.bootstrap_index < len(BOOTSTRAP_ROUTE):
        state.bootstrap_index += 1
    else:
        state.scan_index = (state.scan_index + 1) % len(ENTRY_SCAN_ROUTE)
    state.pending_route_target = None


def _candidate_for_confirmation(
    detections: List[Detection], request: DroneFlybyPredictRequestDto
) -> Optional[Detection]:
    if request.view.resolution_level != 1:
        return None
    candidates = [
        detection
        for detection in detections
        if CONFIRMATION_MIN_CONFIDENCE <= detection[2] <= CONFIRMATION_MAX_CONFIDENCE
    ]
    if not candidates:
        return None
    return min(
        candidates,
        key=lambda detection: (
            abs(detection[2] - 0.42),
            _bbox_diagonal_pixels(detection[1]),
        ),
    )


def _bounded_center(bounds, x: float, y: float) -> Tuple[int, int]:
    return (
        int(round(min(max(x, bounds.minimum_center_x), bounds.maximum_center_x))),
        int(round(min(max(y, bounds.minimum_center_y), bounds.maximum_center_y))),
    )


def _request_if_reachable(
    request: DroneFlybyPredictRequestDto,
    level: int,
    target_x: float,
    target_y: float,
) -> Optional[RequestedViewDto]:
    constraints = request.camera_constraints
    if level not in constraints.allowed_resolution_levels:
        return None
    bounds = constraints.bounds_for_level(level)
    if bounds is None:
        return None
    center_x, center_y = _bounded_center(bounds, target_x, target_y)
    distance = math.hypot(
        center_x - request.view.center_x, center_y - request.view.center_y
    )
    if (
        level != 0
        and distance
        > constraints.maximum_center_delta + CAMERA_DISTANCE_EPSILON
    ):
        return None
    return RequestedViewDto(
        resolution_level=int(level), center_x=center_x, center_y=center_y
    )


def _step_toward_level_one(
    request: DroneFlybyPredictRequestDto, target: Tuple[int, int]
) -> Optional[RequestedViewDto]:
    """Move toward a scan point without ever sending an illegal command."""
    direct = _request_if_reachable(request, 1, target[0], target[1])
    if direct is not None:
        return direct

    bounds = request.camera_constraints.bounds_for_level(1)
    if bounds is None or 1 not in request.camera_constraints.allowed_resolution_levels:
        return None
    dx = target[0] - request.view.center_x
    dy = target[1] - request.view.center_y
    distance = math.hypot(dx, dy)
    if distance <= 0.0:
        return None
    step = request.camera_constraints.maximum_center_delta * 0.96
    center_x = request.view.center_x + dx * step / distance
    center_y = request.view.center_y + dy * step / distance
    center_x, center_y = _bounded_center(bounds, center_x, center_y)
    return _request_if_reachable(request, 1, center_x, center_y)


def choose_next_view(
    request: DroneFlybyPredictRequestDto,
    fresh_detections: Optional[List[Detection]] = None,
) -> Optional[RequestedViewDto]:
    """Bootstrap at L1, scan the entry band, and zoom to confirm candidates."""
    detections = fresh_detections or []
    state = _camera_state_for(request.sequence_id)

    # A rejected command means the evaluator and our planned route diverged.
    # Drop the stale target and briefly suppress another confirmation zoom.
    # The current request view is authoritative for the recovery command.
    if request.camera_command_feedback is not None:
        state.pending_route_target = None
        state.confirming = False
        state.last_confirmation_frame = request.frame

    _advance_route_if_reached(request, state)

    if request.view.resolution_level == 2:
        state.confirming = False
        # At an extreme L2 corner the nearest legal L1 center is 550.73 px
        # away. That is valid under the official 551 px limit, so this must
        # use the exact protocol limit rather than a percentage margin.
        return _request_if_reachable(
            request, 1, request.view.center_x, request.view.center_y
        )

    bootstrap_done = state.bootstrap_index >= len(BOOTSTRAP_ROUTE)
    if (
        bootstrap_done
        and request.frame - state.last_confirmation_frame
        >= CONFIRMATION_COOLDOWN_FRAMES
    ):
        candidate = _candidate_for_confirmation(detections, request)
        if candidate is not None:
            _, bbox, _ = candidate
            center_x = (bbox[0] + bbox[2]) * IMAGE_WIDTH / 2.0
            center_y = (bbox[1] + bbox[3]) * IMAGE_HEIGHT / 2.0
            command = _request_if_reachable(request, 2, center_x, center_y)
            if command is not None:
                state.confirming = True
                state.last_confirmation_frame = request.frame
                return command

    if state.pending_route_target is None:
        if not bootstrap_done:
            state.pending_route_target = BOOTSTRAP_ROUTE[state.bootstrap_index]
        else:
            state.pending_route_target = ENTRY_SCAN_ROUTE[state.scan_index]
    return _step_toward_level_one(request, state.pending_route_target)
