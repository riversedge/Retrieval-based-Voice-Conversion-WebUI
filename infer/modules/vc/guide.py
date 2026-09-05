"""Align Guide Vocals for pronunciation and optional octave-register evidence.

No waveform mixing or guide tuning transfer is involved. Alignment is an
estimate, not a phoneme recognizer.
"""

from dataclasses import dataclass, field
import json

import numpy as np
from scipy.spatial.distance import cdist


SAMPLE_RATE = 16000
FEATURE_HOP = 320  # HuBERT/ContentVec: 20 ms, before RVC's 2x interpolation.
COARSE_FRAMES = 750
REFINE_FRAMES = 500


@dataclass
class GuideInput:
    audio: np.ndarray
    strength: float = 0.5
    mode: str = "retrieval"
    alignment: str = "auto"
    anchors: str = ""
    start: float = 0.0
    end: float = 0.0  # Zero means the end of the source clip.
    report: dict = field(default_factory=dict)

    def validate(self, source):
        if not np.isfinite(self.strength) or not 0 <= self.strength <= 1:
            raise ValueError("Guide strength must be between 0 and 1.")
        if self.mode not in ("content", "retrieval"):
            raise ValueError("Guide mode must be content or retrieval.")
        if self.alignment not in ("auto", "linear"):
            raise ValueError("Guide alignment must be auto or linear.")
        for name, audio in (("Source", source), ("Guide", self.audio)):
            if audio.ndim != 1 or not np.isfinite(audio).all():
                raise ValueError(f"{name} must be finite mono audio.")
            if len(audio) / SAMPLE_RATE < 0.1:
                raise ValueError(f"{name} must contain at least 0.1 seconds of audio.")
            if np.max(np.abs(audio)) < 1e-6:
                raise ValueError(f"{name} audio is silent.")
        duration = len(source) / SAMPLE_RATE
        end = self.end or duration
        if not (np.isfinite(self.start) and np.isfinite(end)
                and 0 <= self.start < end <= duration):
            raise ValueError("Guide region must satisfy 0 <= start < end <= source duration.")


def _sample_frames(features, positions):
    """Linear interpolation along time; never interpolate across feature channels."""
    positions = np.clip(positions, 0, len(features) - 1)
    left = np.floor(positions).astype(np.int64)
    right = np.minimum(left + 1, len(features) - 1)
    fraction = (positions - left).astype(np.float32)[:, None]
    return features[left] * (1 - fraction) + features[right] * fraction


def _activity(audio, count):
    """Conservative silence gate so guide words do not fill instrumental gaps."""
    usable = min(count, len(audio) // FEATURE_HOP)
    frames = audio[:usable * FEATURE_HOP].reshape(usable, FEATURE_HOP)
    rms = np.sqrt(np.mean(np.square(frames, dtype=np.float64), axis=1))
    threshold = max(1e-6, float(np.max(rms)) * 0.01)
    activity = np.clip((rms - threshold) / threshold, 0, 1)
    activity = np.pad(activity, (0, count - usable), mode="edge")
    return np.convolve(np.pad(activity, (2, 2), mode="edge"), np.ones(5) / 5, mode="valid")


def _anchor_mapping(text, source_count, guide_count, source_duration, guide_duration):
    """Explicit anchors override automatic alignment, including optional endpoints."""
    pairs = []
    for line in text.splitlines():
        if not line.strip():
            continue
        try:
            a, b = (float(value.strip()) for value in line.split(","))
        except ValueError as exc:
            raise ValueError("Each timing anchor must be: source_seconds,guide_seconds") from exc
        if not (np.isfinite(a) and np.isfinite(b)
                and 0 <= a <= source_duration and 0 <= b <= guide_duration):
            raise ValueError("Timing anchors must lie inside their respective clips.")
        if pairs and (a <= pairs[-1][0] or b <= pairs[-1][1]):
            raise ValueError("Timing anchors must increase strictly in both recordings.")
        pairs.append((a, b))
    # Endpoints may be explicit to skip leading/trailing material in the guide.
    if not pairs or pairs[0][0] > 0:
        pairs.insert(0, (0.0, 0.0))
    if pairs[-1][0] < source_duration:
        pairs.append((source_duration, guide_duration))
    points = np.asarray(pairs)
    if np.any(np.diff(points[:, 1]) <= 0):
        raise ValueError("Guide anchor times must increase, including clip endpoints.")
    source_times = np.arange(source_count) * FEATURE_HOP / SAMPLE_RATE
    guide_times = np.interp(source_times, points[:, 0], points[:, 1])
    return np.clip(guide_times * SAMPLE_RATE / FEATURE_HOP, 0, guide_count - 1)


def _dtw_mapping(source, target):
    """Small DTW problem. Callers bound the matrix dimensions, not song length."""
    import librosa

    source_unit = source / np.maximum(np.linalg.norm(source, axis=1, keepdims=True), 1e-8)
    target_unit = target / np.maximum(np.linalg.norm(target, axis=1, keepdims=True), 1e-8)
    # Equivalent to cosine distance for nonzero normalized vectors, and avoids
    # spurious float32 Accelerate/BLAS warnings on some macOS NumPy builds.
    cost = np.clip(cdist(source_unit, target_unit, metric="sqeuclidean") * 0.5, 0, 2)
    _, path = librosa.sequence.dtw(C=cost, weights_add=np.array([0, 0.08, 0.08]))
    sums = np.bincount(path[:, 0], weights=path[:, 1], minlength=len(source))
    counts = np.bincount(path[:, 0], minlength=len(source))
    return sums / np.maximum(counts, 1)


def _pooled_features(features):
    edges = np.linspace(0, len(features), min(len(features), COARSE_FRAMES) + 1).astype(int)
    pooled = np.stack([features[a:b].mean(axis=0) for a, b in zip(edges[:-1], edges[1:])])
    centers = (edges[:-1] + edges[1:] - 1) / 2
    return pooled, centers


def automatic_mapping(source, target):
    """Coarse song alignment, then bounded local refinement at 20 ms resolution.

    Matrix memory is independent of track length. Very long/unequal sections are
    subdivided in BOTH timelines; no single full-track quadratic DTW is built.
    Coarse boundaries are estimates and can be overridden with timing anchors.
    """
    if max(len(source), len(target)) <= COARSE_FRAMES:
        return _dtw_mapping(source, target)
    coarse_source, source_centers = _pooled_features(source)
    coarse_target, target_centers = _pooled_features(target)
    coarse_map = _dtw_mapping(coarse_source, coarse_target)
    target_positions = np.interp(coarse_map, np.arange(len(target_centers)), target_centers)
    mapping = np.interp(np.arange(len(source)), source_centers, target_positions)
    mapping[0], mapping[-1] = 0, len(target) - 1

    def refine(a, b, c, d):
        if b - a > REFINE_FRAMES or d - c > REFINE_FRAMES:
            # Split at the existing monotone coarse path. If a path is nearly
            # vertical/horizontal, split its longer dimension to guarantee progress.
            if b - a > 1:
                mid_source = (a + b) // 2
                mid_target = int(np.clip(round(mapping[mid_source]), c, d))
            else:
                mid_source, mid_target = a, (c + d) // 2
            refine(a, mid_source, c, mid_target)
            refine(mid_source, b, mid_target, d)
            return
        if b == a:
            mapping[a] = (c + d) / 2
            return
        if d == c:
            mapping[a:b + 1] = c
            return
        local = _dtw_mapping(source[a:b + 1], target[c:d + 1]) + c
        # Fixed endpoints keep neighboring refined sections monotonic.
        local[0], local[-1] = c, d
        mapping[a:b + 1] = local

    refine(0, len(source) - 1, 0, len(target) - 1)
    return np.maximum.accumulate(mapping)


def align_guide(source_features, guide_features, source_audio, guide):
    """Return source-timed guide features and a compact diagnostic report.

    Automatic DTW uses cosine distance and penalizes insertions/deletions. Manual
    anchors instead define a piecewise linear warp, useful when a slurred vowel
    makes acoustic matching unreliable. Both recordings must contain the same
    words in the same order. Instrumental gaps are allowed; different arrangements
    or repeated verses can need manual anchors.
    """
    guide.validate(source_audio)
    source = np.asarray(source_features, dtype=np.float32)
    target = np.asarray(guide_features, dtype=np.float32)
    if (source.ndim != 2 or target.ndim != 2 or min(len(source), len(target)) < 2
            or source.shape[1] != target.shape[1]
            or not np.isfinite(source).all() or not np.isfinite(target).all()):
        raise ValueError("Cannot align invalid or incompatible content features.")
    source_duration = len(source_audio) / SAMPLE_RATE
    guide_duration = len(guide.audio) / SAMPLE_RATE
    method = guide.alignment
    if guide.anchors.strip():
        mapping = _anchor_mapping(
            guide.anchors, len(source), len(target), source_duration, guide_duration
        )
        method = "anchors"
    elif method == "linear":
        mapping = np.linspace(0, len(target) - 1, len(source))
    else:
        mapping = automatic_mapping(source, target)

    aligned = _sample_frames(target, mapping)
    times = np.arange(len(source)) * FEATURE_HOP / SAMPLE_RATE
    end = guide.end or source_duration
    # Only fade at edit boundaries. Original protection still handles unvoiced
    # frames later, using original F0 and ORIGINAL (unguided) content features.
    weights = np.minimum(
        np.clip((times - guide.start) / 0.04, 0, 1),
        np.clip((end - times) / 0.04, 0, 1),
    ).astype(np.float32) * guide.strength
    weights *= _activity(source_audio, len(source))
    weights *= np.interp(mapping, np.arange(len(target)), _activity(guide.audio, len(target)))
    similarity = np.sum(source * aligned, axis=1) / np.maximum(
        np.linalg.norm(source, axis=1) * np.linalg.norm(aligned, axis=1), 1e-8
    )
    plateau = int(np.count_nonzero(np.diff(mapping) < 0.1))
    report = {
        "mode": guide.mode,
        "strength": guide.strength,
        "alignment": method,
        "source_seconds": round(source_duration, 3),
        "guide_seconds": round(guide_duration, 3),
        "region_seconds": [guide.start, end],
        "mean_feature_similarity": round(float(np.mean(similarity)), 4),
        "stationary_mapping_fraction": round(plateau / max(1, len(mapping) - 1), 4),
        # Frame map is useful for inspecting alignments; similarity is not a
        # calibrated confidence score or an assessment of pronunciation quality.
        "source_frame_hop_seconds": FEATURE_HOP / SAMPLE_RATE,
        "guide_seconds_by_source_frame": (mapping * FEATURE_HOP / SAMPLE_RATE).round(4).tolist(),
    }
    guide.report.update(report)
    return AlignedGuide(aligned, weights, guide.mode, mapping)


@dataclass
class AlignedGuide:
    features: np.ndarray
    weights: np.ndarray
    mode: str
    mapping: np.ndarray = None

    def align_pitch(self, guide_f0, frame_count, frame_seconds=0.01):
        """Warp guide F0 onto the source grid without crossing unvoiced runs."""
        guide_f0 = np.asarray(guide_f0)
        if guide_f0.ndim != 1 or not np.isfinite(guide_f0).all():
            raise ValueError("Guide pitch must be a finite one-dimensional F0 curve.")
        if self.mapping is None:
            raise ValueError("Guide alignment does not include a timing map.")
        if frame_count < 0 or not np.isfinite(frame_seconds) or frame_seconds <= 0:
            raise ValueError("Pitch frame count and duration must be valid.")
        source_feature_positions = (
            np.arange(frame_count) * frame_seconds * SAMPLE_RATE / FEATURE_HOP
        )
        guide_feature_positions = np.interp(
            source_feature_positions,
            np.arange(len(self.mapping)),
            self.mapping,
        )
        guide_pitch_positions = (
            guide_feature_positions * FEATURE_HOP / SAMPLE_RATE / frame_seconds
        )
        result = np.zeros(frame_count, dtype=np.result_type(guide_f0.dtype, np.float32))
        voiced = guide_f0 > 0
        runs = np.flatnonzero(np.diff(np.r_[False, voiced, False])).reshape(-1, 2)
        for start, stop in runs:
            use = ((guide_pitch_positions >= start)
                   & (guide_pitch_positions <= stop - 1))
            if np.any(use):
                result[use] = np.interp(
                    guide_pitch_positions[use], np.arange(start, stop), guide_f0[start:stop]
                )
        return result

    def for_chunk(self, start_sample, pad_samples, frame_count):
        # Chunks can begin on a 10 ms F0 boundary, halfway between content frames.
        # Keep that half-frame offset instead of truncating it. Reflect only for
        # model context padding; reflected weights are zero outside the real clip.
        positions = (start_sample - pad_samples) / FEATURE_HOP + np.arange(frame_count)
        last = len(self.features) - 1
        reflected = np.mod(positions, 2 * last)
        reflected = np.where(reflected > last, 2 * last - reflected, reflected)
        features = _sample_frames(self.features, reflected)
        weights = np.interp(positions, np.arange(len(self.weights)), self.weights, left=0, right=0)
        return features, weights.astype(np.float32)


def guide_summary(report):
    if not report:
        return "Guide disabled."
    summary = {key: value for key, value in report.items() if key != "guide_seconds_by_source_frame"}
    return "Guide Vocals:\n" + json.dumps(summary, indent=2)
