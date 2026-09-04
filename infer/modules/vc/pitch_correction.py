"""Conservative repair of short pitch-doubling/halving excursions.

F0 alone cannot distinguish an intentional brief octave leap from a detector
error. This opt-in heuristic requires agreeing voiced context on BOTH sides;
it does not impose a key, quantize notes, or smooth the expressive pitch curve.
"""

import numpy as np


def correct_brief_octave_jumps(f0, frame_seconds=0.01):
    """Repair octave excursions lasting at most 250 ms, preserving local detail.

    A candidate must enter and leave by approximately one octave (12 +/- 2
    semitones), have at least 50 ms of stable voiced context on each side, and
    fit the neighboring register after an exact x2 or /2 correction. Sustained
    register changes, gradual slides, non-octave errors, and unvoiced gaps stay
    unchanged. Decisions use the unmodified track to avoid cascading repairs.
    """
    values = np.asarray(f0)
    if values.ndim != 1:
        raise ValueError("Pitch must be a one-dimensional F0 curve.")
    if not np.isfinite(frame_seconds) or frame_seconds <= 0:
        raise ValueError("Pitch frame duration must be positive and finite.")
    result = values.copy()
    valid = np.isfinite(values) & (values > 0)
    semitones = np.zeros(len(values), dtype=np.float64)
    semitones[valid] = 12 * np.log2(values[valid].astype(np.float64))
    max_frames = max(1, int(np.floor(0.25 / frame_seconds + 1e-8)))
    context_frames = max(1, int(round(0.12 / frame_seconds)))
    min_context = max(1, int(np.ceil(0.05 / frame_seconds)))
    boundaries = np.flatnonzero(np.diff(np.pad(valid.astype(np.int8), (1, 1))))

    for run_start, run_end in boundaries.reshape(-1, 2):
        if run_end - run_start < 2 * min_context + 1:
            continue
        pitches = semitones[run_start:run_end]
        steps = np.diff(pitches)
        jumps = np.flatnonzero(np.abs(np.abs(steps) - 12) <= 2) + run_start + 1
        repaired_until = run_start
        for position, start in enumerate(jumps):
            if start <= repaired_until or start - run_start < min_context:
                continue
            entry = semitones[start] - semitones[start - 1]
            shift = 12 if entry > 0 else -12
            left = semitones[max(run_start, start - context_frames):start]
            if np.ptp(left) > 2:
                continue
            for end in jumps[position + 1:]:
                if end - start > max_frames:
                    break
                exit_step = semitones[end] - semitones[end - 1]
                if abs(exit_step + shift) > 2 or run_end - end < min_context:
                    continue
                right = semitones[end:min(run_end, end + context_frames)]
                if np.ptp(right) > 2:
                    continue
                left_center, right_center = np.median(left), np.median(right)
                if abs(right_center - left_center) > 3:
                    continue
                corrected = semitones[start:end] - shift
                reference = np.linspace(left_center, right_center, len(corrected) + 2)[1:-1]
                if np.max(np.abs(corrected - reference)) > 2:
                    continue
                if (abs(corrected[0] - semitones[start - 1]) > 3
                        or abs(corrected[-1] - semitones[end]) > 3):
                    continue
                # Exact powers of two retain vibrato, drift, and scoops inside
                # the repaired span; do not replace it with the reference line.
                result[start:end] = values[start:end] * (0.5 if shift > 0 else 2.0)
                repaired_until = end
                break
    return result
