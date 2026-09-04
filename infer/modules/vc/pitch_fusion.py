"""Fuse a primary pitch detector with FCPE and phrase octave continuity."""

from dataclasses import dataclass

import numpy as np

from infer.modules.vc.pitch_correction import _periodicity


@dataclass(frozen=True)
class PitchFusionReport:
    agreement_frames: int
    fcpe_recovered_frames: int
    bridged_frames: int
    octave_corrected_frames: int
    corrected_frames: int


def _validate_curve(f0, name):
    values = np.asarray(f0)
    if values.ndim != 1:
        raise ValueError(f"{name} pitch must be a one-dimensional F0 curve.")
    return values


def _source_silence(audio, sample_rate, count, frame_seconds):
    """Mark sustained near-silence without treating consonants as phrase breaks."""
    silence = np.zeros(count, dtype=bool)
    if audio is None or not len(audio) or not count:
        return silence
    width = max(8, int(round(.01 * sample_rate)))
    offsets = np.arange(width) - width // 2
    rms = np.zeros(count)
    for start in range(0, count, 1024):
        stop = min(start + 1024, count)
        centers = np.rint(np.arange(start, stop) * frame_seconds * sample_rate).astype(np.int64)
        positions = centers[:, None] + offsets
        frames = audio[np.clip(positions, 0, len(audio) - 1)].astype(np.float64)
        frames[(positions < 0) | (positions >= len(audio))] = 0
        rms[start:stop] = np.sqrt(np.mean(frames * frames, axis=1))
    threshold = max(1e-5, float(np.percentile(rms, 95)) * .03)
    quiet = rms < threshold
    minimum = max(1, int(round(.08 / frame_seconds)))
    edges = np.flatnonzero(np.diff(np.r_[False, quiet, False])).reshape(-1, 2)
    for start, stop in edges:
        if stop - start >= minimum:
            silence[start:stop] = True
    return silence


def _same_pitch(primary, secondary, tolerance=.75):
    both = (np.isfinite(primary) & (primary > 0)
            & np.isfinite(secondary) & (secondary > 0))
    agreement = np.zeros(len(primary), dtype=bool)
    agreement[both] = abs(12 * np.log2(primary[both] / secondary[both])) <= tolerance
    return agreement


def _octave_preference(base, secondary, index):
    if secondary is None or not np.isfinite(secondary[index]) or secondary[index] <= 0:
        return None
    distance = 12 * np.log2(secondary[index] / base[index])
    shift = int(np.rint(distance / 12))
    if shift < -1 or shift > 1 or abs(distance - 12 * shift) > .75:
        return None
    return shift + 1


def correct_phrase_octaves(
    f0,
    frame_seconds=.01,
    *,
    audio=None,
    sample_rate=16000,
    alternate=None,
    locked=None,
    silence=None,
):
    """Select a continuous register inside audible phrases.

    Only one-octave alternatives are considered. Reliable notes establish the
    register, while smooth slides and normal melodic intervals remain intact.
    Frames where both detectors agree can be locked to their measured pitch.
    """
    values = _validate_curve(f0, "Primary")
    if not np.isfinite(frame_seconds) or frame_seconds <= 0:
        raise ValueError("Pitch frame duration must be positive and finite.")
    if not np.isfinite(sample_rate) or sample_rate <= 0:
        raise ValueError("Sample rate must be positive and finite.")
    if audio is not None:
        audio = np.asarray(audio)
        if audio.ndim != 1 or not np.all(np.isfinite(audio)):
            raise ValueError("Source audio must be a finite mono waveform.")
    if alternate is not None:
        alternate = _validate_curve(alternate, "Alternate")
        if len(alternate) != len(values):
            raise ValueError("Primary and alternate pitch curves must have equal lengths.")
    locked = np.zeros(len(values), dtype=bool) if locked is None else np.asarray(locked, dtype=bool)
    if locked.shape != values.shape:
        raise ValueError("Locked pitch mask must match the F0 curve.")
    if silence is None:
        silence = _source_silence(audio, sample_rate, len(values), frame_seconds)
    else:
        silence = np.asarray(silence, dtype=bool)
        if silence.shape != values.shape:
            raise ValueError("Silence mask must match the F0 curve.")

    result = values.astype(np.result_type(values.dtype, np.float32), copy=True)
    indices = np.flatnonzero(np.isfinite(values) & (values > 0) & ~silence)
    if len(indices) < 2:
        return result

    notes = 12 * np.log2(values[indices].astype(np.float64))
    shifts = np.arange(-1, 2)
    original = 1
    pitches = notes[:, None] + 12 * shifts
    candidates = values[indices, None] * 2. ** shifts
    allowed = (candidates >= 50) & (candidates <= 1100)
    allowed[:, original] = True
    scores = None
    if audio is not None and len(audio):
        scores = _periodicity(audio, sample_rate, indices * frame_seconds, candidates)

    reset_prefix = np.r_[0, np.cumsum(silence)]
    reset = reset_prefix[indices[1:] + 1] > reset_prefix[indices[:-1]]
    invalid = np.r_[0, np.cumsum(~np.isfinite(values))]
    reset |= invalid[indices[1:]] > invalid[indices[:-1] + 1]
    cuts = np.r_[0, np.flatnonzero(reset) + 1, len(indices)]
    context = max(3, round(.18 / frame_seconds))
    max_gap = max(1, round(.03 / frame_seconds))

    def choose(distances, permitted, preferred, curve_index):
        distances = np.where(permitted, distances, np.inf)
        if locked[curve_index]:
            return original
        if not permitted[preferred]:
            preferred = original
        costs = distances + 2.0 * abs(shifts)
        secondary_state = _octave_preference(values, alternate, curve_index)
        if secondary_state is not None and permitted[secondary_state]:
            costs[secondary_state] -= 2.5
        best = int(np.argmin(costs))
        if distances[preferred] > 7.2 and distances[preferred] - distances[best] >= 3:
            return best
        if secondary_state is not None and distances[secondary_state] <= 7.2:
            if distances[preferred] - distances[secondary_state] >= 1:
                return secondary_state
        return preferred

    for start, stop in zip(cuts[:-1], cuts[1:]):
        anchors = []
        i = start
        while i + context <= stop:
            end = i + context
            window = notes[i:end]
            consensus = np.mean(locked[indices[i:end]]) >= .8
            reliable = scores is None or np.median(scores[i:end, original]) >= .6
            if (np.ptp(window) > 2 or np.any(np.diff(indices[i:end]) > max_gap)
                    or (indices[end - 1] - indices[i]) * frame_seconds > .2
                    or (not reliable and not consensus)):
                i += 1
                continue
            low, high = np.min(window), np.max(window)
            while end < stop and indices[end] - indices[end - 1] <= max_gap:
                new_low, new_high = min(low, notes[end]), max(high, notes[end])
                if new_high - new_low > 2:
                    break
                low, high = new_low, new_high
                end += 1
            anchors.append([i, end, float(np.median(notes[i:end])), original,
                            np.mean(locked[indices[i:end]]) >= .8])
            i = end
        if not anchors:
            continue

        previous = None
        for anchor in anchors:
            a, b, center, state, consensus = anchor
            if previous is not None and not consensus:
                _, q, previous_center, previous_state, _ = previous
                transition = slice(q - 1, a + 1)
                smooth = (np.all(np.diff(indices[transition]) <= max_gap)
                          and np.all(abs(np.diff(notes[transition])) < 3))
                state = previous_state
                permitted = np.all(allowed[a:b], axis=0)
                curve_index = indices[(a + b - 1) // 2]
                if not smooth or not permitted[state]:
                    state = choose(abs(center + 12 * shifts - previous_center),
                                   permitted, state, curve_index)
            if consensus:
                state = original
            anchor[2], anchor[3] = center + 12 * shifts[state], state
            result[indices[a:b]] = values[indices[a:b]] * 2. ** shifts[state]
            anchor_indices = indices[a:b]
            result[anchor_indices[locked[anchor_indices]]] = values[
                anchor_indices[locked[anchor_indices]]
            ]
            if previous is not None:
                _, q, previous_center, previous_state, _ = previous
                for k in range(q, a):
                    weight = (indices[k] - indices[q - 1]) / max(1, indices[a] - indices[q - 1])
                    reference = previous_center * (1 - weight) + anchor[2] * weight
                    selected = choose(abs(pitches[k] - reference), allowed[k],
                                      previous_state, indices[k])
                    result[indices[k]] = values[indices[k]] * 2. ** shifts[selected]
            previous = anchor

        _, end, reference, state, _ = anchors[-1]
        for k in range(end, stop):
            smooth = (indices[k] - indices[k - 1] <= max_gap
                      and abs(notes[k] - notes[k - 1]) < 3)
            if not smooth or not allowed[k, state] or locked[indices[k]]:
                state = choose(abs(pitches[k] - reference), allowed[k], state, indices[k])
            result[indices[k]] = values[indices[k]] * 2. ** shifts[state]
            reference = 12 * np.log2(result[indices[k]])

    result[locked] = values[locked]
    result[silence] = 0
    return result


def _bridge_unpitched(f0, eligible):
    result = f0.copy()
    edges = np.flatnonzero(np.diff(np.r_[False, eligible, False])).reshape(-1, 2)
    bridged = np.zeros(len(result), dtype=bool)
    for start, stop in edges:
        if (start == 0 or stop == len(result)
                or result[start - 1] <= 0 or result[stop] <= 0):
            continue
        left = np.log2(result[start - 1])
        right = np.log2(result[stop])
        result[start:stop] = 2. ** np.linspace(left, right, stop - start + 2)[1:-1]
        bridged[start:stop] = True
    return result, bridged


def fuse_pitch_estimates(primary, fcpe, frame_seconds=.01, *, audio=None, sample_rate=16000):
    """Apply RMVPE/FCPE consensus, recovery, phrase rules, and gap bridging."""
    primary = _validate_curve(primary, "Primary")
    fcpe = _validate_curve(fcpe, "FCPE")
    if len(primary) != len(fcpe):
        raise ValueError("Primary and FCPE pitch curves must have equal lengths.")
    if audio is not None:
        audio = np.asarray(audio)
        if audio.ndim != 1 or not np.all(np.isfinite(audio)):
            raise ValueError("Source audio must be a finite mono waveform.")

    silence = _source_silence(audio, sample_rate, len(primary), frame_seconds)
    primary_valid = np.isfinite(primary) & (primary > 0)
    fcpe_valid = np.isfinite(fcpe) & (fcpe > 0)
    agreement = _same_pitch(primary, fcpe) & ~silence
    recovered = ~primary_valid & fcpe_valid & ~silence
    base = primary.astype(np.result_type(primary.dtype, np.float32), copy=True)
    base[recovered] = fcpe[recovered]
    base[silence] = 0

    corrected = correct_phrase_octaves(
        base, frame_seconds, audio=audio, sample_rate=sample_rate,
        alternate=fcpe, locked=agreement, silence=silence,
    )
    both_missing = ~primary_valid & ~fcpe_valid & ~silence
    corrected, bridged = _bridge_unpitched(corrected, both_missing)
    corrected[silence] = 0

    comparable = primary_valid & (corrected > 0)
    octave_changed = np.zeros(len(primary), dtype=bool)
    octave_changed[comparable] = abs(12 * np.log2(
        corrected[comparable] / primary[comparable])) > 6
    changed = ((primary_valid != (corrected > 0))
               | (comparable & ~np.isclose(primary, corrected, rtol=1e-5, atol=1e-4)))
    report = PitchFusionReport(
        agreement_frames=int(np.count_nonzero(agreement)),
        fcpe_recovered_frames=int(np.count_nonzero(recovered & (corrected > 0))),
        bridged_frames=int(np.count_nonzero(bridged)),
        octave_corrected_frames=int(np.count_nonzero(octave_changed)),
        corrected_frames=int(np.count_nonzero(changed)),
    )
    return corrected, report
