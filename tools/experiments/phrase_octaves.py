"""Offline trial of strict octave continuity; not enabled in the WebUI.

Stable notes establish a phrase's register. Abrupt changes favor the nearest
octave of the next note, even with ambiguous waveform evidence. Real vocal
breaks reset the register; pitch-detector dropouts alone do not. This deliberately
folds some intentional octave leaps. It never fills unvoiced frames.
"""

import numpy as np

from infer.modules.vc.pitch_correction import _periodicity


def _vocal_breaks(audio, sample_rate, count, frame_seconds):
    """Find sustained quiet or aperiodic spans independently of detected F0.

    Aperiodicity is a heuristic for a vocal break, not a breath classifier.
    Analyze bounded batches of 80 ms windows, including detector dropouts.
    """
    width = max(32, round(.08 * sample_rate))
    offsets = np.arange(width) - width // 2
    fft_size = 1 << (2 * width - 1).bit_length()
    lags = np.arange(max(1, int(sample_rate / 1100)),
                     min(width - 1, int(sample_rate / 50)) + 1)
    rms = np.zeros(count)
    quality = np.zeros(count)
    for start in range(0, count, 256):
        stop = min(start + 256, count)
        centers = np.rint(np.arange(start, stop) * frame_seconds * sample_rate).astype(int)
        positions = centers[:, None] + offsets
        frames = audio[np.clip(positions, 0, len(audio) - 1)].astype(np.float64)
        frames[(positions < 0) | (positions >= len(audio))] = 0
        frames -= frames.mean(axis=1, keepdims=True)
        # Short energy windows retain actual pauses rather than smearing them.
        half = max(1, round(.005 * sample_rate))
        rms[start:stop] = np.sqrt(np.mean(frames[:, width // 2-half:width // 2+half] ** 2, axis=1))
        spectrum = np.fft.rfft(frames, n=fft_size, axis=1)
        ac = np.fft.irfft(spectrum * spectrum.conj(), n=fft_size, axis=1)[:, :width]
        energy = np.pad(np.cumsum(frames ** 2, axis=1), ((0, 0), (1, 0)))
        denominator = np.sqrt(np.maximum(
            energy[:, width-lags] * (energy[:, -1, None] - energy[:, lags]), 0))
        quality[start:stop] = np.maximum(0, np.max(ac[:, lags] / np.maximum(denominator, 1e-12), axis=1))
    quiet = (rms < max(1e-5, np.percentile(rms, 95) * .03)) | (quality < .5)
    breaks = np.zeros(count, dtype=bool)
    for start, stop in np.flatnonzero(np.diff(np.r_[False, quiet, False])).reshape(-1, 2):
        if (stop - start) * frame_seconds >= .15:
            breaks[start:stop] = True
    return breaks


def correct_phrase_octaves(f0, frame_seconds=.01, *, audio=None, sample_rate=16000):
    """Apply octave-only phrase continuity for an offline comparison render.

    Reliable 180 ms notes establish anchors. Changes through a fifth remain
    available; larger abrupt intervals fold only when an octave alternative
    improves continuity by at least three semitones. Continuous slides can
    change register. Intervening uncertain frames use both neighboring anchors.
    """
    values = np.asarray(f0)
    if values.ndim != 1 or not np.isfinite(frame_seconds) or frame_seconds <= 0:
        raise ValueError("Expected a one-dimensional F0 curve and positive frame duration.")
    if not np.isfinite(sample_rate) or sample_rate <= 0:
        raise ValueError("Sample rate must be positive and finite.")
    if audio is not None:
        audio = np.asarray(audio)
        if audio.ndim != 1 or not np.all(np.isfinite(audio)):
            raise ValueError("Source audio must be a finite mono waveform.")
    result = values.astype(np.result_type(values.dtype, np.float32), copy=True)
    indices = np.flatnonzero(np.isfinite(values) & (values > 0))
    if len(indices) < 2:
        return result
    notes = 12 * np.log2(values[indices].astype(np.float64))
    shifts = np.arange(-2, 3)
    pitches = notes[:, None] + 12 * shifts
    candidates = values[indices, None] * 2. ** shifts
    allowed = (candidates >= 50) & (candidates <= 1100)
    allowed[:, 2] = True
    scores = None
    if audio is not None and len(audio):
        scores = _periodicity(audio, sample_rate, indices * frame_seconds, candidates)
        breaks = _vocal_breaks(audio, sample_rate, len(values), frame_seconds)
        prefix = np.r_[0, np.cumsum(breaks)]
        reset = prefix[indices[1:] + 1] > prefix[indices[:-1]]
    else:
        reset = (np.diff(indices) - 1) * frame_seconds > .15
    invalid = np.r_[0, np.cumsum(~np.isfinite(values))]
    reset |= invalid[indices[1:]] > invalid[indices[:-1] + 1]
    cuts = np.r_[0, np.flatnonzero(reset) + 1, len(indices)]
    context = max(3, round(.18 / frame_seconds))
    max_gap = max(1, round(.03 / frame_seconds))

    def choose(distances, permitted, preferred):
        distances = np.where(permitted, distances, np.inf)
        if not permitted[preferred]:
            preferred = 2
        # When two registers both connect within a fifth, prefer the smaller
        # change to RMVPE instead of chasing the mathematically nearest anchor.
        # This matters in badly fragmented passages where interpolation alone
        # can favor a two-octave move by less than a semitone.
        best = int(np.argmin(distances + 2.0 * abs(shifts)))
        if distances[preferred] > 7.2 and distances[preferred] - distances[best] >= 3:
            return best
        return preferred

    for start, stop in zip(cuts[:-1], cuts[1:]):
        anchors = []
        i = start
        while i + context <= stop:
            end = i + context
            window = notes[i:end]
            if (np.ptp(window) > 2 or np.any(np.diff(indices[i:end]) > max_gap)
                    or (indices[end - 1] - indices[i]) * frame_seconds > .2
                    or (scores is not None and np.median(scores[i:end, 2]) < .6)):
                i += 1
                continue
            low, high = np.min(window), np.max(window)
            while end < stop and indices[end] - indices[end - 1] <= max_gap:
                new_low, new_high = min(low, notes[end]), max(high, notes[end])
                if new_high - new_low > 2:
                    break
                low, high = new_low, new_high
                end += 1
            anchors.append([i, end, float(np.median(notes[i:end])), 2])
            i = end
        if not anchors:
            continue
        previous = None
        for anchor in anchors:
            a, b, center, state = anchor
            if previous is not None:
                _, q, previous_center, previous_state = previous
                transition = slice(q - 1, a + 1)
                smooth = (np.all(np.diff(indices[transition]) <= max_gap)
                          and np.all(abs(np.diff(notes[transition])) < 3))
                state = previous_state
                permitted = np.all(allowed[a:b], axis=0)
                if not smooth or not permitted[state]:
                    state = choose(abs(center + 12 * shifts - previous_center), permitted, state)
            anchor[2], anchor[3] = center + 12 * shifts[state], state
            result[indices[a:b]] = values[indices[a:b]] * 2. ** shifts[state]
            if previous is not None:
                _, q, previous_center, previous_state = previous
                for k in range(q, a):
                    weight = (indices[k] - indices[q - 1]) / max(1, indices[a] - indices[q - 1])
                    reference = previous_center * (1 - weight) + anchor[2] * weight
                    selected = choose(abs(pitches[k] - reference), allowed[k], previous_state)
                    result[indices[k]] = values[indices[k]] * 2. ** shifts[selected]
            previous = anchor
        _, end, reference, state = anchors[-1]
        for k in range(end, stop):
            smooth = (indices[k] - indices[k - 1] <= max_gap
                      and abs(notes[k] - notes[k - 1]) < 3)
            if not smooth or not allowed[k, state]:
                state = choose(abs(pitches[k] - reference), allowed[k], state)
            result[indices[k]] = values[indices[k]] * 2. ** shifts[state]
            reference = 12 * np.log2(result[indices[k]])
    return result
