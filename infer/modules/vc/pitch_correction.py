"""Opt-in octave selection using phrase continuity and source periodicity."""

import numpy as np


def _periodicity(audio, sample_rate, times, candidates):
    """Normalized autocorrelation at candidate periods, in bounded batches.

    An overtone can be strong even when the waveform repeats at the lower
    fundamental. Longer periods also correlate for many sounds, so continuity
    and a preference for the detector must resolve that ambiguity.
    """
    width = max(32, int(round(0.08 * sample_rate)))
    offsets = np.arange(width) - width // 2
    fft_size = 1 << (2 * width - 1).bit_length()
    scores = np.zeros(candidates.shape, dtype=np.float64)
    for start in range(0, len(times), 256):
        stop = min(start + 256, len(times))
        indices = np.rint(times[start:stop] * sample_rate).astype(np.int64)[:, None] + offsets
        frames = audio[np.clip(indices, 0, len(audio) - 1)].astype(np.float64)
        frames[(indices < 0) | (indices >= len(audio))] = 0
        frames -= frames.mean(axis=1, keepdims=True)
        spectrum = np.fft.rfft(frames, n=fft_size, axis=1)
        correlation = np.fft.irfft(spectrum * spectrum.conj(), n=fft_size, axis=1)[:, :width]
        energy = np.pad(np.cumsum(frames * frames, axis=1), ((0, 0), (1, 0)))
        periods = sample_rate / candidates[start:stop]
        row = np.arange(stop - start)[:, None]
        batch = np.full(periods.shape, -1.0)
        for offset in (-1, 0, 1):
            lag = np.clip(np.rint(periods).astype(np.int64) + offset, 1, width - 1)
            denominator = np.sqrt(np.maximum(energy[row, width - lag] * (energy[:, -1, None] - energy[row, lag]), 0))
            score = correlation[row, lag] / np.maximum(denominator, 1e-12)
            batch = np.maximum(batch, score)
        scores[start:stop] = np.clip(batch, 0, 1)
    return scores


def correct_octave_jumps(f0, frame_seconds=0.01, *, audio=None, sample_rate=16000):
    """Choose an octave path across each phrase; do not flatten its melody.

    Candidates are the original F0 and shifts of one/two octaves, bounded to
    50–1100 Hz (the extraction range). Dynamic programming balances continuity,
    source periodicity, and retaining the detector's octave. Notes on either
    side may differ, and an error may continue through a slide or phrase end.
    There is no maximum correction duration. Gaps up to 150 ms link context;
    longer gaps and invalid values reset it. Unvoiced values are never filled.

    Intentional large leaps may also be folded. This is not scale quantization,
    arbitrary-interval correction, or a way to recover an entirely wrong phrase
    without a trustworthy register reference. Audio starts at F0 frame zero.
    """
    values = np.asarray(f0)
    if values.ndim != 1:
        raise ValueError("Pitch must be a one-dimensional F0 curve.")
    if not np.isfinite(frame_seconds) or frame_seconds <= 0:
        raise ValueError("Pitch frame duration must be positive and finite.")
    if audio is not None:
        audio = np.asarray(audio)
        if audio.ndim != 1 or not np.all(np.isfinite(audio)):
            raise ValueError("Source audio must be a finite mono waveform.")
        if not np.isfinite(sample_rate) or sample_rate <= 0:
            raise ValueError("Sample rate must be positive and finite.")
    result = values.astype(np.result_type(values.dtype, np.float32), copy=True)
    valid = np.isfinite(values) & (values > 0)
    indices = np.flatnonzero(valid)
    if len(indices) < 2:
        return result

    shifts = np.arange(-2, 3)
    candidates = values[indices, None].astype(np.float64) * (2.0 ** shifts)
    pitches = 12 * np.log2(candidates)
    allowed = (candidates >= 50) & (candidates <= 1100)
    allowed[:, 2] = True
    emission = np.broadcast_to(0.015 * np.abs(shifts), candidates.shape).copy()
    if audio is not None and len(audio):
        scores = _periodicity(audio, sample_rate, indices * frame_seconds, candidates)
        best = np.max(np.where(allowed, scores, 0), axis=1, keepdims=True)
        # Weak/noisy frames should not decide the register; tolerate minor
        # differences from vibrato, transients and nonstationary vowels.
        evidence = np.maximum(best - 0.5, 0) * np.maximum(best - scores - 0.08, 0)
        emission += 0.5 * evidence
    emission *= frame_seconds / 0.01
    emission[~allowed] = np.inf

    gaps = np.diff(indices) - 1
    invalid_prefix = np.r_[0, np.cumsum(~np.isfinite(values))]
    invalid_gap = invalid_prefix[indices[1:]] > invalid_prefix[indices[:-1] + 1]
    cuts = np.r_[0, np.flatnonzero((gaps * frame_seconds > 0.15) | invalid_gap) + 1, len(indices)]
    change_cost = 0.8 * np.abs(shifts[:, None] - shifts[None, :])
    for start, stop in zip(cuts[:-1], cuts[1:]):
        if stop - start < 2:
            continue
        # A soft opening anchor favors the original register but still allows
        # correction of a faulty onset when later context is stronger.
        costs = emission[start] + 2.0 * np.abs(shifts)
        back = np.zeros((stop - start, len(shifts)), dtype=np.int8)
        for i in range(start + 1, stop):
            delta = np.abs(pitches[i][None, :] - pitches[i - 1][:, None])
            elapsed = (indices[i] - indices[i - 1]) * frame_seconds
            allowance = 5.5 + min(2.0, max(0.0, elapsed - 0.01) * 12)
            continuity = 0.3 * np.maximum(delta - allowance, 0) ** 2
            transitions = costs[:, None] + continuity + change_cost
            parent = np.argmin(transitions, axis=0)
            back[i - start] = parent
            costs = transitions[parent, np.arange(len(shifts))] + emission[i]
        state = int(np.argmin(costs))
        for i in range(stop - 1, start - 1, -1):
            result[indices[i]] = values[indices[i]] * (2.0 ** shifts[state])
            state = int(back[i - start, state])
    return result


def correct_brief_octave_jumps(f0, frame_seconds=0.01):
    """Compatibility alias for callers of the original continuity helper."""
    return correct_octave_jumps(f0, frame_seconds)
