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


def _supported_path(indices, pitches, proposed, scores, cuts, frame_seconds):
    """Accept proposals against established notes, not transient low points.

    Decisions are sequential: only previously accepted corrections may supply
    context. A new stable note or a consonant gap requires fresh support, so a
    single octave decision cannot silently set the rest of the phrase's octave.
    """
    accepted = np.full(len(indices), 2, dtype=np.int8)
    accepted_quality = np.zeros(len(indices))
    context = max(3, int(round(0.12 / frame_seconds)))
    onset = max(1, int(round(0.04 / frame_seconds)))
    settling = max(3, int(round(0.08 / frame_seconds)))
    for phrase_start, phrase_stop in zip(cuts[:-1], cuts[1:]):
        i = phrase_start
        while i < phrase_stop:
            state = proposed[i]
            if state == 2:
                i += 1
                continue
            end = i + 1
            while end < phrase_stop and proposed[end] == state:
                gap = (indices[end] - indices[end - 1] - 1) * frame_seconds
                step = abs(pitches[end, 2] - pitches[end - 1, 2])
                right = pitches[end:min(end + settling, phrase_stop), 2]
                new_note = (step >= 3 and len(right) >= settling
                            and np.ptp(right) <= 2)
                if gap >= 0.03 or new_note:
                    break
                end += 1

            reference = None
            anchor_quality = 0.0
            continuation = False
            # A detector can briefly recover for one frame in a descending
            # octave error. Resume the supported trajectory across <=40 ms;
            # do not mistake that recovery for a new stable note/register.
            if i > phrase_start and proposed[i - 1] == 2:
                previous = i - 1
                while (previous >= phrase_start and accepted[previous] == 2
                       and (indices[i] - indices[previous]) * frame_seconds <= 0.04):
                    previous -= 1
                if (previous >= phrase_start and accepted[previous] == state
                        and (indices[i] - indices[previous]) * frame_seconds <= 0.05
                        and abs(pitches[i, state] - pitches[previous, state]) <= 3):
                    reference = pitches[previous, state]
                    anchor_quality = accepted_quality[previous]
                    continuation = True
            for anchor_end in range(i, phrase_start + context - 1, -1):
                if reference is not None:
                    break
                anchor_start = anchor_end - context
                if (indices[i] - indices[anchor_start]) * frame_seconds > 0.35:
                    break
                frames = np.arange(anchor_start, anchor_end)
                if np.any(np.diff(indices[frames]) > 1):
                    continue
                notes = pitches[frames, accepted[frames]]
                if np.ptp(notes) > 2:
                    continue
                reference = np.median(notes)
                if scores is not None:
                    anchor_quality = np.median(scores[frames, accepted[frames]])
                break

            if reference is not None:
                head = slice(i, min(i + onset, end))
                raw_distance = abs(np.median(pitches[head, 2]) - reference)
                new_distance = abs(np.median(pitches[head, state]) - reference)
                acoustic_support = False
                reliable = True
                if scores is not None:
                    raw_quality = np.median(scores[head, 2])
                    new_quality = np.median(scores[head, state])
                    # A trustworthy preceding note cannot compensate for a
                    # candidate pitch with weak waveform support. Once an
                    # episode is established, brief recoveries reuse that
                    # decision instead of rejudging each noisy frame.
                    reliable = (continuation or (new_quality >= 0.55
                                and (anchor_quality + new_quality) / 2 >= 0.6))
                    acoustic_support = new_quality - raw_quality >= 0.15
                # Ordinary intervals through a fifth are valid unless there is
                # positive acoustic evidence against the detected octave.
                continuity_support = (raw_distance > 7.2
                                      and raw_distance - new_distance >= 3)
                following_supports_original = False
                # Look ahead only to independently unmodified stable notes.
                # Comparing two proposed shifts would be circular evidence.
                for next_start in range(end, phrase_stop - context + 1):
                    next_end = next_start + context
                    if (indices[next_end - 1] - indices[end - 1]) * frame_seconds > 0.35:
                        break
                    frames = np.arange(next_start, next_end)
                    if (np.any(proposed[frames] != 2)
                            or np.any(np.diff(indices[frames]) > 1)
                            or np.ptp(pitches[frames, 2]) > 2):
                        continue
                    if scores is not None and np.median(scores[frames, 2]) < 0.65:
                        continue
                    following = np.median(pitches[frames, 2])
                    tail = slice(max(i, end - settling), end)
                    raw_exit = abs(np.median(pitches[tail, 2]) - following)
                    new_exit = abs(np.median(pitches[tail, state]) - following)
                    following_supports_original = (raw_exit <= 7.2
                                                   and new_exit - raw_exit >= 3)
                    break
                if (reliable and new_distance <= 7.2
                        and (continuity_support or acoustic_support)
                        and (not following_supports_original or acoustic_support)):
                    accepted[i:end] = state
                    accepted_quality[i:end] = anchor_quality
            i = end
    return accepted


def correct_octave_jumps(f0, frame_seconds=0.01, *, audio=None, sample_rate=16000):
    """Choose an octave path across each phrase; do not flatten its melody.

    Candidates are the original F0 and shifts of one/two octaves, bounded to
    50–1100 Hz (the extraction range). Dynamic programming balances continuity,
    source periodicity, and retaining the detector's octave. Proposals require
    stable preceding notes and reliable evidence; ambiguous transitions stay
    unchanged. Notes on either
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
    emission = np.broadcast_to(0.04 * np.abs(shifts), candidates.shape).copy()
    scores = None
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
    proposed = np.full(len(indices), 2, dtype=np.int8)
    for start, stop in zip(cuts[:-1], cuts[1:]):
        if stop - start < 2:
            continue
        # No prior stable note exists at a phrase opening. Start in the
        # detector's register, matching the acceptance gate below.
        costs = emission[start].copy()
        costs[shifts != 0] = np.inf
        back = np.zeros((stop - start, len(shifts)), dtype=np.int8)
        for i in range(start + 1, stop):
            delta = np.abs(pitches[i][None, :] - pitches[i - 1][:, None])
            elapsed = (indices[i] - indices[i - 1]) * frame_seconds
            allowance = 7.2 + min(2.0, max(0.0, elapsed - 0.01) * 12)
            continuity = 0.3 * np.maximum(delta - allowance, 0) ** 2
            transitions = costs[:, None] + continuity + change_cost
            parent = np.argmin(transitions, axis=0)
            back[i - start] = parent
            costs = transitions[parent, np.arange(len(shifts))] + emission[i]
        state = int(np.argmin(costs))
        for i in range(stop - 1, start - 1, -1):
            proposed[i] = state
            state = int(back[i - start, state])
    accepted = _supported_path(indices, pitches, proposed, scores, cuts, frame_seconds)
    result[indices] = values[indices] * (2.0 ** shifts[accepted])
    return result


def correct_brief_octave_jumps(f0, frame_seconds=0.01):
    """Compatibility alias for callers of the original continuity helper."""
    return correct_octave_jumps(f0, frame_seconds)
