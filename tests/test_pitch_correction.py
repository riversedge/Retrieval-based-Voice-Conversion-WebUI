import unittest
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from infer.modules.vc.pitch_correction import correct_octave_jumps, _periodicity
from infer.modules.vc.pipeline import Pipeline


class OctaveContinuityTests(unittest.TestCase):
    def test_measured_true_errors_and_false_correction_regressions(self):
        fixture = json.loads((Path(__file__).parent / "fixtures/octave_transitions.json").read_text())
        for case in fixture["cases"]:
            with self.subTest(case=case["name"]):
                raw = np.array(case["f0_hz"])
                # Recorded source-waveform evidence makes these realistic
                # decision tests portable, without shipping private audio.
                with patch("infer.modules.vc.pitch_correction._periodicity",
                           return_value=np.array(case["periodicity"])):
                    fixed = correct_octave_jumps(raw, audio=np.zeros(len(raw) * 160))
                for start, stop, factor in case["checks"]:
                    np.testing.assert_array_equal(fixed[start:stop], raw[start:stop] * factor)

    def test_thirds_fourths_and_fifths_remain_melodic_choices(self):
        for interval in (3, 4, 5, 7):
            for direction in (-1, 1):
                middle = 220 * 2 ** (direction * interval / 12)
                raw = np.r_[np.full(40, 220.), np.full(50, middle), np.full(40, 220.)]
                np.testing.assert_array_equal(correct_octave_jumps(raw), raw)

    def test_low_syllable_transition_does_not_reset_next_notes(self):
        raw = np.r_[np.full(50, 220.), np.linspace(220, 110, 12),
                    np.full(50, 246.94), np.full(50, 293.66), np.full(30, 329.63)]
        fixed = correct_octave_jumps(raw)
        np.testing.assert_array_equal(fixed[62:], raw[62:])

    def test_waveform_without_pitch_evidence_cannot_authorize_correction(self):
        raw = np.r_[np.full(50, 220.), np.full(35, 440.), np.full(50, 220.)]
        np.testing.assert_array_equal(correct_octave_jumps(raw, audio=np.zeros(len(raw) * 160)), raw)

    def test_unsupported_onset_does_not_raise_the_following_phrase(self):
        raw = np.r_[np.full(8, 440.), np.full(100, 220.)]
        np.testing.assert_array_equal(correct_octave_jumps(raw), raw)

    def test_doubling_and_halving_preserve_vibrato(self):
        time = np.arange(300) * 0.01
        clean = (220 * 2 ** (0.4 * np.sin(2 * np.pi * 5 * time) / 12)).astype(np.float32)
        wrong = clean.copy()
        wrong[50:85] *= 2
        wrong[150:185] /= 2
        before = wrong.copy()
        np.testing.assert_array_equal(correct_octave_jumps(wrong), clean)
        np.testing.assert_array_equal(wrong, before)

    def test_adjacent_bursts_leave_good_notes_between_them(self):
        clean = np.full(150, 220.0)
        wrong = clean.copy()
        wrong[30:42] *= 2
        wrong[54:66] *= 2
        np.testing.assert_array_equal(correct_octave_jumps(wrong), clean)

    def test_different_notes_on_either_side_and_inside_error(self):
        clean = np.r_[np.full(40, 220.0), np.full(35, 246.94), np.full(35, 293.66), np.full(40, 261.63)]
        wrong = clean.copy()
        wrong[40:110] *= 2
        np.testing.assert_array_equal(correct_octave_jumps(wrong), clean)

    def test_octave_error_through_descending_phrase_end(self):
        clean = np.r_[np.full(50, 480.0), np.linspace(470, 250, 40)]
        wrong = clean.copy()
        wrong[50:] *= 2
        np.testing.assert_array_equal(correct_octave_jumps(wrong), clean)

    def test_representative_measured_tail_after_minus_twelve(self):
        # Representative detector frequencies from the 1:36 descending tail.
        raw = np.r_[np.full(40, 494.0), [487, 478, 937, 903, 880, 840, 704, 650, 608, 580, 534, 566, 498, 495]]
        fixed = correct_octave_jumps(raw)
        np.testing.assert_array_equal(fixed[:42], raw[:42])
        np.testing.assert_array_equal(fixed[42:], raw[42:] / 2)
        self.assertAlmostEqual(fixed[43] / 2, 225.75)  # Correction, then -12.

    def test_sustained_octave_can_change_but_ordinary_note_changes_stay(self):
        curve = np.r_[np.full(50, 220.0), np.full(200, 440.0), np.full(50, 220.0)]
        np.testing.assert_array_equal(correct_octave_jumps(curve), np.full(300, 220.0))
        melody = np.r_[np.full(50, 220.0), np.full(30, 330.0), np.full(50, 220.0)]
        np.testing.assert_array_equal(correct_octave_jumps(melody), melody)

    def test_gradual_octave_slide_stays(self):
        curve = np.r_[np.full(40, 220.0), 220 * 2 ** np.linspace(0, 1, 30), np.full(40, 440.0)]
        np.testing.assert_array_equal(correct_octave_jumps(curve), curve)

    def test_short_gap_links_notes_without_filling_silence(self):
        curve = np.r_[np.full(50, 220.0), np.zeros(5), np.full(40, 440.0)]
        expected = np.r_[np.full(50, 220.0), np.zeros(5), np.full(40, 220.0)]
        np.testing.assert_array_equal(correct_octave_jumps(curve), expected)

    def test_long_breaths_and_invalid_values_reset_register(self):
        for gap in (np.zeros(30), [np.nan], [np.inf]):
            curve = np.r_[np.full(40, 220.0), gap, np.full(40, 440.0)]
            np.testing.assert_array_equal(correct_octave_jumps(curve), curve)

    def test_empty_silent_constant_and_alternate_frame_rate(self):
        for curve in (np.array([]), np.zeros(300), np.array([220.0]), np.full(100, 440.0)):
            np.testing.assert_array_equal(correct_octave_jumps(curve), curve)
        curve = np.r_[np.full(50, 220.0), np.full(40, 440.0), np.full(50, 220.0)]
        np.testing.assert_array_equal(correct_octave_jumps(curve, .02), np.full(140, 220.0))

    def test_waveform_periodicity_distinguishes_strong_overtone(self):
        sr = 16000
        t = np.arange(sr) / sr
        audio = np.sin(2 * np.pi * 220 * t) + 1.5 * np.sin(2 * np.pi * 440 * t)
        scores = _periodicity(audio, sr, np.array([.5]), np.array([[220.0, 440.0]]))
        self.assertGreater(scores[0, 0], .98)
        self.assertGreater(scores[0, 0] - scores[0, 1], .4)
        raw = np.r_[np.full(40, 220.0), np.full(60, 440.0)]
        np.testing.assert_array_equal(correct_octave_jumps(raw, audio=audio), np.full(100, 220.0))

    def test_waveform_evidence_can_resolve_ambiguous_interval(self):
        sr = 16000
        # Five-semitone downward note misdetected an octave high: raw jump is
        # only seven semitones, which continuity alone reasonably preserves.
        raw = np.r_[np.full(40, 220.0), np.full(50, 330.0), np.full(40, 220.0)]
        clean = raw.copy()
        clean[40:90] /= 2
        instantaneous = np.repeat(clean, 160)
        phase = 2 * np.pi * np.cumsum(instantaneous) / sr
        audio = np.sin(phase) + .4 * np.sin(2 * phase)
        np.testing.assert_array_equal(correct_octave_jumps(raw), raw)
        np.testing.assert_array_equal(correct_octave_jumps(raw, audio=audio), clean)

    def test_invalid_arguments(self):
        for duration in (0, -1, np.nan):
            with self.assertRaises(ValueError):
                correct_octave_jumps([220], duration)
        with self.assertRaises(ValueError):
            correct_octave_jumps([[220]])
        with self.assertRaises(ValueError):
            correct_octave_jumps([220], audio=[np.nan])


class PitchPipelineTests(unittest.TestCase):
    def setUp(self):
        self.pipeline = Pipeline(40000, SimpleNamespace(
            x_pad=1, x_query=6, x_center=38, x_max=41, is_half=False, device="cpu",
        ))
        self.raw = np.full(450, 220.0, dtype=np.float32)
        self.raw[180:215] = 440
        self.pipeline.model_rmvpe = SimpleNamespace(infer_from_audio=lambda *a, **kw: self.raw.copy())

    def pitch(self, enabled=False, shift=0, f0_range=None, override=None):
        report = {}
        # Waveform evidence must agree with the known synthetic fundamental.
        time = np.arange(450 * 160) / 16000
        audio = np.sin(2 * np.pi * 220 * time).astype(np.float32)
        _, continuous = self.pipeline.get_f0(
            "fixture", audio, 450,
            shift, "rmvpe", 3, f0_range, override,
            correct_octave_errors=enabled, pitch_report=report,
        )
        return continuous, report

    def test_off_retains_existing_curve_and_on_reports_repairs(self):
        off, report = self.pitch()
        np.testing.assert_array_equal(off, self.raw)
        self.assertEqual(report, {})
        on, report = self.pitch(True)
        np.testing.assert_array_equal(on, np.full(450, 220.0))
        self.assertEqual(report["corrected_frames"], 35)

    def test_pitch_shift_and_range_still_apply_after_repair(self):
        shifted, _ = self.pitch(True, shift=-12)
        np.testing.assert_array_equal(shifted, np.full(450, 110.0))
        ranged, _ = self.pitch(True, shift=-12, f0_range=(150, 300))
        np.testing.assert_array_equal(ranged, np.full(450, 220.0))

    def test_external_curve_is_authoritative(self):
        supplied = np.array([[0, 330], [1, 330]], dtype=np.float32)
        with patch("infer.modules.vc.pipeline.correct_octave_jumps") as correction:
            curve, report = self.pitch(True, override=supplied)
        correction.assert_not_called()
        self.assertEqual(report["skipped"], "supplied F0 curve")
        np.testing.assert_array_equal(curve[100:201], np.full(101, 330.0))

    def test_reflected_padding_is_excluded_from_pitch_and_audio(self):
        with patch("infer.modules.vc.pipeline.correct_octave_jumps", side_effect=lambda f, *a, **kw: f.copy()) as correction:
            self.pitch(True)
        args, kwargs = correction.call_args
        np.testing.assert_array_equal(args[0], self.raw[100:-100])
        self.assertEqual(len(kwargs['audio']), 250 * 160)
        self.assertEqual(kwargs['sample_rate'], 16000)


if __name__ == "__main__":
    unittest.main()
