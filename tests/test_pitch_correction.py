import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from infer.modules.vc.pitch_correction import correct_brief_octave_jumps
from infer.modules.vc.pipeline import Pipeline


class OctaveContinuityTests(unittest.TestCase):
    def test_doubling_and_halving_preserve_vibrato(self):
        time = np.arange(300) * 0.01
        clean = (220 * 2 ** (0.4 * np.sin(2 * np.pi * 5 * time) / 12)).astype(np.float32)
        wrong = clean.copy()
        wrong[50:65] *= 2
        wrong[150:165] /= 2
        before = wrong.copy()
        np.testing.assert_array_equal(correct_brief_octave_jumps(wrong), clean)
        np.testing.assert_array_equal(wrong, before)

    def test_adjacent_bursts_do_not_repair_the_good_notes_between_them(self):
        clean = np.full(150, 220.0)
        wrong = clean.copy()
        wrong[30:42] *= 2
        wrong[54:66] *= 2
        np.testing.assert_array_equal(correct_brief_octave_jumps(wrong), clean)

    def test_sustained_octave_and_non_octave_notes_stay(self):
        for middle in (np.full(70, 440.0), np.full(10, 330.0)):
            curve = np.concatenate((np.full(50, 220.0), middle, np.full(50, 220.0)))
            np.testing.assert_array_equal(correct_brief_octave_jumps(curve), curve)

    def test_gradual_octave_slide_stays(self):
        curve = np.concatenate((np.full(40, 220.0), 220 * 2 ** np.linspace(0, 1, 30), np.full(40, 440.0)))
        np.testing.assert_array_equal(correct_brief_octave_jumps(curve), curve)

    def test_no_correction_across_breaths_or_invalid_frames(self):
        for gap in (0.0, np.nan, np.inf):
            curve = np.concatenate((np.full(40, 220.0), np.full(10, 440.0), [gap], np.full(40, 220.0)))
            np.testing.assert_array_equal(correct_brief_octave_jumps(curve), curve)

    def test_phrase_edges_and_disagreeing_context_stay(self):
        curves = [
            np.r_[np.full(10, 440.0), np.full(50, 220.0)],
            np.r_[np.full(50, 220.0), np.full(10, 440.0)],
            np.r_[np.full(40, 220.0), np.full(10, 440.0), np.full(40, 330.0)],
        ]
        for curve in curves:
            np.testing.assert_array_equal(correct_brief_octave_jumps(curve), curve)

    def test_silence_empty_and_duration_in_seconds(self):
        for curve in (np.array([]), np.zeros(300), np.array([220.0])):
            np.testing.assert_array_equal(correct_brief_octave_jumps(curve), curve)
        # At 20 ms/frame, 10 frames is brief but 20 frames is not.
        brief = np.r_[np.full(30, 220.0), np.full(10, 440.0), np.full(30, 220.0)]
        sustained = np.r_[np.full(30, 220.0), np.full(20, 440.0), np.full(30, 220.0)]
        np.testing.assert_array_equal(correct_brief_octave_jumps(brief, 0.02), np.full(70, 220.0))
        np.testing.assert_array_equal(correct_brief_octave_jumps(sustained, 0.02), sustained)


class PitchPipelineTests(unittest.TestCase):
    def setUp(self):
        self.pipeline = Pipeline(40000, SimpleNamespace(
            x_pad=1, x_query=6, x_center=38, x_max=41, is_half=False, device="cpu",
        ))
        self.raw = np.full(450, 220.0, dtype=np.float32)
        self.raw[180:190] = 440
        self.pipeline.model_rmvpe = SimpleNamespace(infer_from_audio=lambda *a, **kw: self.raw.copy())

    def pitch(self, enabled=False, shift=0, f0_range=None, override=None):
        report = {}
        _, continuous = self.pipeline.get_f0(
            "fixture", np.ones(450 * 160, dtype=np.float32), 450,
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
        self.assertEqual(report["corrected_frames"], 10)

    def test_pitch_shift_and_range_still_apply_after_repair(self):
        shifted, report = self.pitch(True, shift=12)
        np.testing.assert_array_equal(shifted, np.full(450, 440.0))
        ranged, _ = self.pitch(True, shift=12, f0_range=(150, 300))
        np.testing.assert_array_equal(ranged, np.full(450, 220.0))

    def test_external_curve_is_authoritative(self):
        supplied = np.array([[0, 330], [1, 330]], dtype=np.float32)
        with patch("infer.modules.vc.pipeline.correct_brief_octave_jumps") as correction:
            curve, report = self.pitch(True, override=supplied)
        correction.assert_not_called()
        self.assertEqual(report["skipped"], "supplied F0 curve")
        np.testing.assert_array_equal(curve[100:201], np.full(101, 330.0))

    def test_reflected_padding_is_not_used_as_context(self):
        self.raw[:] = 220
        self.raw[100:110] = 440  # First actual samples, preceded only by model padding.
        curve, report = self.pitch(True)
        np.testing.assert_array_equal(curve, self.raw)
        self.assertEqual(report["corrected_frames"], 0)


if __name__ == "__main__":
    unittest.main()
