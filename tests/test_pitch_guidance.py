"""Behavioral tests for selected-detector octave and gap correction."""

import unittest

import numpy as np

from infer.modules.vc.pitch_guidance import correct_pitch_estimates


class PitchGuidanceTests(unittest.TestCase):
    @staticmethod
    def waveform(f0, amplitude=1):
        samples = np.repeat(np.asarray(f0), 160)
        return amplitude * np.sin(2 * np.pi * np.cumsum(samples) / 16000)

    def test_guide_corrects_only_octave_register(self):
        primary = np.r_[np.full(40, 220.), np.full(30, 440.), np.full(30, 293.66)]
        guide = np.r_[np.full(40, 110.), np.full(30, 110.), np.full(30, 138.59)]
        fixed, report = correct_pitch_estimates(
            primary, audio=self.waveform(np.full(len(primary), 220.)), guide=guide
        )
        np.testing.assert_array_equal(fixed[:70], np.full(70, 220.))
        # A one-semitone disagreement is source tuning, not a register error.
        np.testing.assert_array_equal(fixed[70:], primary[70:])
        self.assertEqual(report.guide_register_frames, 30)
        self.assertEqual(report.octave_corrected_frames, 30)

    def test_guide_prevailing_octave_is_normalized(self):
        primary = np.full(100, 440.)
        guide = np.full(100, 220.)
        fixed, report = correct_pitch_estimates(
            primary, audio=self.waveform(primary), guide=guide
        )
        np.testing.assert_array_equal(fixed, primary)
        self.assertEqual(report.corrected_frames, 0)

    def test_guide_shapes_dropout_but_meets_source_anchors(self):
        contour = np.linspace(220, 246.94, 50) + 50 * np.sin(np.linspace(0, np.pi, 50))
        primary = np.r_[np.full(30, 220.), np.zeros(50), np.full(30, 246.94)]
        guide = np.r_[
            np.full(30, 220.),
            contour,
            np.full(30, 246.94),
        ]
        fixed, report = correct_pitch_estimates(
            primary, audio=self.waveform(guide), guide=guide
        )
        self.assertEqual(report.bridged_frames, 50)
        self.assertTrue(np.all(fixed[30:80] > 0))
        self.assertGreater(fixed[54], fixed[30])
        self.assertGreater(fixed[54], fixed[78])
        self.assertLess(abs(12 * np.log2(fixed[30] / primary[29])), 1)
        self.assertLess(abs(12 * np.log2(primary[80] / fixed[79])), 1)

    def test_without_guide_bridges_in_log_frequency(self):
        primary = np.r_[np.full(40, 220.), np.zeros(12), np.full(40, 246.94)]
        audio = self.waveform(np.linspace(220, 246.94, len(primary)))
        fixed, report = correct_pitch_estimates(primary, audio=audio)
        self.assertTrue(np.all(fixed[40:52] > 0))
        self.assertGreater(fixed[51], fixed[40])
        self.assertEqual(report.bridged_frames, 12)

    def test_guide_stabilizes_brief_detector_recovery_after_gap(self):
        primary = np.r_[
            np.full(30, 220.), np.zeros(5),
            [150., 155., 180., 200.], np.full(30, 220.),
        ]
        guide = np.full(len(primary), 220.)
        fixed, report = correct_pitch_estimates(
            primary, audio=self.waveform(guide), guide=guide
        )
        np.testing.assert_allclose(fixed, guide)
        self.assertEqual(report.bridged_frames, 5)
        self.assertEqual(report.stabilized_frames, 4)

    def test_unbounded_audible_dropout_holds_nearest_source_pitch(self):
        primary = np.r_[np.full(40, 220.), np.zeros(20)]
        fixed, report = correct_pitch_estimates(
            primary, audio=self.waveform(np.full(len(primary), 220.))
        )
        np.testing.assert_allclose(fixed, np.full(len(primary), 220.))
        self.assertEqual(report.bridged_frames, 20)

    def test_real_silence_retains_gap_and_resets_register(self):
        primary = np.r_[np.full(40, 220.), np.zeros(20), np.full(40, 440.)]
        audio = self.waveform(primary)
        fixed, report = correct_pitch_estimates(primary, audio=audio, guide=primary)
        np.testing.assert_array_equal(fixed, primary)
        self.assertEqual(report.bridged_frames, 0)

    def test_without_guide_phrase_continuity_fixes_octave_run(self):
        primary = np.r_[np.full(50, 220.), np.full(40, 440.), np.full(40, 220.)]
        fixed, report = correct_pitch_estimates(
            primary, audio=self.waveform(np.full(len(primary), 220.))
        )
        np.testing.assert_array_equal(fixed, np.full(len(primary), 220.))
        self.assertEqual(report.octave_corrected_frames, 40)

    def test_invalid_guide_shape_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "equal lengths"):
            correct_pitch_estimates(np.full(20, 220.), guide=np.full(19, 220.))


if __name__ == "__main__":
    unittest.main()
