"""Behavioral tests for the opt-in RMVPE/FCPE pitch fusion path."""

import unittest

import numpy as np

from infer.modules.vc.pitch_fusion import fuse_pitch_estimates


class PitchFusionTests(unittest.TestCase):
    @staticmethod
    def waveform(f0, amplitude=1):
        samples = np.repeat(np.asarray(f0), 160)
        return amplitude * np.sin(2 * np.pi * np.cumsum(samples) / 16000)

    def test_consensus_is_authoritative(self):
        primary = np.full(100, 220.)
        fixed, report = fuse_pitch_estimates(
            primary, primary.copy(), audio=self.waveform(primary)
        )
        np.testing.assert_array_equal(fixed, primary)
        self.assertEqual(report.agreement_frames, len(primary))
        self.assertEqual(report.corrected_frames, 0)

    def test_sustained_consensus_register_change_is_trusted(self):
        primary = np.r_[np.full(50, 220.), np.full(50, 440.)]
        fixed, report = fuse_pitch_estimates(
            primary, primary.copy(), audio=self.waveform(primary)
        )
        np.testing.assert_array_equal(fixed, primary)
        self.assertEqual(report.agreement_frames, len(primary))
        self.assertEqual(report.corrected_frames, 0)

    def test_fcpe_supplies_missing_primary_frames(self):
        primary = np.r_[np.full(40, 220.), np.zeros(12), np.full(40, 220.)]
        fcpe = np.full(len(primary), 220.)
        fixed, report = fuse_pitch_estimates(
            primary, fcpe, audio=self.waveform(fcpe)
        )
        np.testing.assert_allclose(fixed, fcpe)
        self.assertEqual(report.fcpe_recovered_frames, 12)
        self.assertEqual(report.bridged_frames, 0)

    def test_fcpe_only_run_uses_the_surrounding_register(self):
        primary = np.r_[np.full(40, 220.), np.zeros(12), np.full(40, 220.)]
        fcpe = primary.copy()
        fcpe[40:52] = 440
        fixed, report = fuse_pitch_estimates(
            primary, fcpe, audio=self.waveform(np.full(len(primary), 220.))
        )
        np.testing.assert_allclose(fixed, np.full(len(primary), 220.))
        self.assertEqual(report.fcpe_recovered_frames, 12)

    def test_both_missing_are_bridged_inside_an_audible_phrase(self):
        primary = np.r_[np.full(40, 220.), np.zeros(12), np.full(40, 246.94)]
        fcpe = primary.copy()
        audio = self.waveform(np.linspace(220, 246.94, len(primary)))
        fixed, report = fuse_pitch_estimates(primary, fcpe, audio=audio)
        self.assertTrue(np.all(fixed[40:52] > 0))
        self.assertGreater(fixed[51], fixed[40])
        self.assertEqual(report.bridged_frames, 12)

    def test_long_detector_dropout_still_bridges_when_audio_continues(self):
        primary = np.r_[np.full(40, 220.), np.zeros(50), np.full(40, 220.)]
        fcpe = primary.copy()
        fixed, report = fuse_pitch_estimates(
            primary, fcpe, audio=self.waveform(np.full(len(primary), 220.))
        )
        np.testing.assert_allclose(fixed, np.full(len(primary), 220.))
        self.assertEqual(report.bridged_frames, 50)

    def test_real_silence_retains_the_gap_and_resets_register(self):
        primary = np.r_[np.full(40, 220.), np.zeros(20), np.full(40, 440.)]
        fcpe = primary.copy()
        audio = self.waveform(primary)
        fixed, report = fuse_pitch_estimates(primary, fcpe, audio=audio)
        np.testing.assert_array_equal(fixed, primary)
        self.assertEqual(report.bridged_frames, 0)

    def test_fcpe_octave_evidence_and_phrase_continuity_fix_primary(self):
        primary = np.r_[np.full(50, 220.), np.full(40, 440.), np.full(40, 220.)]
        fcpe = np.full(len(primary), 220.)
        fixed, report = fuse_pitch_estimates(
            primary, fcpe, audio=self.waveform(fcpe)
        )
        np.testing.assert_array_equal(fixed, fcpe)
        self.assertEqual(report.octave_corrected_frames, 40)

    def test_non_octave_disagreement_does_not_replace_the_note(self):
        primary = np.r_[np.full(50, 220.), np.full(40, 293.66), np.full(40, 220.)]
        fcpe = np.r_[np.full(50, 220.), np.full(40, 277.18), np.full(40, 220.)]
        fixed, _ = fuse_pitch_estimates(primary, fcpe, audio=self.waveform(primary))
        np.testing.assert_array_equal(fixed, primary)

    def test_phrase_correction_never_moves_two_octaves_at_once(self):
        primary = np.r_[np.full(50, 220.), np.full(50, 880.)]
        fcpe = np.r_[np.full(50, 220.), np.full(50, 440.)]
        fixed, _ = fuse_pitch_estimates(primary, fcpe, audio=self.waveform(fcpe))
        np.testing.assert_array_equal(fixed[50:], np.full(50, 440.))


if __name__ == "__main__":
    unittest.main()
