"""Behavioral checks for the offline strict-continuity experiment."""

import unittest
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np

from tools.experiments.phrase_octaves import correct_phrase_octaves


class PhraseOctaveExperimentTests(unittest.TestCase):
    @staticmethod
    def waveform(f0):
        samples = np.repeat(f0, 160)
        return np.sin(2 * np.pi * np.cumsum(samples) / 16000) * (samples > 0)

    def test_full_track_errors_and_previously_false_corrections(self):
        fixture = json.loads((Path(__file__).parent / 'fixtures/full_track_octave_transitions.json').read_text())
        expected = {
            'full_track_repeated_recoveries': fixture['cases'][0]['checks'],
            'weak_phrase_ending': fixture['cases'][1]['checks'],
            'following_note_supports_original': fixture['cases'][2]['checks'],
            # Strict mode intentionally folds this mid-phrase octave leap.
            'real_descending_note': [[47, 72, 2]],
        }
        for case in fixture['cases']:
            raw = np.array(case['f0_hz'])
            periodicity = np.array(case['periodicity'])
            with self.subTest(case=case['name']), patch(
                'tools.experiments.phrase_octaves._periodicity', return_value=periodicity
            ), patch(
                'tools.experiments.phrase_octaves._vocal_breaks',
                return_value=np.zeros(len(raw), dtype=bool),
            ):
                fixed = correct_phrase_octaves(raw, audio=np.ones(len(raw) * 160))
                for start, stop, factor in expected[case['name']]:
                    np.testing.assert_array_equal(fixed[start:stop], raw[start:stop] * factor)

    def test_sustained_register_change_folds_even_when_audio_supports_it(self):
        for following in (110., 440.):
            raw = np.r_[np.full(50, 220.), np.full(200, following)]
            fixed = correct_phrase_octaves(raw, audio=self.waveform(raw))
            np.testing.assert_array_equal(fixed, np.full(len(raw), 220.))

    def test_actual_vocal_break_allows_new_register(self):
        raw = np.r_[np.full(50, 220.), np.zeros(30), np.full(50, 440.)]
        np.testing.assert_array_equal(correct_phrase_octaves(raw, audio=self.waveform(raw)), raw)

    def test_voiced_audio_across_detector_dropout_keeps_register_and_zeros(self):
        raw = np.r_[np.full(50, 220.), np.zeros(30), np.full(50, 440.)]
        audio = self.waveform(np.full(len(raw), 220.))
        expected = raw.copy()
        expected[80:] /= 2
        np.testing.assert_array_equal(correct_phrase_octaves(raw, audio=audio), expected)

    def test_gradual_slide_can_change_register(self):
        raw = np.r_[np.full(40, 220.), 220 * 2 ** np.linspace(0, 1, 50), np.full(40, 440.)]
        np.testing.assert_array_equal(correct_phrase_octaves(raw), raw)

    def test_normal_intervals_and_low_syllable_endings(self):
        for interval in (3, 4, 5, 7):
            for direction in (-1, 1):
                raw = np.r_[np.full(40, 220.), np.full(50, 220 * 2 ** (direction * interval / 12)), np.full(40, 220.)]
                np.testing.assert_array_equal(correct_phrase_octaves(raw), raw)
        raw = np.r_[np.full(50, 220.), np.linspace(220, 110, 12), np.full(50, 246.94)]
        np.testing.assert_array_equal(correct_phrase_octaves(raw)[62:], raw[62:])

    def test_repeated_recoveries_during_descending_tail(self):
        clean = np.r_[np.full(50, 480.), np.linspace(470, 250, 40)]
        raw = clean.copy()
        raw[50:] *= 2
        raw[[54, 56]] = clean[[54, 56]]
        np.testing.assert_array_equal(correct_phrase_octaves(raw), clean)

    def test_no_waveform_evidence_does_not_establish_a_register(self):
        raw = np.r_[np.full(50, 220.), np.full(50, 440.)]
        np.testing.assert_array_equal(correct_phrase_octaves(raw, audio=np.zeros(len(raw)*160)), raw)

    def test_nonfinite_frames_reset_and_input_is_not_mutated(self):
        for gap in ([np.nan], [np.inf], np.zeros(30)):
            raw = np.r_[np.full(40, 220.), gap, np.full(40, 440.)]
            before = raw.copy()
            np.testing.assert_array_equal(correct_phrase_octaves(raw), raw)
            np.testing.assert_array_equal(raw, before)


if __name__ == '__main__':
    unittest.main()
