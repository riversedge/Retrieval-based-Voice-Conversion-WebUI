"""Alignment, chunking and inference-routing regression tests (no model downloads)."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from infer.modules.vc.guide import (
    AlignedGuide, GuideInput, align_guide, automatic_mapping, _dtw_mapping,
)
from infer.modules.vc.pipeline import Pipeline


class GuideAlignmentTests(unittest.TestCase):
    def test_full_tracks_and_bounded_dtw(self):
        rng = np.random.default_rng(4)
        phones = rng.normal(size=(40, 16)).astype(np.float32)
        source_lengths = rng.integers(60, 110, len(phones))
        guide_lengths = rng.integers(60, 130, len(phones))
        source = np.repeat(phones, source_lengths, axis=0)
        guide = np.repeat(phones, guide_lengths, axis=0)
        with patch("infer.modules.vc.guide._dtw_mapping", wraps=_dtw_mapping) as dtw:
            mapping = automatic_mapping(source, guide)
        self.assertGreater(len(source) / 50, 30)
        self.assertTrue(all(max(call.args[0].shape[0], call.args[1].shape[0]) <= 751 for call in dtw.call_args_list))
        self.assertTrue(np.all(np.diff(mapping) >= 0))
        source_labels = np.repeat(np.arange(len(phones)), source_lengths)
        guide_labels = np.repeat(np.arange(len(phones)), guide_lengths)
        accuracy = np.mean(source_labels == guide_labels[np.rint(mapping).astype(int)])
        self.assertGreater(accuracy, 0.95)
        audio = np.ones(180 * 16000, dtype=np.float32)
        GuideInput(audio).validate(audio)  # No arbitrary full-song duration cap.

    def test_anchors_and_region(self):
        source = np.ones((400, 3), dtype=np.float32)
        target = np.repeat(np.arange(500, dtype=np.float32)[:, None], 3, axis=1)
        guide = GuideInput(np.ones(160000), strength=1, anchors="2,3\n4,6", start=2, end=4)
        result = align_guide(source, target, np.ones(128000), guide)
        self.assertAlmostEqual(result.features[100, 0], 150)
        self.assertAlmostEqual(result.features[200, 0], 300)
        self.assertTrue(np.all(result.weights[:101] == 0))
        self.assertTrue(np.all(result.weights[200:] == 0))
        self.assertAlmostEqual(result.weights[150], 1)

    def test_bad_anchors_and_audio_fail_explicitly(self):
        for anchors in ("bad", "2,3\n1,4", "2,3\n3,2", "1,nan", "1,20"):
            guide = GuideInput(np.ones(64000), anchors=anchors)
            with self.assertRaises(ValueError):
                align_guide(np.ones((200, 2)), np.ones((200, 2)), np.ones(64000), guide)
        with self.assertRaisesRegex(ValueError, "silent"):
            GuideInput(np.zeros(16000)).validate(np.ones(16000))

    def test_chunk_global_time_and_half_frames(self):
        aligned = AlignedGuide(np.arange(200, dtype=np.float32)[:, None], np.ones(200), "content")
        features, weights = aligned.for_chunk(0, 16000, 100)
        self.assertTrue(np.all(weights[:50] == 0))
        self.assertEqual(features[50, 0], 0)
        features, weights = aligned.for_chunk(16000 + 160, 16000, 10)
        np.testing.assert_allclose(features[:, 0], np.arange(10) + 0.5)
        np.testing.assert_allclose(weights, 1)


class FakeEncoder:
    def extract_features(self, source, padding_mask, output_layer):
        frames = source.unfold(1, 400, 320).mean(dim=-1)
        return (torch.stack((frames, frames + 1, frames + 2), dim=-1),)

    def final_proj(self, features):
        return features * 0.5


class CapturingSynth:
    def infer(self, features, length, pitch, pitchf, sid):
        self.features = features.clone()
        self.pitch = pitch.clone()
        self.pitchf = pitchf.clone()
        return (features.sum(dim=-1)[:, None, :],)


class CapturingIndex:
    ntotal = 8

    def search(self, query, k):
        self.query = query.copy()
        return np.ones((len(query), k), dtype=np.float32), np.tile(np.arange(k), (len(query), 1))


class PipelineGuideTests(unittest.TestCase):
    def setUp(self):
        self.pipeline = Pipeline(40000, SimpleNamespace(
            x_pad=1, x_query=6, x_center=38, x_max=41, is_half=False, device="cpu",
        ))
        self.encoder = FakeEncoder()

    def test_long_feature_extraction_retains_frame_grid(self):
        audio = np.linspace(-1, 1, 65 * 16000 + 127, dtype=np.float32)
        for version in ("v1", "v2"):
            full = self.pipeline.extract_content(self.encoder, audio, version)[0].numpy()
            chunked = self.pipeline.track_content(self.encoder, audio, version)
            np.testing.assert_allclose(chunked, full)

    def test_filtered_audio_negative_strides(self):
        audio = np.linspace(-1, 1, 16000, dtype=np.float32)[::-1]
        np.testing.assert_array_equal(
            self.pipeline.track_content(self.encoder, audio, "v2"),
            self.pipeline.track_content(self.encoder, audio.copy(), "v2"),
        )

    def run_vc(self, guide=None, index=None, protect=0.5, voiced=True):
        synth = CapturingSynth()
        audio = np.ones(16000, dtype=np.float32) * 0.2
        pitch = torch.full((1, 100), 100 if voiced else 1)
        pitchf = torch.full((1, 100), 220.0 if voiced else 0.0)
        self.pipeline.vc(
            self.encoder, synth, torch.tensor([0]), audio, pitch, pitchf,
            [0, 0, 0], index, np.ones((8, 3), dtype=np.float32) * 9 if index else None,
            0.75, "v2", protect, guide, self.pipeline.t_pad,
        )
        return synth

    def test_zero_guidance_is_identical_and_f0_is_original(self):
        baseline = self.run_vc()
        disabled = self.run_vc(AlignedGuide(np.ones((200, 3)) * 8, np.zeros(200), "content"))
        torch.testing.assert_close(disabled.features, baseline.features, rtol=0, atol=0)
        enabled = self.run_vc(AlignedGuide(np.ones((200, 3)) * 8, np.ones(200), "content"))
        self.assertFalse(torch.equal(enabled.features, baseline.features))
        torch.testing.assert_close(enabled.pitch, baseline.pitch)
        torch.testing.assert_close(enabled.pitchf, baseline.pitchf)

    def test_retrieval_only_changes_query_and_original_protection_survives(self):
        index = CapturingIndex()
        baseline = self.run_vc(index=index)
        baseline_query = index.query.copy()
        guided = self.run_vc(AlignedGuide(np.ones((200, 3)) * 8, np.ones(200), "retrieval"), index=index)
        self.assertFalse(np.array_equal(index.query, baseline_query))
        # Fake retrieval returns the same neighbors: only the query should differ.
        torch.testing.assert_close(guided.features, baseline.features)
        protected = self.run_vc(AlignedGuide(np.ones((200, 3)) * 8, np.ones(200), "content"), protect=0, voiced=False)
        original = self.run_vc(protect=0, voiced=False)
        torch.testing.assert_close(protected.features, original.features)

    def test_exact_retrieval_matches_remain_finite(self):
        index = CapturingIndex()
        index.search = lambda query, k: (np.zeros((len(query), k)), np.tile(np.arange(k), (len(query), 1)))
        self.assertTrue(torch.isfinite(self.run_vc(index=index).features).all())


if __name__ == "__main__":
    unittest.main()
