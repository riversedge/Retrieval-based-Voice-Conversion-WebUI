import argparse
import os
import time

import numpy as np
import torch
import soundfile as sf
import librosa
from fairseq import checkpoint_utils
from fairseq.data.dictionary import Dictionary

from infer.lib.device import get_device
from infer.lib.torch_load_compat import torch_load_compat
from infer.lib.rmvpe import RMVPE


def _load_audio(audio_path, target_sr=16000, duration_sec=10):
    if audio_path:
        audio, sr = sf.read(audio_path)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if sr != target_sr:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
        return audio, target_sr
    audio = np.random.randn(int(target_sr * duration_sec)).astype(np.float32)
    return audio, target_sr


def _load_hubert(model_path, device):
    try:
        torch.serialization.add_safe_globals([Dictionary])
    except Exception:
        pass
    original_torch_load = torch.load

    def _torch_load(*args, **kwargs):
        kwargs.setdefault("weights_only_default", False)
        return torch_load_compat(*args, load_fn=original_torch_load, **kwargs)

    torch.load = _torch_load
    try:
        models, _, _ = checkpoint_utils.load_model_ensemble_and_task(
            [model_path], suffix=""
        )
    finally:
        torch.load = original_torch_load
    model = models[0].to(device)
    model.eval()
    return model


def benchmark_hubert(model_path, device, audio_path=None, duration_sec=60):
    audio, _ = _load_audio(audio_path, duration_sec=duration_sec)
    feats = torch.from_numpy(audio).float().unsqueeze(0).to(device)
    padding_mask = torch.zeros_like(feats, dtype=torch.bool).to(device)
    model = _load_hubert(model_path, device)

    start = time.perf_counter()
    with torch.no_grad():
        _ = model.extract_features(
            source=feats, padding_mask=padding_mask, output_layer=9
        )
    elapsed = time.perf_counter() - start
    minutes = duration_sec / 60.0
    return elapsed / minutes


def benchmark_training_step(device, iterations=10, batch_size=4):
    model = torch.nn.Sequential(
        torch.nn.Linear(1024, 1024),
        torch.nn.ReLU(),
        torch.nn.Linear(1024, 1024),
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    losses = []
    start = time.perf_counter()
    for _ in range(iterations):
        data = torch.randn(batch_size, 1024, device=device)
        target = torch.randn(batch_size, 1024, device=device)
        output = model(data)
        loss = torch.nn.functional.mse_loss(output, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    elapsed = time.perf_counter() - start
    return (elapsed / iterations) * 1000.0


def benchmark_inference(device, audio_path=None, rmvpe_path="assets/rmvpe/rmvpe.pt"):
    if not os.path.exists(rmvpe_path):
        return None
    audio, _ = _load_audio(audio_path, duration_sec=10)
    model = RMVPE(rmvpe_path, is_half=False, device=device)
    start = time.perf_counter()
    _ = model.infer_from_audio(audio, thred=0.03)
    elapsed = time.perf_counter() - start
    return elapsed


def main():
    parser = argparse.ArgumentParser(description="macOS benchmark harness")
    parser.add_argument("--audio", type=str, default="", help="Path to audio file")
    parser.add_argument("--hubert", type=str, default="assets/hubert/hubert_base.pt")
    parser.add_argument("--rmvpe", type=str, default="assets/rmvpe/rmvpe.pt")
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    device = get_device(args.device)
    print(f"Device: {device}")

    hubert_speed = benchmark_hubert(args.hubert, device, args.audio or None)
    print(f"HuBERT feature extraction: {hubert_speed:.2f}s / minute audio")

    train_ms = benchmark_training_step(device, args.iterations, args.batch_size)
    print(f"Training step time: {train_ms:.2f} ms/iter ({args.iterations} iters)")

    inference_time = benchmark_inference(device, args.audio or None, args.rmvpe)
    if inference_time is None:
        print("Inference time: RMVPE model not found; skipped")
    else:
        print(f"Inference time (10s clip): {inference_time:.2f}s")


if __name__ == "__main__":
    main()
