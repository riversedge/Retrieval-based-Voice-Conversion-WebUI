import time

import numpy as np
import torch
from fairseq import checkpoint_utils
from fairseq.data.dictionary import Dictionary

from infer.lib.device import get_device
from infer.lib.torch_load_compat import torch_load_compat


def load_hubert(device, model_path="assets/hubert/hubert_base.pt"):
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


def hubert_forward(device):
    model = load_hubert(device)
    audio = np.random.randn(16000).astype(np.float32)
    feats = torch.from_numpy(audio).float().unsqueeze(0).to(device)
    padding_mask = torch.zeros_like(feats, dtype=torch.bool).to(device)
    with torch.no_grad():
        _ = model.extract_features(
            source=feats, padding_mask=padding_mask, output_layer=9
        )


def train_one_step(device):
    model = torch.nn.Linear(512, 256).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    data = torch.randn(2, 512, device=device)
    target = torch.randn(2, 256, device=device)
    output = model(data)
    loss = torch.nn.functional.mse_loss(output, target)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


def main():
    device = get_device("cpu")
    start = time.perf_counter()
    hubert_forward(device)
    train_one_step(device)
    elapsed = time.perf_counter() - start
    print(f"Smoke test complete in {elapsed:.2f}s")


if __name__ == "__main__":
    main()
