# Mac Studio Quickstart

## Supported Python Versions
- Recommended: Python 3.10 / 3.11 / 3.12
- 3.13: may work but expect edge-case dependency issues.

## Install
```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## MPS Enablement
```bash
python - <<'PY'
import torch
print("MPS available:", torch.backends.mps.is_available())
print("MPS built:", torch.backends.mps.is_built())
PY
```

## Recommended Environment Variables
```bash
export RVC_DEVICE=mps
export PYTORCH_ENABLE_MPS_FALLBACK=1
export RVC_AMP=0
```

## Troubleshooting
- **HuBERT load errors / “Weights only load failed”**
  - Update to the latest repo; model loading uses a PyTorch 2.6+ compatibility layer.
- **matplotlib canvas errors**
  - TensorBoard image logging now uses `buffer_rgba()` with a safe fallback. If you still see issues, set `RVC_TB_IMAGES=0`.
- **TensorBoard missing**
  - Install it explicitly: `pip install tensorboard`.
- **.DS_Store in audio folders**
  - macOS metadata files can confuse dataset scans. Delete them or add a cleanup step:
    ```bash
    find logs -name .DS_Store -delete
    ```
