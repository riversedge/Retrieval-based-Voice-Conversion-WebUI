import os

from fairseq import checkpoint_utils


def get_index_path_from_model(sid):
    return next(
        (
            f
            for f in [
                os.path.join(root, name)
                for root, _, files in os.walk(os.getenv("index_root"), topdown=False)
                for name in files
                if name.endswith(".index") and "trained" not in name
            ]
            if sid.split(".")[0] in f
        ),
        "",
    )


def load_hubert(config):
    import torch
    from fairseq import checkpoint_utils

    hubert_path = "assets/hubert/hubert_base.pt"

    # --- BEGIN: torch.load compat for torch>=2.6 + fairseq ---
    _orig_torch_load = torch.load

    def _torch_load_no_weights_only(*args, **kwargs):
        # fairseq doesn't pass weights_only; force False for this load
        kwargs.setdefault("weights_only", False)
        return _orig_torch_load(*args, **kwargs)

    torch.load = _torch_load_no_weights_only
    try:
        models, _, _ = checkpoint_utils.load_model_ensemble_and_task(
            [hubert_path],
            suffix="",
        )
    finally:
        torch.load = _orig_torch_load
    # --- END ---

    hubert_model = models[0].to(config.device)
    hubert_model = hubert_model.half() if config.is_half else hubert_model.float()
    return hubert_model.eval()


def load_hubert_old(config):
    models, _, _ = checkpoint_utils.load_model_ensemble_and_task(
        ["assets/hubert/hubert_base.pt"],
        suffix="",
    )
    hubert_model = models[0]
    hubert_model = hubert_model.to(config.device)
    if config.is_half:
        hubert_model = hubert_model.half()
    else:
        hubert_model = hubert_model.float()
    return hubert_model.eval()
