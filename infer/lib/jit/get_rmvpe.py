import torch

from infer.lib.torch_load_compat import torch_load_compat


def get_rmvpe(model_path="assets/rmvpe/rmvpe.pt", device=torch.device("cpu")):
    from infer.lib.rmvpe import E2E

    model = E2E(4, 1, (2, 2))
    ckpt = torch_load_compat(model_path, map_location=device)
    model.load_state_dict(ckpt)
    model.eval()
    model = model.to(device)
    return model
