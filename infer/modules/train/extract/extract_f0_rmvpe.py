import argparse
import os
import sys
import traceback

now_dir = os.getcwd()
sys.path.append(now_dir)
import logging

import numpy as np

from infer.lib.audio import load_audio
from infer.lib.device import get_rmvpe_device

logging.getLogger("numba").setLevel(logging.WARNING)


def _parse_args(argv):
    if not any(arg.startswith("--") for arg in argv):
        if len(argv) < 5:
            raise ValueError(
                "Expected legacy args: n_part i_part i_gpu exp_dir is_half"
            )
        return {
            "n_part": int(argv[0]),
            "i_part": int(argv[1]),
            "i_gpu": argv[2],
            "exp_dir": argv[3],
            "is_half": argv[4],
            "device": None,
            "dry_run": False,
        }
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-part", type=int, default=1)
    parser.add_argument("--i-part", type=int, default=0)
    parser.add_argument("--i-gpu", default="0")
    parser.add_argument("--exp-dir", required=True)
    parser.add_argument("--is-half", default="true")
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda", "mps", "dml", "gpu"],
        default=None,
        help="RMVPE device preference",
    )
    parser.add_argument("--dry-run", action="store_true")
    return vars(parser.parse_args(argv))


args = _parse_args(sys.argv[1:])
n_part = args["n_part"]
i_part = args["i_part"]
i_gpu = args["i_gpu"]
exp_dir = args["exp_dir"]
use_half = str(args["is_half"]).lower() in {"true", "1", "yes"}
device_prefer = args["device"]
dry_run = args["dry_run"]
os.environ["CUDA_VISIBLE_DEVICES"] = str(i_gpu)
f = open("%s/extract_f0_feature.log" % exp_dir, "a+")


def printt(strr):
    print(strr)
    f.write("%s\n" % strr)
    f.flush()


def _select_rmvpe_device(prefer):
    requested = (
        (prefer or os.getenv("RVC_RMVPE_DEVICE") or os.getenv("RVC_DEVICE") or "")
        .strip()
        .lower()
    )
    if requested == "gpu":
        requested = ""
    device, backend = get_rmvpe_device(prefer or None)
    if requested in {"cuda", "mps", "cpu", "dml"} and backend != requested:
        printt(
            "Requested RMVPE device '%s' is unavailable; falling back to %s."
            % (requested, backend)
        )
    return device, backend


def _log_backend(backend):
    printt("RMVPE backend: %s" % backend)


def _run_dry_run(device, backend, use_half):
    from infer.lib.rmvpe import RMVPE

    _log_backend(backend)
    printt("Loading rmvpe model (dry-run)")
    model = RMVPE("assets/rmvpe/rmvpe.pt", is_half=use_half, device=device)
    dummy_audio = np.zeros(16000, dtype=np.float32)
    _ = model.infer_from_audio(dummy_audio, thred=0.03)
    printt("RMVPE dry-run succeeded")


class FeatureInput(object):
    def __init__(self, device, backend, use_half, samplerate=16000, hop_size=160):
        self.fs = samplerate
        self.hop = hop_size
        self.device = device
        self.backend = backend
        self.use_half = use_half
        self.model_loaded = False
        self.f0_bin = 256
        self.f0_max = 1100.0
        self.f0_min = 50.0
        self.f0_mel_min = 1127 * np.log(1 + self.f0_min / 700)
        self.f0_mel_max = 1127 * np.log(1 + self.f0_max / 700)

    def compute_f0(self, path, f0_method):
        x = load_audio(path, self.fs)
        if f0_method == "rmvpe":
            if not self.model_loaded:
                from infer.lib.rmvpe import RMVPE

                _log_backend(self.backend)
                print("Loading rmvpe model")
                self.model_rmvpe = RMVPE(
                    "assets/rmvpe/rmvpe.pt", is_half=self.use_half, device=self.device
                )
                self.model_loaded = True
            f0 = self.model_rmvpe.infer_from_audio(x, thred=0.03)
        return f0

    def coarse_f0(self, f0):
        f0_mel = 1127 * np.log(1 + f0 / 700)
        f0_mel[f0_mel > 0] = (f0_mel[f0_mel > 0] - self.f0_mel_min) * (
            self.f0_bin - 2
        ) / (self.f0_mel_max - self.f0_mel_min) + 1

        # use 0 or 1
        f0_mel[f0_mel <= 1] = 1
        f0_mel[f0_mel > self.f0_bin - 1] = self.f0_bin - 1
        f0_coarse = np.rint(f0_mel).astype(int)
        assert f0_coarse.max() <= 255 and f0_coarse.min() >= 1, (
            f0_coarse.max(),
            f0_coarse.min(),
        )
        return f0_coarse

    def go(self, paths, f0_method):
        if len(paths) == 0:
            printt("no-f0-todo")
        else:
            printt("todo-f0-%s" % len(paths))
            n = max(len(paths) // 5, 1)  # 每个进程最多打印5条
            for idx, (inp_path, opt_path1, opt_path2) in enumerate(paths):
                try:
                    if idx % n == 0:
                        printt("f0ing,now-%s,all-%s,-%s" % (idx, len(paths), inp_path))
                    if (
                        os.path.exists(opt_path1 + ".npy") == True
                        and os.path.exists(opt_path2 + ".npy") == True
                    ):
                        continue
                    featur_pit = self.compute_f0(inp_path, f0_method)
                    np.save(
                        opt_path2,
                        featur_pit,
                        allow_pickle=False,
                    )  # nsf
                    coarse_pit = self.coarse_f0(featur_pit)
                    np.save(
                        opt_path1,
                        coarse_pit,
                        allow_pickle=False,
                    )  # ori
                except:
                    printt("f0fail-%s-%s-%s" % (idx, inp_path, traceback.format_exc()))


if __name__ == "__main__":
    # exp_dir=r"E:\codes\py39\dataset\mi-test"
    # n_p=16
    # f = open("%s/log_extract_f0.log"%exp_dir, "w")
    printt(" ".join(sys.argv))
    device, backend = _select_rmvpe_device(device_prefer)
    if dry_run:
        _run_dry_run(device, backend, use_half)
        raise SystemExit(0)
    featureInput = FeatureInput(device=device, backend=backend, use_half=use_half)
    paths = []
    inp_root = "%s/1_16k_wavs" % (exp_dir)
    opt_root1 = "%s/2a_f0" % (exp_dir)
    opt_root2 = "%s/2b-f0nsf" % (exp_dir)

    os.makedirs(opt_root1, exist_ok=True)
    os.makedirs(opt_root2, exist_ok=True)
    for name in sorted(list(os.listdir(inp_root))):
        inp_path = "%s/%s" % (inp_root, name)
        if "spec" in inp_path:
            continue
        opt_path1 = "%s/%s" % (opt_root1, name)
        opt_path2 = "%s/%s" % (opt_root2, name)
        paths.append([inp_path, opt_path1, opt_path2])
    try:
        featureInput.go(paths[i_part::n_part], "rmvpe")
    except:
        printt("f0_all_fail-%s" % (traceback.format_exc()))
