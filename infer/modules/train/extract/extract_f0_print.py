import argparse
import os
import sys
import traceback

import parselmouth

now_dir = os.getcwd()
sys.path.append(now_dir)
import logging
import warnings

import numpy as np

from infer.lib.audio import load_audio
from infer.lib.device import get_rmvpe_device

logging.getLogger("numba").setLevel(logging.WARNING)
from multiprocessing import Process


def _parse_args(argv):
    if not any(arg.startswith("--") for arg in argv):
        if len(argv) < 3:
            raise ValueError("Expected legacy args: exp_dir n_p f0method")
        return {
            "exp_dir": argv[0],
            "n_p": int(argv[1]),
            "f0method": argv[2],
            "device": None,
            "quiet": False,
        }
    parser = argparse.ArgumentParser()
    parser.add_argument("exp_dir")
    parser.add_argument("n_p", type=int)
    parser.add_argument("f0method")
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda", "mps", "dml", "gpu"],
        default=None,
        help="RMVPE device preference",
    )
    parser.add_argument("--quiet", action="store_true")
    return vars(parser.parse_args(argv))


args = _parse_args(sys.argv[1:])
exp_dir = args["exp_dir"]
n_p = args["n_p"]
f0method = args["f0method"]
device_prefer = args["device"]
quiet = args["quiet"]
f = open(f"{exp_dir}/extract_f0_feature.log", "a+")


def printt(strr):
    print(strr)
    f.write("%s\n" % strr)
    f.flush()


def _load_pyworld(quiet_mode):
    if not quiet_mode:
        import pyworld

        return pyworld
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="pkg_resources is deprecated as an API",
            category=UserWarning,
        )
        warnings.filterwarnings(
            "ignore",
            message="Deprecated call to `pkg_resources.declare_namespace`",
            category=DeprecationWarning,
        )
        import pyworld

    return pyworld


class FeatureInput(object):
    def __init__(self, device, backend, samplerate=16000, hop_size=160):
        self.fs = samplerate
        self.hop = hop_size
        self.device = device
        self.backend = backend
        self.model_loaded = False

        self.f0_bin = 256
        self.f0_max = 1100.0
        self.f0_min = 50.0
        self.f0_mel_min = 1127 * np.log(1 + self.f0_min / 700)
        self.f0_mel_max = 1127 * np.log(1 + self.f0_max / 700)

    def compute_f0(self, path, f0_method):
        x = load_audio(path, self.fs)
        p_len = x.shape[0] // self.hop
        if f0_method == "pm":
            time_step = 160 / 16000 * 1000
            f0_min = 50
            f0_max = 1100
            f0 = (
                parselmouth.Sound(x, self.fs)
                .to_pitch_ac(
                    time_step=time_step / 1000,
                    voicing_threshold=0.6,
                    pitch_floor=f0_min,
                    pitch_ceiling=f0_max,
                )
                .selected_array["frequency"]
            )
            pad_size = (p_len - len(f0) + 1) // 2
            if pad_size > 0 or p_len - len(f0) - pad_size > 0:
                f0 = np.pad(
                    f0, [[pad_size, p_len - len(f0) - pad_size]], mode="constant"
                )
        elif f0_method == "harvest":
            pyworld = _load_pyworld(quiet)
            f0, t = pyworld.harvest(
                x.astype(np.double),
                fs=self.fs,
                f0_ceil=self.f0_max,
                f0_floor=self.f0_min,
                frame_period=1000 * self.hop / self.fs,
            )
            f0 = pyworld.stonemask(x.astype(np.double), f0, t, self.fs)
        elif f0_method == "dio":
            pyworld = _load_pyworld(quiet)
            f0, t = pyworld.dio(
                x.astype(np.double),
                fs=self.fs,
                f0_ceil=self.f0_max,
                f0_floor=self.f0_min,
                frame_period=1000 * self.hop / self.fs,
            )
            f0 = pyworld.stonemask(x.astype(np.double), f0, t, self.fs)
        elif f0_method == "rmvpe":
            if not self.model_loaded:
                from infer.lib.rmvpe import RMVPE

                printt("RMVPE backend: %s" % self.backend)
                print("Loading rmvpe model")
                self.model_rmvpe = RMVPE(
                    "assets/rmvpe/rmvpe.pt", is_half=False, device=self.device
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
    device, backend = get_rmvpe_device(device_prefer)
    featureInput = FeatureInput(device=device, backend=backend)
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
        featureInput.go(paths, f0method)
    except:
        printt("f0_all_fail-%s" % (traceback.format_exc()))
