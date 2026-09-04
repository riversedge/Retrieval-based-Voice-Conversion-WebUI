import argparse
import os
import sys
import json
from pathlib import Path

import numpy as np
import torch

now_dir = os.getcwd()
sys.path.append(now_dir)
from dotenv import load_dotenv
from scipy.io import wavfile

from configs.config import Config
from infer.modules.vc.modules import VC
from infer.modules.vc.utils import load_hubert

####
# USAGE
#
# In your Terminal or CMD or whatever


def arg_parse() -> tuple:
    parser = argparse.ArgumentParser()
    parser.add_argument("--f0up_key", type=int, default=0)
    parser.add_argument("--input_path", type=str, help="input path")
    parser.add_argument("--index_path", type=str, help="index path")
    parser.add_argument("--f0method", type=str, default="harvest", help="harvest or pm")
    parser.add_argument("--opt_path", type=str, help="opt path")
    parser.add_argument("--model_name", type=str, help="store in assets/weight_root")
    parser.add_argument("--index_rate", type=float, default=0.66, help="index rate")
    parser.add_argument("--device", type=str, help="device")
    parser.add_argument("--is_half", type=bool, help="use half -> True")
    parser.add_argument("--filter_radius", type=int, default=3, help="filter radius")
    parser.add_argument("--resample_sr", type=int, default=0, help="resample sr")
    parser.add_argument("--rms_mix_rate", type=float, default=1, help="rms mix rate")
    parser.add_argument("--protect", type=float, default=0.33, help="protect")
    parser.add_argument(
        "--f0_range",
        type=str,
        default="",
        help="optional f0 range like 'E2 - B4' or '80-400Hz'",
    )
    parser.add_argument("--guide_path", help="optional guide vocal with the same lyrics/verse order")
    parser.add_argument("--guide_strength", type=float, default=0.5)
    parser.add_argument("--guide_mode", choices=["retrieval", "content"], default="retrieval")
    parser.add_argument("--guide_alignment", choices=["auto", "linear"], default="auto")
    parser.add_argument("--guide_anchors_path", help="text file of source_seconds,guide_seconds pairs")
    parser.add_argument("--guide_start", type=float, default=0)
    parser.add_argument("--guide_end", type=float, default=0, help="source region end; 0 means track end")
    parser.add_argument("--seed", type=int, help="repeatable synthesis noise for comparisons")
    parser.add_argument(
        "--compare", action="store_true",
        help="write baseline, retrieval (with index), and three content strengths beside opt_path",
    )

    args = parser.parse_args()
    if not args.input_path or not args.opt_path or not args.model_name:
        parser.error("--input_path, --opt_path, and --model_name are required")
    if args.compare and not args.guide_path:
        parser.error("--compare requires --guide_path")
    sys.argv = sys.argv[:1]

    return args


def main():
    load_dotenv()
    args = arg_parse()
    config = Config()
    config.device = args.device if args.device else config.device
    config.is_half = args.is_half if args.is_half else config.is_half
    vc = VC(config)
    vc.get_vc(args.model_name)
    anchors = Path(args.guide_anchors_path).read_text() if args.guide_anchors_path else ""
    runs = [("guided", args.guide_mode, args.guide_strength if args.guide_path else 0)]
    if args.compare:
        runs = [("baseline", "content", 0)]
        if args.index_path and args.index_rate > 0:
            runs.append(("retrieval_050", "retrieval", 0.5))
        runs.extend((f"content_{round(strength * 100):03d}", "content", strength) for strength in (0.35, 0.7, 1.0))
    seed = args.seed if args.seed is not None else (0 if args.compare else None)
    if seed is not None:
        # Lazy model construction consumes random numbers. Warm it before the
        # per-run reset so the baseline and guided runs get matching synth noise.
        vc.hubert_model = load_hubert(config)
        if vc.if_f0:
            vc.pipeline.get_f0(
                "guide-comparison-warmup", np.zeros(16000, dtype=np.float32),
                100, 0, args.f0method, args.filter_radius,
            )
    output = Path(args.opt_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest = {"settings": vars(args), "seed": seed, "runs": []}
    for label, mode, strength in runs:
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
        report = {}
        info, wav_opt = vc.vc_single(
            0, args.input_path, args.f0up_key, None, args.f0method,
            args.index_path, None, args.index_rate, args.filter_radius,
            args.resample_sr, args.rms_mix_rate, args.protect, args.f0_range,
            args.guide_path, strength, mode, args.guide_alignment, anchors,
            args.guide_start, args.guide_end, guide_report=report,
        )
        if wav_opt is None or wav_opt[0] is None:
            raise RuntimeError(info)
        destination = output.with_name(f"{output.stem}_{label}.wav") if args.compare else output
        wavfile.write(str(destination), wav_opt[0], wav_opt[1])
        print(f"{destination}\n{info}")
        manifest["runs"].append({"file": str(destination), "guide": report})
        if args.guide_path:
            output.with_suffix(".guide.json").write_text(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
