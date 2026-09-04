"""Render the offline phrase-octave experiment without changing the WebUI.

Run from the repository root. Requires the same local models as normal RVC.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import patch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True)
    parser.add_argument('--guide', required=True)
    parser.add_argument('--voice', required=True, help='Voice filename in the configured weights directory')
    parser.add_argument('--index', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--transpose', type=int, default=-12)
    parser.add_argument('--seed', type=int, default=67)
    args = parser.parse_args()
    output = Path(args.output).expanduser().resolve()
    manifest_path = output.with_suffix('.json')
    if output.exists() or manifest_path.exists():
        parser.error('Choose a new output filename; existing exports are preserved.')
    root = Path(__file__).resolve().parents[2]
    os.chdir(root)
    sys.path.insert(0, str(root))
    from dotenv import load_dotenv
    load_dotenv(root / '.env')
    import faiss
    import numpy as np
    import soundfile as sf
    import torch
    from infer.lib.rmvpe import RMVPE
    from infer.modules.vc.modules import VC
    from infer.modules.vc.utils import load_hubert
    from tools.experiments.phrase_octaves import correct_phrase_octaves

    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    faiss.omp_set_num_threads(4)
    config = SimpleNamespace(device='cpu', is_half=False, x_pad=1, x_query=6, x_center=38, x_max=41)
    vc = VC(config)
    vc.get_vc(args.voice)
    vc.hubert_model = load_hubert(config)
    vc.pipeline.model_rmvpe = RMVPE(str(root / 'assets/rmvpe/rmvpe.pt'), False, 'cpu')
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    source = str(Path(args.source).expanduser().resolve())
    guide = str(Path(args.guide).expanduser().resolve())
    index = str(Path(args.index).expanduser().resolve())
    guide_report = {}
    with patch('infer.modules.vc.pipeline.correct_octave_jumps', correct_phrase_octaves):
        info, result = vc.vc_single(
            0, source, args.transpose, None, 'rmvpe', index, None, .75, 3, 0, .25, .33,
            f0_range=None, guide_audio_path=guide, guide_strength=.5,
            guide_mode='retrieval', guide_alignment='auto', guide_report=guide_report,
            correct_octave_errors=True,
        )
    if result is None or result[0] is None:
        raise RuntimeError(info)
    sr, audio = result
    output.parent.mkdir(parents=True, exist_ok=True)
    sf.write(output, audio, sr, subtype='PCM_16')
    manifest = dict(
        experiment='Offline strict phrase octave continuity', source=source, guide=guide,
        voice=args.voice, index=index, transpose=args.transpose, seed=args.seed,
        extractor='rmvpe', index_rate=.75, filter_radius=3, resample_sr=0,
        rms_mix_rate=.25, protect=.33, f0_range=None, speaker=0,
        guide_mode='retrieval', guide_strength=.5, guide_alignment='auto',
        device='cpu', is_half=False, x_pad=1, x_query=6, x_center=38, x_max=41,
        code_sha256=hashlib.sha256(Path(__file__).with_name('phrase_octaves.py').read_bytes()).hexdigest(),
        output=str(output), sample_rate=sr, seconds=len(audio)/sr,
        info=info, guide_report=guide_report,
    )
    manifest_path.write_text(json.dumps(manifest, indent=2) + '\n')
    print(info)
    print(output)


if __name__ == '__main__':
    main()
