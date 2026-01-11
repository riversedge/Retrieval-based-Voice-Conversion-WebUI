import os
import sys
from subprocess import run


now_dir = os.getcwd()
sys.path.append(now_dir)


def main():
    if len(sys.argv) < 2:
        raise ValueError("Expected exp_dir argument")
    exp_dir = sys.argv[1]
    cmd = [
        sys.executable,
        "infer/modules/train/extract/extract_f0_rmvpe.py",
        "--exp-dir",
        exp_dir,
        "--device",
        "dml",
    ]
    run(cmd, check=True, cwd=now_dir)


if __name__ == "__main__":
    main()
