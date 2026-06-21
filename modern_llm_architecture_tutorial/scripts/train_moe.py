from pathlib import Path
import os
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from src.train import train


if __name__ == "__main__":
    train(str(ROOT / "configs" / "moe.yaml"))
