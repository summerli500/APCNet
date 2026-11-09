"""Eval APCNet on AM_PERSON test."""
from __future__ import annotations
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.chdir(ROOT)
import sys
sys.path.insert(0, str(ROOT))

from ultralytics import YOLO

WEIGHTS = ROOT / "weights" / "APCNet_best.pt"
DATA = ROOT / "datasets" / "am_person.yaml"


def main():
    if not DATA.exists():
        raise FileNotFoundError(
            f"Missing {DATA}. Prepare AM_PERSON under datasets/AM_PERSON "
            "(images/{train,val,test} + labels) as described in README."
        )
    model = YOLO(str(WEIGHTS))
    metrics = model.val(data=str(DATA), split="test", imgsz=640, batch=32, device=0, plots=False)
    print("P", float(metrics.box.mp), "R", float(metrics.box.mr),
          "mAP50", float(metrics.box.map50), "mAP50-95", float(metrics.box.map))


if __name__ == "__main__":
    main()
