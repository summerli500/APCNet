"""Train APCNet."""
from __future__ import annotations

import os
import warnings
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("YOLO_AUTOINSTALL", "false")
os.environ.setdefault("WANDB_MODE", "disabled")
os.environ["YOLO_BBOX_IOU"] = "wise_inner_ciou"
warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent
MODEL = ROOT / "ultralytics" / "cfg" / "models" / "11" / "yolo11-C3k2-RFCBAM-CGAFusion.yaml"
DATA = ROOT / "datasets" / "am_person.yaml"


def main() -> None:
    if not DATA.exists():
        raise FileNotFoundError(
            f"Missing {DATA}. Point path: to your AM_PERSON root as described in README."
        )
    if not MODEL.exists():
        raise FileNotFoundError(MODEL)

    from ultralytics import YOLO

    print("APCNet")
    print(f"  model={MODEL.name}")
    print(f"  data={DATA.name}")

    model = YOLO(str(MODEL))
    model.train(
        data=str(DATA),
        imgsz=640,
        epochs=100,
        batch=128,
        workers=8,
        device=0,
        project=str(ROOT / "train"),
        name="AM_PERSON-APCNet-WiseInnerCIoU-b128-lr04",
        pretrained=False,
        optimizer="SGD",
        lr0=0.04,
        warmup_epochs=5,
        cos_lr=True,
        seed=0,
        val=True,
        plots=False,
        patience=0,
    )


if __name__ == "__main__":
    main()
