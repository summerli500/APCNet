# APCNet

## Data

Baidu: https://pan.baidu.com/s/1UpqGkPGhUm8jOJFmb6SEiQ  
Code: `q94r`

```
datasets/AM_PERSON/
  images/{train,val,test}/
  labels/{train,val,test}/
```

Or set `path:` in `datasets/am_person.yaml`.

## Eval

```bash
pip install -r requirements.txt
pip install -e .
python eval_apcnet.py
```

## Train

batch 128, epochs 100, SGD lr0=0.04, cos_lr, imgsz 640, pretrained=False, seed 0.

```bash
pip install -r requirements.txt
pip install -e .
python train_apcnet.py
```

## License

AGPL-3.0. See `LICENSE`.
