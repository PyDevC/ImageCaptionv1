# Image Captioning v1

Image Captioning using VGG-RNN model (with experimental Swin Attention Gated model).

> NOTE: If this sparks any interests in me then I will make it performant.

## Installation

```bash
pip install -r requirements.txt
```

## Train Model

```bash
python3 main.py --train --epoch <num>
```

## Evaluate Model

```bash
python3 main.py --test
```

## Download dataset

Download dataset inside data directory.
All internal files are designed to work with COCO directory being in data directory.

```bash
cd data
python3 download_dataset.py
```

## Results


| Metric | Score |
|---|---|
| Bleu_1 | 0.5953289327600356 |
| Bleu_2 | 0.40645970649523894 |
| Bleu_3 | 0.26734383453626503 |
| Bleu_4 | 0.17662071952569972 |
| METEOR | 0.18679035615274345 |
| ROUGE_L | 0.4412577702224827 |
| CIDEr | 0.5250028088906024 |
