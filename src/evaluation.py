import os
import json
from tqdm import tqdm

import pandas as pd
import torch
from torch.utils.data import DataLoader
from pycocotools.coco import COCO
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer

import src.dataset.ms_coco as ms_coco


def compute_coco_metrics(coco_gt, coco_res, image_ids):
    """BLEU-1..4, METEOR, ROUGE_L, CIDEr and (when its Java jar works) SPICE.

    COCOEvalCap hardcodes its scorer list, so a broken SPICE jar takes every
    other metric down with it; scoring explicitly keeps the rest usable.
    """
    gts = {img_id: coco_gt.imgToAnns[img_id] for img_id in image_ids}
    res = {img_id: coco_res.imgToAnns[img_id] for img_id in image_ids}

    tokenizer = PTBTokenizer()
    gts = tokenizer.tokenize(gts)
    res = tokenizer.tokenize(res)

    metrics = {}

    bleu_scores, _ = Bleu(4).compute_score(gts, res)
    for i, score in enumerate(bleu_scores, start=1):
        metrics[f"Bleu_{i}"] = float(score)

    for name, scorer in (("METEOR", Meteor()), ("ROUGE_L", Rouge()), ("CIDEr", Cider())):
        score, _ = scorer.compute_score(gts, res)
        metrics[name] = float(score)

    try:
        from pycocoevalcap.spice.spice import Spice
        spice_score, _ = Spice().compute_score(gts, res)
        metrics["SPICE"] = float(spice_score)
    except Exception as err:
        print(f"SPICE unavailable ({type(err).__name__}); skipping it.")

    return metrics

def evaluate_model(test_loader: DataLoader,
                   model: torch.nn.Module, 
                   vocab: ms_coco.Vocabulary, 
                   ann_file: str,  # annotation file
                   transform=None, 
                   device: str = "cuda"
                   ):

    model.eval()
    results = {}
    model = model.to(device)

    print("Benchmarking Dataset...")

    special_tokens = {vocab.start_word, vocab.end_word, vocab.pad_word, vocab.unk_word}

    loop = tqdm(test_loader, leave=True)

    for image, _, img_id in loop:
        image = image.to(device)

        with torch.no_grad():
            caption_words = model.caption_image(image, vocab)
        
        caption = " ".join([
            w for w in caption_words 
            if w not in special_tokens
        ])

        image_id = img_id.item() if torch.is_tensor(img_id) else img_id
        results[image_id] = caption

    res_list = [{"image_id": image_id, "caption": caption} for image_id, caption in results.items()]

    res_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "temp", "results")
    os.makedirs(res_dir, exist_ok=True)
    
    res_file = os.path.join(res_dir, "metrics.json")
    with open(res_file, "w") as f:
        json.dump(res_list, f)

    coco_gt = COCO(ann_file)
    coco_res = coco_gt.loadRes(res_file)

    metrics = compute_coco_metrics(coco_gt, coco_res, coco_res.getImgIds())

    for metric, score in metrics.items():
        print(f"{metric}: {score:.4f}")

    metrics_df = pd.DataFrame(list(metrics.items()), columns=['Metric', 'Score'])
    metrics_df.to_csv(os.path.join(res_dir, "metrics.csv"), index=False)
