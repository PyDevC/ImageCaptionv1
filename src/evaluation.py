import os
import json
from tqdm import tqdm

import pandas as pd
import torch
from torch.utils.data import DataLoader
from pycocotools.cocoeval import COCOeval
from pycocotools.coco import COCO
from pycocoevalcap.eval import COCOEvalCap

import src.dataset.ms_coco as ms_coco
import src.models.imagecaptionv1 as imgcap

def evaluate_model(test_loader: DataLoader[ms_coco.CocoDataset],
                   model: imgcap.ImageCaptionModel, 
                   vocab: ms_coco.Vocabulary, 
                   ann_file: str, 
                   transform=None, 
                   device: str="cuda"
                   ):

    model.eval()
    results = []
    model = model.to(device)

    print("BenchMarking Dataset ... .. .")

    loop = tqdm(test_loader, leave=True)
    itos = vocab.itos.values()

    for image, label, img_id in loop:
        image = image.to(device)
        label = label.to(device)

        caption_words = model.caption_image(image, vocab)
        
        caption = " ".join(w for w in caption_words if w in itos)

        results.append({
            "image_id": img_id.item(),
            "caption": caption
        })

    res_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), "temp", "results", "metrics.json")
    with open(res_file, "w") as f:
        json.dump(results, f)

    coco_gt = COCO(ann_file)
    coco_res = coco_gt.loadRes(res_file)
    
    coco_eval = COCOEvalCap(coco_gt, coco_res)
    coco_eval.params['image_id'] = coco_res.getImgIds()

    coco_eval.evaluate()

    for metric, score in coco_eval.eval.items():
        print(f"{metric}: {score:.4f}")

    metrics = pd.DataFrame(coco_eval.eval.items())
    metrics.to_csv(os.path.join(os.path.dirname(os.path.dirname(__file__)), "temp", "results", "metrics.csv"))
