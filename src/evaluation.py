import os
import json
from tqdm import tqdm

import pandas as pd
import torch
from torch.utils.data import DataLoader
from pycocotools.coco import COCO
from pycocoevalcap.eval import COCOEvalCap

import src.dataset.ms_coco as ms_coco
import src.models.imagecaptionv1 as imgcap

def evaluate_model(test_loader: DataLoader,
                   model: imgcap.ImageCaptionModel, 
                   vocab: ms_coco.Vocabulary, 
                   ann_file: str, 
                   transform=None, 
                   device: str = "cuda"
                   ):

    model.eval()
    results = []
    model = model.to(device)

    print("Benchmarking Dataset...")

    # Define special tokens to exclude from the final string
    special_tokens = {vocab.start_word, vocab.end_word, vocab.pad_word, vocab.unk_word}

    loop = tqdm(test_loader, leave=True)

    for image, _, img_id in loop:
        image = image.to(device)

        with torch.no_grad():
            # Assuming model.caption_image returns a list of word strings
            caption_words = model.caption_image(image, vocab)
        
        # Filter special tokens and join
        caption = " ".join([
            w for w in caption_words 
            if w not in special_tokens
        ])

        results.append({
            "image_id": img_id.item() if torch.is_tensor(img_id) else img_id,
            "caption": caption
        })

    # Ensure the results directory exists
    res_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "temp", "results")
    os.makedirs(res_dir, exist_ok=True)
    
    res_file = os.path.join(res_dir, "metrics.json")
    with open(res_file, "w") as f:
        json.dump(results, f)

    # Initialize COCO ground truth and results
    coco_gt = COCO(ann_file)
    coco_res = coco_gt.loadRes(res_file)
    
    # Run evaluation
    coco_eval = COCOEvalCap(coco_gt, coco_res)
    coco_eval.params['image_id'] = coco_res.getImgIds()

    coco_eval.evaluate()

    # Log results
    for metric, score in coco_eval.eval.items():
        print(f"{metric}: {score:.4f}")

    # Export to CSV
    metrics_df = pd.DataFrame(list(coco_eval.eval.items()), columns=['Metric', 'Score'])
    metrics_df.to_csv(os.path.join(res_dir, "metrics.csv"), index=False)
