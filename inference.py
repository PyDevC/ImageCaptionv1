import os

import matplotlib.pyplot as plt
import torch
from PIL import Image

from main import transform as TRANSFORM
from src.models import VGGRnnModel

MODEL_FILE = "temp/models/best_model.pt"
VOCAB_FILE = "vocab.pkl"
OUT_DIR = "temp/preds"
TEST_IMGS = [
    "data/COCO/images/test2017/000000000001.jpg",
    "data/COCO/images/test2017/000000000016.jpg",
    "data/COCO/images/test2017/000000000019.jpg",
    "data/COCO/images/test2017/000000581295.jpg",
]


def caption_text(words, vocab):
    cleaned = []
    for word in words:
        if word in (vocab.start_word, vocab.pad_word, vocab.unk_word):
            continue
        if word == vocab.end_word:
            break
        cleaned.append(word)
    while cleaned and cleaned[-1] in ".!?,;":
        cleaned.pop()
    return " ".join(cleaned).capitalize() + "."


def visualize_prediction(image_path, model, vocab, device):
    """
    Displays image and generated caption using matplotlib.
    """
    raw_image = Image.open(image_path).convert("RGB")
    input_tensor = TRANSFORM(raw_image).unsqueeze(0).to(device)

    words = model.caption_image(input_tensor, vocab)
    text = caption_text(words, vocab)
    print(text)

    os.makedirs(OUT_DIR, exist_ok=True)
    stem = os.path.splitext(os.path.basename(image_path))[0]

    plt.figure(figsize=(8, 8))
    plt.imshow(raw_image)
    plt.axis("off")

    plt.figtext(0.5, 0.05, text, wrap=True, horizontalalignment="center", fontsize=12, fontweight="bold")
    plt.savefig(os.path.join(OUT_DIR, f"{stem}.jpg"))
    plt.close()


def run_visual_inference(image_path, model_path, vocab_file=VOCAB_FILE):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    import pickle
    with open(vocab_file, "rb") as f:
        vocab = pickle.load(f)

    model = VGGRnnModel(
        embed_size=512,
        hidden_size=512,
        vocab_size=len(vocab),
        num_layers=3,
    ).to(device)

    loaded = torch.load(model_path, map_location=device, weights_only=True)
    if isinstance(loaded, dict) and "model_state" in loaded:
        state_dict = loaded["model_state"]
    else:
        state_dict = loaded
    try:
        model.load_state_dict(state_dict)
    except RuntimeError as err:
        raise SystemExit(
            f"{model_path} does not match the current model definition "
            f"(embed=512, hidden=512, num_layers=3).\n"
            f"Checkpoints from the old 10-layer decoder are unusable: that model "
            f"never learned to use the image. Retrain with `python3 main.py --train`.\n"
            f"Original error: {err}"
        ) from err
    model.eval()

    visualize_prediction(image_path, model, vocab, device)


if __name__ == "__main__":
    for test_img in TEST_IMGS:
        run_visual_inference(test_img, MODEL_FILE)
