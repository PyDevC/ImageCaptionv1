import torch
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
from src.models.st_agbilstm import STAGBiLSTMModel

TRANSFORM = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])

def visualize_prediction(image_path, model, vocab, device):
    """
    Displays image and generated caption using matplotlib.
    """
    raw_image = Image.open(image_path).convert("RGB")
    input_tensor = TRANSFORM(raw_image).unsqueeze(0).to(device)
    
    words = model.caption_image(input_tensor, vocab)
    
    cleaned = []
    for word in words:
        if word in [vocab.start_word, vocab.pad_word]:
            continue
        if word == vocab.end_word:
            break
        cleaned.append(word)
    caption_text = " ".join(cleaned).capitalize() + "."
    print(caption_text)

    plt.figure(figsize=(8, 8))
    plt.imshow(raw_image)
    plt.axis("off")
    
    plt.figtext(0.5, 0.05, caption_text, wrap=True, horizontalalignment='center', fontsize=12, fontweight='bold')
    plt.show()

def run_visual_inference(image_path, model_path, vocab_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    with open(vocab_path, "rb") as f:
        import pickle
        vocab = pickle.load(f)

    model = STAGBiLSTMModel(
        embed_size=512,
        hidden_size=512,
        vocab_size=len(vocab),
    ).to(device)
    
    loaded = torch.load(model_path, map_location=device, weights_only=True)
    if isinstance(loaded, dict) and "model_state" in loaded:
        state_dict = loaded["model_state"]
    else:
        state_dict = loaded
    model.load_state_dict(state_dict)
    model.eval()

    visualize_prediction(image_path, model, vocab, device)

if __name__ == "__main__":
    MODEL_FILE = "temp/models/best_model.pt"
    VOCAB_FILE = "vocab.pkl"
    TEST_IMGS = ["data/COCO/images/test2017/000000000001.jpg",
    "data/COCO/images/test2017/000000000016.jpg",
    "data/COCO/images/test2017/000000000019.jpg",
    "data/COCO/images/test2017/000000581295.jpg", ]

    for TEST_IMG in TEST_IMGS:
        run_visual_inference(TEST_IMG, MODEL_FILE, VOCAB_FILE)

