import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader

from src.dataset.ms_coco import CocoDataset, CollateBatch
from src.models.st_agbilstm import STAGBiLSTMModel
from src.trainer import train
from src.evaluation import evaluate_model

embed_size = 512
hidden_size = 512
learning_rate = 3e-4
num_epochs = 10
batch_size = 32
device = "cuda" if torch.cuda.is_available() else "cpu"

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])

PATH = os.path.join(os.path.dirname(__file__), "data", "COCO")
VOCAB_PATH = os.path.join(os.path.dirname(__file__), "vocab.pkl")

def training():
    train_root = os.path.join(PATH, "images", "train2017")
    train_ann = os.path.join(PATH, "annotations", "captions_train2017.json")
    
    train_dataset = CocoDataset(
        root_dir=train_root,
        ann_file=train_ann,
        transform=transform,
        vocab_file=VOCAB_PATH,
        vocab_from_file=False
    )
    
    vocab = train_dataset.vocab
    pad_idx = vocab.word2idx[vocab.pad_word]
    
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=12,
        collate_fn=CollateBatch(pad_idx=pad_idx)
    )
    
    vocab_size = len(vocab)
    
    model = STAGBiLSTMModel(
        embed_size=embed_size,
        hidden_size=hidden_size,
        vocab_size=vocab_size,
    ).to(device)
    
    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    train(
        data_loader=train_loader,
        model=model,
        optimizer=optimizer,
        vocab_size=vocab_size,
        pad_idx=pad_idx,
        num_epoch=num_epochs,
        device=device
    )

def testing():
    from torch.utils.data import SubsetRandomSampler

    test_root = os.path.join(PATH, "images", "val2017")
    test_ann = os.path.join(PATH, "annotations", "captions_val2017.json")
    
    test_dataset = CocoDataset(
        root_dir=test_root,
        ann_file=test_ann,
        transform=transform,
        vocab_file=VOCAB_PATH,
        vocab_from_file=True
    )

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=1,
    )

    vocab = test_dataset.vocab
    vocab_size = len(vocab)

    model = STAGBiLSTMModel(
        embed_size=embed_size,
        hidden_size=hidden_size,
        vocab_size=vocab_size,
    ).to(device)

    model_path = os.path.join(os.path.dirname(__file__), "temp", "models", "best_model.pt")
    
    # 2. Load the state dictionary into the model
    state_dict = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()

    evaluate_model(test_loader, model, test_dataset.vocab, test_ann, transform)

def cli():
    training()
    # testing()

if __name__ == '__main__':
    cli()
