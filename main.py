import os
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader

from src.dataset.ms_coco import CocoDataset, CollateBatch, Vocabulary
from src.models.imagecaptionv1 import ImageCaptionModel
from src.trainer import train
from src.evaluation import evaluate_model

embed_size = 256
hidden_size = 512
num_layers = 5
learning_rate = 1e-6
num_epochs = 10
batch_size = 16
device = "cuda"

## Mentioned in VGG16 docs
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])

PATH = os.path.join(os.path.dirname(__file__), "data", "COCO")

def training():
    train_root = os.path.join(PATH, "images", "train2017")
    train_ann = os.path.join(PATH, "annotations", "captions_train2017.json")
    
    train_dataset = CocoDataset(
        root_dir=train_root,
        ann_file=train_ann,
        transform=transform
    )
    
    pad_idx = train_dataset.vocab.stoi["<PAD>"]
    
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=12,
        collate_fn=CollateBatch(pad_idx=pad_idx)
    )
    
    vocab_size = len(train_dataset.vocab)
    
    model = ImageCaptionModel(
        embed_size=embed_size,
        hidden_size=hidden_size,
        vocab_size=vocab_size,
        num_layers=num_layers
    )
    
    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    train(
        data_loader=train_loader,
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        num_epoch=num_epochs,
        device=device
    )


def testing():
    test_root = os.path.join(PATH, "images", "val2017")
    test_ann = os.path.join(PATH, "annotations", "captions_val2017.json")
    
    test_dataset = CocoDataset(
        root_dir=test_root,
        ann_file=test_ann,
        transform=transform
    )
    
    test_loader = DataLoader(
        dataset=test_dataset
    )

    vocab_size = len(test_dataset.vocab)
    
    model = ImageCaptionModel(
        embed_size=embed_size,
        hidden_size=hidden_size,
        vocab_size=vocab_size,
        num_layers=num_layers
    )

    vocab = test_dataset.vocab

    evaluate_model(test_loader, model, vocab, transform)

def cli():
    training()
    testing()

cli()
