import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import VGG16_Weights

class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        vgg16 = models.vgg16(weights=VGG16_Weights.DEFAULT)
        for param in vgg16.parameters():
            # NOTE: Do not make it True, we don't want to train the weights of vgg16
            param.requires_grad_(False)

        # vgg16 has 3 Major Layers Only First one is used for feature extraction
        self.vgg16 = nn.Sequential(list(vgg16.children())[0])
        self.adaptive_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()

        # Make sure the layer name is same in Decoder too as both have same purpose
        self.embed = nn.Linear(512, embed_size)

    def forward(self, x):
        x = self.vgg16(x)
        x = self.adaptive_avg_pool(x)
        x = self.flatten(x) # Check the size before running
        out = self.embed(x)
        return out

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers) -> None:
        super(DecoderRNN, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)

    # I need to add alias as captions can be anything
    def forward(self, features, captions):
        embeddings = self.embed(captions[:, :-1])
        embeddings = torch.cat((features.unsqueeze(1), embeddings), dim=1)
        hiddens, _ = self.lstm(embeddings)
        return self.linear(hiddens)

class ImageCaptionModel(nn.Module):
    def __init__(self, embed_size: int, hidden_size: int, vocab_size: int, num_layers: int):
        super(ImageCaptionModel, self).__init__()
        self.encoder = EncoderCNN(embed_size)
        self.decoder = DecoderRNN(embed_size, hidden_size, vocab_size, num_layers)

    def forward(self, images: torch.Tensor, captions: torch.Tensor) -> torch.Tensor:
        features = self.encoder(images)
        return self.decoder(features, captions)
