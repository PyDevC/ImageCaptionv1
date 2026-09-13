import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import VGG16_Weights

from .base import ImageCaptionBase

class EncoderVGG(nn.Module):
    def __init__(self, embed_size):
        super(EncoderVGG, self).__init__()
        vgg16 = models.vgg16(weights=VGG16_Weights.DEFAULT)
        for param in vgg16.parameters():
            param.requires_grad_(False)

        self.vgg16 = vgg16.features
        self.adaptive_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        
        self.embed = nn.Linear(512, embed_size)
        self.bn = nn.BatchNorm1d(embed_size, momentum=0.01)

    def forward(self, x):
        x = self.vgg16(x)
        x = self.adaptive_avg_pool(x)
        x = self.flatten(x)
        x = self.embed(x)
        return x

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers):
        super(DecoderRNN, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.RNN(embed_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)

    def forward(self, features, captions):
        embeddings = self.embed(captions[:, :-1])
        embeddings = torch.cat((features.unsqueeze(1), embeddings), dim=1)
        hiddens, _ = self.rnn(embeddings)
        return self.linear(hiddens)

class VGGRnnModel(ImageCaptionBase):
    def __init__(self, embed_size: int, hidden_size: int, vocab_size: int, num_layers: int):
        super(VGGRnnModel, self).__init__(embed_size, hidden_size, vocab_size, num_layers)
        self.encoder = EncoderVGG(embed_size)
        self.decoder = DecoderRNN(embed_size, hidden_size, vocab_size, num_layers)

    def forward(self, images: torch.Tensor, captions: torch.Tensor) -> torch.Tensor:
        features = self.encoder(images)
        return self.decoder(features, captions)

    def caption_image(self, image, vocabulary, max_length=50, temperature=0.5):
        result_caption = []
        with torch.no_grad():
            x = self.encoder(image).unsqueeze(1) 
            h = None 

            for _ in range(max_length):
                hiddens, h = self.decoder.rnn(x, h)
                output = self.decoder.linear(hiddens.squeeze(1))
                
                scaled_logits = output / temperature
                probabilities = torch.softmax(scaled_logits, dim=1)
                predicted = torch.multinomial(probabilities, 1)
                word_idx = predicted.item()
                result_caption.append(word_idx)

                if vocabulary.idx2word[word_idx] == vocabulary.end_word:
                    break
                
                x = self.decoder.embed(predicted) # [batch_size, 1, embed_size]
        
        return [vocabulary.idx2word[i] for i in result_caption]
