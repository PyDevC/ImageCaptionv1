import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import VGG16_Weights

class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        vgg16 = models.vgg16(weights=VGG16_Weights.DEFAULT)
        for param in vgg16.parameters():
            param.requires_grad_(False)

        self.vgg16 = vgg16.features
        self.adaptive_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        
        # Linear layer to map VGG features to embedding space
        self.embed = nn.Linear(512, embed_size)
        # Batch normalization is critical to normalize image features 
        # before they interact with word embeddings
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
        # Standard RNN unit
        self.rnn = nn.RNN(embed_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)

    def forward(self, features, captions):
        # Captions: [batch_size, seq_len]
        # Remove <EOS> from input to keep sequence length consistent 
        # after concatenating image features
        embeddings = self.embed(captions[:, :-1])
        
        # Prepend image features as the first time step
        # features shape: [batch_size, embed_size] -> [batch_size, 1, embed_size]
        embeddings = torch.cat((features.unsqueeze(1), embeddings), dim=1)
        
        hiddens, _ = self.rnn(embeddings)
        return self.linear(hiddens)

class ImageCaptionModel(nn.Module):
    def __init__(self, embed_size: int, hidden_size: int, vocab_size: int, num_layers: int):
        super(ImageCaptionModel, self).__init__()
        self.encoder = EncoderCNN(embed_size)
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
                
                # Apply temperature scaling to logits
                scaled_logits = output / temperature
                probabilities = torch.softmax(scaled_logits, dim=1)
                
                # Sample from probability distribution instead of argmax
                predicted = torch.multinomial(probabilities, 1)
                
                word_idx = predicted.item()
                result_caption.append(word_idx)
                
                if vocabulary.idx2word[word_idx] == vocabulary.end_word:
                    break
                
                x = self.decoder.embed(predicted) # [batch_size, 1, embed_size]
        
        return [vocabulary.idx2word[i] for i in result_caption]
