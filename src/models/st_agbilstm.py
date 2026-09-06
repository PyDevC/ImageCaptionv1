import torch
import torch.nn as nn
import timm


class SwinEncoder(nn.Module):
    def __init__(self, embed_size=512):
        super().__init__()
        self.swin = timm.create_model(
            "swin_small_patch4_window7_224", pretrained=True
        )
        swin_dim = self.swin.num_features  # 768 for Swin-S
        self.proj = nn.Linear(swin_dim, embed_size)
        self.norm = nn.LayerNorm(embed_size)

    def forward(self, x):
        features = self.swin.forward_features(x)  # [B, H, W, swin_dim]
        B, H, W, C = features.shape
        features = features.view(B, H * W, C)  # [B, L, swin_dim]
        return self.norm(self.proj(features))  # [B, L, embed_size]


class AttentionGate(nn.Module):
    def __init__(self, encoder_dim, decoder_dim, attention_dim):
        super().__init__()
        self.W_v = nn.Linear(encoder_dim, attention_dim, bias=False)
        self.W_h = nn.Linear(decoder_dim, attention_dim, bias=False)
        self.v = nn.Linear(attention_dim, 1, bias=False)

    def forward(self, encoder_out, decoder_hidden):
        score = self.v(
            torch.tanh(
                self.W_v(encoder_out) + self.W_h(decoder_hidden).unsqueeze(1)
            )
        ).squeeze(-1)
        weights = torch.softmax(score, dim=-1)
        context = (weights.unsqueeze(-1) * encoder_out).sum(dim=1)
        return context, weights


class AGBiLSTMDecoder(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, attention_dim, encoder_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        bi_hidden = hidden_size * 2

        self.lstm = nn.LSTM(
            embed_size,
            hidden_size,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
        )

        self.attention = AttentionGate(encoder_dim, bi_hidden, attention_dim)
        self.context_proj = nn.Linear(encoder_dim, bi_hidden)
        self.gate = nn.Linear(bi_hidden * 2, bi_hidden)
        self.fc = nn.Linear(bi_hidden, vocab_size)

    def forward(self, features, captions):
        embeds = self.embedding(captions[:, :-1])
        mean_feat = features.mean(dim=1)
        inputs = torch.cat([mean_feat.unsqueeze(1), embeds], dim=1)

        lstm_out, _ = self.lstm(inputs)

        outputs = []
        for t in range(lstm_out.size(1)):
            context, _ = self.attention(features, lstm_out[:, t, :])
            context_proj = self.context_proj(context)
            combined = torch.cat([lstm_out[:, t, :], context_proj], dim=-1)
            gate = torch.sigmoid(self.gate(combined))
            gated = gate * lstm_out[:, t, :] + (1 - gate) * context_proj
            outputs.append(self.fc(gated))

        return torch.stack(outputs, dim=1)

    def forward_step(self, word_idx, hidden, features):
        embed = self.embedding(word_idx)
        lstm_out, hidden = self.lstm(embed, hidden)

        hidden_cat = lstm_out[:, 0, :]
        context, weights = self.attention(features, hidden_cat)
        context_proj = self.context_proj(context)

        combined = torch.cat([hidden_cat, context_proj], dim=-1)
        gate = torch.sigmoid(self.gate(combined))
        gated = gate * hidden_cat + (1 - gate) * context_proj

        output = self.fc(gated)
        return output, hidden, weights


class STAGBiLSTMModel(nn.Module):
    def __init__(
        self,
        embed_size=512,
        hidden_size=512,
        vocab_size=10000,
        attention_dim=256,
    ):
        super().__init__()
        self.encoder = SwinEncoder(embed_size)
        self.decoder = AGBiLSTMDecoder(
            embed_size, hidden_size, vocab_size, attention_dim, embed_size
        )

    def forward(self, images, captions):
        features = self.encoder(images)
        return self.decoder(features, captions)

    def caption_image(self, image, vocabulary, max_length=50, temperature=0.5):
        result_caption = []
        with torch.no_grad():
            features = self.encoder(image)
            word_idx = torch.tensor(
                [vocabulary.word2idx[vocabulary.start_word]]
            ).to(image.device)
            hidden = None

            for _ in range(max_length):
                output, hidden, _ = self.decoder.forward_step(
                    word_idx.unsqueeze(1), hidden, features
                )
                probs = torch.softmax(output / temperature, dim=-1)
                predicted = torch.multinomial(probs, 1).squeeze(1)

                idx = predicted.item()
                result_caption.append(idx)

                if vocabulary.idx2word[idx] == vocabulary.end_word:
                    break

                word_idx = predicted

        return [vocabulary.idx2word[i] for i in result_caption]
