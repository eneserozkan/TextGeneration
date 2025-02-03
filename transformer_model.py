import torch
import torch.nn as nn
from torch.nn import Transformer


class TextTransformer(nn.Module):
    def __init__(self, vocab_size, embed_size, num_heads, num_layers, hidden_size):
        super(TextTransformer, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.transformer = Transformer(
            d_model=embed_size,
            nhead=num_heads,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers
        )
        self.fc_out = nn.Linear(embed_size, vocab_size)

    def forward(self, src, tgt):
        embedded_src = self.embedding(src)
        embedded_tgt = self.embedding(tgt)

        output = self.transformer(embedded_src, embedded_tgt)
        out = self.fc_out(output)
        return out