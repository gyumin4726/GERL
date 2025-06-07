import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer

class NewsTransformer(nn.Module):
    def __init__(self, vocab_size, d_model, nhead, dropout=0.1):
        super(NewsTransformer, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        
        encoder_layers = TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = TransformerEncoder(encoder_layers, num_layers=2)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, src):
        # src: [batch_size, seq_len]
        
        # Embedding + Positional Encoding
        src = self.embedding(src) * (src.size(-1) ** 0.5)  # [batch_size, seq_len, d_model]
        src = self.pos_encoder(src)
        
        # Transformer Encoding
        output = self.transformer_encoder(src)  # [batch_size, seq_len, d_model]
        
        # Pool the output (mean pooling)
        output = output.mean(dim=1)  # [batch_size, d_model]
        
        return output

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x) 