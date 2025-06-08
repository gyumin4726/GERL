import torch
import torch.nn as nn
from typing import Optional, Tuple
from .attention import MultiHeadAttention

class TransformerEncoder(nn.Module):
    """트랜스포머 인코더 레이어"""
    def __init__(self, config):
        super().__init__()
        self.self_attention = MultiHeadAttention(config)
        self.layer_norm1 = nn.LayerNorm(config.hidden_size)
        self.layer_norm2 = nn.LayerNorm(config.hidden_size)
        
        # Position-wise Feed-Forward Network
        self.ff_network = nn.Sequential(
            nn.Linear(config.hidden_size, config.ff_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.ff_dim, config.hidden_size)
        )
        
        self.dropout = nn.Dropout(config.dropout)
        
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Self Attention
        attn_output, attn_weights = self.self_attention(x, x, x, mask)
        attn_output = self.dropout(attn_output)
        out1 = self.layer_norm1(x + attn_output)
        
        # Feed Forward
        ff_output = self.ff_network(out1)
        out2 = self.layer_norm2(out1 + ff_output)
        
        return out2, attn_weights

class NewsTransformer(nn.Module):
    """뉴스 텍스트를 위한 트랜스포머 인코더"""
    def __init__(self, config):
        super().__init__()
        self.position_embedding = nn.Embedding(
            config.max_news_length,
            config.hidden_size
        )
        
        self.encoder_layer = TransformerEncoder(config)
        self.dropout = nn.Dropout(config.dropout)
        
    def forward(
        self,
        input_embeds: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        seq_length = input_embeds.size(1)
        position_ids = torch.arange(seq_length, device=input_embeds.device).unsqueeze(0)
        position_embeds = self.position_embedding(position_ids)
        
        # 입력 임베딩과 위치 임베딩 결합
        hidden_states = input_embeds + position_embeds
        hidden_states = self.dropout(hidden_states)
        
        # 트랜스포머 인코딩
        encoded_states, attention_weights = self.encoder_layer(hidden_states, attention_mask)
        
        return encoded_states, attention_weights 