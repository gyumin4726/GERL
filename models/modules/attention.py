import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

class MultiHeadAttention(nn.Module):
    """멀티헤드 어텐션 구현"""
    def __init__(self, config):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.scaling = self.head_dim ** -0.5
        
        self.q_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.k_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.v_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.out_proj = nn.Linear(config.hidden_size, config.hidden_size)
        
        self.dropout = nn.Dropout(config.dropout)
        
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = query.size(0)
        
        # 선형 변환 및 헤드 분할
        q = self.q_proj(query).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(key).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(value).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        # 어텐션 스코어 계산
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scaling
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))
        
        # 어텐션 가중치 계산
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # 가중치 적용 및 출력 계산
        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.head_dim)
        attn_output = self.out_proj(attn_output)
        
        return attn_output, attn_weights

class AttentiveAggregation(nn.Module):
    """어텐티브 집계 레이어"""
    def __init__(self, config):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(config.hidden_size, config.attention_dim),
            nn.Tanh(),
            nn.Linear(config.attention_dim, 1)
        )
        self.dropout = nn.Dropout(config.dropout)
    
    def forward(
        self,
        inputs: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # inputs: [batch_size, seq_len, hidden_size]
        
        # 어텐션 스코어 계산
        attention_weights = self.attention(inputs).squeeze(-1)
        
        if mask is not None:
            attention_weights = attention_weights.masked_fill(mask == 0, float("-inf"))
        
        # 소프트맥스 적용
        attention_weights = F.softmax(attention_weights, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # 가중치 적용하여 컨텍스트 벡터 계산
        context = torch.bmm(attention_weights.unsqueeze(1), inputs).squeeze(1)
        
        return context, attention_weights 