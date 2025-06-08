import torch
import torch.nn as nn
from typing import Tuple

class NewsEmbedding(nn.Module):
    """뉴스 임베딩 (단어 + 토픽)"""
    def __init__(self, config):
        super().__init__()
        self.word_embedding = nn.Embedding(
            config.vocab_size,
            config.word_embedding_dim,
            padding_idx=0
        )
        self.topic_embedding = nn.Embedding(
            config.num_topics,
            config.topic_embedding_dim,
            padding_idx=0
        )
        
        # 단어 임베딩과 토픽 임베딩을 hidden_size로 투영
        self.word_proj = nn.Linear(
            config.word_embedding_dim,
            config.hidden_size
        )
        self.topic_proj = nn.Linear(
            config.topic_embedding_dim,
            config.hidden_size
        )
        
        self.layer_norm = nn.LayerNorm(config.hidden_size)
        self.dropout = nn.Dropout(config.dropout)
        
    def forward(
        self,
        word_ids: torch.Tensor,
        topic_ids: torch.Tensor
    ) -> torch.Tensor:
        # 단어 임베딩
        word_embeds = self.word_embedding(word_ids)
        word_embeds = self.word_proj(word_embeds)
        
        # 토픽 임베딩
        topic_embeds = self.topic_embedding(topic_ids)
        topic_embeds = self.topic_proj(topic_embeds)
        
        # 임베딩 결합
        embeds = word_embeds + topic_embeds.unsqueeze(1).expand(-1, word_embeds.size(1), -1)
        embeds = self.layer_norm(embeds)
        embeds = self.dropout(embeds)
        
        return embeds

class UserEmbedding(nn.Module):
    """사용자 ID 임베딩"""
    def __init__(self, config):
        super().__init__()
        self.user_embedding = nn.Embedding(
            config.num_users,
            config.user_embedding_dim,
            padding_idx=0
        )
        
        # 사용자 임베딩을 hidden_size로 투영
        self.user_proj = nn.Linear(
            config.user_embedding_dim,
            config.hidden_size
        )
        
        self.layer_norm = nn.LayerNorm(config.hidden_size)
        self.dropout = nn.Dropout(config.dropout)
        
    def forward(self, user_ids: torch.Tensor) -> torch.Tensor:
        # 사용자 임베딩
        user_embeds = self.user_embedding(user_ids)
        user_embeds = self.user_proj(user_embeds)
        user_embeds = self.layer_norm(user_embeds)
        user_embeds = self.dropout(user_embeds)
        
        return user_embeds 