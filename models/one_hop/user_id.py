import torch
import torch.nn as nn
from typing import Optional

class UserIDRepresentation(nn.Module):
    """사용자 ID를 표현 벡터로 변환하는 모듈"""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # 사용자 ID 임베딩 레이어
        self.user_embedding = nn.Embedding(
            num_embeddings=config.num_users,
            embedding_dim=config.id_embedding_dim,
            padding_idx=0
        )
        
        # ID 임베딩을 hidden_size로 변환하는 프로젝션
        self.id_projection = nn.Sequential(
            nn.Linear(config.id_embedding_dim, config.hidden_size),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.LayerNorm(config.hidden_size)
        )
        
    def forward(
        self,
        user_ids: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            user_ids: 사용자 ID [batch_size]
            
        Returns:
            torch.Tensor: 사용자 ID 표현 [batch_size, hidden_size]
        """
        # ID 임베딩
        user_embeddings = self.user_embedding(user_ids)  # [batch_size, id_embedding_dim]
        
        # hidden_size로 프로젝션
        user_id_vector = self.id_projection(user_embeddings)  # [batch_size, hidden_size]
        
        return user_id_vector 