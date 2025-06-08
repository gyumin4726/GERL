import torch
import torch.nn as nn
from typing import Optional, Tuple

from ..modules.attention import AttentiveAggregator

class NeighborUserIDRepresentation(nn.Module):
    """이웃 사용자의 ID 표현을 위한 임베딩과 Attention 기반 모듈
    
    이웃 사용자들의 ID를 임베딩하고 Attention으로 집계하여
    이웃 사용자들의 ID 기반 표현을 생성합니다.
    """
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # 사용자 ID 임베딩
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
        
        # 이웃 사용자들의 어텐티브 어그리게이션
        self.user_attention = AttentiveAggregator(
            input_dim=config.hidden_size,
            dropout=config.dropout
        )
        
    def forward(
        self,
        neighbor_user_ids: torch.Tensor,
        neighbor_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            neighbor_user_ids: 이웃 사용자 ID [batch_size, num_neighbors]
            neighbor_mask: 이웃 사용자 마스크 [batch_size, num_neighbors]
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - neighbor_user_id: 이웃 사용자 ID 표현 [batch_size, hidden_size]
                - attention_weights: 어텐션 가중치 [batch_size, num_neighbors]
        """
        # ID 임베딩
        user_embeddings = self.user_embedding(neighbor_user_ids)  # [batch_size, num_neighbors, id_embedding_dim]
        
        # hidden_size로 프로젝션
        user_vectors = self.id_projection(user_embeddings)  # [batch_size, num_neighbors, hidden_size]
        
        # 이웃 사용자들을 어텐티브하게 집계
        neighbor_user_id, attention_weights = self.user_attention(
            user_vectors,
            neighbor_mask
        )
        
        return neighbor_user_id, attention_weights 