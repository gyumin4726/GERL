import torch
import torch.nn as nn
from typing import Optional, Tuple

from ..modules.transformer import NewsTransformer
from ..modules.attention import AttentiveAggregator

class NeighborNewsSemanticRepresentation(nn.Module):
    """이웃 뉴스의 의미 표현을 위한 Transformer와 Attention 기반 모듈
    
    이웃 뉴스들을 Transformer로 인코딩하고 Attention으로 집계하여
    이웃 뉴스들의 의미적 표현을 생성합니다.
    """
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # 뉴스 인코더 (Transformer)
        self.news_encoder = NewsTransformer(config)
        
        # 이웃 뉴스들의 어텐티브 어그리게이션
        self.news_attention = AttentiveAggregator(
            input_dim=config.hidden_size,
            dropout=config.dropout
        )
        
    def forward(
        self,
        neighbor_news_word_ids: torch.Tensor,
        neighbor_news_topic_ids: torch.Tensor,
        neighbor_news_mask: Optional[torch.Tensor] = None,
        word_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            neighbor_news_word_ids: 이웃 뉴스의 단어 ID [batch_size, num_neighbors, title_length]
            neighbor_news_topic_ids: 이웃 뉴스의 토픽 ID [batch_size, num_neighbors]
            neighbor_news_mask: 이웃 뉴스의 마스크 [batch_size, num_neighbors]
            word_mask: 단어 마스크 [batch_size, num_neighbors, title_length]
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                - neighbor_news_semantic: 이웃 뉴스 의미 표현 [batch_size, hidden_size]
                - news_weights: 뉴스 어텐션 가중치 [batch_size, num_neighbors]
                - word_weights: 단어 어텐션 가중치 [batch_size, num_neighbors, title_length]
        """
        batch_size, num_neighbors = neighbor_news_word_ids.shape[:2]
        
        # 이웃 뉴스들을 각각 인코딩
        neighbor_news_vectors = []
        word_attention_weights = []
        
        for i in range(num_neighbors):
            news_vector, word_weights, _ = self.news_encoder(
                neighbor_news_word_ids[:, i],
                neighbor_news_topic_ids[:, i],
                word_mask[:, i] if word_mask is not None else None
            )
            neighbor_news_vectors.append(news_vector)
            word_attention_weights.append(word_weights)
            
        neighbor_news_vectors = torch.stack(neighbor_news_vectors, dim=1)  # [batch_size, num_neighbors, hidden_size]
        word_attention_weights = torch.stack(word_attention_weights, dim=1)  # [batch_size, num_neighbors, title_length]
        
        # 이웃 뉴스들을 어텐티브하게 집계
        neighbor_news_semantic, news_attention_weights = self.news_attention(
            neighbor_news_vectors,
            neighbor_news_mask
        )
        
        return neighbor_news_semantic, news_attention_weights, word_attention_weights 