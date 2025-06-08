import torch
import torch.nn as nn
from typing import Optional, Tuple

from ..modules.transformer import NewsTransformer
from ..modules.attention import AttentiveAggregator

class UserSemanticRepresentation(nn.Module):
    """사용자의 의미 표현을 위한 Transformer와 Attention 기반 모듈
    
    클릭한 뉴스들을 Transformer로 인코딩하고 Attention으로 집계하여
    사용자의 의미적 표현을 생성합니다.
    """
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # 뉴스 인코더 (Transformer)
        self.news_encoder = NewsTransformer(config)
        
        # 클릭한 뉴스들의 어텐티브 어그리게이션
        self.news_attention = AttentiveAggregator(
            input_dim=config.hidden_size,
            dropout=config.dropout
        )
        
    def forward(
        self,
        clicked_news_word_ids: torch.Tensor,
        clicked_news_topic_ids: torch.Tensor,
        clicked_news_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            clicked_news_word_ids: 클릭한 뉴스의 단어 ID [batch_size, num_clicked, title_length]
            clicked_news_topic_ids: 클릭한 뉴스의 토픽 ID [batch_size, num_clicked]
            clicked_news_mask: 클릭한 뉴스의 마스크 [batch_size, num_clicked]
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                - user_semantic: 사용자 의미 표현 [batch_size, hidden_size]
                - news_weights: 뉴스 어텐션 가중치 [batch_size, num_clicked]
                - word_weights: 단어 어텐션 가중치 [batch_size, num_clicked, title_length]
        """
        batch_size, num_clicked = clicked_news_word_ids.shape[:2]
        
        # 클릭한 뉴스들을 각각 인코딩
        clicked_news_vectors = []
        word_attention_weights = []
        
        for i in range(num_clicked):
            news_vector, word_weights, _ = self.news_encoder(
                clicked_news_word_ids[:, i],
                clicked_news_topic_ids[:, i],
                None  # 개별 뉴스에 대한 마스크는 필요 없음
            )
            clicked_news_vectors.append(news_vector)
            word_attention_weights.append(word_weights)
            
        clicked_news_vectors = torch.stack(clicked_news_vectors, dim=1)  # [batch_size, num_clicked, hidden_size]
        word_attention_weights = torch.stack(word_attention_weights, dim=1)  # [batch_size, num_clicked, title_length]
        
        # 클릭한 뉴스들을 어텐티브하게 집계
        user_semantic, news_attention_weights = self.news_attention(
            clicked_news_vectors,
            clicked_news_mask
        )
        
        return user_semantic, news_attention_weights, word_attention_weights 