import torch
import torch.nn as nn
from typing import Optional, Tuple

from ..modules.transformer import NewsTransformer

class CandidateNewsSemanticRepresentation(nn.Module):
    """후보 뉴스의 의미 표현을 위한 Transformer 기반 모듈
    
    뉴스 제목을 Transformer로 인코딩하여 의미적 표현을 생성합니다.
    """
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # 뉴스 인코더 (Transformer)
        self.news_encoder = NewsTransformer(config)
        
    def forward(
        self,
        news_word_ids: torch.Tensor,
        news_topic_ids: torch.Tensor,
        news_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            news_word_ids: 뉴스 제목의 단어 ID [batch_size, title_length]
            news_topic_ids: 뉴스 토픽 ID [batch_size]
            news_mask: 뉴스 제목의 마스크 [batch_size, title_length]
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                - news_vector: 뉴스 의미 표현 [batch_size, hidden_size]
                - word_weights: 단어 어텐션 가중치 [batch_size, title_length]
                - self_attention: 셀프 어텐션 가중치 [batch_size, num_heads, title_length, title_length]
        """
        return self.news_encoder(news_word_ids, news_topic_ids, news_mask) 