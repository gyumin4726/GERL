import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict
import numpy as np

from .one_hop.candidate_news_semantic import CandidateNewsSemanticRepresentation
from .one_hop.user_id import UserIDRepresentation
from .one_hop.user_semantic import UserSemanticRepresentation
from .two_hop.neighbor_news_semantic import NeighborNewsSemanticRepresentation
from .two_hop.neighbor_news_id import NeighborNewsIDRepresentation
from .two_hop.neighbor_user_id import NeighborUserIDRepresentation

class GERL(nn.Module):
    """Graph Enhanced Representation Learning for News Recommendation
    
    논문의 구조:
    1. One-hop Interaction Learning
        - 후보 뉴스 의미 표현 (Transformer)
        - 사용자 ID 표현 (Embedding)
        - 사용자 의미 표현 (Transformer + Attention)
    
    2. Two-hop Graph Learning
        - 이웃 뉴스 의미 표현 (Transformer + Attention)
        - 이웃 뉴스 ID 표현 (Embedding + Attention)
        - 이웃 사용자 ID 표현 (Embedding + Attention)
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # One-hop 모듈
        self.candidate_news_semantic = CandidateNewsSemanticRepresentation(config)  # 후보 뉴스 의미 표현 (Transformer)
        self.user_id = UserIDRepresentation(config)                                 # 사용자 ID 표현 (Embedding)
        self.user_semantic = UserSemanticRepresentation(config)                    # 사용자 의미 표현 (Transformer + Attention)
        
        # Two-hop 모듈
        self.neighbor_news_semantic = NeighborNewsSemanticRepresentation(config)  # 이웃 뉴스 의미 표현 (Transformer + Attention)
        self.neighbor_news_id = NeighborNewsIDRepresentation(config)             # 이웃 뉴스 ID 표현 (Embedding + Attention)
        self.neighbor_user_id = NeighborUserIDRepresentation(config)             # 이웃 사용자 ID 표현 (Embedding + Attention)
        
        # 사용자 최종 표현을 위한 결합 레이어
        self.user_combine = nn.Sequential(
            nn.Linear(config.hidden_size * 3, config.hidden_size),  # [의미, ID, 이웃ID]
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.LayerNorm(config.hidden_size)
        )
        
        # 뉴스 최종 표현을 위한 결합 레이어
        self.news_combine = nn.Sequential(
            nn.Linear(config.hidden_size * 3, config.hidden_size),  # [의미, 이웃ID, 이웃의미]
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.LayerNorm(config.hidden_size)
        )
        
    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Args:
            batch: 입력 배치 데이터
                - candidate_news_*: 후보 뉴스 관련 데이터
                - user_ids: 사용자 ID
                - clicked_news_*: 클릭한 뉴스 관련 데이터
                - neighbor_news_*: 이웃 뉴스 관련 데이터
                - neighbor_user_*: 이웃 사용자 관련 데이터
        """
        # One-hop Interaction Learning
        # 1. 후보 뉴스 의미 표현 (Transformer)
        candidate_news_vector = self.candidate_news_semantic(
            batch['candidate_news_word_ids'],
            batch['candidate_news_topic_ids'],
            batch.get('candidate_news_mask')
        )
        
        # 2. 사용자 ID 표현 (Embedding)
        user_id_vector = self.user_id(batch['user_ids'])
        
        # 3. 사용자 의미 표현 (Transformer + Attention)
        user_semantic_vector = self.user_semantic(
            batch['clicked_news_word_ids'],
            batch['clicked_news_topic_ids'],
            batch.get('clicked_news_mask')
        )
        
        # Two-hop Graph Learning
        # 1. 이웃 뉴스 의미 표현 (Transformer + Attention)
        neighbor_news_semantic = self.neighbor_news_semantic(
            batch['neighbor_news_word_ids'],
            batch['neighbor_news_topic_ids'],
            batch.get('neighbor_news_mask')
        )
        
        # 2. 이웃 뉴스 ID 표현 (Embedding + Attention)
        neighbor_news_id = self.neighbor_news_id(
            batch['neighbor_news_ids'],
            batch.get('neighbor_news_mask')
        )
        
        # 3. 이웃 사용자 ID 표현 (Embedding + Attention)
        neighbor_user_id = self.neighbor_user_id(
            batch['neighbor_user_ids'],
            batch.get('neighbor_user_mask')
        )
        
        # 최종 표현 결합
        # 1. 사용자 표현: [의미, ID, 이웃ID]
        user_vector = self.user_combine(
            torch.cat([
                user_semantic_vector,  # One-hop 의미
                user_id_vector,       # One-hop ID
                neighbor_user_id      # Two-hop 사용자 ID
            ], dim=-1)
        )
        
        # 2. 뉴스 표현: [의미, 이웃ID, 이웃의미]
        news_vector = self.news_combine(
            torch.cat([
                candidate_news_vector,  # One-hop 의미
                neighbor_news_id,       # Two-hop 뉴스 ID
                neighbor_news_semantic  # Two-hop 뉴스 의미
            ], dim=-1)
        )
        
        # 최종 예측 점수 계산 (내적)
        scores = torch.sum(news_vector * user_vector, dim=-1)
        
        return {
            'scores': scores,
            'user_vector': user_vector,
            'news_vector': news_vector,
            # One-hop 표현
            'user_semantic_vector': user_semantic_vector,
            'user_id_vector': user_id_vector,
            'candidate_news_vector': candidate_news_vector,
            # Two-hop 표현
            'neighbor_user_id': neighbor_user_id,
            'neighbor_news_id': neighbor_news_id,
            'neighbor_news_semantic': neighbor_news_semantic
        } 