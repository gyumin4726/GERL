"""
Section 3.1 "Transformer for Context Understanding"에서 제안된
뉴스 콘텐츠 이해를 위한 Transformer 구현

이 모듈은 뉴스 제목의 단어 간 의존성을 학습하고,
토픽 정보를 통합하여 뉴스의 의미적 표현을 구축합니다.

주요 특징:
1. 단일 계층 Multi-head self-attention
2. 단어 수준 attention
3. 토픽 임베딩 통합
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class NewsTransformer(nn.Module):
    def __init__(self, config):
        """
        Section 3.1에서 설명된 뉴스 transformer 초기화
        
        Args:
            config: 모델 설정으로 다음을 포함:
                - hidden_size: 단어 임베딩 차원
                - num_attention_heads: Attention head 수 (Section 4.5에서 8로 설정)
                - head_dim: 각 attention head의 차원
                - attention_dropout: Attention dropout 비율
                - hidden_dropout: Hidden state dropout 비율
                - topic_dim: 토픽 임베딩 차원
        """
        super().__init__()
        
        # 단어 임베딩 계층 (Section 3.1 첫 번째 단락)
        self.word_embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        
        # 토픽 임베딩 계층 (Section 3.1 마지막 단락)
        self.topic_embedding = nn.Embedding(config.num_topics, config.topic_dim)
        
        # Multi-head self-attention 파라미터
        self.num_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        self.hidden_size = config.hidden_size
        
        # Query, Key, Value 변환을 위한 선형 계층
        self.query = nn.Linear(config.hidden_size, config.hidden_size)
        self.key = nn.Linear(config.hidden_size, config.hidden_size)
        self.value = nn.Linear(config.hidden_size, config.hidden_size)
        
        self.attention_dropout = nn.Dropout(config.attention_dropout)
        self.hidden_dropout = nn.Dropout(config.hidden_dropout)
        
        # 단어 수준 attention (Section 3.1 중간 부분)
        self.word_attention = nn.Sequential(
            nn.Linear(config.hidden_size, config.attention_dim),
            nn.Tanh(),
            nn.Linear(config.attention_dim, 1)
        )
        
    def transpose_for_scores(self, x):
        """
        Multi-head attention 계산을 위한 텐서 변환
        
        입력 텐서를 여러 attention head로 분할하고 
        attention 계산에 적합한 형태로 변환합니다.
        """
        batch_size = x.size(0)
        seq_length = x.size(1)
        x = x.view(batch_size, seq_length, self.num_heads, self.head_dim)
        return x.permute(0, 2, 1, 3)
        
    def forward(self, input_ids, topic_ids=None, attention_mask=None):
        """
        Section 3.1에서 설명된 뉴스 인코딩 과정
        
        1. 단어 임베딩을 통한 초기 표현 생성
        2. Multi-head self-attention으로 단어 간 관계 학습
        3. 단어 수준 attention으로 중요 단어 선택
        4. 토픽 정보 통합 (있는 경우)
        
        Args:
            input_ids: 입력 토큰 ID [batch_size, seq_length]
            topic_ids: 토픽 ID [batch_size]
            attention_mask: Attention 마스크 [batch_size, seq_length]
        """
        # 1. 단어 임베딩
        word_embeds = self.word_embedding(input_ids)
        word_embeds = self.hidden_dropout(word_embeds)
        
        # 2. Multi-head self-attention
        # Query, Key, Value 투영
        query = self.transpose_for_scores(self.query(word_embeds))
        key = self.transpose_for_scores(self.key(word_embeds))
        value = self.transpose_for_scores(self.value(word_embeds))
        
        # Attention 점수 계산 (Section 3.1 수식)
        attention_scores = torch.matmul(query, key.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.head_dim)
        
        if attention_mask is not None:
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            attention_scores = attention_scores.masked_fill(attention_mask == 0, -1e9)
        
        # Attention 가중치 정규화
        attention_probs = F.softmax(attention_scores, dim=-1)
        attention_probs = self.attention_dropout(attention_probs)
        
        # Attention 적용
        context = torch.matmul(attention_probs, value)
        context = context.permute(0, 2, 1, 3).contiguous()
        context = context.view(context.size(0), -1, self.hidden_size)
        
        # 3. 단어 수준 attention
        word_attention = self.word_attention(context)
        word_attention = F.softmax(word_attention, dim=1)
        news_vector = torch.sum(word_attention * context, dim=1)
        
        # 4. 토픽 임베딩 통합
        if topic_ids is not None:
            topic_embeds = self.topic_embedding(topic_ids)
            news_vector = torch.cat([news_vector, topic_embeds], dim=-1)
        
        return news_vector 