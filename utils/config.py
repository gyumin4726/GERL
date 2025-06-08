from dataclasses import dataclass
from typing import Optional

@dataclass
class ModelConfig:
    """모델 구조 관련 설정"""
    
    # 임베딩 관련 설정
    vocab_size: int = 30000
    word_embedding_dim: int = 300  # 논문 기준
    topic_embedding_dim: int = 128  # 논문 기준
    id_embedding_dim: int = 128  # 논문 기준
    num_topics: int = 100
    num_users: int = 10000
    
    # Transformer 관련 설정
    hidden_size: int = 256
    num_attention_heads: int = 8  # 논문 기준
    attention_head_size: int = 16  # 논문 기준
    ff_dim: int = 512
    dropout: float = 0.2
    
    # Attention 관련 설정
    attention_dim: int = 200
    
    # Graph 관련 설정
    max_neighbor_news: int = 15
    max_neighbor_users: int = 15

@dataclass
class TrainingConfig:
    """학습 관련 설정"""
    
    # 데이터 관련 설정
    max_title_length: int = 30  # 논문 기준
    max_clicked_news: int = 50  # 논문 기준
    
    # 학습 하이퍼파라미터
    batch_size: int = 128
    learning_rate: float = 0.001
    num_epochs: int = 10
    negative_sample_ratio: int = 4  # 논문 기준
    
    # 검증 설정
    ndcg_k: list[int] = (5, 10)  # nDCG@k 계산을 위한 k값들

@dataclass
class Config:
    """통합 설정"""
    model: ModelConfig = ModelConfig()
    training: TrainingConfig = TrainingConfig() 