# GERL (Graph Enhanced Representation Learning) 구현 보고서

## 1. 개요

본 보고서는 "Graph Enhanced Representation Learning for News Recommendation" 논문의 핵심 모듈 구현에 대해 설명합니다. GERL은 그래프 기반의 표현 학습을 통해 뉴스 추천의 성능을 개선하는 모델로, One-hop과 Two-hop 상호작용을 모두 고려합니다.

## 2. 구현 구조

### 2.1 기본 모듈 구현

모델의 기본이 되는 세 가지 핵심 모듈을 다음과 같이 구현했습니다:

```python
# Transformer 구현 예시
class TransformerEncoder(nn.Module):
    def __init__(self, config):
        self.attention = nn.MultiheadAttention(
            embed_dim=config.hidden_size,
            num_heads=config.num_attention_heads
        )
        self.feed_forward = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.ReLU(),
            nn.Linear(config.hidden_size, config.hidden_size)
        )

# Embedding 구현 예시
self.word_embedding = nn.Embedding(
    num_embeddings=config.vocab_size,
    embedding_dim=config.word_embedding_dim
)

# Attention 구현 예시
self.attention = nn.MultiheadAttention(
    embed_dim=config.hidden_size,
    num_heads=config.num_attention_heads
)
```

### 2.2 One-hop 모듈 구현

직접적인 사용자-뉴스 상호작용을 위한 세 가지 모듈을 구현했습니다:

1. **후보 뉴스 의미 표현** (`CandidateNewsSemanticRepresentation`)
```python
def forward(self, word_ids, topic_ids, mask=None):
    # 1. 단어 임베딩
    word_embed = self.word_embedding(word_ids)
    # 2. 토픽 임베딩
    topic_embed = self.topic_embedding(topic_ids)
    # 3. Transformer 인코딩
    news_repr = self.transformer(word_embed + topic_embed)
    return news_repr
```

2. **사용자 ID 표현** (`UserIDRepresentation`)
```python
def forward(self, user_ids):
    # 단순 임베딩 레이어를 통한 ID 표현
    return self.user_embedding(user_ids)
```

3. **사용자 의미 표현** (`UserSemanticRepresentation`)
```python
def forward(self, clicked_news_word_ids, clicked_news_topic_ids, mask=None):
    # 1. 클릭한 뉴스들의 의미 표현
    news_reprs = self.news_encoder(clicked_news_word_ids, clicked_news_topic_ids)
    # 2. Attention으로 통합
    user_repr = self.attention_layer(news_reprs, mask)
    return user_repr
```

### 2.3 Two-hop 모듈 구현

간접적인 관계 모델링을 위한 세 가지 모듈을 구현했습니다:

1. **이웃 뉴스 의미 표현** (`NeighborNewsSemanticRepresentation`)
```python
def forward(self, word_ids, topic_ids, mask=None):
    # 1. 각 이웃 뉴스의 의미 표현
    news_reprs = self.news_encoder(word_ids, topic_ids)
    # 2. Attention으로 통합
    neighbor_repr = self.attention_layer(news_reprs, mask)
    return neighbor_repr
```

2. **이웃 뉴스 ID 표현** (`NeighborNewsIDRepresentation`)
```python
def forward(self, news_ids, mask=None):
    # 1. ID 임베딩
    news_embeds = self.news_embedding(news_ids)
    # 2. Attention으로 통합
    neighbor_repr = self.attention_layer(news_embeds, mask)
    return neighbor_repr
```

3. **이웃 사용자 ID 표현** (`NeighborUserIDRepresentation`)
```python
def forward(self, user_ids, mask=None):
    # 1. ID 임베딩
    user_embeds = self.user_embedding(user_ids)
    # 2. Attention으로 통합
    neighbor_repr = self.attention_layer(user_embeds, mask)
    return neighbor_repr
```

### 2.4 최종 표현 통합 구현

각 모듈의 출력을 효과적으로 통합하는 방식을 구현했습니다:

1. **사용자 최종 표현**
```python
user = self.user_combine(
    torch.cat([
        user_semantic,     # One-hop: 의미 표현
        user_id,          # One-hop: ID 표현
        neighbor_user_id   # Two-hop: 이웃 ID 표현
    ], dim=-1)
)
```

2. **뉴스 최종 표현**
```python
news = self.news_combine(
    torch.cat([
        candidate_news_semantic,  # One-hop: 의미 표현
        neighbor_news_id,        # Two-hop: ID 표현
        neighbor_news_semantic   # Two-hop: 의미 표현
    ], dim=-1)
)
```

3. **최종 예측 점수 계산**
```python
scores = torch.sum(news * user, dim=-1)  # 내적을 통한 유사도 계산
```

## 3. 구현 세부사항

### 3.1 주요 하이퍼파라미터

모델의 성능에 중요한 영향을 미치는 하이퍼파라미터들을 다음과 같이 설정했습니다:

```python
config = {
    "word_embedding_dim": 300,    # 단어 임베딩 차원
    "topic_embedding_dim": 128,   # 토픽 임베딩 차원
    "id_embedding_dim": 128,      # ID 임베딩 차원
    "num_attention_heads": 8,     # 어텐션 헤드 수
    "attention_head_size": 16,    # 어텐션 헤드 크기
    "max_clicked_news": 50,       # 최대 클릭 뉴스 수
    "max_title_length": 30,       # 최대 제목 길이
}
```

### 3.2 손실 함수 구현

논문의 대조 학습 손실 함수를 다음과 같이 구현했습니다:

```python
def compute_loss(pos_scores, neg_scores):
    """
    pos_scores: 긍정(클릭) 샘플의 예측 점수
    neg_scores: 부정(비클릭) 샘플들의 예측 점수
    """
    pos_exp = torch.exp(pos_scores)
    neg_exp = torch.exp(neg_scores)
    
    loss = -torch.log(pos_exp / (pos_exp + torch.sum(neg_exp, dim=1)))
    return torch.mean(loss)
```

## 4. 결론

GERL 모델의 구현을 통해 다음과 같은 특징을 가진 뉴스 추천 시스템을 구현했습니다:

1. One-hop과 Two-hop 상호작용을 모두 고려하여 더 풍부한 사용자-뉴스 관계 모델링
2. 효율적인 표현 학습을 위한 다양한 임베딩과 attention 메커니즘 활용
3. 간단하면서도 효과적인 내적 기반 최종 예측 방식

이러한 구현을 통해 논문에서 제시한 방법론을 실제 시스템에 적용 가능한 형태로 구현했습니다. 