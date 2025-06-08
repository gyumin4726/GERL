# Graph Enhanced Representation Learning (GERL) for News Recommendation

GERL은 그래프 기반의 향상된 표현 학습을 통해 뉴스 추천의 성능을 개선하는 딥러닝 모델입니다.

## 모델 구조

### 기본 모듈 (Base Modules)

모델은 다음과 같은 기본 모듈들을 활용합니다:

- **Transformer**: 뉴스 텍스트의 의미적 표현을 학습
- **Embedding**: 사용자와 뉴스의 ID 기반 표현을 학습
- **Attention**: 다양한 표현들 간의 중요도를 학습

### One-hop 모듈

직접적인 사용자-뉴스 상호작용을 모델링합니다:

1. **후보 뉴스 의미 표현** (`candidate_news_semantic.py`)
   - Transformer 기반 표현 모듈
   - 뉴스 제목과 본문의 의미적 특성 추출

2. **사용자 ID 표현** (`user_id.py`)
   - 임베딩 기반 표현 모듈
   - 사용자의 고유 특성 학습

3. **사용자 의미 표현** (`user_semantic.py`)
   - Transformer + Attention 기반 표현 모듈
   - 사용자가 읽은 뉴스들의 의미적 표현을 통합

### Two-hop 모듈

간접적인 사용자-뉴스 관계를 모델링합니다:

1. **이웃 뉴스 의미 표현** (`neighbor_news_semantic.py`)
   - Transformer + Attention 기반 표현 모듈
   - 이웃 뉴스들의 의미적 특성을 통합

2. **이웃 뉴스 ID 표현** (`neighbor_news_id.py`)
   - 임베딩 + Attention 기반 표현 모듈
   - 이웃 뉴스들의 ID 기반 특성을 통합

3. **이웃 사용자 ID 표현** (`neighbor_user_id.py`)
   - 임베딩 + Attention 기반 표현 모듈
   - 이웃 사용자들의 ID 기반 특성을 통합

### 최종 표현 (Final Representation)

각 모듈에서 생성된 표현들은 다음과 같이 통합됩니다:

1. **사용자 최종 표현**
   - 사용자 의미 표현 (One-hop)
   - 사용자 ID 표현 (One-hop)
   - 이웃 사용자 ID 표현 (Two-hop)

2. **뉴스 최종 표현**
   - 후보 뉴스 의미 표현 (One-hop)
   - 이웃 뉴스 ID 표현 (Two-hop)
   - 이웃 뉴스 의미 표현 (Two-hop)

3. **최종 예측**
   - 최종 점수는 사용자와 뉴스 표현의 내적으로 계산
   - scores = user · news

## 설정 (Configuration)

주요 하이퍼파라미터:

```python
{
    # 임베딩 차원
    "word_embedding_dim": 300,    # 단어 임베딩
    "topic_embedding_dim": 128,   # 토픽 임베딩
    "id_embedding_dim": 128,      # ID 임베딩

    # Attention 설정
    "num_attention_heads": 8,     # 어텐션 헤드 수
    "attention_head_size": 16,    # 어텐션 헤드 크기

    # 시퀀스 길이 제한
    "max_clicked_news": 50,       # 최대 클릭 뉴스 수
    "max_title_length": 30,       # 최대 제목 길이

    # 학습 설정
    "negative_sample_ratio": 4    # 부정 샘플링 비율
}
```

## 평가 지표

모델의 성능은 다음 지표들로 평가됩니다:
- AUC (Area Under the ROC Curve)
- MRR (Mean Reciprocal Rank)
- nDCG@5, nDCG@10 (Normalized Discounted Cumulative Gain)

## 손실 함수

모델은 논문의 Equation (5)에 따른 대조 손실 함수를 사용합니다:
```
L = -∑ log(exp(y_i+) / (exp(y_i+) + ∑exp(y_i,j-)))
```
여기서:
- y_i+: 긍정(클릭) 샘플의 예측 점수
- y_i,j-: 부정(비클릭) 샘플들의 예측 점수
- ∑: 배치 내 모든 샘플에 대한 합계

이 손실 함수는 클릭된 뉴스(긍정 샘플)와 클릭되지 않은 뉴스(부정 샘플) 간의 점수 차이를 최대화하도록 학습을 유도합니다. 