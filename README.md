# Graph Enhanced Representation Learning for News Recommendation

이 프로젝트는 "Graph Enhanced Representation Learning for News Recommendation" 논문의 핵심 아이디어를 구현한 것입니다. 뉴스 추천 시스템에서 그래프 기반 표현 학습을 통해 사용자와 뉴스의 고차원적 관련성을 모델링하는 방법을 제안합니다.

## 핵심 아이디어

### 1. 문제 정의
- 기존 뉴스 추천 방법들은 뉴스 콘텐츠와 직접적인 사용자-뉴스 상호작용만을 고려
- 사용자와 뉴스 간의 고차원적 관련성(이웃 사용자, 이웃 뉴스)을 무시
- 뉴스 콘텐츠가 짧고 모호하며 사용자의 과거 행동이 희소한 경우 정확한 표현 학습이 어려움

### 2. 제안하는 해결책
- 사용자-뉴스 상호작용을 이분 그래프로 모델링
- 이웃 뉴스와 이웃 사용자의 정보를 통합하여 표현 학습 향상
- Transformer와 Graph Attention Network를 결합한 하이브리드 아키텍처 제안

## 구현 내용

### 1. 모델 아키텍처

#### One-hop Interaction Learning (`models/gerl.py`, `models/news_transformer.py`)
```python
class GERL(nn.Module):
    def __init__(self, config):
        self.news_encoder = NewsTransformer(config)  # 뉴스 인코더
        self.user_embedding = nn.Embedding(...)      # 사용자 임베딩
        self.graph_learning = TwoHopGraphLearning(config)  # 그래프 학습
```

- **뉴스 인코딩**
  - Transformer를 사용하여 뉴스 제목의 단어 의존성 학습
  - Multi-head self-attention으로 단어 간 상호작용 모델링
  - 토픽 임베딩을 추가하여 뉴스 표현 보강

- **사용자 인코딩**
  - 클릭한 뉴스의 표현을 attention 메커니즘으로 집계
  - 사용자 ID 임베딩을 통합하여 개인화된 표현 학습

#### Two-hop Graph Learning (`models/graph_attention.py`)
```python
class TwoHopGraphLearning(nn.Module):
    def __init__(self, config):
        self.user_gat = GraphAttentionLayer(config)  # 사용자 그래프 어텐션
        self.news_gat = GraphAttentionLayer(config)  # 뉴스 그래프 어텐션
```

- **이웃 뉴스 통합**
  - 그래프 어텐션으로 이웃 뉴스의 의미적 표현 학습
  - ID 임베딩과 텍스트 표현을 모두 활용

- **이웃 사용자 통합**
  - 유사한 뉴스를 클릭한 이웃 사용자의 표현 통합
  - 희소한 사용자 행동 데이터 보완

### 2. 주요 하이퍼파라미터 (`config.py`)
```python
class Config:
    num_attention_heads = 8     # Transformer 어텐션 헤드 수
    max_neighbors = 15          # 최대 이웃 노드 수
    batch_size = 128           # 배치 크기
    negative_samples = 4       # Negative sampling ratio
```

### 3. 평가 메트릭 (`metrics.py`)
- AUC: 클릭/비클릭 분류 성능
- MRR: 순위 예측 정확도
- nDCG@5, nDCG@10: 상위 K개 추천의 품질

## 구현의 특징

1. **모듈화된 설계**
   - 각 컴포넌트(Transformer, Graph Attention, Loss 등)를 독립적인 모듈로 구현
   - 유연한 확장과 실험이 가능한 구조

2. **효율적인 그래프 처리**
   - 배치 단위 그래프 처리로 학습 효율성 향상
   - 희소 행렬 연산 최적화

3. **논문 제안 사항 충실 구현**
   - Multi-head attention 메커니즘
   - Two-hop 그래프 학습
   - Pseudo λ + 1-way 분류 손실 함수

## 사용 방법

1. 환경 설정
```bash
pip install -r requirements.txt
```

2. 학습
```bash
python train_mind.py
```

3. 평가
```bash
python evaluate_mind.py
```

## 데이터셋

- MIND (Microsoft News Dataset)
- 학습: 2018년 12월 13일 ~ 2019년 1월 12일
- 검증: 학습 데이터의 10%
- 테스트: 마지막 1주일 데이터

## 참고 문헌

[1] Graph Enhanced Representation Learning for News Recommendation 