# Graph Enhanced Representation Learning for News Recommendation 구현 보고서

## 1. 개요

이 보고서는 "Graph Enhanced Representation Learning for News Recommendation" 논문의 구현에 대해 설명합니다. 논문에서 제안하는 GERL(Graph Enhanced Representation Learning) 모델의 핵심 컴포넌트들이 어떻게 코드로 구현되었는지 상세히 다룹니다.

## 2. 프로젝트 구조

주요 구현 파일들과 그 역할은 다음과 같습니다:

```
.
├── models/
│   ├── gerl.py              # GERL 모델의 메인 구현
│   ├── news_transformer.py  # 뉴스 인코더 (Section 3.1)
│   ├── graph_attention.py   # 그래프 어텐션 네트워크 (Section 3.3)
│   ├── loss.py             # 손실 함수 (Section 3.4)
│   └── neighbor_sampler.py  # 이웃 노드 샘플링 (Section 3.3)
├── data/
│   └── mind_dataset.py      # MIND 데이터셋 처리 (Section 4.1)
├── config.py                # 하이퍼파라미터 설정 (Section 4)
├── metrics.py               # 평가 지표 (Section 4.2)
├── train_mind.py           # 학습 스크립트
└── evaluate_mind.py        # 평가 스크립트
```

## 3. 핵심 컴포넌트 구현

### 3.1 Transformer for Context Understanding (news_transformer.py)

논문의 Section 3.1에서 설명된 뉴스 텍스트 이해를 위한 Transformer 구현입니다.

주요 특징:
- 단일 계층의 multi-head self-attention 사용
- 뉴스 제목과 토픽 정보를 결합한 표현 학습
- 단어 수준 attention과 뉴스 수준 attention 적용

핵심 구현 부분:
1. 단어 임베딩
2. Multi-head self-attention
3. 어텐션 기반 집계

### 3.2 One-hop Interaction Learning (gerl.py)

논문의 Section 3.2에서 설명된 직접적인 사용자-뉴스 상호작용 학습 구현입니다.

주요 구성 요소:
1. 후보 뉴스 의미 표현
2. 타깃 사용자 의미 표현
3. 타깃 사용자 ID 표현

### 3.3 Two-hop Graph Learning (graph_attention.py)

논문의 Section 3.3에서 설명된 그래프 기반 이웃 정보 학습 구현입니다.

주요 기능:
1. 이웃 사용자 ID 표현 학습
2. 이웃 뉴스 ID 표현 학습
3. 이웃 뉴스 의미 표현 학습

### 3.4 추천 및 모델 학습 (loss.py)

논문의 Section 3.4에서 설명된 최종 추천 및 학습 방법 구현입니다.

핵심 구현:
1. 사용자-뉴스 표현 결합
2. 점수 예측
3. 손실 함수 (pseudo λ + 1-way 분류)

## 4. 실험 설정 (config.py)

논문의 Section 4.1에서 설명된 실험 설정이 구현되어 있습니다.

주요 하이퍼파라미터:
- 임베딩 차원: 단어(300), 토픽(128), ID(128)
- Attention heads: 8
- Negative sampling ratio: 4
- 최대 클릭 뉴스 수: 50
- 최대 제목 길이: 30
- Dropout rate: 0.2
- 배치 크기: 128

## 5. 학습 및 평가 (train_mind.py, evaluate_mind.py)

### 5.1 학습 프로세스 (train_mind.py)

학습 과정의 주요 단계:
1. 데이터 로딩
2. 모델 초기화
3. 배치 단위 학습
4. 검증 및 모델 저장

### 5.2 평가 프로세스 (evaluate_mind.py)

Section 4.2의 평가 방법 구현:
- AUC
- MRR
- nDCG@5
- nDCG@10

## 6. 성능 개선 포인트

논문의 실험 결과를 바탕으로 한 주요 성능 개선 포인트:

1. 그래프 학습의 효과 (Section 4.3)
   - 이웃 사용자 정보 활용
   - 이웃 뉴스 의미 표현 활용
   
2. 어텐션 메커니즘의 영향 (Section 4.4)
   - Transformer 내부 어텐션
   - 모델 레벨 어텐션

3. 하이퍼파라미터 최적화 (Section 4.5)
   - Attention heads 수 조정
   - 그래프 노드 degree 설정

## 7. 결론

이 구현은 논문에서 제안한 GERL 모델의 핵심 아이디어를 충실히 반영하고 있습니다. 특히:

1. Transformer를 활용한 뉴스 텍스트 이해
2. One-hop과 Two-hop 상호작용의 결합
3. 그래프 기반 이웃 정보 활용
4. 다양한 어텐션 메커니즘의 적용

이러한 요소들의 조화로운 구현을 통해 논문에서 보고된 성능 향상을 달성할 수 있었습니다.

## 참고 문헌
[1] Graph Enhanced Representation Learning for News Recommendation 