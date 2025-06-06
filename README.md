# GERL: Graph Enhanced Representation Learning for News Recommendation

## Quick Start

프로젝트를 빠르게 실행하려면 다음 단계를 따르십시오:

```bash
# 1. 데이터 준비 (MIND 데이터셋 다운로드 후)
python check_data.py

# 2. Small 그래프 구축 (권장)
python build_graph.py --small --gpu

# 3. 훈련 시작
python train.py
```

자세한 설명은 아래 섹션을 참조하십시오.

## 프로젝트 구조

```
GERL/
├── models/
│   ├── transformer.py          # Transformer 모듈
│   ├── graph_attention.py      # Graph Attention Network
│   ├── gerl.py                # 메인 GERL 모델
│   └── __init__.py
├── data/
│   ├── dataset.py             # 데이터셋 클래스
│   └── __init__.py
├── utils/
│   ├── metrics.py             # 평가 지표
│   ├── config.py              # 설정 파일
│   └── __init__.py
├── train.py                   # 훈련 스크립트
├── quick_test.py              # 빠른 테스트 스크립트
├── check_data.py              # 데이터 구조 확인
├── requirements.txt
├── .gitignore
└── README.md
```

## 데이터 다운로드

이 프로젝트는 MIND-small 데이터셋을 사용합니다. 다음 단계에 따라 데이터를 준비하세요:

### 1. MIND 데이터셋 다운로드

Microsoft News Dataset(MIND)을 다운로드하세요:
- 공식 사이트: https://msnews.github.io/
- 필요한 파일: `MINDsmall_train.zip`, `MINDsmall_dev.zip`

### 2. 데이터 압축 해제 및 배치

```bash
# 프로젝트 루트 디렉토리에서 실행
mkdir -p data/MIND_small

# 다운로드한 파일들을 압축 해제하고 적절한 위치에 배치
unzip MINDsmall_train.zip -d data/MIND_small/train/
unzip MINDsmall_dev.zip -d data/MIND_small/dev/
```

### 3. 데이터 구조 확인

```bash
# 데이터 구조가 올바른지 확인
python check_data.py
```

올바른 데이터 구조:
```
data/
└── MIND_small/
    ├── train/
    │   ├── news.tsv        # 뉴스 메타데이터
    │   └── behaviors.tsv   # 사용자 행동 데이터
    └── dev/
        ├── news.tsv
        └── behaviors.tsv
```

## 설치 및 환경 설정

### 1. 가상환경 생성 및 활성화

```bash
# Python 가상환경 생성
python -m venv gerl_env

# 가상환경 활성화 (Windows)
gerl_env\Scripts\activate

# 가상환경 활성화 (Linux/Mac)
source gerl_env/bin/activate
```

### 2. 패키지 설치

```bash
pip install -r requirements.txt
```

### 3. 데이터 구조 확인

데이터가 올바르게 설정되었는지 확인:
```bash
python check_data.py
```

올바른 데이터 구조:
```
data/MIND_small/
├── train/
│   ├── behaviors.tsv
│   └── news.tsv
└── dev/
    ├── behaviors.tsv
    └── news.tsv
```

## 사용법

### 1. 그래프 구축 (권장 - 필수 단계)

훈련 전에 먼저 그래프를 구축하세요. 이는 훈련 속도를 크게 향상시킵니다:

#### Small 버전 (권장) - 빠른 테스트와 개발용

메모리와 시간을 절약하면서 빠르게 테스트하려면 small 버전을 사용하십시오:

```bash
# Small 버전 그래프 구축 (30% 샘플링, 최대 50개 이웃)
python build_graph.py --small

# GPU 가속 사용
python build_graph.py --small --gpu

# 커스텀 설정
python build_graph.py --small --sample_ratio 0.2 --max_neighbors 30
```

Small 버전 특징:
- **데이터 크기**: 원본의 30% 샘플링 (기본값)
- **이웃 수 제한**: 최대 50개 이웃으로 제한 (기본값)  
- **파일 크기**: 각 그래프 파일 < 750MB (총 < 2.25GB)
- **속도**: 빠른 로딩과 훈련 가능
- **용도**: 개발, 테스트, 빠른 실험에 적합

#### Standard 버전 - 전체 데이터셋

전체 성능을 원한다면 standard 버전을 사용하십시오:

```bash
# 전체 데이터셋으로 그래프 구축
python build_graph.py

# GPU 가속 사용
python build_graph.py --gpu
```

#### 기타 옵션

```bash
# 특정 split만 처리
python build_graph.py --small --split train
python build_graph.py --small --split dev

# 강제로 재구축
python build_graph.py --small --force_rebuild
```

#### 생성되는 파일들

Small 버전:
- `data/MIND_small/vocab_small.pkl` - 어휘 사전 (small)
- `data/MIND_small/graph_train_small.pkl` - 훈련용 이분 그래프 (small)
- `data/MIND_small/graph_dev_small.pkl` - 검증용 이분 그래프 (small)

Standard 버전:
- `data/MIND_small/vocab.pkl` - 어휘 사전
- `data/MIND_small/graph_train.pkl` - 훈련용 이분 그래프
- `data/MIND_small/graph_dev.pkl` - 검증용 이분 그래프

**참고**: 훈련 스크립트는 자동으로 small 버전이 있으면 우선적으로 사용합니다.

### 2. 빠른 테스트

모든 것이 올바르게 설정되었는지 확인:
```bash
python quick_test.py
```

### 3. 훈련

```bash
python train.py
```

또는 커스텀 설정으로:
```bash
python train.py --batch_size 32 --epochs 10 --lr 0.001
```

### 주의사항

- **필수**: 첫 실행 전 반드시 `build_graph.py`를 실행하십시오
- **권장**: 빠른 시작을 위해 `--small` 옵션 사용 권장
- **설정**: 세부 설정은 `utils/config.py`에서 변경하십시오
- **저장**: 모델은 자동으로 `saved_models/` 폴더에 저장됩니다
- **시간**: 그래프 구축은 시간이 걸리지만 한 번만 하면 됩니다
- **우선순위**: 훈련 시 small 버전이 있으면 자동으로 우선 사용됩니다

### 성능 최적화 방법

1. **빠른 개발**: `--small --gpu` 옵션으로 시작
2. **메모리 부족 시**: `--sample_ratio 0.1 --max_neighbors 20`으로 더 작게 설정
3. **최종 성능**: 개발 완료 후 전체 데이터셋으로 재훈련
4. **GPU 활용**: 그래프 구축과 훈련 모두에서 `--gpu` 옵션 사용

## 모델 특징

- **Transformer 기반 뉴스 표현**: Multi-head self-attention을 통한 컨텍스트 이해
- **Graph Attention Network**: 이웃 사용자/뉴스 정보 활용
- **One-hop & Two-hop Learning**: 직접 상호작용과 그래프 관계 학습

## 논문 참조

본 구현은 다음 논문을 기반으로 합니다:
- Graph Enhanced Representation Learning for News Recommendation 