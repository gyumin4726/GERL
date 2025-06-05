# GERL: Graph Enhanced Representation Learning for News Recommendation

논문 "Graph Enhanced Representation Learning for News Recommendation"에서 제안된 GERL 모델의 PyTorch 구현입니다.

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

**⚠️ 주의사항:** 
- MIND 데이터셋은 저작권이 있으므로 GitHub에 업로드하지 마세요
- `.gitignore`가 자동으로 data/ 폴더를 제외합니다

## 사용법

### 빠른 테스트

먼저 모든 것이 올바르게 설정되었는지 확인:
```bash
python quick_test.py
```

### 훈련

```bash
python train.py
```

또는 커스텀 설정으로:
```bash
python train.py --batch_size 32 --epochs 10 --lr 0.001
```

### 주의사항

- GPU 메모리가 부족한 경우 `config.py`에서 배치 크기를 줄이세요
- 첫 실행 시 데이터 로딩에 시간이 걸릴 수 있습니다
- 모델은 자동으로 `saved_models/` 폴더에 저장됩니다

## 모델 특징

- **Transformer 기반 뉴스 표현**: Multi-head self-attention을 통한 컨텍스트 이해
- **Graph Attention Network**: 이웃 사용자/뉴스 정보 활용
- **One-hop & Two-hop Learning**: 직접 상호작용과 그래프 관계 학습

## 논문 참조

본 구현은 다음 논문을 기반으로 합니다:
- Graph Enhanced Representation Learning for News Recommendation 