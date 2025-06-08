"""
Section 4 "Experiments"에서 설명된 모델 설정과 하이퍼파라미터

이 모듈은 GERL 모델의 모든 설정값을 관리합니다.
논문의 Section 4.1과 4.5에서 제안된 최적의 하이퍼파라미터를
기본값으로 설정하고 있습니다.

주요 설정:
1. 모델 아키텍처 파라미터 (Section 3)
2. Transformer 설정 (Section 3.1)
3. 그래프 학습 설정 (Section 3.3)
4. 학습 파라미터 (Section 4.1)
"""

class Config:
    # 모델 아키텍처 (Section 3 "OUR APPROACH")
    hidden_size = 300  # 단어 임베딩 차원
    topic_dim = 128   # 토픽 임베딩 차원 (Section 3.1 마지막 단락)
    id_dim = 128      # 사용자 ID 임베딩 차원 (Section 3.2)
    attention_dim = 200  # 어텐션 계층의 은닉 차원
    
    # Transformer 파라미터 (Section 3.1 & 4.5)
    num_attention_heads = 8  # Section 4.5에서 제안된 최적값
    head_dim = 16           # hidden_size / num_heads = 128 / 8
    attention_dropout = 0.2  # Attention dropout 비율
    hidden_dropout = 0.2    # Hidden state dropout 비율
    
    # 그래프 파라미터 (Section 3.3 & 4.5)
    graph_hidden_size = 128  # 그래프 어텐션의 은닉 차원
    graph_attention_heads = 8  # Section 4.5에서 제안된 최적값
    graph_dropout = 0.2  # 그래프 레이어의 dropout 비율
    max_neighbors = 15   # Section 4.5에서 제안된 최적의 이웃 노드 수
    
    # 학습 파라미터 (Section 4.1 "Implementation Details")
    batch_size = 128     # Section 4.1에서 제안된 배치 크기
    negative_samples = 4  # Section 3.4의 pseudo λ + 1-way 분류에서의 λ 값
    learning_rate = 0.001  # Adam 옵티마이저의 학습률
    
    # 데이터 파라미터 (Section 4.1)
    max_title_length = 30    # 뉴스 제목의 최대 토큰 수
    max_history_length = 50  # 사용자당 최대 클릭 기록 수
    
    # 어휘 파라미터 (데이터셋에 따라 설정)
    vocab_size = None  # 전체 어휘 크기
    num_users = None   # 전체 사용자 수
    num_topics = None  # 전체 토픽 수
    
    def __init__(self):
        """
        기본 설정으로 Config 객체 초기화
        모든 기본값은 논문의 Section 4에서 제안된 최적값을 따름
        """
        pass
        
    @classmethod
    def from_dict(cls, json_object):
        """
        JSON 형식의 설정을 Config 객체로 변환
        
        Args:
            json_object: 설정값을 담은 딕셔너리
        """
        config = cls()
        for key, value in json_object.items():
            setattr(config, key, value)
        return config 