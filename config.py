class Config:
    # Model Architecture
    d_model = 256  # Transformer의 차원 (num_heads로 나누어 떨어져야 함)
    num_heads = 8  # Transformer의 어텐션 헤드 수
    dropout = 0.1  # Dropout 비율
    
    # Embedding Dimensions
    topic_dim = 128  # 토픽 임베딩 차원
    id_dim = 128    # ID 임베딩 차원
    attention_dim = 200  # 어텐션 레이어의 은닉층 차원
    final_dim = 256  # 최종 표현 차원
    
    # MIND Dataset Parameters
    max_title_length = 30    # 뉴스 제목의 최대 길이
    max_history_length = 50  # 사용자당 최대 클릭 히스토리 길이
    vocab_size = 30522      # BERT tokenizer의 어휘 크기
    
    # Training
    batch_size = 64
    learning_rate = 0.001
    num_epochs = 10
    
    # Graph
    max_neighbors = 15  # 최대 이웃 노드 수 