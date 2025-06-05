class Config:
    """GERL 모델 설정"""
    
    # 데이터 관련 설정
    max_title_len = 30           # 논문: 뉴스 제목 최대 길이
    max_clicked_news = 50        # 논문: 사용자가 클릭한 뉴스 최대 수
    max_neighbors = 15           # 논문: 이웃 노드 degree (Figure 5b에서 15로 설정)
    
    # 임베딩 차원 설정 
    word_embed_dim = 256         # 8로 나누어떨어지는 값 (멀티헤드 호환)
    user_embed_dim = 128         # 논문: 사용자 ID 임베딩 차원
    news_embed_dim = 128         # 논문: 뉴스 ID 임베딩 차원
    topic_embed_dim = 128        # 논문: 토픽 임베딩 차원
    
    # Transformer 설정
    num_heads = 8               # 논문: multi-head attention의 head 수 (Figure 5a에서 8로 설정)
    attention_hidden_dim = 128  # 어텐션 네트워크 은닉 차원
    
    # 훈련 설정
    batch_size = 128            # 논문: 배치 크기
    learning_rate = 0.001       # Adam 학습률
    dropout = 0.2               # 논문: dropout rate
    num_epochs = 10             # 훈련 에포크 수
    negative_sampling_ratio = 4 # 논문: λ = 4 (Eq. 5)
    
    # 데이터셋 크기 (실제 데이터 로딩 후 업데이트 필요)
    vocab_size = 50000          # 어휘 크기
    num_users = 50000           # 사용자 수
    num_news = 100000           # 뉴스 수
    num_topics = 20             # 토픽 카테고리 수
    
    # GloVe 임베딩 설정 (논문에서 사용)
    glove_path = "data/glove.6B.200d.txt"  # GloVe 파일 경로 (256에 가까운 차원)
    use_pretrained_embeddings = True        # 사전 학습된 임베딩 사용 여부
    
    # 기타 설정
    device = 'cuda'             # GPU 사용
    seed = 42                   # 재현성을 위한 시드
    save_dir = 'saved_models'   # 모델 저장 디렉토리
    
    # 평가 설정
    eval_steps = 1000           # 평가 주기
    save_steps = 2000           # 모델 저장 주기
    
    def __init__(self, **kwargs):
        """키워드 인자로 설정 값 업데이트"""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                print(f"Warning: Unknown config parameter '{key}'")
    
    def update_vocab_sizes(self, vocab_size, num_users, num_news, num_topics):
        """데이터 로딩 후 어휘 크기 업데이트"""
        self.vocab_size = vocab_size
        self.num_users = num_users
        self.num_news = num_news
        self.num_topics = num_topics
    
    def display(self):
        """설정 출력"""
        print("=" * 50)
        print("GERL Model Configuration")
        print("=" * 50)
        
        print("\n[데이터 설정]")
        print(f"  최대 제목 길이: {self.max_title_len}")
        print(f"  최대 클릭 뉴스 수: {self.max_clicked_news}")
        print(f"  최대 이웃 수: {self.max_neighbors}")
        
        print("\n[임베딩 차원]")
        print(f"  단어 임베딩: {self.word_embed_dim}")
        print(f"  사용자 임베딩: {self.user_embed_dim}")
        print(f"  뉴스 임베딩: {self.news_embed_dim}")
        print(f"  토픽 임베딩: {self.topic_embed_dim}")
        
        print("\n[모델 구조]")
        print(f"  Attention Head 수: {self.num_heads}")
        print(f"  Attention 은닉 차원: {self.attention_hidden_dim}")
        print(f"  Dropout: {self.dropout}")
        
        print("\n[훈련 설정]")
        print(f"  배치 크기: {self.batch_size}")
        print(f"  학습률: {self.learning_rate}")
        print(f"  에포크 수: {self.num_epochs}")
        
        print("\n[데이터셋 크기]")
        print(f"  어휘 크기: {self.vocab_size}")
        print(f"  사용자 수: {self.num_users}")
        print(f"  뉴스 수: {self.num_news}")
        print(f"  토픽 수: {self.num_topics}")
        
        print("=" * 50) 