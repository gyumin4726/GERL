import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np


def load_glove_embeddings(glove_path, vocab, embed_dim):
    """GloVe 임베딩 로드 및 어휘에 맞게 변환"""
    print(f"Loading GloVe embeddings from {glove_path}...")
    
    # GloVe 임베딩 딕셔너리 로드
    glove_dict = {}
    
    try:
        with open(glove_path, 'r', encoding='utf-8') as f:
            for line in f:
                values = line.strip().split()
                word = values[0]
                vector = np.array(values[1:], dtype=np.float32)
                if len(vector) == embed_dim:  # 차원이 맞는 경우만
                    glove_dict[word] = vector
    except FileNotFoundError:
        print(f"Warning: GloVe file not found at {glove_path}. Using random initialization.")
        return None
    
    print(f"Loaded {len(glove_dict)} GloVe vectors")
    
    # 어휘에 맞는 임베딩 행렬 생성
    vocab_size = len(vocab)
    embedding_matrix = np.random.normal(0, 0.1, (vocab_size, embed_dim))
    
    # 어휘의 각 단어에 대해 GloVe 벡터가 있으면 사용
    found_words = 0
    for word, idx in vocab.items():
        if word in glove_dict:
            embedding_matrix[idx] = glove_dict[word]
            found_words += 1
    
    print(f"Found GloVe vectors for {found_words}/{vocab_size} words ({found_words/vocab_size*100:.1f}%)")
    
    return torch.FloatTensor(embedding_matrix)


class MultiHeadSelfAttention(nn.Module):
    """논문 Eq.(1)의 단순한 Multi-head self-attention"""
    
    def __init__(self, embed_dim, num_heads, dropout=0.2):
        super(MultiHeadSelfAttention, self).__init__()
        assert embed_dim % num_heads == 0
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        # 논문 Eq.(1): α_i,j^k = exp(e_i^T W_s^k e_j) / Σ_m exp(e_i^T W_s^k e_m)
        # 각 헤드별로 하나의 변환 행렬만 사용
        self.W_s = nn.ModuleList([
            nn.Linear(embed_dim, embed_dim, bias=False) for _ in range(num_heads)
        ])
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        """
        Args:
            x: (batch_size, seq_len, embed_dim) - 단어 임베딩 시퀀스
        Returns:
            output: (batch_size, seq_len, embed_dim) - attention 출력
        """
        batch_size, seq_len, embed_dim = x.size()
        head_outputs = []
        
        for head_idx in range(self.num_heads):
            # 논문 Eq.(1) 구현
            # e_i^T W_s^k e_j 계산
            transformed = self.W_s[head_idx](x)  # (batch_size, seq_len, embed_dim)
            
            # attention scores 계산
            scores = torch.matmul(transformed, x.transpose(-2, -1))  # (batch_size, seq_len, seq_len)
            attention_weights = F.softmax(scores, dim=-1)  # (batch_size, seq_len, seq_len)
            attention_weights = self.dropout(attention_weights)
            
            # h_i^k = Σ_j α_i,j^k e_j
            head_output = torch.matmul(attention_weights, x)  # (batch_size, seq_len, embed_dim)
            
            # 논문에 따라 head_dim으로 축소
            head_output = head_output[:, :, :self.head_dim]  # (batch_size, seq_len, head_dim)
            head_outputs.append(head_output)
        
        # 논문 Eq.(1): h_i = [h_i^1; h_i^2; ...; h_i^N] (concatenation)
        output = torch.cat(head_outputs, dim=-1)  # (batch_size, seq_len, embed_dim)
        
        return output



class NewsTransformer(nn.Module):
    """뉴스 텍스트를 위한 Transformer 모듈"""
    
    def __init__(self, vocab_size, embed_dim, num_heads, num_topics, 
                 topic_embed_dim, dropout=0.2, attention_hidden_dim=128,
                 vocab=None, glove_path=None):
        super(NewsTransformer, self).__init__()
        
        # 단어 임베딩 레이어
        self.word_embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        
        # GloVe 임베딩으로 초기화 (논문에서 언급)
        if vocab is not None and glove_path is not None:
            glove_embeddings = load_glove_embeddings(glove_path, vocab, embed_dim)
            if glove_embeddings is not None:
                self.word_embedding.weight.data.copy_(glove_embeddings)
                print("Word embeddings initialized with GloVe")
        
        # 토픽 임베딩 레이어
        self.topic_embedding = nn.Embedding(num_topics, topic_embed_dim)
        
        # Multi-head self-attention
        self.self_attention = MultiHeadSelfAttention(embed_dim, num_heads, dropout)
        
        # 논문 Eq.(2)의 word-level attentive attention (Transformer 내부)
        self.U_w = nn.Linear(embed_dim, attention_hidden_dim)
        self.u_w = nn.Parameter(torch.randn(attention_hidden_dim))
        self.q_w = nn.Parameter(torch.randn(attention_hidden_dim))
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, title_tokens, topic_id):
        """
        Args:
            title_tokens: (batch_size, max_title_len) - 뉴스 제목 토큰
            topic_id: (batch_size,) - 뉴스 토픽 ID
        Returns:
            news_repr: (batch_size, embed_dim + topic_embed_dim) - 뉴스 표현
        """
        # 단어 임베딩
        word_embeds = self.word_embedding(title_tokens)  # (batch_size, seq_len, embed_dim)
        word_embeds = self.dropout(word_embeds)
        
        # Multi-head self-attention (논문 Section 3.1)
        attended_words = self.self_attention(word_embeds)  # (batch_size, seq_len, embed_dim)
        attended_words = self.dropout(attended_words)
        
        # 논문 Eq.(2): word-level attentive attention
        u = torch.tanh(self.U_w(attended_words) + self.u_w)  # (batch_size, seq_len, hidden_dim)
        scores = torch.matmul(u, self.q_w)  # (batch_size, seq_len)
        attention_weights = F.softmax(scores, dim=-1)  # (batch_size, seq_len)
        
        # v_t = Σ_i β_i^w h_i (논문에서 뉴스 제목 표현)
        title_repr = torch.sum(attended_words * attention_weights.unsqueeze(-1), dim=1)  # (batch_size, embed_dim)
        
        # 토픽 임베딩
        topic_repr = self.topic_embedding(topic_id)  # (batch_size, topic_embed_dim)
        
        # 제목과 토픽 표현 연결 (논문 Section 3.1 마지막)
        news_repr = torch.cat([title_repr, topic_repr], dim=-1)
        
        return news_repr 