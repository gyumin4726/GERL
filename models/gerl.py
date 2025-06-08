"""
Graph Enhanced Representation Learning for News Recommendation (GERL)

이 모듈은 논문의 핵심 모델인 GERL을 구현합니다.
논문의 Section 3 "OUR APPROACH"에서 제안된 아키텍처를 구현하며,
One-hop interaction learning과 Two-hop graph learning 모듈로 구성됩니다.

주요 컴포넌트:
1. MultiHeadAttention: Transformer의 핵심 컴포넌트 (Section 3.1)
2. NewsEncoder: 뉴스 콘텐츠 인코딩 (Section 3.1)
3. GraphAttentionLayer: 그래프 기반 표현 학습 (Section 3.3)
4. GERL: 전체 모델 통합 (Section 3)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from .news_transformer import NewsTransformer
from .graph_attention import TwoHopGraphLearning

class MultiHeadAttention(nn.Module):
    """
    Section 3.1 "Transformer for Context Understanding"에서 설명된
    Multi-head self-attention 구현
    
    뉴스 제목의 단어들 간의 관계를 학습하기 위해 사용됩니다.
    예를 들어 "Sparks gives Penny Toler a fire from the organization"에서
    'Sparks'와 'organization' 사이의 관계를 포착할 수 있습니다.
    """
    def __init__(self, config):
        super().__init__()
        self.num_heads = config.num_attention_heads  # 논문 Section 4.5에서 제안된 8
        self.head_dim = config.head_dim
        self.hidden_size = config.hidden_size
        
        # Query, Key, Value 변환을 위한 선형 레이어
        self.query = nn.Linear(config.hidden_size, config.hidden_size)
        self.key = nn.Linear(config.hidden_size, config.hidden_size)
        self.value = nn.Linear(config.hidden_size, config.hidden_size)
        
        self.dropout = nn.Dropout(config.attention_dropout)
        
    def transpose_for_scores(self, x):
        """Multi-head attention을 위한 텐서 변환"""
        batch_size = x.size(0)
        new_shape = (batch_size, -1, self.num_heads, self.head_dim)
        x = x.view(*new_shape)
        return x.permute(0, 2, 1, 3)
        
    def forward(self, hidden_states, attention_mask=None):
        """
        Section 3.1에서 설명된 self-attention 계산
        Q(K^T)/sqrt(d_k)를 계산하고 softmax를 적용하여 attention 가중치 도출
        """
        query = self.transpose_for_scores(self.query(hidden_states))
        key = self.transpose_for_scores(self.key(hidden_states))
        value = self.transpose_for_scores(self.value(hidden_states))
        
        # Attention scores 계산
        attention_scores = torch.matmul(query, key.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.head_dim)
        
        if attention_mask is not None:
            attention_scores = attention_scores.masked_fill(attention_mask == 0, -1e9)
        
        attention_probs = F.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)
        
        context = torch.matmul(attention_probs, value)
        context = context.permute(0, 2, 1, 3).contiguous()
        context = context.view(context.size(0), -1, self.hidden_size)
        
        return context

class NewsEncoder(nn.Module):
    """
    Section 3.1 "Transformer for Context Understanding"에서 설명된
    뉴스 인코더 구현
    
    뉴스 제목과 토픽 정보를 결합하여 뉴스의 의미적 표현을 학습합니다.
    """
    def __init__(self, config):
        super().__init__()
        # 단어와 토픽 임베딩 초기화
        self.word_embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        self.topic_embedding = nn.Embedding(config.num_topics, config.topic_dim)
        
        self.attention = MultiHeadAttention(config)
        # 뉴스 레벨 attention (Section 3.1의 마지막 부분)
        self.news_attention = nn.Sequential(
            nn.Linear(config.hidden_size, config.attention_dim),
            nn.Tanh(),
            nn.Linear(config.attention_dim, 1)
        )
        
        self.dropout = nn.Dropout(config.hidden_dropout)
        
    def forward(self, news_input):
        """
        Section 3.1에서 설명된 뉴스 인코딩 과정
        1. 단어 임베딩
        2. Self-attention
        3. 뉴스 레벨 attention
        4. 토픽 정보 통합
        """
        word_embed = self.word_embedding(news_input['title'])
        word_embed = self.dropout(word_embed)
        
        news_vector = self.attention(word_embed)
        
        attention_weights = self.news_attention(news_vector)
        attention_weights = F.softmax(attention_weights, dim=1)
        news_repr = torch.sum(attention_weights * news_vector, dim=1)
        
        if 'topic' in news_input:
            topic_embed = self.topic_embedding(news_input['topic'])
            news_repr = torch.cat([news_repr, topic_embed], dim=-1)
            
        return news_repr

class GraphAttentionLayer(nn.Module):
    """
    Section 3.3 "Two-hop Graph Learning"에서 설명된
    그래프 어텐션 레이어 구현
    
    이웃 노드(뉴스 또는 사용자)의 정보를 집계하는 데 사용됩니다.
    """
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.graph_hidden_size
        self.attention_heads = config.graph_attention_heads
        
        # 노드 특징 변환을 위한 가중치 행렬 W
        self.W = nn.Linear(config.hidden_size, config.graph_hidden_size * config.graph_attention_heads, bias=False)
        # 어텐션 계산을 위한 가중치 벡터 a
        self.a = nn.Parameter(torch.zeros(size=(2 * config.graph_hidden_size, 1)))
        
        self.leakyrelu = nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout(config.graph_dropout)
        
    def forward(self, node_features, adj):
        """
        Section 3.3에서 설명된 그래프 어텐션 계산
        이웃 노드의 중요도를 계산하고 정보를 집계합니다.
        """
        Wh = self.W(node_features)
        batch_size, N = node_features.size(0), node_features.size(1)
        
        Wh = Wh.view(batch_size, N, self.attention_heads, self.hidden_size)
        
        # 이웃 노드와의 어텐션 계산
        a_input = torch.cat([Wh.repeat(1, 1, N, 1), Wh.repeat(1, N, 1, 1)], dim=3)
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(3))
        
        # 마스킹된 어텐션
        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=2)
        attention = self.dropout(attention)
        
        h_prime = torch.matmul(attention, Wh)
        
        return h_prime

class GERL(nn.Module):
    """
    Section 3 "OUR APPROACH"에서 제안된 전체 모델 구현
    
    One-hop interaction learning과 Two-hop graph learning을
    통합하여 최종적인 사용자와 뉴스의 표현을 학습합니다.
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Section 3.1: 뉴스 인코더
        self.news_encoder = NewsTransformer(config)
        
        # Section 3.2: 사용자 임베딩
        self.user_embedding = nn.Embedding(config.num_users, config.id_dim)
        
        # Section 3.3: Two-hop graph learning
        self.graph_learning = TwoHopGraphLearning(config)
        
        # 최종 투영 레이어
        final_dim = config.hidden_size + config.topic_dim + config.graph_hidden_size
        self.user_projection = nn.Linear(final_dim, config.hidden_size)
        self.news_projection = nn.Linear(final_dim, config.hidden_size)
        
    def encode_news(self, news_input):
        """
        Section 3.1과 3.3에서 설명된 뉴스 인코딩
        텍스트 기반 의미 표현과 그래프 기반 표현을 결합
        """
        news_semantic = self.news_encoder(
            input_ids=news_input['title'],
            topic_ids=news_input.get('topic'),
            attention_mask=news_input.get('attention_mask')
        )
        
        if 'neighbors' in news_input and news_input['neighbors'] is not None:
            news_graph = self.graph_learning.news_gat(
                news_input['neighbors'],
                news_input.get('neighbor_adj')
            )
            news_repr = torch.cat([news_semantic, news_graph], dim=-1)
        else:
            news_repr = news_semantic
            
        return news_repr
        
    def encode_user(self, user_input):
        """
        Section 3.2와 3.3에서 설명된 사용자 인코딩
        클릭 기반 표현과 그래프 기반 표현을 결합
        """
        user_id_embed = self.user_embedding(user_input['user_id'])
        
        clicked_news_repr = []
        for news in user_input['clicked_news']:
            news_repr = self.encode_news(news)
            clicked_news_repr.append(news_repr)
        clicked_news_repr = torch.stack(clicked_news_repr, dim=1)
        
        if user_input.get('history_mask') is not None:
            attention_mask = user_input['history_mask'].unsqueeze(-1)
            clicked_news_repr = clicked_news_repr * attention_mask
            
        user_semantic = torch.mean(clicked_news_repr, dim=1)
        
        if 'neighbors' in user_input and user_input['neighbors'] is not None:
            user_graph = self.graph_learning.user_gat(
                user_input['neighbors'],
                user_input.get('neighbor_adj')
            )
            user_repr = torch.cat([user_id_embed, user_semantic, user_graph], dim=-1)
        else:
            user_repr = torch.cat([user_id_embed, user_semantic], dim=-1)
            
        return user_repr
        
    def forward(self, batch):
        """Forward pass
        
        Args:
            batch: Dictionary containing:
                - candidate_news: Candidate news features
                - clicked_news: User's clicked news features
                - neighbor_news: Neighbor news features (optional)
                - neighbor_users: Neighbor user IDs (optional)
                - news_graph_adj: News graph adjacency matrix (optional)
                - user_graph_adj: User graph adjacency matrix (optional)
                
        Returns:
            scores: Predicted click scores
        """
        # One-hop interaction learning
        candidate_news = self.encode_news(batch['candidate_news'])
        user_repr = self.encode_user({
            'user_id': batch['user_id'],
            'clicked_news': batch['clicked_news'],
            'history_mask': batch.get('history_mask'),
            'neighbors': batch.get('neighbor_users'),
            'neighbor_adj': batch.get('user_graph_adj')
        })
        
        # Two-hop graph learning
        if 'neighbor_news' in batch and batch['neighbor_news'] is not None:
            neighbor_news = self.encode_news(batch['neighbor_news'])
            news_graph_repr = self.graph_learning.news_gat(
                neighbor_news,
                batch['news_graph_adj']
            )
            candidate_news = torch.cat([candidate_news, news_graph_repr], dim=-1)
            
        if 'neighbor_users' in batch and batch['neighbor_users'] is not None:
            neighbor_users = self.user_embedding(batch['neighbor_users'])
            user_graph_repr = self.graph_learning.user_gat(
                neighbor_users,
                batch['user_graph_adj']
            )
            user_repr = torch.cat([user_repr, user_graph_repr], dim=-1)
        
        # Final projections
        user_repr = self.user_projection(user_repr)
        news_repr = self.news_projection(candidate_news)
        
        # Click prediction
        scores = torch.matmul(user_repr, news_repr.transpose(-2, -1))
        
        return scores 