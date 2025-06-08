"""
Section 3.3 "Two-hop Graph Learning"에서 제안된
그래프 어텐션 네트워크 구현

이 모듈은 사용자-뉴스 상호작용 그래프에서
이웃 노드의 정보를 집계하여 표현 학습을 향상시킵니다.

주요 특징:
1. 이웃 노드 정보의 어텐티브 집계
2. 사용자와 뉴스 각각에 대한 그래프 어텐션
3. 이웃 중요도 학습
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class GraphAttentionLayer(nn.Module):
    def __init__(self, config):
        """
        Section 3.3에서 설명된 그래프 어텐션 레이어 초기화
        
        Args:
            config: 모델 설정으로 다음을 포함:
                - hidden_size: 노드 특징 차원
                - graph_hidden_size: 그래프 어텐션 은닉 차원
                - graph_attention_heads: 어텐션 헤드 수
                - graph_dropout: Dropout 비율
        """
        super().__init__()
        self.hidden_size = config.hidden_size
        self.graph_hidden_size = config.graph_hidden_size
        self.num_heads = config.graph_attention_heads
        self.head_dim = config.graph_hidden_size // config.graph_attention_heads
        self.dropout = config.graph_dropout
        
        # 노드 특징 변환을 위한 가중치 행렬 W (Section 3.3 수식)
        self.W = nn.Linear(config.hidden_size, config.graph_hidden_size * config.graph_attention_heads, bias=False)
        
        # 어텐션 계산을 위한 가중치 벡터 a
        self.a = nn.Parameter(torch.zeros(size=(2 * self.head_dim, 1)))
        nn.init.xavier_uniform_(self.a.data)
        
        self.leakyrelu = nn.LeakyReLU(0.2)
        self.dropout_layer = nn.Dropout(self.dropout)
        
    def forward(self, node_features, adj_matrix=None):
        """
        Section 3.3에서 설명된 그래프 어텐션 계산
        
        1. 노드 특징을 선형 변환
        2. 이웃 노드 간 어텐션 점수 계산
        3. 어텐션 가중치로 이웃 정보 집계
        
        Args:
            node_features: 입력 노드 특징 [batch_size, num_nodes, hidden_size]
            adj_matrix: 인접 행렬 [batch_size, num_nodes, num_nodes]
        """
        batch_size, num_nodes = node_features.size(0), node_features.size(1)
        
        # 선형 변환 및 multi-head attention을 위한 reshape
        Wh = self.W(node_features)
        Wh = Wh.view(batch_size, num_nodes, self.num_heads, self.head_dim)
        
        # 이웃 노드와의 어텐션 계산 준비
        Wh1 = Wh.unsqueeze(3)
        Wh2 = Wh.unsqueeze(2)
        
        # 어텐션 점수 계산 (Section 3.3 수식)
        attention = self.leakyrelu(torch.matmul(
            torch.cat([Wh1.expand(-1, -1, -1, num_nodes, -1), 
                      Wh2.expand(-1, -1, num_nodes, -1, -1)], dim=-1),
            self.a.unsqueeze(0).unsqueeze(0).unsqueeze(0)
        )).squeeze(-1)
        
        # 인접 행렬로 어텐션 마스킹
        if adj_matrix is not None:
            adj_matrix = adj_matrix.unsqueeze(3).expand(-1, -1, -1, self.num_heads)
            attention = attention.masked_fill(adj_matrix == 0, float('-inf'))
        
        # 어텐션 가중치 정규화
        attention = F.softmax(attention, dim=2)
        attention = self.dropout_layer(attention)
        
        # 이웃 노드 정보 집계
        out = torch.matmul(attention.transpose(2, 3), Wh)
        out = out.contiguous().view(batch_size, num_nodes, -1)
        
        return out

class TwoHopGraphLearning(nn.Module):
    def __init__(self, config):
        """
        Section 3.3의 Two-hop graph learning 모듈 초기화
        
        사용자와 뉴스 각각에 대해 별도의 그래프 어텐션을 적용하여
        이웃 정보를 학습합니다.
        """
        super().__init__()
        
        # 사용자 그래프 어텐션
        self.user_gat = GraphAttentionLayer(config)
        
        # 뉴스 그래프 어텐션
        self.news_gat = GraphAttentionLayer(config)
        
        # 이웃 정보 집계를 위한 어텐션
        self.user_attention = nn.Sequential(
            nn.Linear(config.graph_hidden_size, config.attention_dim),
            nn.Tanh(),
            nn.Linear(config.attention_dim, 1)
        )
        
        self.news_attention = nn.Sequential(
            nn.Linear(config.graph_hidden_size, config.attention_dim),
            nn.Tanh(),
            nn.Linear(config.attention_dim, 1)
        )
        
    def forward(self, user_features, news_features, user_adj=None, news_adj=None):
        """
        Section 3.3의 이웃 정보 통합 과정
        
        1. 사용자 이웃 정보 학습
        2. 뉴스 이웃 정보 학습
        3. 어텐션 기반 정보 집계
        
        Args:
            user_features: 사용자 노드 특징 [batch_size, num_users, hidden_size]
            news_features: 뉴스 노드 특징 [batch_size, num_news, hidden_size]
            user_adj: 사용자 인접 행렬 [batch_size, num_users, num_users]
            news_adj: 뉴스 인접 행렬 [batch_size, num_news, num_news]
        """
        # 사용자 이웃 정보 학습
        user_neighbor_repr = self.user_gat(user_features, user_adj)
        user_attention_weights = F.softmax(self.user_attention(user_neighbor_repr), dim=1)
        user_repr = torch.sum(user_attention_weights * user_neighbor_repr, dim=1)
        
        # 뉴스 이웃 정보 학습
        news_neighbor_repr = self.news_gat(news_features, news_adj)
        news_attention_weights = F.softmax(self.news_attention(news_neighbor_repr), dim=1)
        news_repr = torch.sum(news_attention_weights * news_neighbor_repr, dim=1)
        
        return user_repr, news_repr 