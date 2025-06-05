import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphAttentionLayer(nn.Module):
    """그래프 어텐션 레이어"""
    
    def __init__(self, in_features, out_features, dropout=0.2):
        super(GraphAttentionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # 논문 Eq.(4)의 파라미터들
        self.W = nn.Linear(in_features, out_features, bias=False)
        self.a = nn.Parameter(torch.empty(size=(2 * out_features, 1)))
        
        self.dropout = nn.Dropout(dropout)
        self.leakyrelu = nn.LeakyReLU(0.2)
        
        self.reset_parameters()
        
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.W.weight)
        nn.init.xavier_uniform_(self.a)
        
    def forward(self, h, adj_mask=None):
        """
        Args:
            h: (batch_size, num_nodes, in_features) - 입력 노드 특징
            adj_mask: (batch_size, num_nodes) - 유효한 이웃 마스크
        Returns:
            output: (batch_size, out_features) - 집계된 특징
        """
        batch_size, num_nodes, _ = h.size()
        
        # 선형 변환
        Wh = self.W(h)  # (batch_size, num_nodes, out_features)
        
        # Attention 계산을 위한 준비
        a_input = self._prepare_attentional_mechanism_input(Wh)
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(-1))  # (batch_size, num_nodes)
        
        # 마스킹 적용 (패딩된 이웃들은 -inf로 설정)
        if adj_mask is not None:
            e.masked_fill_(~adj_mask, float('-inf'))
        
        # Softmax를 통한 attention weights 계산
        attention = F.softmax(e, dim=-1)  # (batch_size, num_nodes)
        attention = self.dropout(attention)
        
        # 가중 평균으로 노드 특징 집계
        h_prime = torch.sum(Wh * attention.unsqueeze(-1), dim=1)  # (batch_size, out_features)
        
        return h_prime
    
    def _prepare_attentional_mechanism_input(self, Wh):
        """Attention 메커니즘을 위한 입력 준비"""
        batch_size, num_nodes, out_features = Wh.size()
        
        # 첫 번째 노드(타겟)를 모든 이웃과 연결
        Wh_repeated_in_chunks = Wh[:, 0:1, :].repeat(1, num_nodes, 1)  # 타겟 노드 반복
        Wh_repeated_alternating = Wh  # 이웃 노드들
        
        # 연결하여 attention input 생성
        all_combinations_matrix = torch.cat(
            [Wh_repeated_in_chunks, Wh_repeated_alternating], dim=-1
        )  # (batch_size, num_nodes, 2 * out_features)
        
        return all_combinations_matrix


class NeighborAggregator(nn.Module):
    """이웃 정보 집계기"""
    
    def __init__(self, embed_dim, hidden_dim, dropout=0.2):
        super(NeighborAggregator, self).__init__()
        
        # 논문 Eq.(4)의 어텐션 네트워크 파라미터들
        self.U = nn.Linear(embed_dim, hidden_dim)
        self.u = nn.Parameter(torch.randn(hidden_dim))
        self.q = nn.Parameter(torch.randn(hidden_dim))
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, neighbor_embeds, neighbor_mask=None):
        """
        Args:
            neighbor_embeds: (batch_size, max_neighbors, embed_dim) - 이웃 임베딩
            neighbor_mask: (batch_size, max_neighbors) - 유효한 이웃 마스크
        Returns:
            output: (batch_size, embed_dim) - 집계된 이웃 표현
        """
        # 논문 Eq.(4) 구현
        u = torch.tanh(self.U(neighbor_embeds) + self.u)  # (batch_size, max_neighbors, hidden_dim)
        scores = torch.matmul(u, self.q)  # (batch_size, max_neighbors)
        
        # 마스킹 적용
        if neighbor_mask is not None:
            scores.masked_fill_(~neighbor_mask, float('-inf'))
        
        # Attention weights 계산
        attention_weights = F.softmax(scores, dim=-1)  # (batch_size, max_neighbors)
        attention_weights = self.dropout(attention_weights)
        
        # 가중 평균으로 이웃 정보 집계
        output = torch.sum(neighbor_embeds * attention_weights.unsqueeze(-1), dim=1)
        
        return output


class GraphAttentionNetwork(nn.Module):
    """Two-hop Graph Learning을 위한 Graph Attention Network"""
    
    def __init__(self, user_embed_dim, news_embed_dim, hidden_dim=128, dropout=0.2):
        super(GraphAttentionNetwork, self).__init__()
        
        # 이웃 사용자 ID 집계기 (논문 Section 3.3.1)
        self.neighbor_user_aggregator = NeighborAggregator(
            user_embed_dim, hidden_dim, dropout
        )
        
        # 이웃 뉴스 ID 집계기 (논문 Section 3.3.2)
        self.neighbor_news_id_aggregator = NeighborAggregator(
            news_embed_dim, hidden_dim, dropout
        )
        
        # 이웃 뉴스 의미 집계기 (논문 Section 3.3.3)
        # 뉴스 의미 임베딩은 transformer output 차원을 사용 (384)
        self.neighbor_news_semantic_aggregator = NeighborAggregator(
            384, hidden_dim, dropout  # news_semantic_dim = 384
        )
        
        self.user_embed_dim = user_embed_dim
        self.news_embed_dim = news_embed_dim
        
    def forward(self, neighbor_users, neighbor_news_ids, neighbor_news_semantic,
                neighbor_user_mask=None, neighbor_news_mask=None):
        """
        Args:
            neighbor_users: (batch_size, max_neighbors, user_embed_dim) - 이웃 사용자 임베딩
            neighbor_news_ids: (batch_size, max_neighbors, news_embed_dim) - 이웃 뉴스 ID 임베딩
            neighbor_news_semantic: (batch_size, max_neighbors, news_embed_dim) - 이웃 뉴스 의미 임베딩
            neighbor_user_mask: (batch_size, max_neighbors) - 유효한 이웃 사용자 마스크
            neighbor_news_mask: (batch_size, max_neighbors) - 유효한 이웃 뉴스 마스크
        Returns:
            aggregated_users: (batch_size, user_embed_dim) - 집계된 이웃 사용자 표현
            aggregated_news_ids: (batch_size, news_embed_dim) - 집계된 이웃 뉴스 ID 표현
            aggregated_news_semantic: (batch_size, news_embed_dim) - 집계된 이웃 뉴스 의미 표현
        """
        
        # 이웃 사용자 ID 표현 집계 (논문 Section 3.3.1)
        aggregated_users = self.neighbor_user_aggregator(
            neighbor_users, neighbor_user_mask
        )
        
        # 이웃 뉴스 ID 표현 집계 (논문 Section 3.3.2)
        aggregated_news_ids = self.neighbor_news_id_aggregator(
            neighbor_news_ids, neighbor_news_mask
        )
        
        # 이웃 뉴스 의미 표현 집계 (논문 Section 3.3.3)
        aggregated_news_semantic = self.neighbor_news_semantic_aggregator(
            neighbor_news_semantic, neighbor_news_mask
        )
        
        return aggregated_users, aggregated_news_ids, aggregated_news_semantic 