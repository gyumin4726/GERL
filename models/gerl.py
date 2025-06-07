import torch
import torch.nn as nn
import torch.nn.functional as F
from .news_transformer import NewsTransformer
from .graph_attention import GraphAttentionLayer

class GERL(nn.Module):
    def __init__(self, config):
        super(GERL, self).__init__()
        
        # News Encoder
        self.news_encoder = NewsTransformer(
            vocab_size=config.vocab_size,
            d_model=config.d_model,
            nhead=config.num_heads,
            dropout=config.dropout
        )
        
        # User Embedding
        self.user_embedding = nn.Embedding(1000000, config.id_dim)  # 큰 값으로 설정
        
        # Graph Attention Layers
        self.user_gat = GraphAttentionLayer(config.id_dim, config.attention_dim)
        self.news_gat = GraphAttentionLayer(config.d_model, config.attention_dim)
        
        # Final Projection Layers
        self.user_projection = nn.Linear(config.id_dim + config.attention_dim, config.final_dim)
        self.news_projection = nn.Linear(config.d_model + config.attention_dim, config.final_dim)
        
        self.final_layer = nn.Linear(config.final_dim, 1)
        self.dropout = nn.Dropout(config.dropout)
    
    def encode_news(self, news_title, news_neighbors=None, news_neighbor_adj=None):
        # 뉴스 인코딩
        news_repr = self.news_encoder(news_title)  # [batch_size, d_model]
        
        # 그래프 어텐션 적용 (이웃 노드가 있는 경우)
        if news_neighbors is not None and news_neighbor_adj is not None:
            neighbor_repr = self.news_encoder(news_neighbors)  # [batch_size, num_neighbors, d_model]
            news_graph_repr = self.news_gat(news_repr.unsqueeze(1), neighbor_repr, news_neighbor_adj)
            news_repr = torch.cat([news_repr, news_graph_repr.squeeze(1)], dim=-1)
        else:
            # 이웃 노드가 없는 경우 zero padding
            news_repr = torch.cat([news_repr, torch.zeros_like(news_repr)], dim=-1)
        
        news_repr = self.news_projection(news_repr)
        return news_repr
    
    def encode_user(self, user_id, clicked_news_titles, user_neighbors=None, user_neighbor_adj=None):
        # 사용자 임베딩
        user_repr = self.user_embedding(user_id)  # [batch_size, id_dim]
        
        # 클릭한 뉴스 인코딩
        batch_size, history_len, title_len = clicked_news_titles.shape
        clicked_news = clicked_news_titles.view(-1, title_len)  # [batch_size * history_len, title_len]
        clicked_news_repr = self.news_encoder(clicked_news)  # [batch_size * history_len, d_model]
        clicked_news_repr = clicked_news_repr.view(batch_size, history_len, -1)  # [batch_size, history_len, d_model]
        
        # 클릭 히스토리의 평균으로 사용자 표현 보강
        history_repr = torch.mean(clicked_news_repr, dim=1)  # [batch_size, d_model]
        user_repr = user_repr + history_repr  # Simple addition for now
        
        # 그래프 어텐션 적용 (이웃 노드가 있는 경우)
        if user_neighbors is not None and user_neighbor_adj is not None:
            neighbor_repr = self.user_embedding(user_neighbors)  # [batch_size, num_neighbors, id_dim]
            user_graph_repr = self.user_gat(user_repr.unsqueeze(1), neighbor_repr, user_neighbor_adj)
            user_repr = torch.cat([user_repr, user_graph_repr.squeeze(1)], dim=-1)
        else:
            # 이웃 노드가 없는 경우 zero padding
            user_repr = torch.cat([user_repr, torch.zeros_like(user_repr)], dim=-1)
        
        user_repr = self.user_projection(user_repr)
        return user_repr
    
    def forward(self, batch):
        # 뉴스 인코딩
        news_repr = self.encode_news(
            batch['news_title'],
            batch.get('news_neighbors', None),
            batch.get('news_neighbor_adj', None)
        )
        
        # 사용자 인코딩
        user_repr = self.encode_user(
            batch['user_id'],
            batch['clicked_news_titles'],
            batch.get('user_neighbors', None),
            batch.get('user_neighbor_adj', None)
        )
        
        # 최종 예측
        logits = self.final_layer(self.dropout(news_repr * user_repr)).squeeze(-1)
        return logits 