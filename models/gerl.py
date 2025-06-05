import torch
import torch.nn as nn
import torch.nn.functional as F

from .transformer import NewsTransformer
from .graph_attention import GraphAttentionNetwork


class GERL(nn.Module):
    """Graph Enhanced Representation Learning for News Recommendation"""
    
    def __init__(self, config):
        super(GERL, self).__init__()
        
        self.config = config
        
        # 임베딩 레이어들
        self.user_embedding = nn.Embedding(
            config.num_users, config.user_embed_dim, padding_idx=0
        )
        self.news_embedding = nn.Embedding(
            config.num_news, config.news_embed_dim, padding_idx=0
        )
        
        # Transformer for Context Understanding (논문 Section 3.1)
        self.news_transformer = NewsTransformer(
            vocab_size=config.vocab_size,
            embed_dim=config.word_embed_dim,
            num_heads=config.num_heads,
            num_topics=config.num_topics,
            topic_embed_dim=config.topic_embed_dim,
            dropout=config.dropout
        )
        
        # 뉴스 의미 표현 차원 계산
        self.news_semantic_dim = config.word_embed_dim + config.topic_embed_dim
        
        # Graph Attention Network for Two-hop Learning (논문 Section 3.3)
        self.graph_attention = GraphAttentionNetwork(
            user_embed_dim=config.user_embed_dim,
            news_embed_dim=config.news_embed_dim,
            hidden_dim=config.attention_hidden_dim,
            dropout=config.dropout
        )
        
        # One-hop Interaction Learning을 위한 어텐션 네트워크들
        # 사용자의 클릭 뉴스 집계를 위한 어텐션 (논문 Eq. 3)
        self.clicked_news_attention = nn.Sequential(
            nn.Linear(self.news_semantic_dim, config.attention_hidden_dim),
            nn.Tanh(),
            nn.Linear(config.attention_hidden_dim, 1)
        )
        
        # 최종 사용자 표현 차원 계산
        self.final_user_dim = (
            self.news_semantic_dim +  # one-hop semantic
            config.user_embed_dim +   # one-hop ID
            config.user_embed_dim     # two-hop graph
        )
        
        # 최종 뉴스 표현 차원 계산
        self.final_news_dim = (
            self.news_semantic_dim +  # n_t^O: semantic representation
            config.news_embed_dim +   # n_e^O: news ID embedding
            config.news_embed_dim +   # n_e^G: two-hop graph news ID
            self.news_semantic_dim    # n_t^G: two-hop graph semantic
        )
        
        self.dropout = nn.Dropout(config.dropout)
        
        # 논문에는 없지만 차원 맞춤을 위해 최소한 필요
        # 더 작은 차원으로 맞춤
        self.final_dim = min(self.final_user_dim, self.final_news_dim)
        if self.final_user_dim != self.final_news_dim:
            self.user_projection = nn.Linear(self.final_user_dim, self.final_dim) if self.final_user_dim > self.final_dim else nn.Identity()
            self.news_projection = nn.Linear(self.final_news_dim, self.final_dim) if self.final_news_dim > self.final_dim else nn.Identity()
        else:
            self.user_projection = nn.Identity()
            self.news_projection = nn.Identity()
        
    def forward(self, batch):
        """
        Args:
            batch: 배치 데이터 딕셔너리
        Returns:
            user_repr: (batch_size, final_embed_dim) - 최종 사용자 표현
            news_repr: (batch_size, final_embed_dim) - 최종 뉴스 표현
        """
        
        # One-hop Interaction Learning (논문 Section 3.2)
        user_one_hop = self._one_hop_user_learning(batch)
        news_one_hop = self._one_hop_news_learning(batch)
        
        # Two-hop Graph Learning (논문 Section 3.3)
        user_two_hop = self._two_hop_user_learning(batch)
        news_two_hop = self._two_hop_news_learning(batch)
        
        # 표현들을 연결하여 최종 표현 생성 (논문 Section 3.4)
        user_repr = torch.cat([
            user_one_hop['semantic'],  # u_t^O
            user_one_hop['id'],        # u_e^O  
            user_two_hop               # u_e^G
        ], dim=-1)
        
        news_repr = torch.cat([
            news_one_hop['semantic'],  # n_t^O
            news_one_hop['id'],        # n_e^O
            news_two_hop['id'],        # n_e^G (논문과 일치하도록 추가)
            news_two_hop['semantic']   # n_t^G
        ], dim=-1)
        
        # 차원 맞춤 (논문에는 없지만 구현상 필요)
        user_repr = self.user_projection(user_repr)
        news_repr = self.news_projection(news_repr)
        
        return user_repr, news_repr
    
    def _one_hop_user_learning(self, batch):
        """One-hop 사용자 학습 (논문 Section 3.2.2, 3.2.3)"""
        
        # Target User Semantic Representations (논문 Section 3.2.2)
        clicked_news_tokens = batch['clicked_news_title']  # (batch_size, max_clicked, max_title_len)
        clicked_news_topics = batch['clicked_news_topic']  # (batch_size, max_clicked)
        clicked_mask = batch['clicked_mask']  # (batch_size, max_clicked)
        
        batch_size, max_clicked, max_title_len = clicked_news_tokens.shape
        
        # 클릭한 뉴스들의 의미 표현 계산
        clicked_news_tokens_flat = clicked_news_tokens.view(-1, max_title_len)
        clicked_news_topics_flat = clicked_news_topics.view(-1)
        
        # 마스크가 True인 뉴스들만 처리
        valid_indices = clicked_mask.view(-1)
        
        if valid_indices.sum() > 0:
            valid_tokens = clicked_news_tokens_flat[valid_indices]
            valid_topics = clicked_news_topics_flat[valid_indices]
            
            # Transformer로 뉴스 표현 계산
            valid_news_reprs = self.news_transformer(valid_tokens, valid_topics)
            
            # 원래 shape으로 복원
            clicked_news_reprs = torch.zeros(
                batch_size * max_clicked, self.news_semantic_dim,
                device=clicked_news_tokens.device
            )
            clicked_news_reprs[valid_indices] = valid_news_reprs
            clicked_news_reprs = clicked_news_reprs.view(
                batch_size, max_clicked, self.news_semantic_dim
            )
        else:
            clicked_news_reprs = torch.zeros(
                batch_size, max_clicked, self.news_semantic_dim,
                device=clicked_news_tokens.device
            )
        
        # Attentive aggregation (논문 Eq. 3)
        attention_logits = self.clicked_news_attention(clicked_news_reprs).squeeze(-1)  # (batch_size, max_clicked)
        attention_logits = attention_logits.masked_fill(~clicked_mask, float('-inf'))
        attention_weights = F.softmax(attention_logits, dim=-1)  # (batch_size, max_clicked)
        
        user_semantic = torch.sum(
            clicked_news_reprs * attention_weights.unsqueeze(-1), dim=1
        )  # (batch_size, news_semantic_dim)
        
        # Target User ID Representations (논문 Section 3.2.3)
        user_ids = batch['user_id']  # (batch_size,)
        user_id_repr = self.user_embedding(user_ids)  # (batch_size, user_embed_dim)
        
        return {
            'semantic': user_semantic,
            'id': user_id_repr
        }
    
    def _one_hop_news_learning(self, batch):
        """One-hop 뉴스 학습 (논문 Section 3.2.1)"""
        
        # Candidate News Semantic Representations
        candidate_news_tokens = batch['candidate_news_title']  # (batch_size, max_title_len)
        candidate_news_topics = batch['candidate_news_topic']  # (batch_size,)
        candidate_news_ids = batch['candidate_news_id']  # (batch_size,)
        
        # Transformer로 의미 표현 계산
        news_semantic = self.news_transformer(candidate_news_tokens, candidate_news_topics)
        
        # 뉴스 ID 임베딩
        news_id_repr = self.news_embedding(candidate_news_ids)
        
        return {
            'semantic': news_semantic,
            'id': news_id_repr
        }
    
    def _two_hop_user_learning(self, batch):
        """Two-hop 사용자 학습 (논문 Section 3.3.1)"""
        
        neighbor_user_ids = batch['neighbor_users']  # (batch_size, max_neighbors)
        neighbor_user_mask = batch['neighbor_user_mask']  # (batch_size, max_neighbors)
        
        # 이웃 사용자 임베딩
        neighbor_user_embeds = self.user_embedding(neighbor_user_ids)
        
        # 사용자 이웃 정보만 집계
        aggregated_users = self.graph_attention.neighbor_user_aggregator(
            neighbor_user_embeds, neighbor_user_mask
        )
        
        return aggregated_users
    
    def _two_hop_news_learning(self, batch):
        """Two-hop 뉴스 학습 (논문 Section 3.3.2, 3.3.3)"""
        
        neighbor_news_ids = batch['neighbor_news']  # (batch_size, max_neighbors)
        neighbor_news_mask = batch['neighbor_news_mask']  # (batch_size, max_neighbors)
        
        # 이웃 뉴스들의 의미 표현 계산
        neighbor_news_tokens = batch['neighbor_news_title']  # (batch_size, max_neighbors, max_title_len)
        neighbor_news_topics = batch['neighbor_news_topic']  # (batch_size, max_neighbors)
        
        batch_size, max_neighbors, max_title_len = neighbor_news_tokens.shape
        
        # Flatten and process
        neighbor_tokens_flat = neighbor_news_tokens.view(-1, max_title_len)
        neighbor_topics_flat = neighbor_news_topics.view(-1)
        valid_indices = neighbor_news_mask.view(-1)
        
        if valid_indices.sum() > 0:
            valid_tokens = neighbor_tokens_flat[valid_indices]
            valid_topics = neighbor_topics_flat[valid_indices]
            
            valid_news_semantic = self.news_transformer(valid_tokens, valid_topics)
            
            neighbor_news_semantic = torch.zeros(
                batch_size * max_neighbors, self.news_semantic_dim,
                device=neighbor_news_tokens.device
            )
            neighbor_news_semantic[valid_indices] = valid_news_semantic
            neighbor_news_semantic = neighbor_news_semantic.view(
                batch_size, max_neighbors, self.news_semantic_dim
            )
        else:
            neighbor_news_semantic = torch.zeros(
                batch_size, max_neighbors, self.news_semantic_dim,
                device=neighbor_news_tokens.device
            )
        
        # 이웃 뉴스 ID 임베딩
        neighbor_news_id_embeds = self.news_embedding(neighbor_news_ids)
        
        # 뉴스 이웃 정보들을 각각 집계
        aggregated_news_ids = self.graph_attention.neighbor_news_id_aggregator(
            neighbor_news_id_embeds, neighbor_news_mask
        )
        
        aggregated_news_semantic = self.graph_attention.neighbor_news_semantic_aggregator(
            neighbor_news_semantic, neighbor_news_mask
        )
        
        return {
            'id': aggregated_news_ids,
            'semantic': aggregated_news_semantic
        }
    
    def predict(self, user_repr, news_repr):
        """예측 점수 계산 (논문 Section 3.4)"""
        scores = torch.sum(user_repr * news_repr, dim=-1)  # (batch_size,)
        return scores
    
    def compute_loss(self, batch, negative_samples=None):
        """손실 함수 계산 (논문 Eq. 5)"""
        
        user_repr, news_repr = self.forward(batch)
        
        # 긍정 샘플 점수
        pos_scores = self.predict(user_repr, news_repr)  # (batch_size,)
        
        # 논문에 따른 negative sampling (λ=4)
        # 실제 구현에서는 배치 내 다른 뉴스를 negative로 사용
        if negative_samples is not None:
            # 명시적 negative samples가 제공된 경우
            neg_user_repr = user_repr.unsqueeze(1).expand(-1, negative_samples.size(1), -1)  # (batch_size, num_neg, dim)
            neg_scores = torch.sum(neg_user_repr * negative_samples, dim=-1)  # (batch_size, num_neg)
        else:
            # 배치 내 다른 뉴스들을 negative로 사용 (실용적 접근)
            neg_scores = torch.matmul(user_repr, news_repr.transpose(0, 1))  # (batch_size, batch_size)
            # 자기 자신 제외
            mask = torch.eye(neg_scores.size(0), device=neg_scores.device).bool()
            neg_scores = neg_scores.masked_fill(mask, float('-inf'))
        
        # 논문 Eq.(5) - 최대 우도 손실
        pos_exp = torch.exp(pos_scores)  # (batch_size,)
        neg_exp = torch.exp(neg_scores)  # (batch_size, num_neg)
        neg_sum = torch.sum(neg_exp, dim=-1)  # (batch_size,)
        
        # log-likelihood 최대화 = negative log-likelihood 최소화
        loss = -torch.log(pos_exp / (pos_exp + neg_sum + 1e-8))
        return loss.mean() 