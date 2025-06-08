import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from .news_transformer import NewsTransformer
from .graph_attention import TwoHopGraphLearning

class MultiHeadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        self.hidden_size = config.hidden_size
        
        self.query = nn.Linear(config.hidden_size, config.hidden_size)
        self.key = nn.Linear(config.hidden_size, config.hidden_size)
        self.value = nn.Linear(config.hidden_size, config.hidden_size)
        
        self.dropout = nn.Dropout(config.attention_dropout)
        
    def transpose_for_scores(self, x):
        batch_size = x.size(0)
        new_shape = (batch_size, -1, self.num_heads, self.head_dim)
        x = x.view(*new_shape)
        return x.permute(0, 2, 1, 3)
        
    def forward(self, hidden_states, attention_mask=None):
        query = self.transpose_for_scores(self.query(hidden_states))
        key = self.transpose_for_scores(self.key(hidden_states))
        value = self.transpose_for_scores(self.value(hidden_states))
        
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
    def __init__(self, config):
        super().__init__()
        self.word_embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        self.topic_embedding = nn.Embedding(config.num_topics, config.topic_dim)
        
        self.attention = MultiHeadAttention(config)
        self.news_attention = nn.Sequential(
            nn.Linear(config.hidden_size, config.attention_dim),
            nn.Tanh(),
            nn.Linear(config.attention_dim, 1)
        )
        
        self.dropout = nn.Dropout(config.hidden_dropout)
        
    def forward(self, news_input):
        # Word embeddings
        word_embed = self.word_embedding(news_input['title'])
        word_embed = self.dropout(word_embed)
        
        # Self attention
        news_vector = self.attention(word_embed)
        
        # News attention
        attention_weights = self.news_attention(news_vector)
        attention_weights = F.softmax(attention_weights, dim=1)
        news_repr = torch.sum(attention_weights * news_vector, dim=1)
        
        # Add topic embeddings if available
        if 'topic' in news_input:
            topic_embed = self.topic_embedding(news_input['topic'])
            news_repr = torch.cat([news_repr, topic_embed], dim=-1)
            
        return news_repr

class GraphAttentionLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.graph_hidden_size
        self.attention_heads = config.graph_attention_heads
        
        self.W = nn.Linear(config.hidden_size, config.graph_hidden_size * config.graph_attention_heads, bias=False)
        self.a = nn.Parameter(torch.zeros(size=(2 * config.graph_hidden_size, 1)))
        
        self.leakyrelu = nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout(config.graph_dropout)
        
    def forward(self, node_features, adj):
        Wh = self.W(node_features)
        batch_size, N = node_features.size(0), node_features.size(1)
        
        Wh = Wh.view(batch_size, N, self.attention_heads, self.hidden_size)
        
        # Self-attention on the nodes
        a_input = torch.cat([Wh.repeat(1, 1, N, 1), Wh.repeat(1, N, 1, 1)], dim=3)
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(3))
        
        # Masked attention
        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=2)
        attention = self.dropout(attention)
        
        h_prime = torch.matmul(attention, Wh)
        
        return h_prime

class GERL(nn.Module):
    def __init__(self, config):
        """Initialize GERL model
        
        Args:
            config: Model configuration
        """
        super().__init__()
        self.config = config
        
        # News encoder
        self.news_encoder = NewsTransformer(config)
        
        # User embedding
        self.user_embedding = nn.Embedding(config.num_users, config.id_dim)
        
        # Two-hop graph learning
        self.graph_learning = TwoHopGraphLearning(config)
        
        # Final projections
        final_dim = config.hidden_size + config.topic_dim + config.graph_hidden_size
        self.user_projection = nn.Linear(final_dim, config.hidden_size)
        self.news_projection = nn.Linear(final_dim, config.hidden_size)
        
    def encode_news(self, news_input):
        """Encode news using transformer and graph attention
        
        Args:
            news_input: Dictionary containing:
                - title: News title token IDs
                - topic: News topic IDs
                - neighbors: Neighbor news features
                - neighbor_mask: Mask for neighbor padding
                
        Returns:
            news_repr: News representations
        """
        # Get news semantic representation from transformer
        news_semantic = self.news_encoder(
            input_ids=news_input['title'],
            topic_ids=news_input.get('topic'),
            attention_mask=news_input.get('attention_mask')
        )
        
        # Get graph-enhanced representation if neighbors exist
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
        """Encode user using click history and graph attention
        
        Args:
            user_input: Dictionary containing:
                - user_id: User IDs
                - clicked_news: Clicked news features
                - history_mask: Mask for history padding
                - neighbors: Neighbor user features
                - neighbor_mask: Mask for neighbor padding
                
        Returns:
            user_repr: User representations
        """
        # Get user ID embedding
        user_id_embed = self.user_embedding(user_input['user_id'])
        
        # Encode clicked news
        clicked_news_repr = []
        for news in user_input['clicked_news']:
            news_repr = self.encode_news(news)
            clicked_news_repr.append(news_repr)
        clicked_news_repr = torch.stack(clicked_news_repr, dim=1)
        
        # Apply attention over clicked news
        if user_input.get('history_mask') is not None:
            attention_mask = user_input['history_mask'].unsqueeze(-1)
            clicked_news_repr = clicked_news_repr * attention_mask
            
        user_semantic = torch.mean(clicked_news_repr, dim=1)
        
        # Get graph-enhanced representation if neighbors exist
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