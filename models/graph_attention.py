import torch
import torch.nn as nn
import torch.nn.functional as F

class GraphAttentionLayer(nn.Module):
    def __init__(self, config):
        """Initialize graph attention layer
        
        Args:
            config: Model configuration containing:
                - hidden_size: Hidden dimension size
                - graph_hidden_size: Graph attention hidden size
                - graph_attention_heads: Number of attention heads
                - graph_dropout: Dropout rate
        """
        super().__init__()
        self.hidden_size = config.hidden_size
        self.graph_hidden_size = config.graph_hidden_size
        self.num_heads = config.graph_attention_heads
        self.head_dim = config.graph_hidden_size // config.graph_attention_heads
        self.dropout = config.graph_dropout
        
        # Transformations for node features
        self.W = nn.Linear(config.hidden_size, config.graph_hidden_size * config.graph_attention_heads, bias=False)
        
        # Attention mechanism
        self.a = nn.Parameter(torch.zeros(size=(2 * self.head_dim, 1)))
        nn.init.xavier_uniform_(self.a.data)
        
        self.leakyrelu = nn.LeakyReLU(0.2)
        self.dropout_layer = nn.Dropout(self.dropout)
        
    def forward(self, node_features, adj_matrix=None):
        """Forward pass
        
        Args:
            node_features: Input node features [batch_size, num_nodes, hidden_size]
            adj_matrix: Adjacency matrix [batch_size, num_nodes, num_nodes]
            
        Returns:
            output: Updated node features [batch_size, num_nodes, graph_hidden_size]
        """
        batch_size, num_nodes = node_features.size(0), node_features.size(1)
        
        # Linear transformation and reshape for multi-head attention
        Wh = self.W(node_features)  # [batch_size, num_nodes, num_heads * head_dim]
        Wh = Wh.view(batch_size, num_nodes, self.num_heads, self.head_dim)
        
        # Self-attention on the nodes
        # Prepare for attention
        Wh1 = Wh.unsqueeze(3)  # [batch_size, num_nodes, num_heads, 1, head_dim]
        Wh2 = Wh.unsqueeze(2)  # [batch_size, num_nodes, 1, num_heads, head_dim]
        
        # Attention scores
        # [batch_size, num_nodes, num_nodes, num_heads]
        attention = self.leakyrelu(torch.matmul(
            torch.cat([Wh1.expand(-1, -1, -1, num_nodes, -1), 
                      Wh2.expand(-1, -1, num_nodes, -1, -1)], dim=-1),
            self.a.unsqueeze(0).unsqueeze(0).unsqueeze(0)
        )).squeeze(-1)
        
        # Mask attention scores using adjacency matrix if provided
        if adj_matrix is not None:
            adj_matrix = adj_matrix.unsqueeze(3).expand(-1, -1, -1, self.num_heads)
            attention = attention.masked_fill(adj_matrix == 0, float('-inf'))
        
        # Normalize attention scores
        attention = F.softmax(attention, dim=2)
        attention = self.dropout_layer(attention)
        
        # Apply attention to node features
        out = torch.matmul(attention.transpose(2, 3), Wh)  # [batch_size, num_nodes, num_heads, head_dim]
        out = out.contiguous().view(batch_size, num_nodes, -1)  # [batch_size, num_nodes, graph_hidden_size]
        
        return out

class TwoHopGraphLearning(nn.Module):
    def __init__(self, config):
        """Initialize two-hop graph learning module
        
        Args:
            config: Model configuration
        """
        super().__init__()
        
        # User graph attention
        self.user_gat = GraphAttentionLayer(config)
        
        # News graph attention
        self.news_gat = GraphAttentionLayer(config)
        
        # Attention for aggregating neighbor information
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
        """Forward pass
        
        Args:
            user_features: User node features [batch_size, num_users, hidden_size]
            news_features: News node features [batch_size, num_news, hidden_size]
            user_adj: User adjacency matrix [batch_size, num_users, num_users]
            news_adj: News adjacency matrix [batch_size, num_news, num_news]
            
        Returns:
            user_repr: Updated user representations
            news_repr: Updated news representations
        """
        # Two-hop graph learning for users
        user_neighbor_repr = self.user_gat(user_features, user_adj)
        user_attention_weights = F.softmax(self.user_attention(user_neighbor_repr), dim=1)
        user_repr = torch.sum(user_attention_weights * user_neighbor_repr, dim=1)
        
        # Two-hop graph learning for news
        news_neighbor_repr = self.news_gat(news_features, news_adj)
        news_attention_weights = F.softmax(self.news_attention(news_neighbor_repr), dim=1)
        news_repr = torch.sum(news_attention_weights * news_neighbor_repr, dim=1)
        
        return user_repr, news_repr 