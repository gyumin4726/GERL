import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class NewsTransformer(nn.Module):
    def __init__(self, config):
        """Initialize news transformer
        
        Args:
            config: Model configuration containing:
                - hidden_size: Word embedding dimension
                - num_attention_heads: Number of attention heads
                - head_dim: Dimension of each attention head
                - attention_dropout: Dropout rate for attention
                - hidden_dropout: Dropout rate for hidden states
                - topic_dim: Topic embedding dimension
        """
        super().__init__()
        
        # Word embedding
        self.word_embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        
        # Topic embedding
        self.topic_embedding = nn.Embedding(config.num_topics, config.topic_dim)
        
        # Multi-head self-attention
        self.num_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        self.hidden_size = config.hidden_size
        
        self.query = nn.Linear(config.hidden_size, config.hidden_size)
        self.key = nn.Linear(config.hidden_size, config.hidden_size)
        self.value = nn.Linear(config.hidden_size, config.hidden_size)
        
        self.attention_dropout = nn.Dropout(config.attention_dropout)
        self.hidden_dropout = nn.Dropout(config.hidden_dropout)
        
        # Word-level attention
        self.word_attention = nn.Sequential(
            nn.Linear(config.hidden_size, config.attention_dim),
            nn.Tanh(),
            nn.Linear(config.attention_dim, 1)
        )
        
    def transpose_for_scores(self, x):
        """Transpose and reshape tensor for attention computation"""
        batch_size = x.size(0)
        seq_length = x.size(1)
        x = x.view(batch_size, seq_length, self.num_heads, self.head_dim)
        return x.permute(0, 2, 1, 3)
        
    def forward(self, input_ids, topic_ids=None, attention_mask=None):
        """Forward pass
        
        Args:
            input_ids: Input token IDs [batch_size, seq_length]
            topic_ids: Topic IDs [batch_size]
            attention_mask: Attention mask [batch_size, seq_length]
            
        Returns:
            news_vector: News representation vector [batch_size, hidden_size + topic_dim]
        """
        # Word embeddings
        word_embeds = self.word_embedding(input_ids)  # [batch_size, seq_length, hidden_size]
        word_embeds = self.hidden_dropout(word_embeds)
        
        # Multi-head self-attention
        # Compute query, key, value projections
        query = self.transpose_for_scores(self.query(word_embeds))
        key = self.transpose_for_scores(self.key(word_embeds))
        value = self.transpose_for_scores(self.value(word_embeds))
        
        # Compute attention scores
        attention_scores = torch.matmul(query, key.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.head_dim)
        
        if attention_mask is not None:
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            attention_scores = attention_scores.masked_fill(attention_mask == 0, -1e9)
        
        # Normalize attention scores
        attention_probs = F.softmax(attention_scores, dim=-1)
        attention_probs = self.attention_dropout(attention_probs)
        
        # Apply attention to values
        context = torch.matmul(attention_probs, value)
        context = context.permute(0, 2, 1, 3).contiguous()
        context = context.view(context.size(0), -1, self.hidden_size)
        
        # Word-level attention for news representation
        word_attention = self.word_attention(context)  # [batch_size, seq_length, 1]
        word_attention = F.softmax(word_attention, dim=1)
        news_vector = torch.sum(word_attention * context, dim=1)  # [batch_size, hidden_size]
        
        # Add topic embeddings if provided
        if topic_ids is not None:
            topic_embeds = self.topic_embedding(topic_ids)  # [batch_size, topic_dim]
            news_vector = torch.cat([news_vector, topic_embeds], dim=-1)
        
        return news_vector 