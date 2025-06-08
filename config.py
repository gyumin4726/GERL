class Config:
    # Model Architecture
    hidden_size = 300  # Word embedding dimension
    topic_dim = 128    # Topic embedding dimension
    id_dim = 128      # ID embedding dimension
    attention_dim = 200
    
    # Transformer parameters
    num_attention_heads = 8  # From paper section 4.5
    head_dim = 16           # 128 / 8 = 16
    attention_dropout = 0.2
    hidden_dropout = 0.2
    
    # Graph parameters
    graph_hidden_size = 128
    graph_attention_heads = 8
    graph_dropout = 0.2
    max_neighbors = 15      # From paper section 4.5
    
    # Training parameters
    batch_size = 128        # From paper section 4.1
    negative_samples = 4    # λ in paper
    learning_rate = 0.001
    
    # Data parameters
    max_title_length = 30   # From paper section 4.1
    max_history_length = 50 # From paper section 4.1
    
    # Vocab parameters (to be set based on dataset)
    vocab_size = None
    num_users = None
    num_topics = None
    
    def __init__(self):
        pass
        
    @classmethod
    def from_dict(cls, json_object):
        config = cls()
        for key, value in json_object.items():
            setattr(config, key, value)
        return config 