import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from transformers import BertTokenizer
import json
from tqdm import tqdm
import os
import networkx as nx

class MINDDataset(Dataset):
    def __init__(self, data_dir, max_title_length=30, max_history_length=50, num_neighbors=15, tokenizer=None):
        self.data_dir = data_dir
        self.max_title_length = max_title_length
        self.max_history_length = max_history_length
        self.num_neighbors = num_neighbors
        
        # Initialize tokenizer
        self.tokenizer = tokenizer if tokenizer else BertTokenizer.from_pretrained('bert-base-uncased')
        
        # Load data
        print("Loading news data...")
        self.news_df = pd.read_csv(os.path.join(data_dir, 'news.tsv'), 
                                 sep='\t',
                                 names=['news_id', 'category', 'subcategory', 'title', 'abstract', 'url', 'title_entities', 'abstract_entities'])
        
        print("Loading behaviors data...")
        self.behaviors_df = pd.read_csv(os.path.join(data_dir, 'behaviors.tsv'),
                                      sep='\t',
                                      names=['impression_id', 'user_id', 'time', 'history', 'impressions'])
        
        # Create ID mappings
        print("Creating ID mappings...")
        self.news2idx = {nid: idx for idx, nid in enumerate(self.news_df['news_id'].unique(), start=1)}
        self.news2idx['[PAD]'] = 0
        self.user2idx = {uid: idx for idx, uid in enumerate(self.behaviors_df['user_id'].unique(), start=1)}
        self.user2idx['[PAD]'] = 0
        
        # Build interaction graphs
        print("Building interaction graphs...")
        self.build_graphs()
        
        print("Processing news content...")
        self.process_news_content()
        
        print("Dataset initialization completed")
        
    def build_graphs(self):
        """Build user-news interaction graphs"""
        self.user_news_graph = nx.Graph()
        self.news_news_graph = nx.Graph()
        self.user_user_graph = nx.Graph()
        
        # Build user-news edges
        for _, row in tqdm(self.behaviors_df.iterrows(), desc="Building graphs"):
            user_id = row['user_id']
            if pd.isna(row['history']):
                continue
                
            history = row['history'].split()
            
            # Add user-news edges
            for news_id in history:
                self.user_news_graph.add_edge(user_id, news_id)
                
            # Add news-news edges (co-occurrence in history)
            for i in range(len(history)):
                for j in range(i+1, len(history)):
                    self.news_news_graph.add_edge(history[i], history[j])
        
        # Build user-user edges based on common news
        for user1 in tqdm(self.user_news_graph.nodes(), desc="Building user-user edges"):
            if not isinstance(user1, str) or user1.startswith('N'):  # Skip news nodes
                continue
            for user2 in self.user_news_graph.nodes():
                if not isinstance(user2, str) or user2.startswith('N'):
                    continue
                if user1 >= user2:
                    continue
                    
                user1_news = set(self.user_news_graph.neighbors(user1))
                user2_news = set(self.user_news_graph.neighbors(user2))
                common_news = len(user1_news & user2_news)
                
                if common_news > 0:
                    similarity = common_news / len(user1_news | user2_news)  # Jaccard similarity
                    self.user_user_graph.add_edge(user1, user2, weight=similarity)
    
    def process_news_content(self):
        """Process and tokenize news content"""
        self.news_title_tokens = {}
        self.news_topic_ids = {}
        
        for _, news in tqdm(self.news_df.iterrows(), desc="Processing news"):
            news_id = news['news_id']
            
            # Tokenize title
            title_tokens = self.tokenizer.encode(
                news['title'],
                add_special_tokens=True,
                max_length=self.max_title_length,
                padding='max_length',
                truncation=True
            )
            self.news_title_tokens[news_id] = title_tokens
            
            # Convert category to topic ID
            # You might want to implement more sophisticated topic modeling here
            self.news_topic_ids[news_id] = hash(news['category']) % 100
    
    def get_neighbors(self, node_id, graph, k):
        """Get top-k neighbors from graph"""
        if node_id not in graph:
            return []
        
        neighbors = sorted(graph[node_id].items(), key=lambda x: x[1]['weight'] if 'weight' in x[1] else 1.0, reverse=True)
        neighbors = [n[0] for n in neighbors[:k]]
        
        if len(neighbors) < k:
            neighbors.extend(['[PAD]'] * (k - len(neighbors)))
        return neighbors
    
    def __len__(self):
        return len(self.behaviors_df)
    
    def __getitem__(self, idx):
        behavior = self.behaviors_df.iloc[idx]
        
        # Process history
        history = [] if pd.isna(behavior['history']) else behavior['history'].split()
        if len(history) > self.max_history_length:
            history = history[-self.max_history_length:]
        history_idx = [self.news2idx.get(h, 0) for h in history]
        history_mask = [1] * len(history_idx) + [0] * (self.max_history_length - len(history_idx))
        history_idx.extend([0] * (self.max_history_length - len(history_idx)))
        
        # Get user neighbors
        user_neighbors = self.get_neighbors(behavior['user_id'], self.user_user_graph, self.num_neighbors)
        user_neighbor_idx = [self.user2idx.get(u, 0) for u in user_neighbors]
        
        # Process impressions
        impressions = []
        labels = []
        for impression in behavior['impressions'].split():
            news_id, label = impression.split('-')
            impressions.append(news_id)
            labels.append(int(label))
        
        # Get news content and neighbors
        news_title_tokens = [self.news_title_tokens.get(nid, [0] * self.max_title_length) for nid in impressions]
        news_topic_ids = [self.news_topic_ids.get(nid, 0) for nid in impressions]
        news_neighbors = [self.get_neighbors(nid, self.news_news_graph, self.num_neighbors) for nid in impressions]
        news_neighbor_idx = [[self.news2idx.get(n, 0) for n in neighbors] for neighbors in news_neighbors]
        
        return {
            'user_id': self.user2idx.get(behavior['user_id'], 0),
            'history_idx': torch.tensor(history_idx),
            'history_mask': torch.tensor(history_mask),
            'user_neighbors': torch.tensor(user_neighbor_idx),
            'news_title': torch.tensor(news_title_tokens),
            'news_topic': torch.tensor(news_topic_ids),
            'news_neighbors': torch.tensor(news_neighbor_idx),
            'labels': torch.tensor(labels)
        } 