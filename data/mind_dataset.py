import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from transformers import BertTokenizer
import json
import os
from tqdm import tqdm

class MINDDataset(Dataset):
    def __init__(self, data_dir, max_title_length=30, max_history_length=50, num_neighbors=5):
        self.data_dir = data_dir
        self.max_title_length = max_title_length
        self.max_history_length = max_history_length
        self.num_neighbors = num_neighbors
        
        print("Initializing BERT tokenizer...")
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        
        # 데이터 로드
        self.load_data()
        
    def load_data(self):
        print("\nLoading news data...")
        news_df = pd.read_csv(os.path.join(self.data_dir, 'news.tsv'), 
                            sep='\t', 
                            names=['news_id', 'category', 'subcategory', 'title', 'abstract', 'url', 'title_entities', 'abstract_entities'])
        self.news_df = news_df
        
        # 뉴스 ID를 인덱스로 매핑
        print("Creating news index mapping...")
        self.news2idx = {nid: idx for idx, nid in enumerate(tqdm(news_df['news_id'].unique()))}
        self.idx2news = {idx: nid for nid, idx in self.news2idx.items()}
        
        # 엔티티 임베딩 로드
        print("\nLoading entity embeddings...")
        self.entity_embedding = {}
        with open(os.path.join(self.data_dir, 'entity_embedding.vec'), 'r', encoding='utf-8') as f:
            total_lines = sum(1 for _ in f)
        
        with open(os.path.join(self.data_dir, 'entity_embedding.vec'), 'r', encoding='utf-8') as f:
            for line in tqdm(f, total=total_lines, desc="Processing entity embeddings"):
                tokens = line.strip().split()
                if len(tokens) > 2:
                    entity = tokens[0]
                    embedding = np.array([float(x) for x in tokens[1:]])
                    self.entity_embedding[entity] = embedding
        
        # 뉴스-엔티티 그래프 구성
        print("\nConstructing news-entity graph...")
        self.news_entities = {}
        for idx, row in tqdm(news_df.iterrows(), total=len(news_df), desc="Processing news entities"):
            news_id = row['news_id']
            entities = []
            
            # 제목의 엔티티 추출
            if isinstance(row['title_entities'], str):
                try:
                    title_ents = json.loads(row['title_entities'])
                    entities.extend([e['WikidataId'] for e in title_ents])
                except:
                    pass
            
            # 본문의 엔티티 추출
            if isinstance(row['abstract_entities'], str):
                try:
                    abstract_ents = json.loads(row['abstract_entities'])
                    entities.extend([e['WikidataId'] for e in abstract_ents])
                except:
                    pass
            
            self.news_entities[news_id] = list(set(entities))
        
        # 행동 데이터 로드
        print("\nLoading user behaviors...")
        behaviors_df = pd.read_csv(os.path.join(self.data_dir, 'behaviors.tsv'),
                                 sep='\t',
                                 names=['impression_id', 'user_id', 'time', 'history', 'impressions'])
        self.behaviors_df = behaviors_df
        
        # 사용자 ID를 인덱스로 매핑
        print("Creating user index mapping...")
        self.user2idx = {uid: idx for idx, uid in enumerate(tqdm(behaviors_df['user_id'].unique()))}
        self.idx2user = {idx: uid for uid, idx in self.user2idx.items()}
        
        # 사용자-뉴스 그래프 구성
        print("\nConstructing user-news graph...")
        self.user_news_graph = {}
        for _, row in tqdm(behaviors_df.iterrows(), total=len(behaviors_df), desc="Processing user behaviors"):
            user_id = row['user_id']
            if isinstance(row['history'], str) and row['history'].strip() != '':
                clicked_news = row['history'].split()
                if user_id not in self.user_news_graph:
                    self.user_news_graph[user_id] = set()
                self.user_news_graph[user_id].update(clicked_news)
        
        # 데이터 전처리
        print("\nPreprocessing training samples...")
        self.process_behaviors()
    
    def find_news_neighbors(self, news_id):
        """뉴스의 이웃 노드를 찾습니다."""
        # 현재 뉴스의 엔티티들
        current_entities = set(self.news_entities.get(news_id, []))
        if not current_entities:
            return []
        
        # 엔티티 기반으로 이웃 뉴스 찾기
        neighbor_scores = {}
        for other_id, other_entities in self.news_entities.items():
            if other_id != news_id:
                # 엔티티 교집합 크기로 유사도 계산
                common_entities = current_entities.intersection(set(other_entities))
                if common_entities:
                    neighbor_scores[other_id] = len(common_entities)
        
        # 상위 K개 이웃 선택
        neighbors = sorted(neighbor_scores.items(), key=lambda x: x[1], reverse=True)[:self.num_neighbors]
        return [n[0] for n in neighbors]
    
    def find_user_neighbors(self, user_id):
        """사용자의 이웃 노드를 찾습니다."""
        if user_id not in self.user_news_graph:
            return []
        
        # 현재 사용자의 클릭 뉴스
        current_news = self.user_news_graph[user_id]
        
        # 클릭 패턴 유사도로 이웃 사용자 찾기
        neighbor_scores = {}
        for other_id, other_news in self.user_news_graph.items():
            if other_id != user_id:
                # 자카드 유사도 계산
                intersection = len(current_news.intersection(other_news))
                union = len(current_news.union(other_news))
                if union > 0:
                    neighbor_scores[other_id] = intersection / union
        
        # 상위 K개 이웃 선택
        neighbors = sorted(neighbor_scores.items(), key=lambda x: x[1], reverse=True)[:self.num_neighbors]
        return [n[0] for n in neighbors]
    
    def process_behaviors(self):
        self.samples = []
        total_impressions = sum(len(row['impressions'].split()) 
                              for _, row in self.behaviors_df.iterrows())
        
        pbar = tqdm(total=total_impressions, desc="Processing samples")
        
        for _, row in self.behaviors_df.iterrows():
            user_id = row['user_id']
            user_idx = self.user2idx[user_id]
            
            # 클릭 히스토리 처리
            history = []
            if isinstance(row['history'], str) and row['history'].strip() != '':
                history = [self.news2idx[h] for h in row['history'].split()]
            history = history[-self.max_history_length:]  # 최근 N개만 사용
            
            # 히스토리 패딩
            if len(history) < self.max_history_length:
                history = history + [0] * (self.max_history_length - len(history))
            
            # 사용자 이웃 찾기
            user_neighbors = self.find_user_neighbors(user_id)
            user_neighbor_indices = [self.user2idx[n] for n in user_neighbors]
            if len(user_neighbor_indices) < self.num_neighbors:
                user_neighbor_indices += [0] * (self.num_neighbors - len(user_neighbor_indices))
            
            # 임프레션 처리
            impressions = row['impressions'].split()
            for imp in impressions:
                news_id, label = imp.split('-')
                if news_id in self.news2idx:
                    news_idx = self.news2idx[news_id]
                    
                    # 뉴스 이웃 찾기
                    news_neighbors = self.find_news_neighbors(news_id)
                    news_neighbor_indices = [self.news2idx[n] for n in news_neighbors]
                    if len(news_neighbor_indices) < self.num_neighbors:
                        news_neighbor_indices += [0] * (self.num_neighbors - len(news_neighbor_indices))
                    
                    # 뉴스 제목 토큰화
                    news_title = self.news_df.iloc[news_idx]['title']
                    news_tokens = self.tokenize_title(news_title)
                    
                    # 히스토리 뉴스 제목 토큰화
                    history_titles = []
                    for h_idx in history:
                        if h_idx == 0:
                            title_tokens = [0] * self.max_title_length
                        else:
                            title = self.news_df.iloc[h_idx]['title']
                            title_tokens = self.tokenize_title(title)
                        history_titles.append(title_tokens)
                    
                    # 이웃 뉴스 제목 토큰화
                    neighbor_titles = []
                    for n_idx in news_neighbor_indices:
                        if n_idx == 0:
                            title_tokens = [0] * self.max_title_length
                        else:
                            title = self.news_df.iloc[n_idx]['title']
                            title_tokens = self.tokenize_title(title)
                        neighbor_titles.append(title_tokens)
                    
                    self.samples.append({
                        'user_id': user_idx,
                        'news_title': news_tokens,
                        'clicked_news_titles': history_titles,
                        'user_neighbors': user_neighbor_indices,
                        'news_neighbors': neighbor_titles,
                        'label': int(label)
                    })
                pbar.update(1)
        
        pbar.close()
        print(f"\nTotal samples created: {len(self.samples)}")
    
    def tokenize_title(self, title):
        tokens = self.tokenizer.encode(
            title,
            add_special_tokens=True,
            max_length=self.max_title_length,
            padding='max_length',
            truncation=True
        )
        return tokens
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # 인접 행렬 생성
        user_neighbor_adj = torch.ones(self.num_neighbors, self.num_neighbors)  # 완전 연결 그래프로 가정
        news_neighbor_adj = torch.ones(self.num_neighbors, self.num_neighbors)  # 완전 연결 그래프로 가정
        
        return {
            'user_id': torch.tensor(sample['user_id']),
            'news_title': torch.tensor(sample['news_title']),
            'clicked_news_titles': torch.tensor(sample['clicked_news_titles']),
            'user_neighbors': torch.tensor(sample['user_neighbors']),
            'news_neighbors': torch.tensor(sample['news_neighbors']),
            'user_neighbor_adj': user_neighbor_adj,
            'news_neighbor_adj': news_neighbor_adj,
            'label': torch.tensor(sample['label'])
        } 