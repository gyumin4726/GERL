"""
Section 4.1 "Datasets and Experimental Settings"에서 사용된
MIND 데이터셋 처리 모듈

MIND 데이터셋을 로드하고 전처리하여 모델 학습에 사용할 수 있는 형태로 변환합니다.
"""

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
        """
        MIND 데이터셋 초기화
        
        Args:
            data_dir: 데이터 디렉토리 경로
            max_title_length: 뉴스 제목 최대 길이 (Section 4.1)
            max_history_length: 사용자 기록 최대 길이 (Section 4.1)
            num_neighbors: 이웃 노드 수 (Section 4.5)
        """
        self.data_dir = data_dir
        self.max_title_length = max_title_length
        self.max_history_length = max_history_length
        self.num_neighbors = num_neighbors
        
        # BERT 토크나이저 초기화
        self.tokenizer = tokenizer if tokenizer else BertTokenizer.from_pretrained('bert-base-uncased')
        
        # 데이터 로드 및 전처리
        print("뉴스 데이터 로딩 중...")
        self.news_df = pd.read_csv(os.path.join(data_dir, 'news.tsv'), 
                                 sep='\t',
                                 names=['news_id', 'category', 'subcategory', 'title', 'abstract', 'url', 'title_entities', 'abstract_entities'])
        
        print("사용자 행동 데이터 로딩 중...")
        self.behaviors_df = pd.read_csv(os.path.join(data_dir, 'behaviors.tsv'),
                                      sep='\t',
                                      names=['impression_id', 'user_id', 'time', 'history', 'impressions'])
        
        # ID 매핑 생성
        print("ID 매핑 생성 중...")
        self.news2idx = {nid: idx for idx, nid in enumerate(self.news_df['news_id'].unique(), start=1)}
        self.news2idx['[PAD]'] = 0
        self.user2idx = {uid: idx for idx, uid in enumerate(self.behaviors_df['user_id'].unique(), start=1)}
        self.user2idx['[PAD]'] = 0
        
        # 상호작용 그래프 구축 (Section 3.3)
        print("상호작용 그래프 구축 중...")
        self.build_graphs()
        
        print("뉴스 콘텐츠 처리 중...")
        self.process_news_content()
        
        print("데이터셋 초기화 완료")
        
    def build_graphs(self):
        """
        Section 3.3의 이분 그래프 구축
        
        사용자-뉴스 상호작용을 바탕으로 세 가지 그래프를 구축:
        1. 사용자-뉴스 이분 그래프
        2. 뉴스-뉴스 그래프 (같은 사용자가 클릭한 뉴스 간 연결)
        3. 사용자-사용자 그래프 (같은 뉴스를 클릭한 사용자 간 연결)
        """
        self.user_news_graph = nx.Graph()
        self.news_news_graph = nx.Graph()
        self.user_user_graph = nx.Graph()
        
        # 사용자-뉴스 엣지 구축
        for _, row in tqdm(self.behaviors_df.iterrows(), desc="그래프 구축 중"):
            user_id = row['user_id']
            if pd.isna(row['history']):
                continue
                
            history = row['history'].split()
            
            # 사용자-뉴스 엣지 추가
            for news_id in history:
                self.user_news_graph.add_edge(user_id, news_id)
                
            # 뉴스-뉴스 엣지 추가 (동일 사용자 클릭 기반)
            for i in range(len(history)):
                for j in range(i+1, len(history)):
                    self.news_news_graph.add_edge(history[i], history[j])
        
        # 사용자-사용자 엣지 구축 (공통 뉴스 기반)
        for user1 in tqdm(self.user_news_graph.nodes(), desc="사용자-사용자 엣지 구축 중"):
            if not isinstance(user1, str) or user1.startswith('N'):  # 뉴스 노드 제외
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
                    similarity = common_news / len(user1_news | user2_news)  # Jaccard 유사도
                    self.user_user_graph.add_edge(user1, user2, weight=similarity)
    
    def process_news_content(self):
        """
        Section 3.1에 따른 뉴스 콘텐츠 처리
        
        1. 뉴스 제목을 BERT 토크나이저로 인코딩
        2. 카테고리 정보를 토픽 ID로 변환
        """
        self.news_title_tokens = {}
        self.news_topic_ids = {}
        
        for _, news in tqdm(self.news_df.iterrows(), desc="뉴스 처리 중"):
            news_id = news['news_id']
            
            # 제목 토크나이징
            title_tokens = self.tokenizer.encode(
                news['title'],
                add_special_tokens=True,
                max_length=self.max_title_length,
                padding='max_length',
                truncation=True
            )
            self.news_title_tokens[news_id] = title_tokens
            
            # 카테고리를 토픽 ID로 변환
            self.news_topic_ids[news_id] = hash(news['category']) % 100
    
    def get_neighbors(self, node_id, graph, k):
        """
        Section 3.3의 이웃 노드 추출
        
        그래프에서 주어진 노드의 상위 k개 이웃을 추출합니다.
        이웃은 엣지 가중치를 기준으로 정렬됩니다.
        
        Args:
            node_id: 대상 노드 ID
            graph: 그래프 객체
            k: 추출할 이웃 수
        """
        if node_id not in graph:
            return []
        
        neighbors = sorted(graph[node_id].items(), key=lambda x: x[1]['weight'] if 'weight' in x[1] else 1.0, reverse=True)
        neighbors = [n[0] for n in neighbors[:k]]
        
        if len(neighbors) < k:
            neighbors.extend(['[PAD]'] * (k - len(neighbors)))
        return neighbors
    
    def __len__(self):
        """데이터셋의 총 샘플 수 반환"""
        return len(self.behaviors_df)
    
    def __getitem__(self, idx):
        """
        Section 4.1의 데이터 형식에 따른 샘플 반환
        
        한 샘플은 다음을 포함:
        - 사용자 ID
        - 클릭 기록
        - 이웃 사용자
        - 뉴스 제목
        - 뉴스 토픽
        - 이웃 뉴스
        - 클릭 여부 레이블
        """
        behavior = self.behaviors_df.iloc[idx]
        
        # 클릭 기록 처리
        history = [] if pd.isna(behavior['history']) else behavior['history'].split()
        if len(history) > self.max_history_length:
            history = history[-self.max_history_length:]
        history_idx = [self.news2idx.get(h, 0) for h in history]
        history_mask = [1] * len(history_idx) + [0] * (self.max_history_length - len(history_idx))
        history_idx.extend([0] * (self.max_history_length - len(history_idx)))
        
        # 이웃 사용자 가져오기
        user_neighbors = self.get_neighbors(behavior['user_id'], self.user_user_graph, self.num_neighbors)
        user_neighbor_idx = [self.user2idx.get(u, 0) for u in user_neighbors]
        
        # 노출 정보 처리
        impressions = []
        labels = []
        for impression in behavior['impressions'].split():
            news_id, label = impression.split('-')
            impressions.append(news_id)
            labels.append(int(label))
        
        # 뉴스 콘텐츠와 이웃 정보
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