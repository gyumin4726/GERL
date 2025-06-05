import pandas as pd
import torch
from torch.utils.data import Dataset
import numpy as np
from collections import defaultdict
import random


class MINDDataset(Dataset):
    """MIND 데이터셋 클래스 - 간단한 버전"""
    
    def __init__(self, data_dir="data/MIND_small", split="train", max_title_len=30, max_clicked_news=50):
        self.data_dir = data_dir
        self.split = split
        self.max_title_len = max_title_len
        self.max_clicked_news = max_clicked_news
        
        # 데이터 로드
        self.load_data()
        
    def load_data(self):
        """데이터 로드"""
        print(f"Loading {self.split} data...")
        
        # 뉴스 데이터 로드
        news_path = f"{self.data_dir}/{self.split}/news.tsv"
        self.news_df = pd.read_csv(
            news_path,
            sep='\t',
            header=None,
            usecols=[0, 1, 3]  # news_id, category, title
        )
        self.news_df.columns = ['news_id', 'category', 'title']
        
        # 행동 데이터 로드
        behaviors_path = f"{self.data_dir}/{self.split}/behaviors.tsv"
        self.behaviors_df = pd.read_csv(
            behaviors_path,
            sep='\t',
            header=None
        )
        self.behaviors_df.columns = ['impression_id', 'user_id', 'time', 'clicked_news', 'impressions']
        
        # 뉴스 정보를 딕셔너리로 변환
        self.news_dict = {}
        for _, row in self.news_df.iterrows():
            self.news_dict[row['news_id']] = {
                'category': row['category'],
                'title': row['title'] if pd.notna(row['title']) else "",
            }
        
        # 사용자 클릭 히스토리 구축
        self.user_clicked_news = defaultdict(list)
        for _, row in self.behaviors_df.iterrows():
            if pd.notna(row['clicked_news']):
                clicked_list = row['clicked_news'].split()
                self.user_clicked_news[row['user_id']].extend(clicked_list)
        
        # 샘플 생성
        self.samples = []
        self.create_samples()
        
        print(f"Loaded {len(self.samples)} samples")
        
    def create_samples(self):
        """학습 샘플 생성"""
        for _, row in self.behaviors_df.iterrows():
            if pd.isna(row['impressions']):
                continue
                
            user_id = row['user_id']
            impressions = row['impressions'].split()
            
            for impression in impressions:
                parts = impression.split('-')
                if len(parts) != 2:
                    continue
                    
                news_id, label = parts[0], int(parts[1])
                
                if news_id not in self.news_dict:
                    continue
                
                self.samples.append({
                    'user_id': user_id,
                    'news_id': news_id,
                    'label': label
                })
    
    def tokenize_simple(self, text, max_len):
        """간단한 토큰화 (단어 단위)"""
        if not text:
            return [0] * max_len
        
        # 간단한 전처리
        words = text.lower().split()[:max_len]
        
        # 간단한 해시 기반 인덱싱 (실제로는 vocab 사용해야 함)
        token_ids = [hash(word) % 10000 + 1 for word in words]
        
        # 패딩
        while len(token_ids) < max_len:
            token_ids.append(0)
        
        return token_ids
    
    def get_user_history(self, user_id):
        """사용자 클릭 히스토리 가져오기"""
        clicked_news = self.user_clicked_news.get(user_id, [])
        
        # 최근 뉴스들만 선택
        clicked_news = clicked_news[-self.max_clicked_news:]
        
        clicked_titles = []
        clicked_categories = []
        
        for news_id in clicked_news:
            if news_id in self.news_dict:
                clicked_titles.append(self.news_dict[news_id]['title'])
                clicked_categories.append(self.news_dict[news_id]['category'])
        
        return clicked_titles, clicked_categories
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        user_id = sample['user_id']
        news_id = sample['news_id']
        label = sample['label']
        
        # 후보 뉴스 정보
        candidate_title = self.news_dict[news_id]['title']
        candidate_category = self.news_dict[news_id]['category']
        
        # 사용자 히스토리
        clicked_titles, clicked_categories = self.get_user_history(user_id)
        
        # 토큰화
        candidate_tokens = self.tokenize_simple(candidate_title, self.max_title_len)
        
        # 클릭한 뉴스들 토큰화
        clicked_tokens_list = []
        for title in clicked_titles[:self.max_clicked_news]:
            clicked_tokens_list.append(self.tokenize_simple(title, self.max_title_len))
        
        # 패딩
        while len(clicked_tokens_list) < self.max_clicked_news:
            clicked_tokens_list.append([0] * self.max_title_len)
        
        # 마스크 생성
        clicked_mask = [1 if i < len(clicked_titles) else 0 for i in range(self.max_clicked_news)]
        
        # 간단한 카테고리 인덱싱
        candidate_topic = hash(candidate_category) % 20 if candidate_category else 0
        
        # 이웃 정보는 간단하게 랜덤으로 생성 (실제로는 그래프 기반)
        neighbor_users = [random.randint(1, 1000) for _ in range(15)]
        neighbor_news = [random.randint(1, 1000) for _ in range(15)]
        neighbor_user_mask = [1] * 15
        neighbor_news_mask = [1] * 15
        
        # 이웃 뉴스 제목들 (랜덤 샘플링)
        neighbor_news_titles = []
        for _ in range(15):
            random_news = random.choice(list(self.news_dict.keys()))
            neighbor_title = self.news_dict[random_news]['title']
            neighbor_news_titles.append(self.tokenize_simple(neighbor_title, self.max_title_len))
        
        neighbor_news_topics = [random.randint(0, 19) for _ in range(15)]
        
        return {
            'user_id': torch.tensor(hash(user_id) % 10000, dtype=torch.long),
            'candidate_news_id': torch.tensor(hash(news_id) % 10000, dtype=torch.long),
            'candidate_news_title': torch.tensor(candidate_tokens, dtype=torch.long),
            'candidate_news_topic': torch.tensor(candidate_topic, dtype=torch.long),
            'clicked_news_title': torch.tensor(clicked_tokens_list, dtype=torch.long),
            'clicked_news_topic': torch.tensor([hash(cat) % 20 for cat in clicked_categories[:self.max_clicked_news]] + [0] * (self.max_clicked_news - len(clicked_categories)), dtype=torch.long),
            'clicked_mask': torch.tensor(clicked_mask, dtype=torch.bool),
            'neighbor_users': torch.tensor(neighbor_users, dtype=torch.long),
            'neighbor_user_mask': torch.tensor(neighbor_user_mask, dtype=torch.bool),
            'neighbor_news': torch.tensor(neighbor_news, dtype=torch.long),
            'neighbor_news_title': torch.tensor(neighbor_news_titles, dtype=torch.long),
            'neighbor_news_topic': torch.tensor(neighbor_news_topics, dtype=torch.long),
            'neighbor_news_mask': torch.tensor(neighbor_news_mask, dtype=torch.bool),
            'label': torch.tensor(label, dtype=torch.float)
        }


def create_data_loaders(data_dir="data/MIND_small", batch_size=32):
    """데이터 로더 생성"""
    from torch.utils.data import DataLoader
    
    train_dataset = MINDDataset(data_dir, split="train")
    dev_dataset = MINDDataset(data_dir, split="dev")
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, dev_loader


if __name__ == "__main__":
    # 테스트
    dataset = MINDDataset()
    print(f"Dataset size: {len(dataset)}")
    
    sample = dataset[0]
    print("Sample keys:", sample.keys())
    print("Candidate title shape:", sample['candidate_news_title'].shape)
    print("Clicked news shape:", sample['clicked_news_title'].shape) 