import pandas as pd
import torch
from torch.utils.data import Dataset
import numpy as np
from collections import defaultdict
import random
import pickle
import os
from tqdm import tqdm


class MINDDataset(Dataset):
    """MIND 데이터셋 클래스 - 논문에 따른 그래프 기반 이웃 생성"""
    
    def __init__(self, data_dir="data/MIND_small", split="train", max_title_len=30, 
                 max_clicked_news=50, max_neighbors=15, rebuild_graph=False):
        self.data_dir = data_dir
        self.split = split
        self.max_title_len = max_title_len
        self.max_clicked_news = max_clicked_news
        self.max_neighbors = max_neighbors
        
        # 어휘 사전 경로 (small 버전 우선)
        self.vocab_small_path = os.path.join(data_dir, "vocab_small.pkl")
        self.graph_small_path = os.path.join(data_dir, f"graph_{split}_small.pkl")
        self.vocab_path = os.path.join(data_dir, "vocab.pkl")
        self.graph_path = os.path.join(data_dir, f"graph_{split}.pkl")
        
        # Small 버전이 있으면 우선 사용
        if os.path.exists(self.vocab_small_path) and os.path.exists(self.graph_small_path):
            self.vocab_path = self.vocab_small_path
            self.graph_path = self.graph_small_path
            self.use_small = True
        else:
            self.use_small = False
        
        # 어휘 사전 로드 (build_graph.py에서 생성된 것 사용)
        self.load_vocab()
        
        # 그래프와 데이터 로드 (build_graph.py에서 생성된 것 우선 사용)
        if os.path.exists(self.graph_path):
            version_type = "small" if self.use_small else "standard"
            print(f"Using pre-built graph and data from build_graph.py ({version_type} version)")
            self.load_graph_and_data()
        else:
            print(" Pre-built graph not found! Please run:")
            print(f"   python build_graph.py --small --data_dir {data_dir} --split {split}  (recommended)")
            print(f"   python build_graph.py --data_dir {data_dir} --split {split}          (full dataset)")
            print("   Falling back to in-memory graph building...")
            # 기존 방식으로 폴백
            self.load_data()
            if rebuild_graph or not os.path.exists(self.graph_path):
                self.build_graph()
            else:
                self.load_graph()
        
        # 샘플 생성
        self.samples = []
        self.create_samples()
        
        print(f"Dataset ready: {len(self.samples)} samples")
    
    def load_vocab(self):
        """어휘 사전 로드 (build_graph.py에서 생성된 것)"""
        if os.path.exists(self.vocab_path):
            version_type = "small" if self.use_small else "standard"
            print(f"Loading vocabulary from build_graph.py ({version_type} version)...")
            with open(self.vocab_path, 'rb') as f:
                self.vocab = pickle.load(f)
            print(f"Vocabulary loaded: {len(self.vocab)} words")
        else:
            print(" Vocabulary not found! Please run build_graph.py first")
            print("   Creating basic vocabulary...")
            self.vocab = {'<PAD>': 0, '<UNK>': 1}
    
    def _load_pickle_with_progress(self, file_path):
        """tqdm을 사용한 pickle 파일 로딩"""
        file_size = os.path.getsize(file_path)
        
        with open(file_path, 'rb') as f:
            with tqdm(total=file_size, unit='B', unit_scale=True, 
                     desc="   Loading graph", ncols=80) as pbar:
                
                # 작은 청크로 읽으면서 진행률 업데이트
                data = bytearray()
                chunk_size = 8192  # 8KB씩 읽기
                
                while True:
                    chunk = f.read(chunk_size)
                    if not chunk:
                        break
                    data.extend(chunk)
                    pbar.update(len(chunk))
                
                # 메모리에서 pickle 로드
                pbar.set_description("   Deserializing")
                return pickle.loads(data)
    
    def load_graph_and_data(self):
        """build_graph.py에서 생성된 그래프와 데이터 로드"""
        print(" Loading pre-built graph and data...")
        
        # 파일 크기 확인하여 예상 로딩 시간 안내
        file_size_mb = os.path.getsize(self.graph_path) / (1024 * 1024)
        print(f"   Graph file size: {file_size_mb:.1f} MB")
        
        # tqdm으로 파일 로딩 진행률 표시
        graph_data = self._load_pickle_with_progress(self.graph_path)
        print("   Graph file loaded successfully!")
        
        # 그래프 정보 로드
        print("   Converting graph data structures...")
        self.news_to_users = defaultdict(set, {k: set(v) for k, v in tqdm(graph_data['news_to_users'].items(), desc="   News->Users")})
        self.user_to_news = defaultdict(set, {k: set(v) for k, v in tqdm(graph_data['user_to_news'].items(), desc="   User->News")})
        self.news_neighbors = defaultdict(set, {k: set(v) for k, v in tqdm(graph_data['news_neighbors'].items(), desc="   News neighbors")})
        self.user_neighbors = defaultdict(set, {k: set(v) for k, v in tqdm(graph_data['user_neighbors'].items(), desc="   User neighbors")})
        
        # 뉴스 정보와 사용자 클릭 히스토리 로드
        print("   Loading news dictionary and user histories...")
        self.news_dict = graph_data['news_dict']
        self.user_clicked_news = defaultdict(list, graph_data['user_clicked_news'])
        
        # 행동 데이터도 로드 (샘플 생성용)
        print("   Loading behavior data...")
        behaviors_path = f"{self.data_dir}/{self.split}/behaviors.tsv"
        self.behaviors_df = pd.read_csv(behaviors_path, sep='\t', header=None)
        self.behaviors_df.columns = ['impression_id', 'user_id', 'time', 'clicked_news', 'impressions']
        
        print(f"Graph loaded:")
        print(f"   News nodes: {len(self.news_neighbors)}")
        print(f"   User nodes: {len(self.user_neighbors)}")
        print(f"   News database: {len(self.news_dict)}")
        print(f"   User histories: {len(self.user_clicked_news)}")
    
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
    
    def load_or_build_vocab(self):
        """어휘 사전 구축 또는 로드 (DEPRECATED - 이제 load_vocab 사용)"""
        if os.path.exists(self.vocab_path):
            print("Loading existing vocabulary...")
            with open(self.vocab_path, 'rb') as f:
                self.vocab = pickle.load(f)
        else:
            print("Building vocabulary...")
            self.build_vocab()
            with open(self.vocab_path, 'wb') as f:
                pickle.dump(self.vocab, f)
        
        print(f"Vocabulary size: {len(self.vocab)}")
    
    def build_vocab(self):
        """어휘 사전 구축"""
        word_count = defaultdict(int)
        
        # 모든 뉴스 제목에서 단어 수집
        for _, row in self.news_df.iterrows():
            if pd.notna(row['title']):
                words = row['title'].lower().split()
                for word in words:
                    word_count[word] += 1
        
        # 빈도 기준으로 어휘 구축 (최소 빈도 2 이상)
        self.vocab = {'<PAD>': 0, '<UNK>': 1}
        for word, count in word_count.items():
            if count >= 2:  # 최소 빈도
                self.vocab[word] = len(self.vocab)
    
    def build_graph(self):
        """논문에 따른 이분 그래프 구축"""
        print("Building bipartite graph for neighbor finding...")
        
        # 논문 Section 1: 사용자-뉴스 이분 그래프
        # "사용자와 뉴스가 모두 노드로 간주되고 그들 간의 상호작용은 엣지로 간주"
        
        # 뉴스 -> 클릭한 사용자들
        self.news_to_users = defaultdict(set)
        # 사용자 -> 클릭한 뉴스들  
        self.user_to_news = defaultdict(set)
        
        for user_id, clicked_news_list in self.user_clicked_news.items():
            for news_id in clicked_news_list:
                if news_id in self.news_dict:
                    self.news_to_users[news_id].add(user_id)
                    self.user_to_news[user_id].add(news_id)
        
        # 논문: "동일한 사용자가 본 뉴스로 간주되어 이웃 뉴스로 정의"
        self.news_neighbors = defaultdict(set)
        for user_id, news_set in self.user_to_news.items():
            news_list = list(news_set)
            for i, news1 in enumerate(news_list):
                for j, news2 in enumerate(news_list):
                    if i != j:
                        self.news_neighbors[news1].add(news2)
        
        # 논문: "특정 사용자는 동일한 클릭한 뉴스를 공유하여 이웃 사용자로 정의"
        self.user_neighbors = defaultdict(set)
        for news_id, user_set in self.news_to_users.items():
            user_list = list(user_set)
            for i, user1 in enumerate(user_list):
                for j, user2 in enumerate(user_list):
                    if i != j:
                        self.user_neighbors[user1].add(user2)
        
        # 그래프 저장
        graph_data = {
            'news_to_users': dict(self.news_to_users),
            'user_to_news': dict(self.user_to_news), 
            'news_neighbors': dict(self.news_neighbors),
            'user_neighbors': dict(self.user_neighbors)
        }
        
        with open(self.graph_path, 'wb') as f:
            pickle.dump(graph_data, f)
        
        print(f"Graph built: {len(self.news_neighbors)} news nodes, {len(self.user_neighbors)} user nodes")
    
    def load_graph(self):
        """저장된 그래프 로드 (DEPRECATED - load_graph_and_data 사용)"""
        print("Loading existing graph...")
        with open(self.graph_path, 'rb') as f:
            graph_data = pickle.load(f)
        
        self.news_to_users = defaultdict(set, {k: set(v) for k, v in graph_data['news_to_users'].items()})
        self.user_to_news = defaultdict(set, {k: set(v) for k, v in graph_data['user_to_news'].items()})
        self.news_neighbors = defaultdict(set, {k: set(v) for k, v in graph_data['news_neighbors'].items()})
        self.user_neighbors = defaultdict(set, {k: set(v) for k, v in graph_data['user_neighbors'].items()})
        
        print(f"Graph loaded: {len(self.news_neighbors)} news nodes, {len(self.user_neighbors)} user nodes")
    
    def get_neighbor_users(self, user_id, max_neighbors=None):
        """이웃 사용자 가져오기 (논문 Section 3.3.1)"""
        if max_neighbors is None:
            max_neighbors = self.max_neighbors
            
        neighbors = list(self.user_neighbors.get(user_id, set()))
        
        if len(neighbors) > max_neighbors:
            # 논문: "클릭한 뉴스의 수에 따라 순위를 매김"
            # 여기서는 단순히 랜덤 샘플링 (실제로는 클릭 수 기준 정렬)
            neighbors = random.sample(neighbors, max_neighbors)
        elif len(neighbors) < max_neighbors:
            # 패딩 (0으로 채움)
            neighbors.extend([None] * (max_neighbors - len(neighbors)))
        
        return neighbors
    
    def get_neighbor_news(self, news_id, max_neighbors=None):
        """이웃 뉴스 가져오기 (논문 Section 3.3.2, 3.3.3)"""
        if max_neighbors is None:
            max_neighbors = self.max_neighbors
            
        neighbors = list(self.news_neighbors.get(news_id, set()))
        
        if len(neighbors) > max_neighbors:
            neighbors = random.sample(neighbors, max_neighbors)
        elif len(neighbors) < max_neighbors:
            neighbors.extend([None] * (max_neighbors - len(neighbors)))
        
        return neighbors
    
    def tokenize_with_vocab(self, text, max_len):
        """어휘 사전을 사용한 토큰화"""
        if not text:
            return [0] * max_len
        
        words = text.lower().split()[:max_len]
        token_ids = []
        
        for word in words:
            if word in self.vocab:
                token_ids.append(self.vocab[word])
            else:
                token_ids.append(self.vocab['<UNK>'])  # Unknown 토큰
        
        # 패딩
        while len(token_ids) < max_len:
            token_ids.append(0)  # PAD 토큰
        
        return token_ids
    
    def create_samples(self):
        """학습 샘플 생성"""
        print("   Creating training samples...")
        for _, row in tqdm(self.behaviors_df.iterrows(), total=len(self.behaviors_df), desc="Processing behaviors for samples"):
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
    
    def get_category_id(self, category):
        """카테고리를 ID로 변환"""
        # 간단한 카테고리 매핑 (실제로는 별도 어휘 구축)
        category_map = {
            'news': 0, 'sports': 1, 'entertainment': 2, 'finance': 3,
            'lifestyle': 4, 'travel': 5, 'foodanddrink': 6, 'health': 7,
            'auto': 8, 'tv': 9, 'movies': 10, 'music': 11, 'northamerica': 12,
            'weather': 13, 'video': 14, 'kids': 15, 'middleeast': 16,
            'europe': 17, 'southamerica': 18, 'asia': 19
        }
        return category_map.get(category, 0)
    
    def get_user_id(self, user_str):
        """사용자 문자열을 정수 ID로 변환"""
        return hash(user_str) % 50000  # 적당한 범위로 해싱
    
    def get_news_id(self, news_str):
        """뉴스 문자열을 정수 ID로 변환"""
        return hash(news_str) % 100000  # 적당한 범위로 해싱
    
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
        
        # 토큰화 (개선된 어휘 사용)
        candidate_tokens = self.tokenize_with_vocab(candidate_title, self.max_title_len)
        
        # 클릭한 뉴스들 토큰화
        clicked_tokens_list = []
        for title in clicked_titles[:self.max_clicked_news]:
            clicked_tokens_list.append(self.tokenize_with_vocab(title, self.max_title_len))
        
        # 패딩
        while len(clicked_tokens_list) < self.max_clicked_news:
            clicked_tokens_list.append([0] * self.max_title_len)
        
        # 마스크 생성
        clicked_mask = [1 if i < len(clicked_titles) else 0 for i in range(self.max_clicked_news)]
        
        # 카테고리 ID 변환
        candidate_topic = self.get_category_id(candidate_category)
        clicked_topics = [self.get_category_id(cat) for cat in clicked_categories[:self.max_clicked_news]]
        clicked_topics.extend([0] * (self.max_clicked_news - len(clicked_topics)))
        
        # 논문에 따른 실제 이웃 정보 가져오기
        neighbor_users = self.get_neighbor_users(user_id)
        neighbor_news = self.get_neighbor_news(news_id)
        
        # 이웃 사용자 ID 변환 및 마스크
        neighbor_user_ids = []
        neighbor_user_mask = []
        for neighbor in neighbor_users:
            if neighbor is not None:
                neighbor_user_ids.append(self.get_user_id(neighbor))
                neighbor_user_mask.append(1)
            else:
                neighbor_user_ids.append(0)
                neighbor_user_mask.append(0)
        
        # 이웃 뉴스 ID 변환 및 마스크
        neighbor_news_ids = []
        neighbor_news_mask = []
        neighbor_news_titles = []
        neighbor_news_topics = []
        
        for neighbor in neighbor_news:
            if neighbor is not None and neighbor in self.news_dict:
                neighbor_news_ids.append(self.get_news_id(neighbor))
                neighbor_news_mask.append(1)
                neighbor_title = self.news_dict[neighbor]['title']
                neighbor_category = self.news_dict[neighbor]['category']
                neighbor_news_titles.append(self.tokenize_with_vocab(neighbor_title, self.max_title_len))
                neighbor_news_topics.append(self.get_category_id(neighbor_category))
            else:
                neighbor_news_ids.append(0)
                neighbor_news_mask.append(0)
                neighbor_news_titles.append([0] * self.max_title_len)
                neighbor_news_topics.append(0)
        
        return {
            'user_id': torch.tensor(self.get_user_id(user_id), dtype=torch.long),
            'candidate_news_id': torch.tensor(self.get_news_id(news_id), dtype=torch.long),
            'candidate_news_title': torch.tensor(candidate_tokens, dtype=torch.long),
            'candidate_news_topic': torch.tensor(candidate_topic, dtype=torch.long),
            'clicked_news_title': torch.tensor(clicked_tokens_list, dtype=torch.long),
            'clicked_news_topic': torch.tensor(clicked_topics, dtype=torch.long),
            'clicked_mask': torch.tensor(clicked_mask, dtype=torch.bool),
            'neighbor_users': torch.tensor(neighbor_user_ids, dtype=torch.long),
            'neighbor_user_mask': torch.tensor(neighbor_user_mask, dtype=torch.bool),
            'neighbor_news': torch.tensor(neighbor_news_ids, dtype=torch.long),
            'neighbor_news_title': torch.tensor(neighbor_news_titles, dtype=torch.long),
            'neighbor_news_topic': torch.tensor(neighbor_news_topics, dtype=torch.long),
            'neighbor_news_mask': torch.tensor(neighbor_news_mask, dtype=torch.bool),
            'label': torch.tensor(label, dtype=torch.float)
        }


def create_data_loaders(data_dir="data/MIND_small", batch_size=32, rebuild_graph=False):
    """데이터 로더 생성 - build_graph.py 사용 권장"""
    from torch.utils.data import DataLoader
    
    # build_graph.py 실행 여부 확인 (small 버전 우선)
    vocab_small_path = os.path.join(data_dir, "vocab_small.pkl")
    train_graph_small_path = os.path.join(data_dir, "graph_train_small.pkl")
    vocab_path = os.path.join(data_dir, "vocab.pkl")
    train_graph_path = os.path.join(data_dir, "graph_train.pkl")
    
    # Small 버전이 있는지 먼저 확인
    if os.path.exists(vocab_small_path) and os.path.exists(train_graph_small_path):
        print("Found small pre-built graph files - using for fast loading!")
    elif os.path.exists(vocab_path) and os.path.exists(train_graph_path):
        print("Found standard pre-built graph files")
    else:
        print(" Pre-built graph files not found!")
        print(" For optimal performance, please run:")
        print(f"   python build_graph.py --small  (recommended, faster)")
        print(f"   python build_graph.py          (full dataset)")
        print("   This will significantly speed up data loading!")
        print()
    
    train_dataset = MINDDataset(data_dir, split="train", rebuild_graph=rebuild_graph)
    dev_dataset = MINDDataset(data_dir, split="dev", rebuild_graph=False)
    
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