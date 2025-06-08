import torch
import numpy as np
from collections import defaultdict

class NeighborSampler:
    def __init__(self, config):
        self.max_neighbors = config.max_neighbors
        self.user_click_dict = defaultdict(list)  # user_id -> [news_ids]
        self.news_user_dict = defaultdict(list)   # news_id -> [user_ids]
        
    def build_graph(self, behaviors):
        """사용자-뉴스 상호작용 그래프 구축
        
        Args:
            behaviors: List of (user_id, news_id) tuples
        """
        for user_id, news_id in behaviors:
            self.user_click_dict[user_id].append(news_id)
            self.news_user_dict[news_id].append(user_id)
            
    def sample_news_neighbors(self, news_id):
        """뉴스의 이웃 뉴스 샘플링 (같은 사용자가 클릭한 뉴스)
        
        Args:
            news_id: Target news ID
            
        Returns:
            List of neighbor news IDs
        """
        neighbor_news = []
        users = self.news_user_dict[news_id]
        
        for user in users:
            neighbor_news.extend(self.user_click_dict[user])
            
        # 자기 자신 제거
        neighbor_news = [n for n in neighbor_news if n != news_id]
        
        # 빈도수로 정렬하고 상위 K개 선택
        if len(neighbor_news) > self.max_neighbors:
            news_freq = defaultdict(int)
            for n in neighbor_news:
                news_freq[n] += 1
            sorted_news = sorted(news_freq.items(), key=lambda x: x[1], reverse=True)
            neighbor_news = [n[0] for n in sorted_news[:self.max_neighbors]]
        
        # Padding if necessary
        while len(neighbor_news) < self.max_neighbors:
            neighbor_news.append(0)  # 0 for padding
            
        return neighbor_news[:self.max_neighbors]
        
    def sample_user_neighbors(self, user_id):
        """사용자의 이웃 사용자 샘플링 (같은 뉴스를 클릭한 사용자)
        
        Args:
            user_id: Target user ID
            
        Returns:
            List of neighbor user IDs
        """
        neighbor_users = []
        news_clicked = self.user_click_dict[user_id]
        
        for news in news_clicked:
            neighbor_users.extend(self.news_user_dict[news])
            
        # 자기 자신 제거
        neighbor_users = [u for u in neighbor_users if u != user_id]
        
        # 빈도수로 정렬하고 상위 K개 선택
        if len(neighbor_users) > self.max_neighbors:
            user_freq = defaultdict(int)
            for u in neighbor_users:
                user_freq[u] += 1
            sorted_users = sorted(user_freq.items(), key=lambda x: x[1], reverse=True)
            neighbor_users = [u[0] for u in sorted_users[:self.max_neighbors]]
            
        # Padding if necessary
        while len(neighbor_users) < self.max_neighbors:
            neighbor_users.append(0)  # 0 for padding
            
        return neighbor_users[:self.max_neighbors]
        
    def get_neighbor_mask(self, neighbors):
        """이웃 노드에 대한 마스크 생성
        
        Args:
            neighbors: List of neighbor IDs
            
        Returns:
            Binary mask (1 for valid neighbors, 0 for padding)
        """
        return [1 if n != 0 else 0 for n in neighbors] 