"""
Section 3.3 "Two-hop Graph Learning"에서 설명된
이웃 노드 샘플링 구현

이 모듈은 사용자-뉴스 상호작용 그래프에서 이웃 노드를 샘플링합니다.
논문의 Figure 1에서 보여주는 것처럼, 사용자와 뉴스는 클릭 행동을 통해
이분 그래프를 형성하며, 이를 통해 이웃 관계가 정의됩니다.

주요 기능:
1. 이웃 뉴스 샘플링: 같은 사용자가 클릭한 다른 뉴스
2. 이웃 사용자 샘플링: 같은 뉴스를 클릭한 다른 사용자
3. Section 4.5에서 제안된 대로 최대 15개의 이웃만 유지
"""

import torch
import numpy as np
from collections import defaultdict

class NeighborSampler:
    def __init__(self, config):
        """
        Section 3.3의 이웃 샘플링을 위한 초기화
        
        Args:
            config: 모델 설정으로 다음을 포함:
                - max_neighbors: 최대 이웃 수 (Section 4.5에서 15로 설정)
        """
        self.max_neighbors = config.max_neighbors
        self.user_click_dict = defaultdict(list)  # user_id -> [news_ids]
        self.news_user_dict = defaultdict(list)   # news_id -> [user_ids]
        
    def build_graph(self, behaviors):
        """
        Section 3.3에서 설명된 이분 그래프 구축
        
        사용자의 뉴스 클릭 행동을 바탕으로 이분 그래프를 구성합니다.
        논문의 Figure 1에서 보여주는 것처럼, 사용자와 뉴스는
        클릭 행동을 통해 연결됩니다.
        
        Args:
            behaviors: (user_id, news_id) 튜플의 리스트
        """
        for user_id, news_id in behaviors:
            self.user_click_dict[user_id].append(news_id)
            self.news_user_dict[news_id].append(user_id)
            
    def sample_news_neighbors(self, news_id):
        """
        Section 3.3의 Two-hop 관계에서 뉴스의 이웃 뉴스 샘플링
        
        예를 들어, 논문의 Figure 1에서 n1과 n5는
        같은 사용자 u2가 클릭했기 때문에 이웃 관계입니다.
        
        Args:
            news_id: 대상 뉴스 ID
            
        Returns:
            이웃 뉴스 ID 리스트 (최대 max_neighbors개)
        """
        neighbor_news = []
        users = self.news_user_dict[news_id]
        
        # Two-hop 이웃 수집: 같은 사용자가 클릭한 다른 뉴스
        for user in users:
            neighbor_news.extend(self.user_click_dict[user])
            
        # 자기 자신 제거
        neighbor_news = [n for n in neighbor_news if n != news_id]
        
        # Section 4.5에 따라 빈도수 기반으로 상위 K개 선택
        if len(neighbor_news) > self.max_neighbors:
            news_freq = defaultdict(int)
            for n in neighbor_news:
                news_freq[n] += 1
            sorted_news = sorted(news_freq.items(), key=lambda x: x[1], reverse=True)
            neighbor_news = [n[0] for n in sorted_news[:self.max_neighbors]]
        
        # Padding 추가 (그래프 어텐션을 위한 고정 길이 유지)
        while len(neighbor_news) < self.max_neighbors:
            neighbor_news.append(0)  # 0은 패딩을 의미
            
        return neighbor_news[:self.max_neighbors]
        
    def sample_user_neighbors(self, user_id):
        """
        Section 3.3의 Two-hop 관계에서 사용자의 이웃 사용자 샘플링
        
        예를 들어, 논문의 Figure 1에서 u1과 u2는
        같은 뉴스 n1을 클릭했기 때문에 이웃 관계입니다.
        
        Args:
            user_id: 대상 사용자 ID
            
        Returns:
            이웃 사용자 ID 리스트 (최대 max_neighbors개)
        """
        neighbor_users = []
        news_clicked = self.user_click_dict[user_id]
        
        # Two-hop 이웃 수집: 같은 뉴스를 클릭한 다른 사용자
        for news in news_clicked:
            neighbor_users.extend(self.news_user_dict[news])
            
        # 자기 자신 제거
        neighbor_users = [u for u in neighbor_users if u != user_id]
        
        # Section 4.5에 따라 빈도수 기반으로 상위 K개 선택
        if len(neighbor_users) > self.max_neighbors:
            user_freq = defaultdict(int)
            for u in neighbor_users:
                user_freq[u] += 1
            sorted_users = sorted(user_freq.items(), key=lambda x: x[1], reverse=True)
            neighbor_users = [u[0] for u in sorted_users[:self.max_neighbors]]
            
        # Padding 추가 (그래프 어텐션을 위한 고정 길이 유지)
        while len(neighbor_users) < self.max_neighbors:
            neighbor_users.append(0)  # 0은 패딩을 의미
            
        return neighbor_users[:self.max_neighbors]
        
    def get_neighbor_mask(self, neighbors):
        """
        Section 3.3의 그래프 어텐션을 위한 마스크 생성
        
        패딩된 이웃은 어텐션 계산에서 제외되어야 하므로,
        유효한 이웃과 패딩을 구분하는 이진 마스크를 생성합니다.
        
        Args:
            neighbors: 이웃 ID 리스트
            
        Returns:
            이진 마스크 (유효한 이웃은 1, 패딩은 0)
        """
        return [1 if n != 0 else 0 for n in neighbors] 