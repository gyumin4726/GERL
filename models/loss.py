"""
Section 3.4 "Recommendation and Model Training"에서 제안된
손실 함수 구현

이 모듈은 뉴스 추천을 위한 학습 목적 함수를 구현합니다.
클릭 예측 문제를 pseudo λ + 1-way 분류 작업으로 공식화하여,
클릭된 뉴스(양성)와 클릭되지 않은 뉴스(음성) 간의 구분을 학습합니다.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class NewsClickPredictionLoss(nn.Module):
    def __init__(self, config):
        """
        Section 3.4에서 설명된 손실 함수 초기화
        
        Args:
            config: 모델 설정으로 다음을 포함:
                - negative_samples: λ 값 (논문에서 4로 설정)
        """
        super().__init__()
        self.negative_samples = config.negative_samples
        
    def forward(self, pos_scores, neg_scores):
        """
        Section 3.4의 수식 (5)에서 제안된 손실 함수 계산
        
        클릭된 뉴스의 예측 점수와 클릭되지 않은 뉴스들의 예측 점수를 받아
        negative log likelihood를 최소화합니다.
        
        Args:
            pos_scores: 클릭된 뉴스의 예측 점수 [batch_size, 1]
            neg_scores: 클릭되지 않은 뉴스의 예측 점수 [batch_size, negative_samples]
        """
        # 모든 점수를 하나의 텐서로 결합
        all_scores = torch.cat([pos_scores.unsqueeze(1), neg_scores], dim=1)
        
        # 클릭된 뉴스(양성)의 로그 우도 계산
        pos_prob = torch.exp(pos_scores) / torch.sum(torch.exp(all_scores), dim=1)
        
        # Negative log likelihood 최소화
        loss = -torch.mean(torch.log(pos_prob))
        
        return loss

class MarginRankingLoss(nn.Module):
    def __init__(self, margin=0.1):
        """
        Section 3.4의 보조 손실 함수로 사용될 수 있는
        마진 기반 랭킹 손실 구현
        
        Args:
            margin: 양성과 음성 샘플 간의 최소 마진
        """
        super().__init__()
        self.margin = margin
        
    def forward(self, pos_scores, neg_scores):
        """
        양성 샘플이 음성 샘플보다 margin 이상 높은 점수를 갖도록 학습
        
        Args:
            pos_scores: 클릭된 뉴스의 예측 점수 [batch_size, 1]
            neg_scores: 클릭되지 않은 뉴스의 예측 점수 [batch_size, negative_samples]
        """
        # 각 음성 샘플에 대해 마진 손실 계산
        loss = torch.mean(torch.relu(self.margin - pos_scores.unsqueeze(1) + neg_scores))
        
        return loss 