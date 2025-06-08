import torch
import numpy as np
from typing import Dict

class RecommendationMetrics:
    """뉴스 추천 시스템을 위한 평가 메트릭
    
    구현된 메트릭:
    - AUC: Area Under the ROC Curve
    - MRR: Mean Reciprocal Rank
    - nDCG@k: Normalized Discounted Cumulative Gain at k
    """
    
    @staticmethod
    def calculate_metrics(scores: torch.Tensor, labels: torch.Tensor) -> Dict[str, float]:
        """모든 평가 메트릭 계산
        
        Args:
            scores: 예측 점수
            labels: 실제 레이블
            
        Returns:
            metrics: 계산된 메트릭 딕셔너리
        """
        metrics = RecommendationMetrics()
        return {
            'auc': metrics._calculate_auc(scores, labels),
            'mrr': metrics._calculate_mrr(scores, labels),
            'ndcg@5': metrics._calculate_ndcg(scores, labels, k=5),
            'ndcg@10': metrics._calculate_ndcg(scores, labels, k=10)
        }
    
    @staticmethod
    def _calculate_auc(scores: torch.Tensor, labels: torch.Tensor) -> float:
        """AUC 계산
        
        Args:
            scores: 예측 점수
            labels: 실제 레이블
            
        Returns:
            float: AUC 점수
        """
        scores = scores.detach().cpu().numpy()
        labels = labels.detach().cpu().numpy()
        return float(np.mean([
            1 if s_pos > s_neg else 0
            for s_pos, s_neg in zip(scores[labels == 1], scores[labels == 0])
        ]))
    
    @staticmethod
    def _calculate_mrr(scores: torch.Tensor, labels: torch.Tensor) -> float:
        """MRR (Mean Reciprocal Rank) 계산
        
        Args:
            scores: 예측 점수
            labels: 실제 레이블
            
        Returns:
            float: MRR 점수
        """
        scores = scores.detach().cpu().numpy()
        labels = labels.detach().cpu().numpy()
        ranks = np.argsort(-scores)
        pos_ranks = np.where(labels[ranks] == 1)[0] + 1
        return float(np.mean(1.0 / pos_ranks))
    
    @staticmethod
    def _calculate_ndcg(scores: torch.Tensor, labels: torch.Tensor, k: int) -> float:
        """nDCG (normalized Discounted Cumulative Gain) 계산
        
        Args:
            scores: 예측 점수
            labels: 실제 레이블
            k: 상위 k개 항목 고려
            
        Returns:
            float: nDCG@k 점수
        """
        scores = scores.detach().cpu().numpy()
        labels = labels.detach().cpu().numpy()
        
        # ideal DCG
        idcg = np.sum([1.0 / np.log2(i + 2) for i in range(min(k, np.sum(labels == 1)))])
        
        # actual DCG
        ranks = np.argsort(-scores)[:k]
        dcg = np.sum([
            (1.0 / np.log2(i + 2)) * labels[rank]
            for i, rank in enumerate(ranks)
        ])
        
        return float(dcg / idcg) if idcg > 0 else 0.0 