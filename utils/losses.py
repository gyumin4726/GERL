import torch
import torch.nn as nn

class GERLLoss(nn.Module):
    """Graph Enhanced Representation Learning Loss
    
    논문의 Equation (5)에 따른 Loss 계산:
    L = -∑ log(exp(y_i+) / (exp(y_i+) + ∑exp(y_i,j-)))
    """
    def __init__(self):
        super().__init__()
    
    def forward(self, scores: torch.Tensor, positive_mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            scores: 예측 점수
            positive_mask: 양성 샘플 마스크
            
        Returns:
            loss: 계산된 loss 값
        """
        positive_scores = scores[positive_mask]
        negative_scores = scores[~positive_mask].view(len(positive_scores), -1)
        
        exp_pos = torch.exp(positive_scores)
        exp_neg = torch.exp(negative_scores)
        
        denominator = exp_pos + torch.sum(exp_neg, dim=1)
        loss = -torch.mean(torch.log(exp_pos / denominator))
        
        return loss 