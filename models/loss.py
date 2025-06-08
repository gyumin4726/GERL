import torch
import torch.nn as nn
import torch.nn.functional as F

class GERLLoss(nn.Module):
    def __init__(self, num_negatives):
        """Initialize GERL loss function
        
        Args:
            num_negatives (int): Number of negative samples (λ in paper)
        """
        super().__init__()
        self.num_negatives = num_negatives
    
    def forward(self, user_repr, news_repr, labels):
        """Compute loss according to Eq.(5) in paper
        
        Args:
            user_repr: User representations [batch_size, hidden_size]
            news_repr: News representations [batch_size, num_candidates, hidden_size]
            labels: Ground truth labels [batch_size, num_candidates]
            
        Returns:
            loss: Negative log-likelihood loss
        """
        # Compute prediction scores
        # [batch_size, num_candidates]
        scores = torch.bmm(user_repr.unsqueeze(1), news_repr.transpose(1, 2)).squeeze(1)
        
        # Split positive and negative samples
        pos_mask = (labels == 1)
        pos_scores = scores[pos_mask]  # [num_positives]
        
        # For each positive sample, randomly sample λ negative samples
        batch_size = user_repr.size(0)
        neg_candidates = scores[~pos_mask].view(batch_size, -1)  # [batch_size, num_neg_candidates]
        
        neg_samples = []
        for i in range(batch_size):
            if neg_candidates[i].size(0) >= self.num_negatives:
                # Randomly sample λ negatives without replacement
                idx = torch.randperm(neg_candidates[i].size(0))[:self.num_negatives]
                neg_samples.append(neg_candidates[i][idx])
            else:
                # If not enough negatives, sample with replacement
                idx = torch.randint(0, neg_candidates[i].size(0), (self.num_negatives,))
                neg_samples.append(neg_candidates[i][idx])
        
        neg_scores = torch.stack(neg_samples)  # [batch_size, num_negatives]
        
        # Compute loss according to Eq.(5)
        pos_exp = torch.exp(pos_scores)
        neg_exp = torch.exp(neg_scores).sum(dim=1)  # Sum over negative samples
        denominator = pos_exp + neg_exp
        
        loss = -torch.log(pos_exp / denominator).mean()
        
        return loss 