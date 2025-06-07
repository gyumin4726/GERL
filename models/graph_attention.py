import torch
import torch.nn as nn
import torch.nn.functional as F

class GraphAttentionLayer(nn.Module):
    def __init__(self, in_features, out_features, dropout=0.1, alpha=0.2):
        super(GraphAttentionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.alpha = alpha
        
        # 변환 행렬
        self.W = nn.Linear(in_features, out_features, bias=False)
        
        # 어텐션 메커니즘
        self.a = nn.Linear(2 * out_features, 1, bias=False)
        
        # LeakyReLU
        self.leakyrelu = nn.LeakyReLU(self.alpha)
    
    def forward(self, node, neighbors, adj):
        """
        node: [batch_size, 1, in_features]
        neighbors: [batch_size, num_neighbors, in_features]
        adj: [batch_size, num_neighbors, num_neighbors]
        """
        batch_size = node.size(0)
        num_neighbors = neighbors.size(1)
        
        # 선형 변환
        Wh = self.W(torch.cat([node, neighbors], dim=1))  # [batch_size, 1+num_neighbors, out_features]
        
        # 어텐션 계산
        a_input = torch.cat([
            Wh[:, :1].repeat(1, num_neighbors, 1),  # [batch_size, num_neighbors, out_features]
            Wh[:, 1:]  # [batch_size, num_neighbors, out_features]
        ], dim=2)  # [batch_size, num_neighbors, 2*out_features]
        
        e = self.leakyrelu(self.a(a_input))  # [batch_size, num_neighbors, 1]
        
        # Masked Attention
        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        
        # 최종 출력
        h_prime = torch.bmm(attention.transpose(1, 2), Wh[:, 1:])  # [batch_size, 1, out_features]
        
        return h_prime 