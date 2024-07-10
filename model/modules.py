import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class LayerNorm(nn.Module):
    """Layer normalization module.
       https://arxiv.org/abs/1607.06450
    """
    def __init__(self, features, eps=1e-8) -> None:
        super(LayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(features))
        self.beta = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        var = x.var(-1, keepdim=True)
        return self.gamma * (x - mean) / torch.sqrt(var + self.eps) + self.beta
    
class ScaledDotProductAttention(nn.Module):
    def __init__(self, dropout_rate=0.):
        super(ScaledDotProductAttention, self).__init__()
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, Q, K, V):
        """
        Compute scaled dot-product attention
        
        Args:
        Q: Queries tensor with shape [N, T_q, d_k]
        K: Keys tensor with shape [N, T_k, d_k]
        V: Values tensor with shape [N, T_k, d_v]
        
        Where:
        N is the batch size
        T_q is the length of the query sequence
        T_k is the length of the key/value sequence
        d_k is the dimension of the keys (and queries)
        d_v is the dimension of the values
        
        Returns:
        outputs: Attention outputs with shape [N, T_q, d_v]
        attention: Attention weights with shape [N, T_q, T_k]
        """
        # Get the dimension of the keys
        d_k = K.size(-1)
        
        # Dot product between Q and K
        # (N, T_q, d_k) x (N, d_k, T_k) -> (N, T_q, T_k)
        scores = torch.matmul(Q, K.transpose(-2, -1))
        
        # Scale the scores
        scores = scores / torch.sqrt(torch.tensor(d_k, dtype=torch.float32))
        
        # Apply softmax to get attention weights
        # (N, T_q, T_k)
        attention = F.softmax(scores, dim=-1)
        
        # Apply dropout to attention weights
        attention = self.dropout(attention)
        
        # Multiply attention weights with values
        # (N, T_q, T_k) x (N, T_k, d_v) -> (N, T_q, d_v)
        outputs = torch.matmul(attention, V)
        
        return outputs, attention



class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads=8, dropout_rate=0.):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.W_Q = nn.Linear(d_model, d_model, bias=True)
        self.W_K = nn.Linear(d_model, d_model, bias=True)
        self.W_V = nn.Linear(d_model, d_model, bias=True)
        self.W_O = nn.Linear(d_model, d_model, bias=True)
        
        self.attention = ScaledDotProductAttention(dropout_rate)
        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout_rate)

    def split_heads(self, x):
        batch_size, seq_length, _ = x.size()
        return x.view(batch_size, seq_length, self.num_heads, self.d_k).transpose(1, 2)

    def combine_heads(self, x):
        batch_size, _, seq_length, _ = x.size()
        return x.transpose(1, 2).contiguous().view(batch_size, seq_length, self.d_model)

    def forward(self, queries, keys, values):
        # Linear projections
        Q = self.W_Q(queries)
        K = self.W_K(keys)
        V = self.W_V(values)
        
        # Split heads
        Q = self.split_heads(Q)
        K = self.split_heads(K)
        V = self.split_heads(V)
        
        # Scaled Dot-Product Attention
        outputs, attention = self.attention(Q, K, V)
        
        # Combine heads
        outputs = self.combine_heads(outputs)
        
        # Final linear projection
        outputs = self.W_O(outputs)
        
        # Residual connection and layer normalization
        outputs = self.layer_norm(outputs + queries)
        
        return outputs, attention
