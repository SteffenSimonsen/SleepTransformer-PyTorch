import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class LayerNorm(nn.Module):
    """Layer normalization module.
       https://arxiv.org/abs/1607.06450
    """
    def __init__(self, features, eps=1e-8):
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
        d_k = K.size(-1)  # Dimension of the key vectors
        
        # Compute the scaled dot-product attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(d_k, dtype=torch.float32))
        
        # Apply softmax to get the attention weights
        attention = F.softmax(scores, dim=-1)
        
        # Apply dropout to the attention weights
        attention = self.dropout(attention)
        
        # Compute the final output by multiplying the attention weights with the value matrix
        outputs = torch.matmul(attention, V)
        
        return outputs, attention



class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, d_model, dropout_rate=0.):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        self.depth = d_model // num_heads

        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.WQ = nn.Linear(d_model, d_model)
        self.WK = nn.Linear(d_model, d_model)
        self.WV = nn.Linear(d_model, d_model)
        self.WO = nn.Linear(d_model, d_model)
        
        self.attention = ScaledDotProductAttention(dropout_rate)

    def split_heads(self, x, batch_size):
        # Split the last dimension into (num_heads, depth)
        x = x.view(batch_size, -1, self.num_heads, self.depth)
        # Transpose the result to (batch_size, num_heads, seq_length, depth)
        return x.transpose(1, 2)

    def forward(self, Q, K, V):
        batch_size = Q.size(0)

        # Linear projections and split into heads
        Q = self.split_heads(self.WQ(Q), batch_size)
        K = self.split_heads(self.WK(K), batch_size)
        V = self.split_heads(self.WV(V), batch_size)

        # Scaled dot-product attention
        scaled_attention, attention_weights = self.attention(Q, K, V)

        # Transpose and concatenate heads
        scaled_attention = scaled_attention.transpose(1, 2).contiguous()
        scaled_attention = scaled_attention.view(batch_size, -1, self.d_model)

        # Final linear projection
        output = self.WO(scaled_attention)
        return output, attention_weights
    

class PositionWiseFeedforward(nn.Module):
    def __init__(self, d_model, d_ff, dropout_rate=0.1):
        super(PositionWiseFeedforward, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout_rate)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, x):
        
        
        # First linear layer
        out = self.linear1(x)
        
        # ReLU activation
        out = self.relu(out)
        
        # Second linear layer
        out = self.linear2(out)
        
        # Adding residual connection
        out += x
        
        # Layer normalization
        out = self.layer_norm(out)

        # Dropout
        out = self.dropout(out)
        
        return out

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        
        # Create a long enough P matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        
        x = x + self.pe[:x.size(0), :]

        return x
