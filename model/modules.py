import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple
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

    """Scaled dot-product attention mechanism.
    """
    def __init__(self, dropout_rate=0.):
        super(ScaledDotProductAttention, self).__init__()
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, Q, K, V):
        d_k = K.size(-1)  # dimension of the keys
        
        # scaled dot-product attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(d_k, dtype=torch.float32)) #maybe add small epsilon here
        
        # softmax produces attention weights
        attention_weights = F.softmax(scores, dim=-1)
        
        # dropout to the attention weights (maybe not necessary here?)
        attention_weights = self.dropout(attention_weights)
        
        #  final output by multiplying the attention weights with the value matrix
        outputs = torch.matmul(attention_weights, V)
        
        return outputs, attention_weights



class MultiHeadAttention(nn.Module):
    """Multi-head attention module.
    """
    def __init__(self, num_heads, d_model, dropout_rate=0.):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        self.d_k = d_model // num_heads

        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        # separate projections for each head
        self.W_q = nn.ModuleList([nn.Linear(d_model, self.d_k) for _ in range(num_heads)])
        self.W_k = nn.ModuleList([nn.Linear(d_model, self.d_k) for _ in range(num_heads)])
        self.W_v = nn.ModuleList([nn.Linear(d_model, self.d_k) for _ in range(num_heads)])
        
        # output projection
        self.W_o = nn.Linear(d_model, d_model)
        
        self.attention = ScaledDotProductAttention(dropout_rate)


    def forward(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:

        
        # erform linear projections and split into heads
        Q_heads = [self.W_q[i](Q) for i in range(self.num_heads)]  #  [batch_size, seq_len, d_k]
        K_heads = [self.W_k[i](K) for i in range(self.num_heads)]
        V_heads = [self.W_v[i](V) for i in range(self.num_heads)]

        # scaled dot product attention to each head
        outputs = []
        attentions = []
        for Q_h, K_h, V_h in zip(Q_heads, K_heads, V_heads):
            output, attention = self.attention(Q_h, K_h, V_h)
            outputs.append(output) # [batch_size, seq_len, d_k]
            attentions.append(attention) # [batch_size, seq_len, seq_len]
        
        # concat outputsfrom  heads
        output = torch.cat(outputs, dim=-1)
        
        # output projection
        output = self.W_o(output)
        

        return output, attentions
    

class PositionWiseFeedforward(nn.Module):
    def __init__(self, d_model, d_ff, dropout_rate=0.1):
        super(PositionWiseFeedforward, self).__init__()
        
        # Two linear transformations with a ReLU activation in between
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
