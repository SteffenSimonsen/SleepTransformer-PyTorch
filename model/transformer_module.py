import torch
import torch.nn as nn
from modules import MultiHeadAttention, PositionWiseFeedforward, LayerNorm, PositionalEncoding, Attention
from typing import Optional

class TransformerEncoder(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout_rate: float = 0.1):
        super(TransformerEncoder, self).__init__()

        self.multi_head_attention = MultiHeadAttention(num_heads, d_model, dropout_rate)
        self.positionwise_feedforward = PositionWiseFeedforward(d_model, d_ff, dropout_rate)

        self.layer_norm1 = LayerNorm(d_model)
        self.layer_norm2 = LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x: torch.Tensor, return_attention: bool = False) -> torch.Tensor:

        if return_attention:
            mha_output, attention = self.multi_head_attention(x, x, x, return_attention=True)# multi head attention with attention weights returned for vizualization
        else:
            mha_output = self.multi_head_attention(x, x, x)
        
        x = x + self.dropout(mha_output) # residual connection and dropout
        x = self.layer_norm1(x) # layer normalization

        ffn_output = self.positionwise_feedforward(x) # feed forward network
        x = x + self.dropout(ffn_output) # residual connection and dropout
        x = self.layer_norm2(x) # layer normalization

        if return_attention:
            return x, attention
        else:   
            return x  
    

class TransformerHeap(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, num_layers: int, dropout_rate: float = 0.1):
        super(TransformerHeap, self).__init__()

        self.positional_encoding = PositionalEncoding(d_model) # positional encoding

        self.layers = nn.ModuleList([
            TransformerEncoder(d_model, num_heads, d_ff, dropout_rate) #stack multiple transformer encoder layers into heap of num_layers
            for _ in range(num_layers)
        ])

    def forward(self, x: torch.Tensor, return_attention: bool = False) -> torch.Tensor:
        x = self.positional_encoding(x)
        for i, layer in enumerate(self.layers):
            if return_attention and i == len(self.layers) - 1:  # Only for the last layer
                x, attention_weights = layer(x, return_attention=True)
            else:
                x = layer(x)
        
        if return_attention:
            return x, attention_weights
        return x
    
    
