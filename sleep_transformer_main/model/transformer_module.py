import torch
import torch.nn as nn
from modules import MultiHeadAttention, PositionWiseFeedforward, LayerNorm, PositionalEncoding, Attention, RMSNorm

class TransformerEncoder(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout_rate: float = 0.1, norm_type: str = "RMSNorm"):
        super(TransformerEncoder, self).__init__()

        #self.multi_head_attention = MultiHeadAttention(num_heads, d_model, dropout_rate) # My own implementation of multi head attention
        self.multi_head_attention = nn.MultiheadAttention(d_model, num_heads, dropout=dropout_rate, batch_first=True) # PyTorch's implementation of multi head attention
        self.positionwise_feedforward = PositionWiseFeedforward(d_model, d_ff, dropout_rate)

        
        if norm_type == 'RMSNorm':
            print("applying RMSNorm")
            self.norm1 = RMSNorm(d_model)
            self.norm2 = RMSNorm(d_model)
        elif norm_type == 'LayerNorm':  
            print("applying LayerNorm")
            self.norm1 = nn.LayerNorm(d_model)
            self.norm2 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x: torch.Tensor, return_attention: bool = False) -> torch.Tensor:
        # using my own implementation of multi head attention
        # if return_attention:
        #     mha_output, attention = self.multi_head_attention(x, x, x, return_attention=True)# multi head attention with attention weights returned for vizualization
        # else:
        #     mha_output = self.multi_head_attention(x, x, x)

        # using PyTorch's implementation of multi head attention
        if return_attention:
            mha_output, attention = self.multi_head_attention(x, x, x, need_weights=True, average_attn_weights=False)
        else:
            mha_output, _ = self.multi_head_attention(x, x, x, need_weights=False)
        
        x = x + self.dropout(mha_output) # residual connection and dropout
        x = self.norm1(x) # layer normalization

        ffn_output = self.positionwise_feedforward(x) # feed forward network
        x = x + self.dropout(ffn_output) # residual connection and dropout
        x = self.norm2(x) # layer normalization

        if return_attention:
            return x, attention
        else:   
            return x  
    

class TransformerHeap(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, num_layers: int, dropout_rate: float = 0.1, norm_type: str = 'RMSNorm'):
        super(TransformerHeap, self).__init__()

        self.positional_encoding = PositionalEncoding(d_model) # positional encoding

        self.layers = nn.ModuleList([
            TransformerEncoder(d_model, num_heads, d_ff, dropout_rate, norm_type) #stack multiple transformer encoder layers into heap of num_layers
            for _ in range(num_layers)
        ])

    def forward(self, x: torch.Tensor, return_attention: bool = False) -> torch.Tensor:
        x = self.positional_encoding(x)
        for i, layer in enumerate(self.layers):
            if return_attention and i == len(self.layers) - 1:  
                x, attention_weights = layer(x, return_attention=True)
            else:
                x = layer(x)
        
        if return_attention:
            return x, attention_weights
        return x
    
    
