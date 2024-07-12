import torch
import torch.nn as nn
from modules import MultiHeadAttention, PositionWiseFeedforward, LayerNorm

class TransformerEncoder(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout_rate=0.1):
        super(TransformerEncoder, self).__init__()

        self.multi_head_attention = MultiHeadAttention(num_heads, d_model, dropout_rate)
        self.positionwise_feedforward = PositionWiseFeedforward(d_model, d_ff, dropout_rate)

        self.layer_norm1 = LayerNorm(d_model)
        self.layer_norm2 = LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):

        
        mha_output, _ = self.multi_head_attention(x, x, x) # multi head attention
        x = x + self.dropout(mha_output) # residual connection and dropout
        x = self.layer_norm1(x) # layer normalization

        ffn_output = self.positionwise_feedforward(x) # feed forward network
        x = x + self.dropout(ffn_output) # residual connection and dropout
        x = self.layer_norm2(x) # layer normalization

        return x
    
    

