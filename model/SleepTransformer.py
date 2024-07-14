import torch 
import torch.nn as nn
from modules import Attention
from transformer_module import TransformerHeap

class SleepTransformer(nn.Module):
    """Sleep Transformer model.
    """
    def __init__(self, config):
        super(SleepTransformer, self).__init__()
        self.config = config
        
        self.epoch_transformer = TransformerHeap( # epoch-level transformer
            d_model=config.epoch_d_model,
            num_heads=config.epoch_num_heads,
            d_ff=config.epoch_d_ff,
            num_layers=config.epoch_num_layers,
            dropout_rate=config.dropout_rate
        )
        
        self.epoch_attention = Attention(config.epoch_d_model, config.epoch_d_attention) # Attention module
        
        self.sequence_transformer = TransformerHeap( # sequence-level transformer
            d_model=config.seq_d_model,
            num_heads=config.seq_num_heads,
            d_ff=config.seq_d_ff,
            num_layers=config.seq_num_layers,
            dropout_rate=config.dropout_rate
        )
        
        # fc layers
        self.fc1 = nn.Linear(config.seq_d_model, config.fc_hidden_size)
        self.fc2 = nn.Linear(config.fc_hidden_size, config.fc_hidden_size)
        
        # Final classification layer
        self.classifier = nn.Linear(config.fc_hidden_size, config.num_classes)
