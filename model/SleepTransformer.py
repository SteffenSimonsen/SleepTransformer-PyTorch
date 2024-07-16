import torch.nn as nn
from torch.nn import functional as F
from modules import Attention
from transformer_module import TransformerHeap

class SleepTransformer(nn.Module):
    """Sleep Transformer model.
    """
    def __init__(self, config):
        super(SleepTransformer, self).__init__()
        self.config = config
        
        self.epoch_transformer = TransformerHeap( # epoch-level transformer
            d_model=config.d_model,
            num_heads=config.epoch_num_heads,
            d_ff=config.epoch_d_ff,
            num_layers=config.epoch_num_layers,
            dropout_rate=config.dropout_rate
        )
        
        self.epoch_attention = Attention(config.d_model, config.epoch_d_attention) # Attention module
        
        self.sequence_transformer = TransformerHeap( # sequence-level transformer
            d_model=config.d_model,
            num_heads=config.seq_num_heads,
            d_ff=config.seq_d_ff,
            num_layers=config.seq_num_layers,
            dropout_rate=config.dropout_rate
        )
        
        
        self.fc1 = nn.Linear(config.d_model, config.fc_hidden_size) 
        self.fc2 = nn.Linear(config.fc_hidden_size, config.num_classes) 

    def forward(self, x, return_attention=False):
        
        batch_size, epoch_seq_len, frame_seq_len, ndim, nchannel = x.shape # x: (batch_size, epoch_seq_len, frame_seq_len, ndim, nchannel)
        
        # reshape for epoch transformer
        # combine batch_size and epoch_seq_len, and flatten ndim and nchannel
        x = x.view(-1, frame_seq_len, ndim * nchannel) #  x : (batch_size * epoch_seq_len, frame_seq_len, ndim * nchannel)
        
        
        #epoch transformer
        epoch_output = self.epoch_transformer(x) # epoch_output : (batch_size * epoch_seq_len, frame_seq_len, d_model)
        
    
        # attention mechanism 
        # Todo : implement return of attention weights throughout the model
        # this is only for the epoch level context vector attention
        if return_attention:
            epoch_context, attention_weights = self.epoch_attention(epoch_output, return_alphas=True) # epoch_context : (batch_size * epoch_seq_len, d_model)
        else:
            epoch_context = self.epoch_attention(epoch_output)
       
        
        # reshape for sequence transformer
        sequence_input = epoch_context.view(batch_size, epoch_seq_len, -1) # sequence_input : (batch_size, epoch_seq_len, d_model)
        
        
        # sequence transformer
        sequence_output = self.sequence_transformer(sequence_input) # sequence_output : (batch_size, epoch_seq_len, d_model)
      
        
        #  FC layers
        x = F.relu(self.fc1(sequence_output)) # x : (batch_size, epoch_seq_len, fc_hidden_size)
        
        logits = self.fc2(x) # logits : (batch_size, epoch_seq_len, num_classes)
        
        
        # softmax
        output = F.softmax(logits, dim=-1) # output : (batch_size, epoch_seq_len, num_classes)
        
        
        return output
    
    
    def count_parameters(self):
        """Count and print the number of parameters in each component and total."""
        total_params = 0
        for name, module in self.named_children():
            params = sum(p.numel() for p in module.parameters() if p.requires_grad)
            print(f"{name}: {params:,} parameters")
            total_params += params
        print(f"Total: {total_params:,} parameters")
        return total_params
