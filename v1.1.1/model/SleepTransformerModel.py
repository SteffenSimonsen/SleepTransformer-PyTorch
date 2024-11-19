import torch
import torch.nn as nn
from torch.nn import functional as F
from modules import Attention
from transformer_module import TransformerHeap
from SpectrogramCondenser import SpectrogramCondenser

class SleepTransformer(nn.Module):
    """Sleep Transformer model.
    """
    def __init__(self, config):
        super(SleepTransformer, self).__init__()
        self.config = config
        self.return_attention = config.return_attention_weights

        
        self.condenser = SpectrogramCondenser( # add the SpectrogramCondenser 
            in_channels=config.input_channels, 
            in_freq=config.input_freq,
            out_freq=config.d_model
        )
        
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
        self.fc2 = nn.Linear(config.fc_hidden_size, config.fc_hidden_size)
        self.output = nn.Linear(config.fc_hidden_size, config.num_classes) 

    def forward(self, x, return_attention=False):

        #print("Input spectrogram", x.shape)

        # x: (batch_size, epoch_seq_len, frame_seq_len, input_freq, input_channels)
        x = self.condenser(x)
        # x: (batch_size, epoch_seq_len, frame_seq_len, d_model, 1)

        #print("Condensed spectrogram", x.shape)
        
        
        batch_size, epoch_seq_len, frame_seq_len, ndim, nchannel = x.shape # x: (batch_size, epoch_seq_len, frame_seq_len, ndim, nchannel)
        
        # reshape for epoch transformer
        # combine batch_size and epoch_seq_len, and flatten ndim and nchannel
        x = x.view(-1, frame_seq_len, ndim * nchannel) #  x : (batch_size * epoch_seq_len, frame_seq_len, ndim * nchannel)
        
        
        #epoch transformer
        if self.return_attention:
            epoch_output, epoch_attention_weights = self.epoch_transformer(x, return_attention=True)
        else:
            epoch_output = self.epoch_transformer(x) # epoch_output : (batch_size * epoch_seq_len, frame_seq_len, d_model)
        
    
        # attention mechanism 
        # Todo : implement return of attention weights throughout the model
        # this is only for the epoch level context vector attention
        if self.return_attention:
            epoch_context, context_attention_weights = self.epoch_attention(epoch_output, return_alphas=True) # epoch_context : (batch_size * epoch_seq_len, d_model)
        else:
            epoch_context = self.epoch_attention(epoch_output)
       
        
        # reshape for sequence transformer
        sequence_input = epoch_context.view(batch_size, epoch_seq_len, -1) # sequence_input : (batch_size, epoch_seq_len, d_model)
        
        
        # sequence transformer
        if self.return_attention:
            sequence_output, sequence_attention_weights = self.sequence_transformer(sequence_input, return_attention=True)
        else:
            sequence_output = self.sequence_transformer(sequence_input) # sequence_output : (batch_size, epoch_seq_len, d_model)
      
        
        #  FC layers
        x = F.relu(self.fc1(sequence_output)) # x : (batch_size, epoch_seq_len, fc_hidden_size)

        x = F.relu(self.fc2(x)) # x : (batch_size, epoch_seq_len, fc_hidden_size)
        
        logits = self.output(x) # logits : (batch_size, epoch_seq_len, num_classes)
        
        
        # softmax
        #output = F.softmax(logits, dim=-1) # output : (batch_size, epoch_seq_len, num_classes)
        
        
        # if self.return_attention:
        #     return output, {
        #         'epoch_transformer': epoch_attention_weights,
        #         'epoch_attention': context_attention_weights,
        #         'sequence_transformer': sequence_attention_weights
        #     }
        # else:
        #     return output

        if self.return_attention:
            return logits, {
                'epoch_transformer': epoch_attention_weights,
                'epoch_attention': context_attention_weights,
                'sequence_transformer': sequence_attention_weights
            }
        else:
            return logits
    
    
    def count_parameters(self):
        """Count and print the number of parameters in each component and total."""
        total_params = 0
        for name, module in self.named_children():
            params = sum(p.numel() for p in module.parameters() if p.requires_grad)
            print(f"{name}: {params:,} parameters")
            total_params += params
        print(f"Total: {total_params:,} parameters")
        return total_params
