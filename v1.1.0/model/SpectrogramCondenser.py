import torch
import torch.nn as nn
class SpectrogramCondenser(nn.Module):
    def __init__(self, in_channels, in_freq, out_freq=128):
        super(SpectrogramCondenser, self).__init__()
        self.in_channels = in_channels
        self.in_freq = in_freq
        self.out_freq = out_freq
        
        if in_freq != out_freq:
            # Calculate the frequency dimension for each channel
            self.channel_freq = out_freq // in_channels
            self.padding = out_freq % in_channels
            
            # Create a linear layer for each channel
            self.channel_layers = nn.ModuleList([
                nn.Linear(in_freq, self.channel_freq)
                for _ in range(in_channels)
            ])
        else:
            self.channel_layers = None
        
    def forward(self, x):
        # x shape: (batch_size, epoch_seq_len, frame_seq_len, in_freq, in_channels)
        if self.channel_layers is None:
            return x  # If in_freq == out_freq, no need to condense
        
        batch_size, epoch_seq_len, frame_seq_len, _, _ = x.shape
        
        # Process each channel
        processed_channels = []
        for i in range(self.in_channels):
            channel = x[..., i]
            processed = self.channel_layers[i](channel)
            processed_channels.append(processed)
        
        # Stack the processed channels
        output = torch.cat(processed_channels, dim=-1)
    
        # Add padding if necessary
        if self.padding > 0:
            padding = torch.zeros(batch_size, epoch_seq_len, frame_seq_len, self.padding, device=x.device)
            output = torch.cat([output, padding], dim=-1)
        
        # Reshape to add the channel dimension
        output = output.unsqueeze(-1)
        return output
