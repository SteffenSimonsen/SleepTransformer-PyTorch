class SleepTransformerConfig:
    def __init__(self):
        # Input and model dimensions
        self.input_channels = 1  # Single EEG channel
        self.epoch_seq_len = 21  # L = 21 in the paper
        self.frame_seq_len = 29  # T = 29 time frames
        self.d_model = 128  # F = 128 frequency bins

        # epoch ransformer parameters
        self.epoch_num_heads = 8  # H = 8 
        self.epoch_d_ff = 1024  
        self.epoch_num_layers = 4  # N_E = 4 

        # seq transformer parameters
        self.seq_num_heads = 8  # H = 8 
        self.seq_d_ff = 1024  
        self.seq_num_layers = 4  # N_S = 4 

        self.epoch_d_attention = 64

        self.fc_hidden_size = 1024  
        
        self.num_classes = 5  

        self.dropout_rate = 0.1  
        
        # return attention weights for visualization
        
        self.return_attention_weights = True

    def __str__(self):
        return "\n".join(f"{key}: {value}" for key, value in self.__dict__.items())
