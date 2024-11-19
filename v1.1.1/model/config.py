class SleepTransformerConfig:
    def __init__(self):
        """Configuration class for the Sleep Transformer model. 
        Mostly based on https://arxiv.org/abs/2105.11043"""

        #spectrogram input frequency bins
        self.input_freq = 129 # 

        # input and model dimensions
        self.input_channels = 2  # Number of channels
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

        #various model parameters
        self.epoch_d_attention = 64

        self.fc_hidden_size = 1024  
        
        self.num_classes = 5  

        self.dropout_rate = 0.1  

        # training parameters
        self.max_epochs = 1
        self.batch_size = 32
        self.learning_rate = 1e-4
        self.optimizer_beta1 = 0.9
        self.optimizer_beta2 = 0.999
        self.optimizer_eps = 1e-7
        
        # learning rate scheduler parameters
        self.scheduler_patience = 40
        self.scheduler_factor = 0.001
        self.scheduler_min_lr = 1e-6
        self.scheduler_monitor = 'val_kappa'  # Metric to monitor for LR scheduling

        #metrics to monitor during training

        self.monitor_metrics = {
            'kappa': {'mode': 'max', 'metric': 'val_kappa'},
            'f1': {'mode': 'max', 'metric': 'val_f1'},
            'accuracy': {'mode': 'max', 'metric': 'val_acc'},
            'loss': {'mode': 'min', 'metric': 'val_loss'}
        }

        # return attention weights for visualization
        
        self.return_attention_weights = False

    def __str__(self):
        return "\n".join(f"{key}: {value}" for key, value in self.__dict__.items())
