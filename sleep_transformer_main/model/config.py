class SleepTransformerConfig:
    def __init__(self):
        """Configuration class for the Sleep Transformer model. 
        Mostly based on https://arxiv.org/abs/2105.11043"""

        self.experiment_name = "testing"

        self.data_path = os.getenv('SLEEP_DATA_PATH', './data/')
        self.split_path = os.getenv('SLEEP_SPLIT_PATH', './splits/abc_split.json')

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
        self.epoch_d_attention = 1024

        self.fc_hidden_size = 1024  
        
        self.num_classes = 5  

        self.dropout_rate = 0.1  

        self.norm_type = "LayerNorm"

        # training parameters
        self.max_epochs = 1
        self.batch_size = 64
        self.learning_rate = 1e-3
        self.optimizer_beta1 = 0.9
        self.optimizer_beta2 = 0.999
        self.optimizer_eps = 1e-7

        self.use_virtual_epochs = True
        self.steps_per_epoch = self.batch_size * 883  # number of training steps per virtual epoch
        
        
        # learning rate scheduler parameters
        self.scheduler_patience = 40
        self.scheduler_factor = 0.5
        self.scheduler_min_lr = 1e-6
        self.scheduler_monitor = 'validation_kappa'  

       
        # return attention weights for visualization
        
        self.return_attention_weights = False

    def __str__(self):
        return "\n".join(f"{key}: {value}" for key, value in self.__dict__.items())
