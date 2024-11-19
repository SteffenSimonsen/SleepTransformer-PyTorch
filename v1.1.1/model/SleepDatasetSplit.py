import torch
from torch.utils.data import Dataset
from split import Split
from sampler import DeterministicSampler, RandomSampler, SamplerConfig, TestSampler
import numpy as np
import os

class SleepDatasetSplit(Dataset):
    def __init__(self, split_file=None, data_path=None, split_type="train", seq_length=21):
        self.seq_length = seq_length
        
        if split_file is not None:
            if not os.path.exists(split_file):
                raise FileNotFoundError(f"Split file not found: {split_file}")
            self.split = Split.file(split_file)
        elif data_path is not None:
            self.split = Split.random(data_path, "random_split", (0.8, 0.1, 0.1))
        else:
            raise ValueError("Either split_file or data_path must be provided")
        
        if split_type == "train":

            config = SamplerConfig(
                split_data=self.split,
                split_type=split_type,
                seq_length=seq_length
            )

            self.sampler = RandomSampler(config)

        elif split_type == "val":

            config = SamplerConfig(
                split_data=self.split,
                split_type=split_type,
                seq_length=seq_length
            )

            self.sampler = DeterministicSampler(config)

        elif split_type == "test":

            config = SamplerConfig(
                split_data=self.split,
                split_type=split_type,
                seq_length=seq_length,
                get_all_channels=True  # all channels for test set
            )

            self.sampler = TestSampler(config)

        else:
            raise ValueError(f"Invalid split_type: {split_type}")
        
        self.is_test = split_type == "test"
    
    def __len__(self):
        return self.sampler.num_samples

    def __getitem__(self, idx):
        sample = self.sampler.get_sample(idx)

        eeg = np.array(sample.eeg)
        eog = np.array(sample.eog)
        
        # convert to tensors
        
        eeg = torch.as_tensor(eeg).float()  # [1, 21, 29, 129]
        eog = torch.as_tensor(eog).float()  # [1, 21, 29, 129]
        labels = torch.as_tensor(sample.labels).long()  # [21]

        if not self.is_test:
            
            # move channel dimension (first) to last: [1, 21, 29, 129] -> [21, 29, 129, 1]
            eeg = eeg.permute(1, 2, 3, 0)
            eog = eog.permute(1, 2, 3, 0)

            # stack along the last dimension
            data = torch.cat([eeg, eog], dim=-1)  # [21, 29, 129, 2]
            
            if self.transform:
                data = self.transform(data)
        
            return data, labels
        else:
        
            # keep channels separate - no stacking
            # used for testing, all channels are used
            # tensor is manipulated in the training step
            eeg = torch.as_tensor(sample.eeg).float()  # [num_eeg_channels, 21, 29, 129]
            eog = torch.as_tensor(sample.eog).float()  # [num_eog_channels, 21, 29, 129]
            labels = torch.as_tensor(sample.labels).long()
            
            return (eeg, eog), labels
    
    def save_split(self, path, name):
        """Save the current split configuration to a JSON file"""
        os.makedirs(path, exist_ok=True)
        full_path = os.path.join(path, f"{name}.json")
        self.split.dump_file(path, name)
        return full_path

