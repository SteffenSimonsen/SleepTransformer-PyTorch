import torch
from torch.utils.data import Dataset
from split import Split
from samplers import RandomSampler, DetermSampler
import numpy as np
import os

class SleepDatasetSplit(Dataset):
    def __init__(self, split_file=None, data_path=None, split_type="train", seq_length=21, use_virtual_epochs=False, steps_per_epoch=883):
        self.seq_length = seq_length
        
        if split_file is not None:
            if not os.path.exists(split_file):
                raise FileNotFoundError(f"Split file not found: {split_file}")
            self.split = Split.file(split_file)
        elif data_path is not None:
            self.split = Split.random(data_path, "random_split", (0.8, 0.1, 0.1))
        else:
            raise ValueError("Either split_file or data_path must be provided")
        
        # configure samplers based on split type
        if split_type == "train":
            # random sampler for training
            num_iterations = steps_per_epoch if use_virtual_epochs else 883
            
            self.sampler = RandomSampler(
                split_data=self.split,
                split_type=split_type,
                num_epochs=seq_length,
                num_iterations=num_iterations
            )

        elif split_type == "val":
            # deterministic sampler for validation
            self.sampler = DetermSampler(
                split_data=self.split,
                split_type=split_type,
                subject_percentage=1.0  
            )

        elif split_type == "test":
            # deterministic sampler for testing, getting all channels
            self.sampler = DetermSampler(
                split_data=self.split,
                split_type=split_type,
                subject_percentage=1.0,
                get_all_channels=True  
            )
        else:
            raise ValueError(f"Invalid split_type: {split_type}")

        
        self.is_test = split_type == "test"
    
    def __len__(self):
        return self.sampler.num_samples

    def __getitem__(self, idx):
        sample = self.sampler.get_sample(idx)

        metadata = {
        'dataset': sample.tag.dataset,
        'subject': sample.tag.subject,
        'record': sample.tag.record
        }

        

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
        
            return data, labels, metadata
        else:
        
            # keep channels separate - no stacking
            # used for testing, all channels are used
            # tensor is manipulated in the training step
            eeg = torch.as_tensor(sample.eeg).float()  # [num_eeg_channels, 21, 29, 129]
            eog = torch.as_tensor(sample.eog).float()  # [num_eog_channels, 21, 29, 129]
            labels = torch.as_tensor(sample.labels).long()
            
            return (eeg, eog), labels, metadata
    
    def save_split(self, path, name):
        """Save the current split configuration to a JSON file"""
        os.makedirs(path, exist_ok=True)
        full_path = os.path.join(path, f"{name}.json")
        self.split.dump_file(path, name)
        return full_path

