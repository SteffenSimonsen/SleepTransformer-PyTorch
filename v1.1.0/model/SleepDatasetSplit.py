import torch
from torch.utils.data import Dataset, DataLoader
from split import Split
from sampler import Determ_Sampler, Random_Sampler
import os

class SleepDatasetSplit(Dataset):
    def __init__(self, split_file=None, data_path=None, split_type="train", seq_length=21, transform=None):
        self.transform = transform
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
            self.sampler = Random_Sampler(
                split_data=self.split,
                split_type=split_type,
                num_epochs=seq_length,
                num_iterations=100000
            )

            #self.sampler.print_report()
        else:
            self.sampler = Determ_Sampler(
                split_data=self.split,
                split_type=split_type
            )

            #self.sampler.print_report()
    
    def __len__(self):
        return self.sampler.num_samples

    def __getitem__(self, idx):
        sample = self.sampler.get_sample(idx)
        
        # Convert to tensors
        eeg = torch.as_tensor(sample.eeg).float()  # [1, 21, 29, 129]
        eog = torch.as_tensor(sample.eog).float()  # [1, 21, 29, 129]
        labels = torch.as_tensor(sample.labels).long()  # [21]
        
        # Move channel dimension (first) to last: [1, 21, 29, 129] -> [21, 29, 129, 1]
        eeg = eeg.permute(1, 2, 3, 0)
        eog = eog.permute(1, 2, 3, 0)
        
        # Stack EEG and EOG along the last dimension
        data = torch.cat([eeg, eog], dim=-1)  # [21, 29, 129, 2]
        
        if self.transform:
            data = self.transform(data)
        
        return data, labels
    
    def save_split(self, path, name):
        """Save the current split configuration to a JSON file"""
        os.makedirs(path, exist_ok=True)
        full_path = os.path.join(path, f"{name}.json")
        self.split.dump_file(path, name)
        return full_path

