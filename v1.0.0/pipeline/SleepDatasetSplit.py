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
            print("Created new random split. Consider saving it with split.dump_file() for reproducibility")
        else:
            raise ValueError("Either split_file or data_path must be provided")
        
        if split_type == "train":
            self.sampler = Random_Sampler(
                split_data=self.split,
                split_type=split_type,
                num_epochs=seq_length
            )
        else:
            self.sampler = Determ_Sampler(
                split_data=self.split,
                split_type=split_type
            )
    
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

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    splits_dir = os.path.join(script_dir, "splits")
    os.makedirs(splits_dir, exist_ok=True)
    
    data_path = '/home/steff/SleepTransformer/pipeline/data'
    train_dataset = SleepDatasetSplit(
        data_path=data_path,
        split_type="train"
    )
    
    split_path = train_dataset.save_split(splits_dir, "experiment1_split")
    print(f"Split saved to: {split_path}")
    
    val_dataset = SleepDatasetSplit(
        split_file=split_path,
        split_type="val"
    )
    
    # Inspect a single sample
    data, labels = train_dataset[0]
    print(f"Data shape: {data.shape}")  # Should be [21, 29, 129, 2]
    print(f"Labels shape: {labels.shape}")  # Should be [21]
    
    # Create dataloader and inspect a batch
    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    batch_data, batch_labels = next(iter(val_loader))
    # print(f"Batch data shape: {batch_data.shape}")  # Should be [32, 21, 29, 129, 2]
    # print(f"Batch labels shape: {batch_labels.shape}")  # Should be [32, 21]
    # print(f"Validation Batch labels: {batch_labels}")  # Should be [32, 21]

    for i, (data, labels) in enumerate(val_loader):
        print(f"Batch {i}: {data.shape}, {labels.shape}"
              f"\n{labels}")

    # batch_data, batch_labels = next(iter(train_loader))
    # print(f"Batch data shape: {batch_data.shape}")  # Should be [32, 21, 29, 129, 2]
    # print(f"Batch labels shape: {batch_labels.shape}")  # Should be [32, 21]
    # print(f"Train Batch labels: {batch_labels}")  # Should be [32, 21]
