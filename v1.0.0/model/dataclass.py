import torch
from torch.utils.data import Dataset, DataLoader
import h5py
import numpy as np

class SleepDatasetHDF5(Dataset):
    def __init__(self, hdf5_path, config, subject_ids=None, transform=None):
        """
        Initialize the sleep dataset with configuration parameters.
        
        Args:
            hdf5_path (str): Path to the HDF5 data file
            config (SleepTransformerConfig): Configuration object containing model parameters
            subject_ids (list, optional): List of subject IDs to include. Defaults to None (all subjects).
            transform (callable, optional): Optional transform to be applied to the data
        """
        self.hdf5_path = hdf5_path
        self.transform = transform
        self.config = config
        
        # Get sequence length from config
        self.seq_length = config.epoch_seq_len
        
        with h5py.File(self.hdf5_path, 'r') as hf:
            if subject_ids is None:
                self.subject_ids = [key for key in hf.keys() if key != 'subject_ids']
            else:
                self.subject_ids = list(subject_ids)  # Convert to list if it's a Subset

        # Build data index
        self.data_info = []
        with h5py.File(self.hdf5_path, 'r') as hf:
            for subject_id in self.subject_ids:
                num_epochs = len(hf[subject_id]['spectrograms']) - self.seq_length + 1
                for i in range(num_epochs):
                    self.data_info.append((subject_id, i))
    
    def __len__(self):
        return len(self.data_info)

    def __getitem__(self, idx):
        subject_id, start_idx = self.data_info[idx]
        
        with h5py.File(self.hdf5_path, 'r') as hf:
            # Load sequence of spectrograms and labels
            data = hf[subject_id]['spectrograms'][start_idx:start_idx+self.seq_length]
            labels = hf[subject_id]['stages'][start_idx:start_idx+self.seq_length]
        
        # Convert to torch tensors
        data = torch.from_numpy(data).float()
        labels = torch.from_numpy(labels).long()
        
        # Add or adjust channel dimension to match config.input_channels
        if data.shape[-1] != self.config.input_channels:
            if len(data.shape) == 3:  # If no channel dimension
                data = data.unsqueeze(-1)
            # Repeat the channel if needed to match desired number of channels
            if data.shape[-1] == 1 and self.config.input_channels > 1:
                data = data.repeat(1, 1, 1, self.config.input_channels)
            elif data.shape[-1] != self.config.input_channels:
                raise ValueError(f"Data has {data.shape[-1]} channels but config specifies {self.config.input_channels} channels")
        
        if self.transform:
            data = self.transform(data)
        
        # Convert one-hot encoded labels to class indices
        labels = labels.argmax(dim=-1)

        return data, labels

def inspect_dataset(dataset, num_samples=5):
    """
    Inspect the dataset by printing information about a few samples.
    
    Args:
        dataset (SleepDatasetHDF5): Dataset to inspect
        num_samples (int): Number of samples to inspect
    """
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    
    for i, (data, labels) in enumerate(dataloader):
        print(f"Sample {i+1}:")
        print(f"  Data shape: {data.shape}")
        print(f"  Labels shape: {labels.shape}")
        print(f"  Data type: {data.dtype}")
        print(f"  Labels type: {labels.dtype}")
        print(f"  Data range: [{data.min().item():.2f}, {data.max().item():.2f}]")
        print(f"  Unique labels: {torch.unique(labels).tolist()}")
        print()
        
        if i+1 >= num_samples:
            break

# Example usage:
if __name__ == "__main__":
    from config import SleepTransformerConfig
    
    # Create configuration
    config = SleepTransformerConfig()
    
    # Create the dataset with config
    hdf5_path = '/home/steff/SleepTransformer/model/data/all_subjects.h5'
    dataset = SleepDatasetHDF5(hdf5_path, config)
    
    # Inspect the dataset
    print("Inspecting dataset:")
    inspect_dataset(dataset)
    
    # Get a single sample
    single_data, single_labels = dataset[0]
    print("\nSingle sample inspection:")
    print(f"  Data shape: {single_data.shape}")
    print(f"  Labels shape: {single_labels.shape}")
    
    # Create a DataLoader and inspect a batch
    batch_size = 32
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    batch_data, batch_labels = next(iter(dataloader))
    print("\nBatch inspection:")
    print(f"  Batch data shape: {batch_data.shape}")
    print(f"  Batch labels shape: {batch_labels.shape}")
    print(f"  Batch data: {batch_labels}")
