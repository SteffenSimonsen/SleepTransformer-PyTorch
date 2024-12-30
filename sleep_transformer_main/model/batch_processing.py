import torch
from dataclasses import dataclass

@dataclass
class BatchedRecord:
    """Holds windowed sleep staging data and metadata"""
    data: torch.Tensor      # Shape: (n_windows, window_size, 29, 128, 2) 
    labels: torch.Tensor    # Shape: (n_windows, window_size)
    indices: torch.Tensor   # Start index of each window
    remaining_window: tuple[torch.Tensor, torch.Tensor] | None = None
    
    def has_remainder(self) -> bool:
        """Returns True if this record has a partial window that needs processing"""
        return self.remaining_window is not None
    
    def get_remainder(self) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.has_remainder():
            raise ValueError("No remaining window exists")
        return self.remaining_window

def create_sliding_windows(data: torch.Tensor, labels: torch.Tensor, 
                         window_size: int, step_size: int) -> BatchedRecord:
    """Creates sliding windows over sleep staging data"""

    # remove batch dimension since we're processing one record at a time
    data = data.squeeze(0)  
    labels = labels.squeeze(0)  

    num_epochs = data.shape[0]

    
    num_complete_windows = (num_epochs - window_size) // step_size + 1
    remaining_epochs = num_epochs % step_size
    
    
    batched_data = torch.zeros(num_complete_windows, window_size, *data.shape[1:], 
                              dtype=data.dtype, device=data.device)
    batched_labels = torch.zeros(num_complete_windows, window_size, 
                                dtype=torch.long, device=labels.device)
    indices = torch.zeros(num_complete_windows, dtype=torch.long, device=data.device)

    
    for i in range(num_complete_windows):
        start_idx = i * step_size
        end_idx = start_idx + window_size
        
        batched_data[i] = data[start_idx:end_idx]
        batched_labels[i] = labels[start_idx:end_idx]
        indices[i] = start_idx

    
    remaining_window = None
    if remaining_epochs > 0:
        start_idx = num_complete_windows * step_size
        remaining_window = (data[start_idx:], labels[start_idx:])
    
    return BatchedRecord(
        data=batched_data,
        labels=batched_labels,
        indices=indices,
        remaining_window=remaining_window
    )
