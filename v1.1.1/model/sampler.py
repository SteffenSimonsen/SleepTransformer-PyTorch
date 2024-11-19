import torch
import os
import h5py
import random
import numpy as np
from dataclasses import dataclass
from abc import ABC, abstractmethod
from split import Split, Dataset_Split, ISample, ITag
from typing import Tuple, List, Dict



@dataclass
class SamplerConfig:
    """Configuration class for samplers"""
    split_data: Split
    split_type: str
    seq_length: int = 21
    subject_percentage: float = 1.0
    get_all_channels: bool = False


class BaseSampler(ABC):
    """Abstract base class for sleep data samplers"""
    def __init__(self, config: SamplerConfig):
        self.config = config
        self.split_type = config.split_type
        self.seq_length = config.seq_length
        self.split_datasets = self._filter_valid_datasets()
        self.records = self._list_records()
        self._initialize()

    @abstractmethod
    def _initialize(self):
        """Initialize the sampler"""
        pass
    
    @abstractmethod
    def get_sample(self, index: int) -> ISample:
        """Get a sample from the dataset"""
        pass

    @property
    def num_samples(self) -> int:
        """Return total number of available samples"""
        return self._get_num_sequences()
    
    def _get_num_sequences(self) -> int:
        """Count total number of non-overlapping sequences across records"""
        total_sequences = 0
        for filepath, subject, record in self.records:
            with h5py.File(filepath, "r") as hdf5:
                hypnogram = hdf5["data"][subject][record]["hypnogram"][()]
                num_sequences = len(hypnogram) // self.seq_length
                total_sequences += num_sequences
        return total_sequences

    
    def _filter_valid_datasets(self) -> List[Dataset_Split]:
        """Filter datasets based on split type"""
        return list(filter(lambda x: len(x.get_subjects_from_string(self.split_type)) > 0, 
                         self.config.split_data.dataset_splits))
    
    def _list_records(self) -> List[Tuple[str, str, str]]:
        """List all available records across datasets"""
        records = []
        for dataset in self.split_datasets:
            with h5py.File(dataset.dataset_filepath, "r") as hdf5:
                hdf5 = hdf5["data"]
                subjects = dataset.get_subjects_from_string(self.split_type)
                num_subjects = len(subjects)
                num_subjects_to_use = int(np.ceil(num_subjects * self.config.subject_percentage))
                subjects = subjects[:num_subjects_to_use]
                
                for subject in subjects:
                    try:
                        for record in hdf5[subject].keys():
                            records.append((dataset.dataset_filepath, subject, record))
                    except KeyError:
                        print(f"Subject {subject} not found in {dataset.dataset_filepath}")
                        continue
        return records
    

    
class DeterministicSampler(BaseSampler):
    """Deterministic sampler for sleep data that generates non-overlapping sequences.
    Used for validation"""
    def _initialize(self):
        """Initialize non-overlapping sequences"""
        self.sequences = []
        num_expected_sequences = self._get_num_sequences() 

        
        for record_idx, (filepath, subject, record) in enumerate(self.records):
            with h5py.File(filepath, "r") as hdf5:
                num_epochs = len(hdf5["data"][subject][record]["hypnogram"][()])
                num_sequences = num_epochs // self.seq_length
                # start indices for non-overlapping sequences
                self.sequences.extend([
                    (record_idx, i * self.seq_length) 
                    for i in range(num_sequences)
                ])
        
        # expected number of sequences
        assert len(self.sequences) == num_expected_sequences, \
            f"Expected {num_expected_sequences} sequences but got {len(self.sequences)}"

    def get_sample(self, index: int) -> ISample:
        """Get specific sequence at index"""
        record_idx, start_idx = self.sequences[index]
        filepath, subject, record = self.records[record_idx]
        
        with h5py.File(filepath, "r") as hdf5:
            hdf5 = hdf5["data"]
            
            # load labels
            labels = torch.tensor(
                hdf5[subject][record]["hypnogram"][start_idx:start_idx + self.seq_length]
            )
            
            # load PSG data
            psg_channels = list(hdf5[subject][record]["psg"].keys())
            eeg_channels = [x for x in psg_channels if not x.startswith("EOG_")]
            eog_channels = [x for x in psg_channels if x.startswith("EOG_")]
            
            if not self.config.get_all_channels:
                eeg_channels = eeg_channels[:1] if eeg_channels else []
                eog_channels = eog_channels[:1] if eog_channels else []
            
            # load EEG data
            eeg_data = []
            for ch in eeg_channels:
                data = hdf5[subject][record]["psg"][ch][start_idx:start_idx + self.seq_length]
                eeg_data.append(data)

            eeg_data = np.array(eeg_data)
            eeg_data = torch.tensor(eeg_data)
            
            # load EOG data
            eog_data = []
            for ch in eog_channels:
                data = hdf5[subject][record]["psg"][ch][start_idx:start_idx + self.seq_length]
                eog_data.append(data)

            eog_data = np.array(eog_data)
            eog_data = torch.tensor(eog_data)

        sample = ISample(index)
        sample.eeg = eeg_data
        sample.eog = eog_data
        sample.labels = labels
        sample.tag = ITag(
            dataset=os.path.basename(filepath),
            subject=subject,
            record=record,
            eeg=eeg_channels,
            eog=eog_channels,
            start_idx=start_idx,
            end_idx=start_idx + self.seq_length
        )
        return sample


class RandomSampler(BaseSampler):
    """Random sampler for sleep data that implements stratified sampling across datasets.
    Used for training
    """

    def __init__(self, config: SamplerConfig, pick_function=None):
        """Initialize with optional custom pick function for channel selection"""
        self.pick_function = pick_function if pick_function else self.__pick_random_EEG_and_EOG
        super().__init__(config)
        
    
    def _initialize(self):
        """Initialize sampler with dataset probabilities"""
        self.num_records = self._count_records()
        self.sampling_probs = self._calculate_sampling_probs()
        self.records_by_dataset = self._group_records_by_dataset()
        self._print_report()
    
    def _count_records(self) -> List[int]:
        """Count records per dataset using self.records from base class"""
        dataset_counts = []
        for dataset in self.split_datasets:
            count = sum(1 for record in self.records 
                       if record[0] == dataset.dataset_filepath)
            dataset_counts.append(count)
        return dataset_counts
    
    def _group_records_by_dataset(self) -> Dict[str, List[Tuple[str, str, str]]]:
        """Group records by dataset path for efficient sampling"""
        grouped = {}
        for filepath, subject, record in self.records:
            if filepath not in grouped:
                grouped[filepath] = []
            grouped[filepath].append((filepath, subject, record))
        return grouped
    
    def _calculate_sampling_probs(self) -> List[float]:
        """Calculate sampling probabilities based on dataset sizes"""
        total_datasets = len(self.split_datasets)
        total_records = sum(self.num_records)
        
        probs = []
        for num_records in self.num_records:
            size_weight = num_records / total_records
            uniform_weight = 1.0 / total_datasets
            prob = 0.5 * size_weight + 0.5 * uniform_weight
            probs.append(prob)
            
        return probs
    
    def _print_report(self):
        """Print sampler configuration details"""
        datasets = [split.dataset_filepath for split in self.split_datasets]
        print(f"Training on datasets {datasets}")
        print(f"Number of records per dataset: {self.num_records}")
        print(f"Sampling probabilities: {self.sampling_probs}")
        print("\nTraining subjects overview:")
        for dataset in self.split_datasets:
            print(f"{dataset.dataset_filepath}: {dataset.train}")

    def __pick_random_EEG_and_EOG(self, channel_list):
        """Default channel picking function"""
        eeg_channels = [x for x in channel_list if not x.startswith("EOG_")]
        eog_channels = [x for x in channel_list if x.startswith("EOG_")]
        
        selected_eeg = np.random.choice(eeg_channels, size=1) if eeg_channels else []
        selected_eog = np.random.choice(eog_channels, size=1) if eog_channels else []
        
        return selected_eeg, selected_eog
    
    def get_sample(self, index: int) -> ISample:
        success = False
            
        while not success:
            sample: ISample = self.__get_sample()

            if sample != None:
                success = True

        return sample

    
    def __get_sample(self):
        """Internal method to generate a random sample"""
        
        # randomly select dataset based on probabilities
        r_dataset: Dataset_Split = np.random.choice(self.split_datasets, size=1, p=self.sampling_probs)[0]
        r_dataset_path = r_dataset.dataset_filepath
        
        if r_dataset_path not in self.records_by_dataset:
            return None
        
        # randomly select a record from the dataset
        filepath, subject, record = random.choice(self.records_by_dataset[r_dataset_path])
        
        with h5py.File(filepath, "r") as hdf5:
            hdf5 = hdf5["data"]

            #randomly select a label
            hypnogram = hdf5[subject][record]["hypnogram"][()]
            label_set = np.unique(hypnogram)
            r_label = np.random.choice(label_set, size=1)[0]
            
            # find a random index with the selected label
            indexes = np.where(hypnogram == r_label)[0]
            r_index = np.random.choice(indexes, size=1)[0]
            
            # randomly shift the start index
            r_shift = np.random.choice(range(self.seq_length), size=1)[0]
            
            
            start_index = r_index - r_shift
            if start_index < 0:
                start_index = 0
            elif start_index + self.seq_length >= len(hypnogram):
                start_index = len(hypnogram) - self.seq_length
            
            labels = torch.tensor(
                hypnogram[start_index:start_index + self.seq_length]
            )
            
            # use pick function to select channels
            psg_channels = list(hdf5[subject][record]["psg"].keys())
            try:
                selected_eeg, selected_eog = self.pick_function(psg_channels)
            except:
                print(f"Could not pick channels from dataset {r_dataset}, subject: {subject}, record: {record}")
                return None
            
            if len(selected_eeg) == 0 and len(selected_eog) == 0:
                print(f"No EEG or EOG available. Available channels: {psg_channels} from {subject}, {record}")
                return None
            
            # load channel data
            eeg_data = []
            for ch in selected_eeg:
                try:
                    data = hdf5[subject][record]["psg"][ch][start_index:start_index + self.seq_length]
                    eeg_data.append(data)
                except:
                    eeg_data = []
            eeg_data = np.array(eeg_data)
            eeg_data = torch.tensor(eeg_data)
            
            eog_data = []
            for ch in selected_eog:
                try:
                    data = hdf5[subject][record]["psg"][ch][start_index:start_index + self.seq_length]
                    eog_data.append(data)
                except:
                    eog_data = []

            eeg_data = np.array(eeg_data)
            eeg_data = torch.tensor(eeg_data)

            
            sample = ISample(-1)
            sample.eeg = eeg_data
            sample.eog = eog_data
            sample.labels = labels
            sample.tag = ITag(
                dataset=os.path.basename(filepath),
                subject=subject,
                record=record,
                eeg=list(selected_eeg),
                eog=list(selected_eog),
                start_idx=start_index,
                end_idx=start_index + self.seq_length
            )
            
            return sample
        

class TestSampler(BaseSampler):
    """Test sampler that generates overlapping sequences and includes all channels.
    Used for final model evaluation and prediction."""
    
    def _initialize(self):
        """Initialize overlapping sequences for all recordings"""
        self.sequences = []
        
        for record_idx, (filepath, subject, record) in enumerate(self.records):
            with h5py.File(filepath, "r") as hdf5:
                # number of epochs in this recording
                num_epochs = len(hdf5["data"][subject][record]["hypnogram"][()])
                
                # generate all possible overlapping sequences
                # each sequence starts one epoch later than the previous
                sequence_starts = range(0, num_epochs - self.seq_length + 1)
                
                # (record_idx, start_idx) for each sequence
                self.sequences.extend([
                    (record_idx, start_idx)
                    for start_idx in sequence_starts
                ])
    
    def _filter_channels(self, psg_channels: List[str]) -> Tuple[List[str], List[str]]:
        """Filter PSG channels into EEG and EOG channels"""
        eeg_channels = [x for x in psg_channels if not x.startswith("EOG_")]
        eog_channels = [x for x in psg_channels if x.startswith("EOG_")]
        return eeg_channels, eog_channels
    
    def get_sample(self, index: int) -> ISample:
        """Get specific sequence at index with all channels"""
        record_idx, start_idx = self.sequences[index]
        filepath, subject, record = self.records[record_idx]
        
        with h5py.File(filepath, "r") as hdf5:
            hdf5 = hdf5["data"]
            
            # load labels
            labels = torch.tensor(
                hdf5[subject][record]["hypnogram"][start_idx:start_idx + self.seq_length]
            )
            
            # tetrieve all available channels
            psg_channels = list(hdf5[subject][record]["psg"].keys())
            eeg_channels, eog_channels = self._filter_channels(psg_channels)
            
            # load all EEG channels
            eeg_data = []
            for ch in eeg_channels:
                data = hdf5[subject][record]["psg"][ch][start_idx:start_idx + self.seq_length]
                eeg_data.append(data)
            eeg_data = np.array(eeg_data)
            eeg_data = torch.tensor(eeg_data)
            
            # load all EOG channels
            eog_data = []
            for ch in eog_channels:
                data = hdf5[subject][record]["psg"][ch][start_idx:start_idx + self.seq_length]
                eog_data.append(data)
            eog_data = np.array(eog_data)
            eog_data = torch.tensor(eog_data)
            
            # no EOG channels found, duplicate EEG 
            if eog_data.shape[0] == 0:
                eog_data = eeg_data
                eog_channels = eeg_channels

        sample = ISample(index)
        sample.eeg = eeg_data
        sample.eog = eog_data
        sample.labels = labels
        sample.tag = ITag(
            dataset=os.path.basename(filepath),
            subject=subject,
            record=record,
            eeg=eeg_channels,
            eog=eog_channels,
            start_idx=start_idx,
            end_idx=start_idx + self.seq_length
        )
        
        return sample

    @property
    def num_samples(self) -> int:
        """Return total number of available sequences"""
        return len(self.sequences)
