import torch
import os
import h5py
import math
import numpy as np
from abc import ABC, abstractmethod
from split import Split, Dataset_Split, ISample, ITag

def filter_channels(channel_list):
        #Choose random eeg and eog
    eog_channels = [x for x in channel_list if x.startswith("EOG_")]
    eeg_channels = [x for x in channel_list if x not in eog_channels]
    return eeg_channels, eog_channels

class ISampler:
    @abstractmethod
    def get_sample(self, index) -> ISample:
        pass

    num_samples: int

class SamplerConfiguration:
    def __init__(self, 
                 train: ISampler,
                 val: ISampler, 
                 test: ISampler):
        self.train_sampler = train
        self.val_sampler = val
        self.test_sampler = test

    def get_sampler_by_stage(self, stage: str):
        assert stage == "train" or stage == "val" or stage == "test"

        if stage == "train":
            return self.train_sampler
        elif stage == "val":
            return self.val_sampler
        else:
            return self.test_sampler


class Random_Sampler(ISampler):
    def __init__(self,
                 split_data: Split,
                 split_type: str, 
                 num_epochs: int, 
                 num_iterations: int,
                 pick_function = None):
        assert split_type == "train"

        if pick_function == None:
            self.pick_function = self.__pick_random_EEG_and_EOG
        else:
            self.pick_function = pick_function

        #Remove splits if they do not have training
        #split_data.dataset_splits = filter(lambda x: len(x.train) > 0, split_data.dataset_splits)
        
        self.split_type = split_type
        self.split_datasets: list[Dataset_Split] = list(filter(lambda x: len(x.train) > 0, split_data.dataset_splits))

        self.num_records = self.__count_records()

        try:
            self.probs = self.calc_probs()
        except:
            self.probs = []


        self.print_report()
            
        self.epoch_length = num_epochs
        self.num_samples = num_iterations

    def print_report(self):
            probs = self.probs
            datasets = [split.dataset_filepath for split in self.split_datasets]
            num_records = self.num_records

            print(f"Training on datasets {datasets}. The number of records are {num_records}, which yields stratified sampling probabilities {probs}")
            print("Training subjects overview:")

            for dataset_split in self.split_datasets:
                print(f"{dataset_split.dataset_filepath}: {dataset_split.train}")

    def get_sample(self, index: int) -> ISample:
        success = False
            
        while not success:
            sample: ISample = self.__get_sample()

            if sample != None:
                success = True

        return sample
    
    def calc_probs(self):

        total_num_datsets = len(self.split_datasets)
        total_num_records = sum(self.num_records)

        probs = []

        for i, _ in enumerate(self.split_datasets):
            num_records = self.num_records[i]
            strat_prob = num_records / total_num_records
            dis_prob = 1 / total_num_datsets

            prob_d = 0.5 * strat_prob + 0.5 * dis_prob 
            probs.append(prob_d)

        return probs
    
    def __get_sample(self):

        possible_sets: list[Dataset_Split] = self.split_datasets

        probs = self.probs

        #Choose random dataset
        r_dataset: Dataset_Split = np.random.choice(possible_sets, 1, p=probs)[0]

        #choose random subject
        subjects = r_dataset.get_subjects_from_string(self.split_type)

        r_subject = np.random.choice(subjects, 1)[0]

        if len(subjects) == 0:
            raise ValueError(f"No subjects in split type: {self.split_type} for dataset {r_dataset}")
        
        with h5py.File(r_dataset.dataset_filepath, "r") as hdf5:
            hdf5 = hdf5["data"]

            # choose random record
            records = list(hdf5[r_subject].keys())
            r_record = np.random.choice(records, 1)[0]

            # pull out hypnogram and spectrogram
            hyp = hdf5[r_subject][r_record]["hypnogram"][()]
            psg = list(hdf5[r_subject][r_record]["psg"].keys())


            try:
                eegs, eogs = self.pick_function(psg)
            except:
                print(f"Could not pick eeg or eog from dataset {r_dataset}, subject: {r_subject}, record: {r_record}")
                return None
            
            if len(eegs) == 0 and len(eogs) == 0:
                print(f"No EEG or EOG available. Available channels: {psg} from {r_subject}, {r_record}")
                return None

            
            #choose random sleep stage (label)
            label_set = np.unique(hyp)
            r_label = np.random.choice(label_set, 1)[0]

            #choose random index of that label
            indexes = [i for i in range (len(hyp)) if hyp[i] == r_label]
            r_index = np.random.choice(indexes, 1)[0]

            #Randomly shift the position of the random label index
            #TODO: shift with index at center
            r_shift = np.random.choice(list(range(0, self.epoch_length)), 1)[0]

            start_index = r_index - r_shift

            if start_index < 0:
                start_index = 0
            elif (start_index + self.epoch_length) >= len(hyp):
                start_index = len(hyp) - self.epoch_length


            #Get the data

            y = hyp[start_index:start_index+self.epoch_length]
            
            y = torch.tensor(y)

            x_start_index = start_index

            eeg_segments = []
            eog_segments = []


            for eeg in eegs:
                try:
                    eeg_segment = hdf5[r_subject][r_record]["psg"][eeg][x_start_index:x_start_index+(self.epoch_length)]
                except:
                    eeg_segment = []
                eeg_segments.append(eeg_segment)

            for eog in eogs:
                try:
                    eog_segment = hdf5[r_subject][r_record]["psg"][eog][x_start_index:x_start_index+(self.epoch_length)]
                except:
                    eog_segment = []
                eog_segments.append(eog_segment)

        eeg_segments = np.array(eeg_segments)
        eog_segments = np.array(eog_segments)

        x_eeg = torch.tensor(eeg_segments)
        x_eog = torch.tensor(eog_segments)


        sample = ISample(-1)
        sample.eeg = x_eeg
        sample.eog = x_eog
        sample.labels = y
        sample.tag = ITag(os.path.basename(r_dataset.dataset_filepath),
                          r_subject,
                          r_record,
                          [],
                          [],
                          x_start_index,
                          x_start_index+(self.epoch_length))
        
        return sample

    def __pick_random_EEG_and_EOG(self, channel_list):
        eeg_channels, eog_channels = filter_channels(channel_list)

        if len(eeg_channels) > 0:
            r_eeg = np.random.choice(eeg_channels, 1)
        else:
            r_eeg = []

        if len(eog_channels) > 0:
            r_eog = np.random.choice(eog_channels, 1)
        else:
            r_eog = []
            
        return r_eeg, r_eog

    def __count_records(self):
        num_records = []
        
        for f in self.split_datasets:
            file_path = f.dataset_filepath

            with h5py.File(file_path, "r") as hdf5:
                hdf5 = hdf5["data"]

                subs = f.get_subjects_from_string(self.split_type)

                tot_records = 0
                
                for subj_key in subs:
                    try:
                        subj = hdf5[subj_key]
                    except:
                        print(f"Did not find subject {subj_key} in dataset {f} for splittype {self.split_type}")
                        continue
                        
                    records = len(subj.keys())
                    tot_records += records
                
                num_records.append(tot_records)

        return num_records


class Determ_Sampler(ISampler):
    def __init__(self,
                 split_data: Split,
                 split_type: str,
                 seq_length: int = 21,
                 subject_percentage: float = 1.0,
                 get_all_channels = False):
        self.split_type = split_type
        self.split_data = split_data
        self.subject_percentage = subject_percentage
        self.seq_length = seq_length
        self.get_all_channels = get_all_channels
        
        # Get list of all records
        self.records = self.list_records()
        
        # Prepare sequence information
        self.sequences = self._prepare_sequences()
        self.num_samples = len(self.sequences)
        
        self.print_report()

    def print_report(self):
        print(f"Records for {self.split_type}: {self.records}")
    
    def list_records(self):
        """Lists all records available in the split"""
        list_of_records = []
        datasets = self.split_data.dataset_splits

        for f in datasets:
            with h5py.File(f.dataset_filepath, "r") as hdf5:
                hdf5 = hdf5["data"]

                subjects = f.get_subjects_from_string(self.split_type)
                
                num_subjects = len(subjects)
                num_subjects_to_use = math.ceil(num_subjects*self.subject_percentage)
                subjects = subjects[0:num_subjects_to_use]
                
                for s in subjects:
                    try:
                        records = list(hdf5[s])
                    except:
                        print(f"Did not find subject {s} in dataset {f} for splittype {self.split_type}")
                        continue

                    for r in records:
                        list_of_records.append((f.dataset_filepath,s,r))
        
        return list_of_records
    
    def _prepare_sequences(self):
        """Create list of all possible sequences from all records"""
        sequences = []
        
        for record_idx, record in enumerate(self.records):
            with h5py.File(record[0], "r") as hdf5:
                hdf5 = hdf5["data"]
                # Get length of hypnogram
                num_epochs = len(hdf5[record[1]][record[2]]["hypnogram"][()])
                # Calculate number of complete sequences
                num_sequences = num_epochs - self.seq_length + 1
                
                # Store record and start index for each possible sequence
                for start_idx in range(num_sequences):
                    sequences.append((record_idx, start_idx))
        
        return sequences
    
    def get_sample(self, index: int) -> ISample:
        # Get sequence information
        record_idx, start_idx = self.sequences[index]
        record = self.records[record_idx]
        
        with h5py.File(record[0], "r") as hdf5:
            hdf5 = hdf5["data"]
            
            # Get the sequence of labels
            y = hdf5[record[1]][record[2]]["hypnogram"][start_idx:start_idx + self.seq_length]
            y = torch.tensor(y)
            
            psg_channels = list(hdf5[record[1]][record[2]]["psg"].keys())
            eeg_data, eog_data, eeg_tag, eog_tag = self.__load_sequence_data(
                hdf5, record[1], record[2], psg_channels, start_idx
            )

        sample = ISample(index)
        sample.eeg = eeg_data
        sample.eog = eog_data
        sample.labels = y
        sample.tag = ITag(os.path.basename(record[0]),
                         record[1],
                         record[2],
                         eeg_tag,
                         eog_tag)
        
        return sample
    
    def determine_single_key(self, keys):
        """Determines a single key from a list of keys"""
        if len(keys) > 0:
            key = keys[0]
            tag = key
            keys = [key]
        else:
            tag = "none"
        
        return keys, tag

    def __load_sequence_data(self, hdf5, subject, rec, psg_channels, start_idx):
        """Load EEG and EOG data for a specific sequence"""
        eeg_data = []
        eog_data = []
        
        available_eeg_keys, available_eog_keys = filter_channels(psg_channels)
        
        if self.get_all_channels == False:
            eeg_keys, eeg_tag = self.determine_single_key(available_eeg_keys)
            eog_keys, eog_tag = self.determine_single_key(available_eog_keys)
        else:
            eeg_keys = available_eeg_keys
            eog_keys = available_eog_keys
            eeg_tag = available_eeg_keys
            eog_tag = available_eog_keys
            
        for ch in eeg_keys:
            # Load only the sequence we want
            data = hdf5[subject][rec]["psg"][ch][start_idx:start_idx + self.seq_length]
            eeg_data.append(data)
            
        for ch in eog_keys:
            data = hdf5[subject][rec]["psg"][ch][start_idx:start_idx + self.seq_length]
            eog_data.append(data)
            
        eog_data = np.array(eog_data)
        eog_data = torch.Tensor(eog_data)
        
        eeg_data = np.array(eeg_data)
        eeg_data = torch.Tensor(eeg_data)
        
        return eeg_data, eog_data, eeg_tag, eog_tag
    

