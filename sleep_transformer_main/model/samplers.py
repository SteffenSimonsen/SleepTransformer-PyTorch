import torch
import os
import h5py
import math
import numpy as np
from abc import ABC, abstractmethod
from split import ISample, ITag, Split, Dataset_Split


def filter_channels(channel_list):
    #return lists of eeg and eog channels
    eog_channels = [x for x in channel_list if x.startswith("EOG_")]
    eeg_channels = [x for x in channel_list if x not in eog_channels]

    return eeg_channels, eog_channels


class ISampler:
    @abstractmethod
    def get_sample(self, index) -> ISample:
        pass
    
    num_samples: int

    def get_sampler_by_stage(self, stage: str):

        if stage == "train":
            return self.train_sampler
        elif stage == "val":
            return self.val_sampler
        elif stage == "test":
            return self.test_sampler
        else:
            raise ValueError(f"Invalid stage: {stage}")
        

class RandomSampler(ISampler):
    def __init__(self,
                 split_data: Split,
                 split_type: str,
                 num_epochs: int,
                 num_iterations: int,
                 pick_function = None):
        
        if not split_type == "train":
            raise ValueError(f"RandomSampler only supports train split, not {split_type}")
        
        if pick_function is None:
            self.pick_function = self._pick_random_EEG_and_EOG
        else:
            self.pick_function = pick_function

        
        self.split_type = split_type
        self.split_datasets: list[Dataset_Split] = list(filter(lambda x: len(x.train) > 0, split_data.dataset_splits))

        self.num_records = self.__count_records()

        try: 
            self.probs = self.calc_probs()
        except:
            print("Error calculating probabilities")
            self.probs = []

        #self.print_report()

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

    def __count_records(self):
        num_records = []

        for f in self.split_datasets:
            file_path = f.dataset_filepath

            with h5py.File(file_path, 'r') as hdf5:
                hdf5 = hdf5["data"]
                
                subs = f.get_subjects_from_string(self.split_type)

                tot_records = 0

                for subj_key in subs:
                    try:
                        subj = hdf5[subj_key]
                    except:
                        print(f"Did not find subject {subj_key} in dataset {f.dataset_filepath} for splittype {self.split_type}")
                        continue

                    records = len(subj.keys())

                    tot_records += records


                num_records.append(tot_records)

        return num_records
    
    def calc_probs(self):

        total_num_datasets = len(self.split_datasets)
        total_num_records = sum(self.num_records)
        
        probs = []
        
        for i, _ in enumerate(self.split_datasets):
            num_records = self.num_records[i]
            
            strat_prob = num_records/total_num_records

            dis_prob = 1 / total_num_datasets
            
            prob_d = 0.5 * strat_prob + 0.5*dis_prob
            probs.append(prob_d)

        return probs
    
    def _pick_random_EEG_and_EOG(self, channel_list):
        #pick random eeg and eog channels from channel list
        eeg_channels, eog_channels = filter_channels(channel_list)

        if len(eeg_channels) > 0:
            r_eeg_channel = np.random.choice(eeg_channels, 1)
        else:
            print("Could not find any EEG channels")
            r_eeg_channel = []

        if len(eog_channels) > 0:
            r_eog_channel = np.random.choice(eog_channels, 1)
        else:
            print("Could not find any EOG channels")
            r_eog_channel = []

        return r_eeg_channel, r_eog_channel
    
    def get_sample(self, index: int) -> ISample:
        success = False
            
        while not success:
            sample: ISample = self.__get_sample()

            if sample != None:
                success = True

        return sample
    

    def __get_sample(self) -> ISample:

        possible_sets: list[Dataset_Split] = self.split_datasets
    
        probs = self.probs
        
        # Choose random dataset
        r_dataset: Dataset_Split = np.random.choice(possible_sets, 1, p=probs)[0]
        
        #get subjects
        subjects = r_dataset.get_subjects_from_string(self.split_type)

        if len(subjects) == 0:
            raise ValueError(f"No subjects in split type: {self.split_type} for dataset {r_dataset}")

        #choose random subject
        r_subject = np.random.choice(subjects, 1)[0]


        with h5py.File(r_dataset.dataset_filepath, "r") as hdf5:
            hdf5 = hdf5["data"]

            #retrieve random subject records
            records = list(hdf5[r_subject].keys())

            #choose random record
            r_record = np.random.choice(records, 1)[0]

            # retrieve hypnogram (labels) for random record 
            hyp = hdf5[r_subject][r_record]["hypnogram"][()]

            # retrieve psg channels (keys for) random record
            psg = list(hdf5[r_subject][r_record]["psg"].keys())

            try:
                eegs, eogs = self.pick_function(psg)
            except:
                print(f"Could not pick eeg or eog from dataset {r_dataset}, subject: {r_subject}, record: {r_record}")
                return None

            if len(eegs) == 0:
                print(f"No EEG. Available channels: {psg} from {r_subject}, {r_record}")
                return None
            
            if len(eogs) == 0:
                print(f"No EOG. Available channels: {psg} from {r_subject}, {r_record}")
                return None
            
            # choose a random label 
            label_set = np.unique(hyp)
            r_label = np.random.choice(label_set, 1)[0]

            #choose random index
            indexes = [i for i in range(len(hyp)) if hyp[i] == r_label]
            r_index = np.random.choice(indexes, 1)[0]

            # Randomly shift the position of the random label index
            r_shift = np.random.choice(list(range(0,self.epoch_length)), 1)[0]
            
            start_index = r_index-r_shift
        
            if start_index < 0:
                start_index = 0
            elif (start_index + self.epoch_length) >= len(hyp):
                start_index = len(hyp) - self.epoch_length
            
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


class DetermSampler(ISampler):
    def __init__(self,
                 split_data: Split,
                 split_type: str,
                 subject_percentage: float = 1.0,
                 get_all_channels = False):
        self.split_type = split_type
        self.split_data = split_data
        self.subject_percentage = subject_percentage
        self.records = self.list_records()
        self.num_samples = len(self.records)
        self.get_all_channels = get_all_channels

        #self.print_report()

    def get_sample(self, index: int):
        sample: ISample = self.__get_sample(index)

        # if any(dim == 0 for dim in sample.eog.shape):
        #     print("Found no EOG channel, duplicating EEG instead")
        #     sample.eog = sample.eeg
        #     sample.tag.eog = sample.tag.eeg
        
        return sample
    
    def print_report(self):
        print(f"Records for {self.split_type}: {self.records}")

    def list_records(self):
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
                        print(f"Did not find subject {s} in dataset {f.dataset_filepath} for splittype {self.split_type}")
                        continue

                    for r in records:
                        list_of_records.append((f.dataset_filepath,s,r))
        
        return list_of_records

    def __get_sample(self, index: int) -> ISample:
        r = self.records[index]

        dataset = r[0]
        subject = r[1]
        rec = r[2]

        with h5py.File(dataset, "r") as hdf5:
            hdf5 = hdf5["data"]

            y = hdf5[subject][rec]["hypnogram"][()]

            psg_channels = list(hdf5[subject][rec]["psg"].keys())

            eeg_data, eog_data, eeg_tag, eog_tag = self.__load_data(hdf5, subject, rec, psg_channels)

        sample = ISample(index)
        sample.eeg = eeg_data
        sample.eog = eog_data
        sample.labels = torch.tensor(y)
        sample.tag = ITag(os.path.basename(dataset),
                          subject,
                          rec,
                          eeg_tag,
                          eog_tag)

        return sample
    
    def determine_single_key(self, keys):
        if len(keys) > 0:
            key = keys[0]
            tag = key
            keys = [key]
        else:
            tag = "none"
        
        return keys, tag

    def __load_data(self, hdf5, subject, rec, psg_channels):
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
            data = hdf5[subject][rec]["psg"][ch][:]
            eeg_data.append(data)
            
        for ch in eog_keys:
            data = hdf5[subject][rec]["psg"][ch][:]
            eog_data.append(data)

        eog_data = np.array(eog_data)
        eog_data = torch.Tensor(eog_data)

        eeg_data = np.array(eeg_data)
        eeg_data = torch.Tensor(eeg_data)
        
        return eeg_data, eog_data, eeg_tag, eog_tag



    
