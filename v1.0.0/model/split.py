import torch
import os
import json
import h5py

from sklearn.model_selection import train_test_split


class Dataset_Split():
    dataset_filepath: str
    train: list[str]
    val: list[str]
    test: list[str]

    def __init__(self, 
                 dataset_filepath: str,
                 train: list[str] = [],
                 val: list[str] = [],
                 test: list[str] = []):
        self.dataset_filepath = dataset_filepath
        self.train = train
        self.val = val
        self.test = test


    def get_subjects_from_string(self, str):
        if str == "train":
            return self.train
        elif str == "val":
            return self.val
        else: 
            return self.test
        

class Split():
    id:str
    dataset_splits: list[Dataset_Split]
    base_data_path: str

    @classmethod
    def file(cls, split_file_path):

        dataset_splits = []

        with open(split_file_path) as f:
            data = json.load(f)

        base_data_path = data["base_data_path"]
        id = os.path.basename(split_file_path.rstrip(".json"))
        inner_data = data["datasets"]

        datasets = list(map(lambda x: x[0], inner_data.items()))

        for dataset in datasets:
            s = Dataset_Split(f"{base_data_path}/{dataset}",
                                train = inner_data[dataset]["train"],
                                val = inner_data[dataset]["val"],
                                test = inner_data[dataset]["test"])

            dataset_splits.append(s)

        return cls(id, 
                    dataset_splits=dataset_splits, 
                    base_data_path=base_data_path)
    

    @classmethod
    def random(cls, 
                base_hdf5_path,
                split_name = "random",
                split_percentages = (0.8, 0.1, 0.1)):
        hdf5_paths = os.listdir(base_hdf5_path)
        hdf5_paths = [f"{base_hdf5_path}/{path}" for path in hdf5_paths]

        dataset_splits : list[Dataset_Split] = []

        for path in hdf5_paths:
            with h5py.File(path, "r") as hdf5:
                subs= list(hdf5["data"].keys())

                train, test = train_test_split(subs, test_size=1-split_percentages[0])
                val, test = train_test_split(test, test_size=split_percentages[2]/(split_percentages[1]+split_percentages[2]))
                
                split = Dataset_Split(path, 
                                        train=train, 
                                        val=val, 
                                        test=test)
                
                dataset_splits.append(split)

        return cls(split_name, 
                    dataset_splits=dataset_splits, 
                    base_data_path=base_hdf5_path)
        
    def dump_file(self, path, name):
        dic = self.get_dict()

        with open(f"{path}/{name}.json", "w") as outfile: 
            json.dump(dic, outfile)

    def __init__(self,
                 id="",
                 dataset_splits=[],
                 base_data_path=""):
        self.id = id
        self.dataset_splits = dataset_splits
        self.base_data_path = base_data_path

    def __repr__(self):
        return f"{self.__class__.__name__}(id={self.id}, dataset_splits={self.dataset_splits}, base_data_path={self.base_data_path})"
    
    def get_dict(self) -> dict:
        dic = dict()
        dic["datasets"] = {}
        
        for split in self.dataset_splits:
            dic["datasets"][os.path.basename(split.dataset_filepath)] = {"train": split.train,
                                                                         "val": split.val,
                                                                         "test": split.test}
        
        dic["base_data_path"] = self.base_data_path

        return dic



class ITag:
    dataset: str
    subject: str
    record: str
    eeg: [str]
    eog: [str]
    start_idx: str
    end_idx: str

    def __init__(self,
                 dataset: str = "",
                 subject: str = "",
                 record : str = "",
                 eeg = [],
                 eog = [],
                 start_idx = -1,
                 end_idx = -1):
        self.dataset = dataset
        self.subject = subject
        self.record = record
        self.eeg = eeg
        self.eog = eog
        self.start_idx = start_idx
        self.end_idx = end_idx

class ISample:
    def __init__(self, index: int):
        self.index = index

    index: int
    eeg: torch.Tensor | None
    eog: torch.Tensor | None
    labels: torch.Tensor | None
    tag: ITag | None

