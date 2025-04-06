import os
import numpy as np
import pandas as pd
import torch
import json

from torch.utils.data import Dataset


class BaseDataset(Dataset):
    def __init__(self, config, train):
        part = 'train' if train else 'val'
        self.data_path = config["path"]

        # label name
        self.target_name = config["target_name"]

        # read table
        table_path = os.path.join(self.data_path, f'{part}.csv')
        self.data_table = pd.read_csv(table_path)

        # features loading, mini processing
        self.data_table[self.target_name] = self.data_table[self.target_name].map(json.loads)
        self.features = list(self.data_table.columns)
        self.features.remove(self.target_name)

        # second input features
        pref = config["second_inp_prefix"] if "second_inp_prefix" in config else None
        if pref != None:
            self.features_si = [i for i in self.features if i.startswith(pref)]
        else:
            self.features_si = []
        # exclude from features
        self.features = list(set(self.features).difference(set(self.features_si)))


    def __len__(self):
        return len(self.data_table)


    def __getitem__(self, item):
        item = self.data_table.iloc[item]

        sample = item[self.features]
        sample = sample.to_numpy().astype(float)
        sample = torch.from_numpy(sample).float()
        sample = [sample]

        if len(self.features_si) != 0:
            sec_sample = item[self.features_si]
            sec_sample = sec_sample.to_numpy().astype(float)
            sec_sample = torch.from_numpy(sec_sample).float()
            sample.append(sec_sample)

        label = item[self.target_name]
        label = np.array(label)
        label = torch.from_numpy(label).float()

        return {'sample': sample, 'label': label}