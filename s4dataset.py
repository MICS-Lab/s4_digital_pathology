import os
import torch
import pandas as pd
from torch.utils.data import Dataset

class S4Dataset(Dataset):
    def __init__(self, config, dataset_type):
        super(S4Dataset, self).__init__()
        assert dataset_type in ['train', 'val', 'test'], f'dataset_type must be in [train, val, test], not f{dataset_type}.'

        fold_data = pd.read_csv(os.path.join(config.data.folds_path, f'fold{config.data.fold}.csv'))
        self.data = fold_data.loc[:, f'{dataset_type}'].dropna().tolist()
        self.label = fold_data.loc[:, f'{dataset_type}_label'].dropna().tolist()
        self.data_path = config.data.data_path
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        slide_id = self.data[idx]
        label = int(self.label[idx])
        features = torch.load(os.path.join(self.data_path, f'{slide_id}.pt'))
        return features, label