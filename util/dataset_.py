import torch
from torch.utils.data import Dataset


class SigirDataset(Dataset):
    def __init__(self, df, usecols, lag_targets=None, lag_urls=None, lag_categories=None, lag_descriptions=None, lag_imgs=None, target=None):
        super().__init__()
        self.usecols = usecols
        self.len = len(df)

        if lag_targets:
            setattr(self, 'lag_products', df[lag_targets].values)
        if lag_urls:
            setattr(self, f'lag_urls', df[lag_urls].values)
        if lag_categories:
            setattr(self, f'lag_categories', df[lag_categories].values)
        if lag_descriptions:
            setattr(self, f'lag_descriptions', df[lag_descriptions].values)
        if lag_imgs:
            setattr(self, f'lag_imgs', df[lag_imgs].values)
        for c in usecols:
            if not c in ['lag_products', 'lag_urls', 'lag_categories', 'lag_descriptions', 'lag_imgs']:
                setattr(self, c, df[c].values)

        if target is None:
            self.target = None
        else:
            self.target = df[target].values

    def __len__(self):
        return self.len

    def __getitem__(self, idx: int):
        input_dict = {}
        for c in self.usecols:
            if getattr(self, c).dtype in ['float32', 'float64']:
                dtype = torch.float
            else:
                dtype = torch.long
            if c in ['lag_products', 'lag_urls', 'lag_categories']:
                input_dict[c] = torch.tensor(getattr(self, c)[idx], dtype=torch.long)
            elif c in ['lag_descriptions', 'lag_imgs']:
                input_dict[c] = torch.tensor(getattr(self, c)[idx], dtype=torch.float)
            else:
              input_dict[c] = torch.tensor([getattr(self, c)[idx]], dtype=dtype)
        if not self.target is None:
            input_dict['target'] = torch.tensor([self.target[idx]], dtype=torch.long)
        return input_dict

    def addattr(self, key, value):
        setattr(self, key, value)
