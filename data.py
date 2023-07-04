import librosa
import numpy as np

import torch
from torch.utils.data import  Dataset
from torch.nn.utils.rnn import pad_sequence

def load_image():
    pass

class AudioDataSet(Dataset):
    def __init__(self, file_list, y=None):
        self.features = [load_image(file) for file in file_list]
        self.max_length_file = file_list.iloc[0] #need to resize this to 224 or maybe not
        
        if y is not None:
            self.y = torch.tensor(y.values, dtype=torch.long)
        else:
            self.y = torch.zeros(len(self.features)) #dummy y for Compatibility

        self.scaler = None

    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, index):
        return torch.tensor(self.features[index], dtype=torch.float), self.y[index]