import librosa
import numpy as np
from PIL import Image

import torch
from torch.utils.data import  Dataset

#image transformation 추가하기

def load_image(file):
    return np.array(Image.open(file))

def rle_encode(mask):
    pixels = mask.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)

def rle_decode(mask_rle, shape):
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape)

class ImageDataSet(Dataset):
    def __init__(self, file_list, y=None):
        self.features = [load_image(file) for file in file_list]
        self.max_length_file = file_list.iloc[0] #need to resize this to 224 or maybe not
        #rle_decode함수 추가 필요
        if y is not None:
            self.y = torch.tensor(y.values, dtype=torch.long)
        else:
            self.y = torch.zeros(len(self.features)) #dummy y for Compatibility

    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, index):
        return self.features[index], self.y[index]