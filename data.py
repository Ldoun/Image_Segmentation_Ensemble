import librosa
import numpy as np
from PIL import Image
import albumentations as A

import torch
from torch.utils.data import  Dataset

#image transformation 추가하기

def load_image(file):
    return np.array(Image.open(file)).astype(np.float32)

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

train_transform = A.Compose([ #need to change TO random crop
    A.ColorJitter(p=0.5),
    A.Flip(p=0.5),
    A.RandomResizedCrop(height=224, width=224, ratio=(0.5, 1.5), always_apply=True),
    #A.CropNonEmptyMaskIfExists(224, 224, ignore_values=[0], ignore_channels=None, always_apply=True, p=1.0)
])

valid_transform = A.Compose([
    #A.CropNonEmptyMaskIfExists(224, 224, ignore_values=[0], ignore_channels=None, always_apply=True, p=1.0)
])

class ImageDataSet(Dataset):
    def __init__(self, file_list, transform=None, mask=None, label=None):
        self.features = [load_image(file) for file in file_list]
        self.max_length_file = file_list.iloc[0] #need to resize this to 224 or maybe not
        self.transform = transform
        
        self.mask = None
        if mask is not None:
            self.mask = [rle_decode(m, (1024, 1024)) for m in mask]
        
        if label is not None:
            self.label = torch.tensor(label, dtype=torch.long)
        else:
            self.label = torch.zeros(len(file_list), dtype=torch.long)

    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, index):
        transformed = self.transform(image=self.features[index], mask=self.mask[index])
        return torch.tensor(transformed['image'], dtype=torch.float), torch.tensor(transformed['mask'], dtype=torch.long), self.label[index]