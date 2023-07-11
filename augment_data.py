import os
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image
from copy import deepcopy
import albumentations as A

from data import rle_decode, rle_encode

def get_720_image(image, mask):
    images = []
    masks = []
    size = 1024

    images.append(deepcopy(image[:720, :720, :]))
    images.append(deepcopy(image[:720, size-720:, :]))
    images.append(deepcopy(image[size-720:, size-720:, :]))
    images.append(deepcopy(image[size-720:, :720, :]))

    masks.append(deepcopy(mask[:720, :720]))
    masks.append(deepcopy(mask[:720, size-720:]))
    masks.append(deepcopy(mask[size-720:, size-720:]))
    masks.append(deepcopy(mask[size-720:, :720]))

    return images, masks


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=72)
    parser.add_argument("--train_csv", required=True)
    parser.add_argument("--path", default='./')
    parser.add_argument("--output_path", default='./aug')

    args = parser.parse_args()

    data = pd.read_csv(args.train_csv)
    data['img_path'] = data['img_path'].apply(lambda x: os.path.join(args.path, x))

    transform = A.RandomResizedCrop(height=224, width=224, ratio=(0.4, 1.6), always_apply=True)

    img_ids = []
    mask_rles = []
    img_paths = []
    has_mask = []

    for i, row in tqdm(data.iterrows()):
        mask = rle_decode(row['mask_rle'], (1024, 1024))
        image = np.array(Image.open(row['img_path']))
        
        images, masks = get_720_image(image, mask)
        for i in range(4):
            croped = transform(image=images[i], mask=masks[i])

            img_id = row['img_path'].split('/')[-1][:-4]+f'_{i}'
            path = os.path.join(args.output_path, img_id+'.png')
            Image.fromarray(croped['image']).save(path)

            img_ids.append(img_id)
            mask_rles.append(rle_encode(croped['mask']))
            img_paths.append(path)
            has_mask.append(croped['mask'].any()==True)

    df = pd.DataFrame()
    df['img_id'] = img_ids
    df['mask_rle'] = mask_rles
    df['img_path'] = img_paths
    df['has_mask'] = has_mask

    df.to_csv(os.path.join(args.path, "augmented_images.csv"), index=False)