import os
import torch
import random
import numpy as np

from data import rle_decode, rle_encode

def seed_everything(seed: int): #for deterministic result; currently wav2vec2 model and torch.use_deterministic_algorithms is incompatible
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    #os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    #torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = True

def dice_score(prediction: np.array, ground_truth: np.array, smooth=1e-7) -> float:
    '''
    Calculate Dice Score between two binary masks.
    '''
    intersection = np.sum(prediction * ground_truth)
    return (2.0 * intersection + smooth) / (np.sum(prediction) + np.sum(ground_truth) + smooth)

def batch_dice_score(predictions, ground_truths):
    return np.mean(dice_score(predictions[i], ground_truths[i]) for i in range(len(predictions)))


def calculate_dice_scores(ground_truth_df, prediction_df, img_shape=(224, 224)):
    '''
    Calculate Dice scores for a dataset.
    '''

    # Keep only the rows in the prediction dataframe that have matching img_ids in the ground truth dataframe
    prediction_df = prediction_df[prediction_df.iloc[:, 0].isin(ground_truth_df.iloc[:, 0])]
    prediction_df.index = range(prediction_df.shape[0])

    # Extract the mask_rle columns
    pred_mask_rle = prediction_df.iloc[:, 1]
    gt_mask_rle = ground_truth_df.iloc[:, 1]

    def calculate_dice(pred_rle, gt_rle):
        pred_mask = rle_decode(pred_rle, img_shape)
        gt_mask = rle_decode(gt_rle, img_shape)


        if np.sum(gt_mask) > 0 or np.sum(pred_mask) > 0:
            return dice_score(pred_mask, gt_mask)
        else:
            return None  # No valid masks found, return None

    dice_scores = Parallel(n_jobs=-1)(
        delayed(calculate_dice)(pred_rle, gt_rle) for pred_rle, gt_rle in zip(pred_mask_rle, gt_mask_rle)
    )

    dice_scores = [score for score in dice_scores if score is not None]  # Exclude None values
    return np.mean(dice_scores)
