import os
import logging
import numpy as np
import pandas as pd
from functools import partial
from sklearn.model_selection import StratifiedKFold, KFold
from types import SimpleNamespace

import torch
from torch import optim, nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from warmup_scheduler import GradualWarmupScheduler

from config import get_args
from trainer import Trainer
from dice_loss import dice_loss
from utils import seed_everything, rle_encode
from models import HuggingFace, AutoImageProcessor
from data import ImageDataSet, load_image, train_transform, valid_transform
from auto_batch_size import max_gpu_batch_size

if __name__ == "__main__":
    args = get_args()
    seed_everything(args.seed) #fix seed
    device = torch.device('cuda:0') #use cuda:0

    if args.continue_train > 0:
        result_path = args.continue_from_folder
    else:
        result_path = os.path.join(args.result_path, args.pretrained_model.replace('/', '_') +'_'+str(len(os.listdir(args.result_path))))
        os.makedirs(result_path)
    
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    logger = logging.getLogger()
    logger.addHandler(logging.FileHandler(os.path.join(result_path, 'log.log')))    
    logger.info(args)
    #logger to log result of every output

    train_data = pd.read_csv(args.train)
    train_data['np_path'] = train_data['np_path'].apply(lambda x: os.path.join(args.path, x))
    test_data = pd.read_csv(args.test)
    test_data['np_path'] = test_data['np_path'].apply(lambda x: os.path.join(args.path, x))
    #fix path based on the data dir

    processor = AutoImageProcessor.from_pretrained(args.pretrained_model)#normalization은 유지 #, reduce_labels=True) #reduce_label remove background class
    #process image using pretrained model's AutoImageProcessor

    post_processor = processor.post_process_semantic_segmentation
    processor = partial(processor, return_tensors='pt', do_normalize=True, do_rescale=True, do_resize=False, do_center_crop=False) 

    input_size = (224, 224)
    output_size = 224*224

    skf = StratifiedKFold(n_splits=args.cv_k, random_state=args.seed, shuffle=True) #Using StratifiedKFold for cross-validation    
    for fold, (train_index, valid_index) in enumerate(skf.split(train_data['np_path'], train_data['has_mask'])): #by skf every fold will have similar label distribution
        if args.continue_train > fold+1:
            logger.info(f'skipping {fold+1}-fold')
            continue
        fold_result_path = os.path.join(result_path, f'{fold+1}-fold')
        os.makedirs(fold_result_path)
        fold_logger = logger.getChild(f'{fold+1}-fold')
        fold_logger.handlers.clear()
        fold_logger.addHandler(logging.FileHandler(os.path.join(fold_result_path, 'log.log')))    
        fold_logger.info(f'start training of {fold+1}-fold')
        #logger to log current n-fold output

        kfold_train_data = train_data.iloc[train_index]
        kfold_valid_data = train_data.iloc[valid_index]

        train_dataset = ImageDataSet(file_list=kfold_train_data['np_path'].values, transform=train_transform, mask=kfold_train_data['mask_rle'].values, label=kfold_train_data['has_mask'].values) #label -> True if the image contains Building 
        valid_dataset = ImageDataSet(file_list=kfold_valid_data['np_path'].values, transform=valid_transform, mask=kfold_valid_data['mask_rle'].values, label=kfold_valid_data['has_mask'].values)

        model = HuggingFace(args, {0: 'Neg', 1:'Pos'}, {'Neg':0, 'Pos':1}).to(device) #make model based on the model name and args
        loss_fn = dice_loss if args.dice_loss > 0.0 else lambda *x, **y: 0 #args.dice_loss = 0 -> not using dice loss for it
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
        scheduler_warmup = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=7, after_scheduler=scheduler)


        if args.batch_size == None: #if batch size is not defined -> calculate the appropriate batch size
            args.batch_size = max_gpu_batch_size(device, load_image, logger, model, loss_fn, train_dataset.max_length_file)
            model = HuggingFace(args, {0: 'Neg', 1:'Pos'}, {'Neg':0, 'Pos':1}).to(device)
            optimizer = optim.Adam(model.parameters(), lr=args.lr)
            scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
            scheduler_warmup = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=7, after_scheduler=scheduler)

        train_loader = DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True
        )
        valid_loader = DataLoader(
            valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True
        )

        trainer = Trainer(
            train_loader, valid_loader, model, loss_fn, optimizer, scheduler_warmup, device, processor, post_processor, args.patience, args.epochs, fold_result_path, fold_logger, len(train_dataset), len(valid_dataset), args.dice_loss)
        trainer.train() #start training

        test_dataset = ImageDataSet(file_list=test_data['np_path'].values, transform=valid_transform, mask=None, label=None)
        test_loader = DataLoader(
            test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers
        ) #make test data loader
        np.savez_compressed(os.path.join(fold_result_path, 'test_prediction'), trainer.test(test_loader)) #softmax applied output; accumulate test prediction of current fold model
        np.savez_compressed(os.path.join(fold_result_path, 'valid_prediction'), trainer.test(valid_loader))
        np.savez(os.path.join(fold_result_path, 'valid_index'), valid_index)