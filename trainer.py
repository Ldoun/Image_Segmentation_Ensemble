import os
import sys
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm

from utils import batch_dice_score

class Trainer():
    def __init__(self, train_loader, valid_loader, model, loss_fn, optimizer, device, processor, post_processor, patience, epochs, result_path, fold_logger, len_train, len_valid):
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.device = device
        self.processor = processor
        self.post_processor = post_processor
        self.patience = patience
        self.epochs = epochs
        self.logger = fold_logger
        self.best_model_path = os.path.join(result_path, 'best_model.pt')
        self.len_train = len_train
        self.len_valid = len_valid
    
    def train(self):
        best = np.inf
        for epoch in range(1,self.epochs+1):
            loss_train, score_train = self.train_step()
            loss_val, score_val = self.valid_step()

            self.logger.info(f'Epoch {str(epoch).zfill(5)}: t_loss:{loss_train:.3f} t_score:{score_train:.3f} v_loss:{loss_val:.3f} v_score:{score_val:.3f}')

            if loss_val < best:
                best = loss_val
                torch.save(self.model.state_dict(), self.best_model_path)
                bad_counter = 0

            else:
                bad_counter += 1

            if bad_counter == self.patience:
                break

    def train_step(self):
        self.model.train()

        total_loss = 0
        correct = 0
        for batch in tqdm(self.train_loader, file=sys.stdout): #tqdm output will not be written to logger file(will only written to stdout)
            x = self.processor(batch['image'], segmentation_maps=batch['mask'])
            x, y = x.to(self.device), batch['label'].to(self.device)
            
            self.optimizer.zero_grad()
            if 'labels' not in x.keys():# need to check once more 
                mask = batch['mask'].to(self.device)
                output = self.model(pixel_values=x['pixel_values'], labels=mask)
            else:
                output = self.model(**x)            
            
            loss = output.loss#self.loss_fn(output, y)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item() # *x.shape[0]
            segmentatation_result = self.post_processor(output, target_sizes=[[224, 224]]*x['pixel_values'].shape[0])
            correct += batch_dice_score(segmentatation_result, mask.detach().cpu().numpy())
        
        return total_loss/self.len_train, correct/self.len_train
    
    def valid_step(self):
        self.model.eval()
        with torch.no_grad():
            total_loss = 0
            correct = 0
            for batch in self.valid_loader:
                x = self.processor(batch['image'], segmentation_maps=batch['mask'])
                x, y = x.to(self.device), batch['label'].to(self.device)

                if 'labels' not in x.keys():# need to check once more 
                    mask = batch['mask'].to(self.device)
                    output = self.model(pixel_values=x['pixel_values'], labels=mask)
                else:
                    output = self.model(**x)
                
                loss = output.loss
                total_loss += loss.item() 
                segmentatation_result = self.post_processor(output, target_sizes=[[224, 224]]*x['pixel_values'].shape[0])
                correct += batch_dice_score(segmentatation_result, mask.detach().cpu().numpy())
                
        return total_loss/self.len_valid, correct/self.len_valid

    def test(self, test_loader):
        self.model.load_state_dict(torch.load(self.best_model_path))
        self.model.eval()
        with torch.no_grad():
            result = []
            for batch in tqdm(test_loader, file=sys.stdout):
                x = self.processor(batch['image']).to(self.device)
                output = self.model(pixel_values=x['pixel_values']).logits.detach().reshape(x['pixel_values'].shape[0], -1).cpu().numpy()
                #segmentatation_result = self.post_processor(output, target_sizes=[[224, 224]]*x['pixel_values'].shape[0]) #need fix for high temperature softmax value
                
                result.append(output)
        return np.concatenate(result,axis=0)