import argparse
from models import args_for_HuggingFace

def args_for_data(parser):
    parser.add_argument('--train', type=str, default='../data/train.csv')
    parser.add_argument('--test', type=str, default='../data/test.csv')
    parser.add_argument('--submission', type=str, default='../data/sample_submission.csv')
    parser.add_argument('--path', type=str, default='../data')
    parser.add_argument('--result_path', type=str, default='./result')
    
def args_for_image(parser):
    pass

def args_for_train(parser):
    parser.add_argument('--cv_k', type=int, default=10, help='k-fold stratified cross validation')
    parser.add_argument('--num_workers', type=int, default=4, help='num_workers')
    parser.add_argument('--batch_size', type=int, default=None, help='batch_size')
    parser.add_argument('--epochs', type=int, default=100, help='max epochs')
    parser.add_argument('--patience', type=int, default=7, help='patience for early stopping')    
    parser.add_argument('--lr', type=float, default=5e-4, help='learning rate for the optimizer')
    parser.add_argument('--warmup_epochs', type=int, default=3, help='number of warmup epoch of lr scheduler')

    parser.add_argument('--dice_loss', type=float, default=0.0, help='ratio of dice loss in total loss')

    parser.add_argument('--continue_train', type=int, default=-1, help='continue training from fold x') 
    parser.add_argument('--continue_from_folder', type=str, help='continue training from args.continue_from')

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', default=42, type=int)

    args_for_data(parser)
    args_for_train(parser)
    args_for_image(parser)
    args_for_HuggingFace(parser)

    args = parser.parse_args()
    return args
