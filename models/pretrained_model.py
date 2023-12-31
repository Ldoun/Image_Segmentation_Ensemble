import torch.nn as nn
from transformers import AutoModelForImageSegmentation, AutoModelForInstanceSegmentation, AutoModelForSemanticSegmentation, AutoModelForUniversalSegmentation, AutoImageProcessor

class HuggingFace(nn.Module):
    def __init__(self, args, id2label, label2id):
        super().__init__()

        if args.model_type == 'image':
            self.model = AutoModelForImageSegmentation.from_pretrained(args.pretrained_model)
        elif args.model_type == 'instance':
            self.model = AutoModelForInstanceSegmentation.from_pretrained(args.pretrained_model) #instance Segmentation: R channel classify caegory, G channel classify instance
        elif args.model_type == 'semantic':
            self.model = AutoModelForSemanticSegmentation.from_pretrained(args.pretrained_model, id2label=id2label, label2id=label2id, ignore_mismatched_sizes=True) #ignore_mismatched_sizes=True
        elif args.model_type == 'universal':
            self.model = AutoModelForUniversalSegmentation.from_pretrained(args.pretrained_model)
        else:
            raise 'model unknown'

    def forward(self, **kwargs):
        return self.model(**kwargs)