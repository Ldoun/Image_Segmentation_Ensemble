from models.pretrained_model import HuggingFace, AutoFeatureExtractor

def args_for_HuggingFace(parser):
    parser.add_argument('--pretrained_model', type=str, default="facebook/wav2vec2-base", help="pretrained model name")
    parser.add_argument('--model_type', type=str, default="image", choices=['image', 'instance', 'semantic', 'universal'])