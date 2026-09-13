from .st_agbilstm import STAGBiLSTMModel
from .vgg_rnn import VGGRnnModel

## Get your models from here
__all__ = [
    "VGGRnnModel",
    "STAGBiLSTMModel"
]

def get_model_list():
    """List of Available models"""
    return __all__
