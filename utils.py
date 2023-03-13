import os
import yaml
import models
import torch.optim
from easydict import EasyDict as edict

def read_config(config_path):
    """
        Read the config file and check the relevance of the data.
        args:
            config_path [str]: path to the .yaml config file
        outputs:
            config [easydict.Easydict]: dictionary of the config
    """
    with open(config_path, 'r') as f:
        config = edict(yaml.safe_load(f))
    
    # check the model section
    assert hasattr(models, config.model.model_type), f'model.model_config ({config.model.model_type}) could not be found.'
    assert isinstance(config.model.model_dim, int), f'model.model_dim must be of type int, not {type(config.model.model_dim).__name__}.'
    assert isinstance(config.model.state_dim, int), f'model.state_dim must be of type int, not {type(config.model.state_dim).__name__}.'

    # check the data section
    assert os.path.exists(config.data.data_path), f'data.data_path does not exist.'
    assert os.path.exists(config.data.labels_path), f'data.labels_path does not exist.'
    assert isinstance(config.data.n_classes, int), f'data.n_classes must be of type int, not {type(config.data.n_classes).__name__}.'
    assert isinstance(config.data.input_dim, int), f'data.input_dim must be of type int, not {type(config.data.input_dim).__name__}.'

    # check the training section
    assert hasattr(torch.optim, config.training.optimizer), f'model.training.optimizer ({config.training.optimizer}) could not be found in torch.optim.'

    return config