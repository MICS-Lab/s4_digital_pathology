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

    return config

def check_config(config):
    """
        Check the config file (that may have been modified since reading).
        args:
            config [easydict.Easydict]: dictionary of the config
        outputs:
            None
    """
    # check the model section
    assert hasattr(models, config.model.model_type), f'model.model_config ({config.model.model_type}) could not be found.'
    assert isinstance(config.model.model_dim, int), f'model.model_dim must be of type int, not {type(config.model.model_dim).__name__}.'
    assert isinstance(config.model.state_dim, int), f'model.state_dim must be of type int, not {type(config.model.state_dim).__name__}.'

    # check the data section
    assert os.path.exists(config.data.data_path), f'data.data_path does not exist.'
    assert os.path.exists(config.data.folds_path), f'data.folds_path does not exist.'
    assert isinstance(config.data.n_classes, int), f'data.n_classes must be of type int, not {type(config.data.n_classes).__name__}.'
    assert isinstance(config.data.input_dim, int), f'data.input_dim must be of type int, not {type(config.data.input_dim).__name__}.'
    assert isinstance(config.data.fold, int), f'data.fold must be of type int, not {type(config.data.fold).__name__}.'
    assert 0 < config.data.fold < config.data.n_fold, f'config.data.fold msut be in [0, config.data.n_fold[.'

    # check the training section
    assert hasattr(torch.optim, config.training.optimizer), f'model.training.optimizer ({config.training.optimizer}) could not be found in torch.optim.'