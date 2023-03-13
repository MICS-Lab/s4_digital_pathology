import os
import yaml
import models
import torch.nn
import numpy as np
import torch.optim
from tqdm import tqdm
from easydict import EasyDict as edict
from sklearn.metrics import accuracy_score, roc_auc_score

def read_config(config_path):
    """
        Read the config file and check the relevance of the data.
        args:
            config_path [str]: path to the .yaml config file
        returns:
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
        returns:
            None
    """
    # check the seed
    assert isinstance(config.seed, int), f'seed must be of type int, not {type(config.seed).__name__}.'

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
    assert config.data.fold >= 0, f'config.data.fold must be positive.'
    assert isinstance(config.data.batch_size, int), f'data.batch_size must be of type int, not {type(config.data.batch_size).__name__}.'
    
    # check the training section
    assert hasattr(torch.optim, config.training.optimizer), f'model.training.optimizer ({config.training.optimizer}) could not be found in torch.optim.'
    assert isinstance(config.training.lr, float), f'config.training.lr must be of type float, not {type(config.config.training.lr).__name__}.'
    assert isinstance(config.training.wd, float), f'config.training.wd must be of type float, not {type(config.config.training.wd).__name__}.'
    assert isinstance(config.training.patience, int), f'config.training.patience must be of type int, not {type(config.config.training.patience).__name__}.'
    assert isinstance(config.training.max_epochs, int), f'config.training.max_epochs must be of type int, not {type(config.config.training.max_epochs).__name__}.'
    assert hasattr(torch.nn, config.training.loss), f'config.training.loss ({config.training.loss}) could not be found in torch.nn.'
    assert os.path.exists(config.training.save_path), f'config.training.save_path does not exist.'

def get_epoch_metrics(n_classes, labels, y_probs, y_hats):
    """
        Compute the accuracy and the AUC at the end of an epoch.
        args:
            n_classes [int]: number of classes
            labels [torch.Tensor]: stacked labels over the epoch
            y_probs [torch.Tensor]: stacked Y_prob over the epoch
            y_hats [torch.Tensor]: stacked Y_hat over the epoch
        returns:
            accuracy [float]: epoch accuracy
            auc [float]: epoch AUC
    """
    accuracy = accuracy_score(labels, y_hats)
    if n_classes == 2:
        auc = roc_auc_score(labels, y_probs[:, -1])
    else:
        auc = roc_auc_score(labels, y_probs, multi_class='ovr', average='weighted')
    return accuracy, auc

def train(config, model, device, train_dataloader, val_dataloader):
    """
        Train a given model on a single fold.
        args:
            config [easydict.Easydict]: dictionary of the config
            model [torch.nn.Module]: model to train
            device [torch.device]: device on which to train
            train_dataloader [torch.utils.data.dataloader.DataLoader]: train dataloader
            val_dataloader [torch.utils.data.dataloader.DataLoader]: validation dataloader
        returns:
            None
    """
    format_epoch_width = 1+int(np.log10(config.training.max_epochs))
    criterion = getattr(torch.nn, config.training.loss)()
    optimizer = getattr(torch.optim, config.training.optimizer)(model.parameters(), lr=config.training.lr, weight_decay=config.training.wd)
    best_val_loss = np.inf
    patience = 0

    for epoch in range(config.training.max_epochs):
        # training phase
        model.train()
        training_progress = tqdm(train_dataloader)
        training_progress.set_description(f'Train [{str(epoch+1).zfill(format_epoch_width)}/{config.training.max_epochs}] | Loss {0:.4f} | Accuracy ...... | AUC ......')
        train_loss, train_labels, train_y_probs, train_y_hats = [], [], [], []
        for train_data, train_label in training_progress:
            optimizer.zero_grad()
            train_pred = model(train_data.to(device))
            loss = criterion(train_pred['Y_prob'], train_label)
            loss.backward()
            optimizer.step()

            train_labels.append(train_label)
            train_y_probs.append(train_pred['Y_prob'].detach().cpu())
            train_y_hats.append(train_pred['Y_hat'].detach().cpu())

            train_loss.append([loss.detach().cpu().tolist()] * len(train_data))
            training_progress.set_description(f'Train [{str(epoch+1).zfill(format_epoch_width)}/{config.training.max_epochs}] | Loss {np.mean(train_loss):.4f} | Accuracy ...... | AUC ......')

            if len(train_labels) == len(training_progress):
                train_labels, train_y_probs, train_y_hats = torch.cat(train_labels), torch.vstack(train_y_probs), torch.cat(train_y_hats)
                epoch_accuracy, epoch_auc = get_epoch_metrics(config.data.n_classes, train_labels, train_y_probs, train_y_hats)
                training_progress.set_description(f'Train [{str(epoch+1).zfill(format_epoch_width)}/{config.training.max_epochs}] | Loss {np.mean(train_loss):.4f} | Accuracy {epoch_accuracy:.4f} | AUC {epoch_auc:.4f}')

        # validation phase
        
        model.eval()
        val_progress = tqdm(val_dataloader)
        val_progress.set_description(f'Valid [{str(epoch+1).zfill(format_epoch_width)}/{config.training.max_epochs}] | Loss {0:.4f} | Accuracy ...... | AUC ......')
        val_loss, val_labels, val_y_probs, val_y_hats = [], [], [], []
        for val_data, val_label in val_progress:
            with torch.no_grad():
               val_pred = model(val_data.to(device))
            loss = criterion(val_pred['Y_prob'], val_label)

            val_labels.append(val_label)
            val_y_probs.append(val_pred['Y_prob'])
            val_y_hats.append(val_pred['Y_hat'])

            val_loss.append([loss.tolist()] * len(val_data))
            val_progress.set_description(f'Valid [{str(epoch+1).zfill(format_epoch_width)}/{config.training.max_epochs}] | Loss {np.mean(val_loss):.4f} | Accuracy ...... | AUC ......')

            if len(val_labels) == len(val_progress):
                val_labels, val_y_probs, val_y_hats = torch.cat(val_labels), torch.vstack(val_y_probs), torch.cat(val_y_hats)
                epoch_accuracy, epoch_auc = get_epoch_metrics(config.data.n_classes, val_labels, val_y_probs, val_y_hats)
                val_progress.set_description(f'Valid [{str(epoch+1).zfill(format_epoch_width)}/{config.training.max_epochs}] | Loss {np.mean(val_loss):.4f} | Accuracy {epoch_accuracy:.4f} | AUC {epoch_auc:.4f}')
        
        mean_val_loss = np.mean(val_loss)
        if mean_val_loss < best_val_loss:
            model_save_path = os.path.join(config.training.save_path, f'loss_{mean_val_loss:.8f}.pt')
            print(f'New best validation loss ({best_val_loss:.4f} -> {mean_val_loss:.4f}). Saving model to {model_save_path}.')
            best_val_loss = mean_val_loss
            patience = 0
        else:
            patience += 1
            print(f'Best validation loss was not beaten. Patience [{patience}/{config.training.patience}].')
        
        if patience == config.training.patience:
            print(f'Patience limit was reached. Ending model training. Best model at {model_save_path}')
            break