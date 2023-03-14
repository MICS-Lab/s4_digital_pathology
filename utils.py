import os
import yaml
import models
import torch.nn
import numpy as np
import torch.optim
from tqdm import tqdm
from torch.optim import Optimizer
from collections import defaultdict
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
    if config.training.use_lookahead:
        optimizer = Lookahead(optimizer)
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
            loss = criterion(train_pred['Y_prob'], train_label.to(device))
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
            loss = criterion(val_pred['Y_prob'], val_label.to(device))

            val_labels.append(val_label)
            val_y_probs.append(val_pred['Y_prob'].cpu())
            val_y_hats.append(val_pred['Y_hat'].cpu())

            val_loss.append([loss.detach().cpu().tolist()] * len(val_data))
            val_progress.set_description(f'Valid [{str(epoch+1).zfill(format_epoch_width)}/{config.training.max_epochs}] | Loss {np.mean(val_loss):.4f} | Accuracy ...... | AUC ......')

            if len(val_labels) == len(val_progress):
                val_labels, val_y_probs, val_y_hats = torch.cat(val_labels), torch.vstack(val_y_probs), torch.cat(val_y_hats)
                epoch_accuracy, epoch_auc = get_epoch_metrics(config.data.n_classes, val_labels, val_y_probs, val_y_hats)
                val_progress.set_description(f'Valid [{str(epoch+1).zfill(format_epoch_width)}/{config.training.max_epochs}] | Loss {np.mean(val_loss):.4f} | Accuracy {epoch_accuracy:.4f} | AUC {epoch_auc:.4f}')
        
        mean_val_loss = np.mean(val_loss)
        if mean_val_loss < best_val_loss:
            if 'model_save_path' in locals():
                os.unlink(model_save_path)
            model_save_path = os.path.join(config.training.save_path, f'fold_{config.data.fold}_loss_{mean_val_loss:.8f}.pt')
            print(f'New best validation loss ({best_val_loss:.4f} -> {mean_val_loss:.4f}). Saving model to {model_save_path}.')
            torch.save(model.state_dict(), model_save_path)
            best_val_loss = mean_val_loss
            patience = 0
        else:
            patience += 1
            print(f'Best validation loss was not beaten. Patience [{patience}/{config.training.patience}].')
        
        if patience == config.training.patience:
            print(f'Patience limit was reached. Ending model training. Best model at {model_save_path}')
            break

def eval(config, model, device, test_dataloader, model_path):
    """
        Train a given model on a single fold.
        args:
            config [easydict.Easydict]: dictionary of the config
            model [torch.nn.Module]: model to evaluate
            device [torch.device]: device on which to run the model
            test_dataloader [torch.utils.data.dataloader.DataLoader]: test dataloader
            model_path [str]: path to the model (if None, then the model with the smallest loss is chosen)
        returns:
            None
    """
    if model_path is None:
        possible_paths = [path for path in os.listdir(config.training.save_path) if f'fold_{config.data.fold}' in path]
        possible_losses = [float(os.path.splitext(path.split('_')[-1])[0]) for path in possible_paths]
        assert len(possible_losses) > 0, f'No trained model could be found at {config.training.save_path}.'
        model_path = os.path.join(config.training.save_path, possible_paths[np.argmin(possible_losses)])
    assert os.path.exists(model_path), f'The model path ({model_path}) does not exist.'
    model.load_state_dict(torch.load(model_path))
    model.eval()
    print(f'Loading model from {model_path}.')

    test_progress = tqdm(test_dataloader)
    test_labels, test_y_probs, test_y_hats = [], [], []
    for test_data, test_label in test_progress:
        with torch.no_grad():
            test_pred = model(test_data.to(device))
        test_labels.append(test_label)
        test_y_probs.append(test_pred['Y_prob'].cpu())
        test_y_hats.append(test_pred['Y_hat'].cpu())
    
    test_labels, test_y_probs, test_y_hats = torch.cat(test_labels), torch.vstack(test_y_probs), torch.cat(test_y_hats)
    test_accuracy, test_auc = get_epoch_metrics(config.data.n_classes, test_labels, test_y_probs, test_y_hats)
    print(f'Fold {config.data.fold} test accuracy: {test_accuracy:.8f}.')
    print(f'Fold {config.data.fold} test AUC: {test_auc:.8f}.')
    print(f'Fold {config.data.fold} confusion matrix:')
    for label in np.unique(test_labels):
        print(f'\tFor label {int(label)}')
        for pred in np.unique(test_labels):
            pred_count = int(sum(test_y_hats[np.where(test_labels == label)] == pred))
            print(f'\t\tPredicted {int(pred)}: {pred_count}')

# The Lookahead class is taken from TransMIL's implementation of Lookahead https://github.com/szc19990412/TransMIL/blob/3f6bbe868ac39e7d861a111398b848ba3b943ca8/MyOptimizer/lookahead.py
class Lookahead(Optimizer):
    def __init__(self, base_optimizer, alpha=0.5, k=6):
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f'Invalid slow update rate: {alpha}')
        if not 1 <= k:
            raise ValueError(f'Invalid lookahead steps: {k}')
        defaults = dict(lookahead_alpha=alpha, lookahead_k=k, lookahead_step=0)
        self.base_optimizer = base_optimizer
        self.param_groups = self.base_optimizer.param_groups
        self.defaults = base_optimizer.defaults
        self.defaults.update(defaults)
        self.state = defaultdict(dict)
        # manually add our defaults to the param groups
        for name, default in defaults.items():
            for group in self.param_groups:
                group.setdefault(name, default)

    def update_slow(self, group):
        for fast_p in group["params"]:
            if fast_p.grad is None:
                continue
            param_state = self.state[fast_p]
            if 'slow_buffer' not in param_state:
                param_state['slow_buffer'] = torch.empty_like(fast_p.data)
                param_state['slow_buffer'].copy_(fast_p.data)
            slow = param_state['slow_buffer']
            slow.add_(fast_p.data - slow, alpha=group['lookahead_alpha'])
            fast_p.data.copy_(slow)

    def sync_lookahead(self):
        for group in self.param_groups:
            self.update_slow(group)

    def step(self, closure=None):
        #assert id(self.param_groups) == id(self.base_optimizer.param_groups)
        loss = self.base_optimizer.step(closure)
        for group in self.param_groups:
            group['lookahead_step'] += 1
            if group['lookahead_step'] % group['lookahead_k'] == 0:
                self.update_slow(group)
        return loss

    def state_dict(self):
        fast_state_dict = self.base_optimizer.state_dict()
        slow_state = {
            (id(k) if isinstance(k, torch.Tensor) else k): v
            for k, v in self.state.items()
        }
        fast_state = fast_state_dict['state']
        param_groups = fast_state_dict['param_groups']
        return {
            'state': fast_state,
            'slow_state': slow_state,
            'param_groups': param_groups,
        }

    def load_state_dict(self, state_dict):
        fast_state_dict = {
            'state': state_dict['state'],
            'param_groups': state_dict['param_groups'],
        }
        self.base_optimizer.load_state_dict(fast_state_dict)

        # We want to restore the slow state, but share param_groups reference
        # with base_optimizer. This is a bit redundant but least code
        slow_state_new = False
        if 'slow_state' not in state_dict:
            print('Loading state_dict from optimizer without Lookahead applied.')
            state_dict['slow_state'] = defaultdict(dict)
            slow_state_new = True
        slow_state_dict = {
            'state': state_dict['slow_state'],
            'param_groups': state_dict['param_groups'],  # this is pointless but saves code
        }
        super(Lookahead, self).load_state_dict(slow_state_dict)
        self.param_groups = self.base_optimizer.param_groups  # make both ref same container
        if slow_state_new:
            # reapply defaults to catch missing lookahead specific ones
            for name, default in self.defaults.items():
                for group in self.param_groups:
                    group.setdefault(name, default)