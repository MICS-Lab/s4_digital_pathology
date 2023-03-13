import torch
import utils
import models
import random
import argparse
from s4dataset import S4Dataset
from torch.utils.data import DataLoader


parser = argparse.ArgumentParser()
parser.add_argument('--config', required=True, type=str, help='Path to the .yaml config file.')
parser.add_argument('--fold', required=False, type=int, default=None, help='Fold on which to launch training.')
args = parser.parse_args()

# read the config file and update if needed
config = utils.read_config(args.config)
utils.check_config(config)
if args.fold is not None:
    config.data.fold = args.fold
random.seed(config.seed)
torch.manual_seed(config.seed)

# create the model (note the S4Model is not yet compatible with MPS)
device = torch.device('gpu') if torch.cuda.is_available() else torch.device('cpu')
model = getattr(models, config.model.model_type)(config).to(device)

# create the datasets and dataloaders
train_dataset = S4Dataset(config, 'train')
val_dataset = S4Dataset(config, 'val')
train_dataloader = DataLoader(train_dataset, batch_size=config.data.batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=config.data.batch_size, shuffle=False)

utils.train(config, model, device, train_dataloader, val_dataloader)