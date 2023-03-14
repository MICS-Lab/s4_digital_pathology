import torch
import utils
import models
import random
import argparse
from s4dataset import S4Dataset
from torch.utils.data import DataLoader

parser = argparse.ArgumentParser()
parser.add_argument('--config', required=True, type=str, help='Path to the .yaml config file.')
parser.add_argument('--model_path', required=False, type=str, default=None, help='Path to the .pt file of the trained model (default is the one with the lowest loss).')
parser.add_argument('--fold', required=False, type=int, default=None, help='Fold on which to launch the evaluation of the model.')
args = parser.parse_args()

# read the config file and update if needed
config = utils.read_config(args.config)
utils.check_config(config)
if args.fold is not None:
    config.data.fold = args.fold
random.seed(config.seed)
torch.manual_seed(config.seed)

# create the model (note the S4Model is not yet compatible with MPS)
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(f'Evaluating on {str(device)}.')
model = getattr(models, config.model.model_type)(config).to(device)

# create the datasets and dataloaders
test_dataset = S4Dataset(config, 'test')
test_dataloader = DataLoader(test_dataset, batch_size=config.data.batch_size, shuffle=False)

utils.eval(config, model, device, test_dataloader, args.model_path)