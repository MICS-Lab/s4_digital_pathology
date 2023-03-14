# Structured state space models for multiple instance learning in digital pathology

## Getting started
The correct conda environment can be setup using the `environment.yaml` file.
```bash
conda env create --file environment.yaml
# if you want to run the code on CUDA, run the following line as well
conda install pytorch torchvision torchaudio pytorch-cuda=11.6 -c pytorch -c nvidia
```

## Running the code
### Data
The splits for cross validation should be placed in `.csv` files under the names `fold{k}.csv` with six columns (train, train_label, val, val_label, test, and test_label). They should be placed in `folds/{dataset_name}`. All `.pt` files containing the sequence of patch features for the WSIs should be placed in `data/{dataset_name}`. An example `.csv` file has been placed in `folds/camelyon16`.

### Configuration
The `.yaml` configuration file specifies everything needed for the training and evaluation of the model.
```yaml
seed: the random seed
model:
  model_type: the name of the model (currently on S4Model is supported)
  model_dim: the dimension onto which the input data is projected
  state_dim: the state dimension of the S4D module
data:
  data_path: the path to the directoy containing the .pt files of the dataset
  folds_path: the path to the directory containing the .csv files of the folds
  n_classes: the number of classes in the dataset
  input_dim: the dimension of the input feature vectors
  fold: the fold number (may be overwritten during training)
  batch_size: the batch size (currently only 1 is supported)
training:
  optimizer: the name of the optimizer (should be in torch.optim)
  use_lookahead: whether or not to use the Lookahead optimizer on the of the chosen optimizer
  lr: the learning rate
  wd: the weight decay
  patience: the patience used for early stopping
  max_epochs: the maximum number of epochs to train for
  loss: the loss used for training (should be in torch.nn)
  save_path: the path to which the models are saved
```

### Training
To launch a training, specify the configuration file as well as the split (by default, the split is the one specified in the configuration file, but the argument overwrites it). The model with the lowest validation loss is saved in the path specified in the configuration file under the name `fold_{k}_loss_{lowest_val_loss}.pt`.
```bash
python train.py --config <path to config.yaml file> --fold <fold number>
```

### Evalutation
To evaluate a trained model, specify the configuration file which was used to train it, the fold number, and optionally the path to the saved model (by default, the model loads the model with the lowest validation loss).
```bash
python eval.py --config <path to the config.yaml file> --fold <fold number> --model_path <optional model path>
```

## Credits
The code for the S4D module (`models/s4.py`) was taken from the [original S4 repository](https://github.com/HazyResearch/state-spaces).

The code for the lookahead optimizer (`Lookahead` class in `utils.py`) was taken from [TransMIL's repository](https://github.com/szc19990412/TransMIL/blob/3f6bbe868ac39e7d861a111398b848ba3b943ca8/MyOptimizer/lookahead.py).