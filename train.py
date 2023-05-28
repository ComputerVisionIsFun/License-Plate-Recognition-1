import yaml
import utils
from torch.utils.data import DataLoader
import myDataset as md



# Load config for training
with open('./config.yaml') as f:
    config = yaml.safe_load(f)
    
utils.show_train_info('config', config)


# Assign dataset and dataloader


# Assign a model, loss function and optimizer


# training



