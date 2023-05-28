from torch.utils.data import Dataset
from itertools import product
import XML



Axs = [16+i*32 for i in range(13)]
Ays = [16+i*32 for i in range(13)]


class myDataset(Dataset):
    def __init__(self, ays=Ays, axs=Axs):
        self.anchors = product(ays, axs)
        self.axs, self.ays = axs, ays

        
    def __len__(self):


    def __getitem__(self):
