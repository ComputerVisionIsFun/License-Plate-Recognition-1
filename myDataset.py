from torch.utils.data import Dataset
from itertools import product
import XML
import encoding as E
import utils as U


# Axs = [16+i*32 for i in range(13)]
# Ays = [16+i*32 for i in range(13)]


class myDataset(Dataset):
    def __init__(self, ays:list, axs:list, xml_img_folder:str = '', rWidth = 416, rHeight = 416, oWidth=1280, oHeight=720):
        self.anchors = U.Anchors(ays=ays, axs=axs)
        
        # find all annotations in the xml folder [{'filename':str, 'objs':['xmin':,... ]}, {}, ...]



        # resize: xmin, xmax, ymin, ymax-->xmin = xmin/(oWidth/rWidth), ...


        
        # encoding


        
    def __len__(self):



    def __getitem__(self):
