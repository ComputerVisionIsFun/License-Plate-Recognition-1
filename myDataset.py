from torch.utils.data import Dataset
from itertools import product
import XML
import encoding as E
import utils as U
import cv2
import os
from torchvision.transforms import  ToTensor, Compose, Normalize
import torch



class myDataset(Dataset):
    def __init__(self, ays:list, axs:list, xml_img_folder:str = '', rWidth = 416, rHeight = 416, oWidth=1280, oHeight=720, mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225]):
        self.anchors = U.Anchors(ays=ays, axs=axs)
        self.xml_img_folder = xml_img_folder
        self.rWidth, self.rHeight = rWidth, rHeight
        self.mean = mean
        self.std = std

        # find all annotations(self.annos) in the xml folder [{'filename':str, 'objs':[{'xmin':, ...},... ], {}, ...]
        
        self.annos, _ = XML.xmlDecodingInTheFolder(self.xml_img_folder)
        
        # resize: xmin, xmax, ymin, ymax-->xmin = xmin/(oWidth/rWidth), ...
        self.rannos = []
        resize_w_ratio, resize_h_ratio = oWidth/rWidth, oHeight/rHeight

        for anno in self.annos:
            ranno = {'filename':anno['filename'], 'objs':[]}
            for obj in anno['objs']:
                r_xmin, r_xmax = int(obj['xmin']/resize_w_ratio + .5), int(obj['xmax']/resize_w_ratio + .5)
                r_ymin, r_ymax = int(obj['ymin']/resize_h_ratio + .5), int(obj['ymax']/resize_h_ratio + .5)
                
                ranno['objs'].append({'xmin':r_xmin, 'xmax':r_xmax, 'ymin':r_ymin, 'ymax':r_ymax})
            
            self.rannos.append(ranno)
        self.transform = Compose(ToTensor(), Normalize(mean=self.mean, std=self.std)) 

    def __len__(self):
        return len(self.rannos['filename'])


    def __getitem__(self, idx):
        '''
        
        Return: a dict of the form {'image':3 x H x W tensor, 'target':1 x (H/cell_h) x (W/cell_w), 'area':1 x (H/cell_h) x (W/cell_w)}
        where H and W are the width and height of an image while cell_w and cell_h are the width and height of cells.
        
        '''
        
        filepath = os.path.join(self.xml_img_folder, self.rannos[idx]['filename'])
        image = cv2.imread(filepath, 1)# H x W x 3
        image = cv2.resize(image, (self.rWidth, self.rHeight), 1)
        objs = self.rannos[idx]['objs']
        
        target, area = E.encoding_area(objs, self.anchors)

        return {'image':self.transform(image), 'target':torch.from_numpy(target), 'area':torch.from_numpy(area)}

    

