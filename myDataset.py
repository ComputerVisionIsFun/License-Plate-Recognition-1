from torch.utils.data import Dataset
from itertools import product
import XML
import encoding as E
import utils as U
import cv2
import os
from torchvision.transforms import  ToTensor
import torch


class myDataset(Dataset):
    def __init__(self, ays:list, axs:list, xml_img_folder:str = '', rWidth = 416, rHeight = 416, oWidth=1280, oHeight=720):
        self.anchors = U.Anchors(ays=ays, axs=axs)
        self.xml_img_folder = xml_img_folder
        self.rWidth, self.rHeight = rWidth, rHeight

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
        self.transform = ToTensor()

    def __len__(self):
        return len(self.rannos['filename'])


    def __getitem__(self, idx):
        
        filepath = os.path.join(self.xml_img_folder, self.rannos[idx]['filename'])
        image = cv2.imread(filepath, 1)# H x W x 3
        image = cv2.resize(image, (self.rWidth, self.rHeight), 1)
        objs = self.rannos[idx]['objs']
        # print(objs,'---------------------')
        target, area = E.encoding_area(objs, self.anchors)

        return {'image':self.transform(image), 'target':torch.from_numpy(target), 'area':torch.from_numpy(area)}

    

