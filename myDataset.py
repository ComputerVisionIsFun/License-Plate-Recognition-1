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
        self.xml_img_folder = xml_img_folder

        # find all annotations in the xml folder [{'filename':str, 'objs':['xmin':,... ]}, {}, ...]
        '''
        self.annos is a list, and for example, self.annos[0] = {'filename': 'load_2397.png', 'objs': [{'xmin': 442, 'xmax': 568, 'ymin': 242, 'ymax': 291}]}
        '''
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
                

        
        # encoding


        
    def __len__(self):
        return 


    def __getitem__(self):
        return 
