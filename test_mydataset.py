import myDataset as md
import yaml
import utils as U
import cv2
import numpy as np
import visualize as V
import decoding as D


with open('config.yaml') as f:
    parameters = yaml.safe_load(f)

U.show_train_info(title = 'parameters of training', parameters=parameters) 

# parameters for md.myDataset

axs = [16+i*32 for i in range(13)]
ays = [16+i*32 for i in range(13)]
xml_img_folder = parameters['TRAINING_FOLDER']
rWidth, rHeight = parameters['RWIDTH'], parameters["RHEIGHT"]
oWidth, oHeight = parameters['OWIDTH'], parameters["OHEIGHT"]

# 

ds = md.myDataset(ays,axs,xml_img_folder,rWidth,rHeight,oWidth,oHeight)
idx = 156
img = D.inverse_normalize(ds[idx]['image'], mean = [0.485,0.456,0.406], std = [0.229,0.224,0.225])

target = ds[idx]['target']

# plot grid
V.draw_grid(img, ds.anchors.axs, ds.anchors.ays)

# # plot anchors
V.draw_anchors(img, ds.anchors)

# plot objects
img = D.target_to_vis(ds.anchors, img, target)

cv2.namedWindow("", 2)
cv2.imshow("", img)
cv2.waitKey(0)

