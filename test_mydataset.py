import myDataset as md
import yaml
import utils as U
import cv2
import numpy as np
import visualize as V



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
idx = 10
img = ds[idx]['image'].detach().cpu().numpy().transpose(1,2,0)
target = ds[idx]['target'].detach().cpu().numpy()
objness = np.where(target[0, :, :]==1)
cxs = target[1, :, :]
cys = target[2, :, :]


img = (img*255).astype('uint8').copy()

# img = np.zeros((500,500,3),dtype='uint8')

# plot grid
V.draw_grid(img, ds.anchors.axs, ds.anchors.ays)

# plot anchors
V.draw_anchors(img, ds.anchors)


# plot objects
objness_xs, objness_ys = objness[1], objness[0]
num_objs = len(objness[1])
cell_w_half = (ds.anchors.axs[1] - ds.anchors.axs[0])/2
cell_h_half = (ds.anchors.ays[1] - ds.anchors.ays[0])/2

objs = []
for obj_i in range(num_objs):
    






for i, ax in enumerate(objness_xs):
    ay = objness_ys[i]
    # print((ds.anchors.axs[ax], ds.anchors.ays[ay]))
    cv2.circle(img, (ds.anchors.axs[ax], ds.anchors.ays[ay]),3, (255,255,0),-1,8,0)
    







cv2.namedWindow("", 2)
cv2.imshow("", img)
cv2.waitKey(0)

