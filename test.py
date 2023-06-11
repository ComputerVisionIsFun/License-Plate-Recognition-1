import myDataset as md
import os 
import cv2
import matplotlib.pyplot as plt
import utils as U
import encoding as E
import numpy as np
import test_paras as tp
import pandas as pd

ds = md.myDataset(axs=tp.axs,ays=tp.ays,xml_img_folder=tp.xml_img_folder,rWidth=tp.rWidth,rHeight=tp.rHeight,oWidth=tp.oWidth,oHeight=tp.oHeight)
ind = 379#40 for bad annotation
iou_th = 199


obj_widths, obj_heights = [], []

obj_sizes = pd.DataFrame()


for ranno in ds.rannos:
    # read image
    image_path = os.path.join(tp.xml_img_folder, ranno['filename'])
    image = cv2.imread(image_path, 1)
    rimage = cv2.resize(image, (tp.rWidth,tp.rHeight), 1)
    # draw objs
    objs = ranno['objs']
    for i, obj in enumerate(objs):
        xmin, xmax = int(obj['xmin']), int(obj['xmax'])
        ymin, ymax = int(obj['ymin']), int(obj['ymax'])
        if (xmax - xmin)>=32:
            cv2.rectangle(rimage, (xmin, ymin), (xmax, ymax), (255,1,1),3,8,0)
        
        obj_widths.append(xmax - xmin)
        obj_heights.append(ymax - ymin)
        
    # plot grid
    rimage = tp.draw_grid(rimage, xticks=[i for i in range(0, tp.rHeight, 32)], yticks=[i for i in range(0, tp.rHeight, 32)])

    # plot anchors
    rimage = tp.draw_anchors(rimage, ds.anchors)

    # plt.yticks([i for i in range(0, tp.rHeight, 32)])
    # plt.xticks([i for i in range(0, tp.rWidth, 32)])
    # plt.grid(color='black', alpha=.5)        

    cv2.imshow("", rimage)
    cv2.waitKey(0)


    # plot anchors

    # for anchor in ds.anchors.anchors:
    #     plt.scatter(anchor[1], anchor[0], color='green', s = 3)

    save_path = os.path.join(tp.draw_folder, ranno['filename'])
    # rimage = cv2.cvtColor(rimage, cv2.COLOR_BGR2RGB)
    # plt.tight_layout()
    # plt.imshow(rimage)
    # plt.show()
    # plt.savefig(save_path)
    # plt.close()
    # cv2.imshow("", rimage)
    # cv2.waitKey(0)
    # cv2.imwrite(save_path, rimage)


    # 
obj_sizes['width'] = obj_widths
obj_sizes['height'] = obj_heights

obj_sizes.to_excel(os.path.join(tp.draw_folder, 'obj_size.xlsx'))



# ranno = ds.rannos[ind]
# image_path = os.path.join(tp.xml_img_folder, ranno['filename'])

# objs = ranno['objs']
# # print(objs)
# image = cv2.imread(image_path, 1)
# rimage = cv2.resize(image, (tp.rWidth,tp.rHeight), 1)
# for i, obj in enumerate(objs):
#     xmin, xmax = int(obj['xmin']), int(obj['xmax'])
#     ymin, ymax = int(obj['ymin']), int(obj['ymax'])
#     cv2.rectangle(rimage, (xmin, ymin), (xmax, ymax), (255,1,1),3,8,0)
#     # print(i, obj)


# plt.yticks([i for i in range(0, tp.rHeight, 32)])
# plt.xticks([i for i in range(0, tp.rWidth, 32)])
# plt.grid(color='black', alpha=.5)

# # plot anchors
# for anchor in ds.anchors.anchors:
#     plt.scatter(anchor[1], anchor[0], color='green', s = 3)

'''
test annotation
'''


# target = E.encoding(objs, ds.anchors, tp.rWidth, tp.rHeight, iou_th, tp.cell_w, tp.cell_h)
# objness = np.where(target[0, :, :]>0)
# obj_xs = target[1, :, :]
# obj_ys = target[2, :, :]
# obj_ws = target[3, :, :]
# obj_hs = target[4, :, :]
# num_objness = objness[0].shape[0]
# print(objness)
# for i in range(num_objness):
#     # plot cell
#     # print('---------------', i)
#     row, col = objness[0][i], objness[1][i]
#     ax, ay = ds.anchors.axs[col], ds.anchors.ays[row]
#     cell_xmin, cell_xmax = int(ax - tp.cell_w/2), int(ax + tp.cell_w/2)
#     cell_ymin, cell_ymax = int(ay - tp.cell_h/2), int(ay + tp.cell_h/2)
#     cv2.rectangle(rimage,(cell_xmin,cell_ymin),(cell_xmax,cell_ymax),(0,0,255),3,8,0)

# cv2.imshow("", rimage)
# cv2.waitKey(0)

# plt.imshow(rimage)

# plt.show()


