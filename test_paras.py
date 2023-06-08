import numpy as np
import cv2
import utils as U


axs = [16+i*32 for i in range(13)]
ays = [16+i*32 for i in range(13)]
axs = [16+i*16 for i in range(26)]
ays = [16+i*16 for i in range(26)]
xml_img_folder = '/Users/chiang-en/Desktop/LPR_dataset/xml/'
draw_folder = './draw_1/'


rWidth = 416
rHeight = 416
oWidth=1280
oHeight=720 
cell_w, cell_h = 64, 48



def draw_grid(image:np.array, xticks:list, yticks:list):
    xlim = [0, image.shape[1] - 1]
    ylim = [0, image.shape[0] - 1]

    for xtick in xticks:
        cv2.line(image, (xtick, ylim[0]), (xtick, ylim[1]), (128, 128, 128), 1, 8, 0)

    for ytick in yticks:
        cv2.line(image, (xlim[0], ytick), (xlim[1], ytick), (128, 128, 128), 1, 8, 0)

    return image

def draw_anchors(image:np.array, anchors:U.Anchors):
    for anchor in anchors.anchors:
        cx, cy = anchor[1], anchor[0]
        cv2.circle(image, (cx, cy), 2, (0, 255, 0), -1, 8, 0)

    return image









