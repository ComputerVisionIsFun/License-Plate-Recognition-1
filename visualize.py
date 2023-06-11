import cv2
import numpy as np
import utils as U
from typing import List









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



def draw_objs(image:np.array, objs:List[dict]):
    '''
    For each element in the objs, its of type dict with keys, 'xmin',
    'xmax', 'ymin' and 'ymax'.

    '''
    for obj in objs:
        xmin, xmax = int(obj['xmin']), int(obj['xmax'])
        ymin, ymax = int(obj['ymin']), int(obj['ymax'])
        cv2.rectangle(image, (xmin,ymin),(xmax,ymax),(100,200,36), 2, 8, 0)
