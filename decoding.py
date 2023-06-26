import utils as U
import numpy as np
import torch 
import cv2


def target_to_vis(anchors:U.Anchors, img:np.array, target:torch.Tensor):
    '''
    target is of size 3 x H x W 
    '''
    target_cpu_np = target.detach().cpu().numpy()
    objness = np.where(target[0, :, :])


    # 
    objness_xs, objness_ys= objness[1], objness[0]
    num_bojs = len(objness_xs)
    cell_w_half = (anchors.axs[1] - anchors.axs[0])/2
    cell_h_half = (anchors.ays[1] - anchors.ays[0])/2


    # 
    for obj_i in range(num_bojs):
        row, col = objness_ys[obj_i], objness_xs[obj_i]
        axy = anchors.at(row, col)
        ax, ay = axy.x, axy.y
        cx = int(target_cpu_np[1, row, col]*cell_w_half + ax + .5)
        cy = int(target_cpu_np[2, row, col]*cell_h_half + ay + .5)

        cv2.circle(img, (cx, cy), 2, (0,0,255), -1, 8, 0)


    return img


def inverse_normalize(img:torch.Tensor, mean = [0.485,0.456,0.406], std = [0.229,0.224,0.225]):
    '''
    img is of size 3 x H x W 
    '''
    img_cpu_np = img.detach().cpu().numpy().transpose(1, 2, 0)
    img_cpu_np = img_cpu_np*std + mean

    img_cpu_np = (img_cpu_np*255).astype('uint8').copy()

    return img_cpu_np

