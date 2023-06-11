import utils as U
import numpy as np
from typing import List
import debug as D
import cv2
from termcolor import colored


def encoding(objs:List[dict], anchors:U.Anchors, w_normalize = 416, h_normalize = 416, iou_th = .3, cell_width=64, cell_height=32):
    '''
    Ecoding an image to an array of shape 5 x rows x cols, where (rows, cols) is the shape of anchors.

    ## Parameters:
        * objs: It's a list of dicts with keys xmin, xmax, ymin, ymax.
        * anchors: variable of type Anchors.
        * w_normalize and h_normalize: In general, they are the width and height of an image, respectively.
        
    ## return:
        * target: an array of shape 5 x rows x cols, where (rows, cols) is the shape of anchors.\n
        Given appreciate row, col,\n 
        target[0, row, col]: objectness,\n
        target[1, row, col]: normalized cx,\n
        target[2, row, col]: normalized cy,\n
        target[3, row, col]: normalized w,\n
        target[4, row, col]: normalized h w.r.t the position (row, col), respectively.
        
    '''
    
    x_normalize, y_normalize = (anchors.at(0,0).x - anchors.at(0,1).x)/2, (anchors.at(1,0).y - anchors.at(0,0).y)/2
    target = np.zeros(shape=(5, anchors.rows, anchors.cols))#objness, cx, cy, w, h
    # print(iou_th, '---')
    for obj in objs:
        # find nearest anchors
        nearest_anchors = U.find_the_nearest_anchor(anchors, obj)
        obj_center = U.Point((obj['xmin'] + obj['xmax'])/2, (obj['ymin'] + obj['ymax'])/2)

        for na_i in range(nearest_anchors[0].shape[0]):
            row, col = nearest_anchors[0][na_i], nearest_anchors[1][na_i]
            target[0, row, col] = 1
            target[1, row, col] = (obj_center.x - anchors.axs[col])/x_normalize
            target[2, row, col] = (obj_center.y - anchors.axs[row])/y_normalize
            target[3, row, col] = (obj['xmax'] - obj['xmin'])/w_normalize
            target[4, row, col] = (obj['ymax'] - obj['ymin'])/h_normalize

        # larger iou
        ayx_tuples_iou = U.argmax_iou(anchors, cell_width, cell_height,obj['xmin'],obj['xmax'],obj['ymin'],obj['ymax'],iou_th)
        # print(colored('iou', 'yellow'), ayx_tuples_iou)
        for i, ays in enumerate(ayx_tuples_iou[0]):
            # print('iou:', i)
            # print(ays)
            row, col = ays, ayx_tuples_iou[1][i]
            target[0, row, col] = 1
            target[1, row, col] = (obj_center.x - anchors.axs[col])/x_normalize
            target[2, row, col] = (obj_center.y - anchors.axs[row])/y_normalize
            target[3, row, col] = (obj['xmax'] - obj['xmin'])/w_normalize
            target[4, row, col] = (obj['ymax'] - obj['ymin'])/h_normalize


    return target




def encoding_area(objs:List[dict], anchors:U.Anchors):
    n_axs, n_ays = len(anchors.axs), len(anchors.ays)
    
    target = np.zeros((3, n_ays, n_axs), dtype='float')
    area = np.zeros((1, n_ays, n_axs), dtype='float')
    # 
    cell_w_half, cell_h_half = (anchors.axs[1] - anchors.axs[0])/2, (anchors.ays[1] - anchors.ays[0])/2
    cell_area = (anchors.axs[1] - anchors.axs[0])*(anchors.ays[1] - anchors.ays[0])
    for obj in objs:
        # find the nearest anchors
        nearest_anchors = U.find_the_nearest_anchor(anchors, obj)
        obj_center = U.Point((obj['xmin'] + obj['xmax'])/2, (obj['ymin'] + obj['ymax'])/2)
        
        for na_i in range(nearest_anchors[0].shape[0]):
            row, col = nearest_anchors[0][na_i], nearest_anchors[1][na_i]
            target[0, row, col] = 1
            target[1, row, col] = (obj_center.x - anchors.axs[col])/cell_w_half
            target[2, row, col] = (obj_center.y - anchors.axs[row])/cell_h_half
            
        # fill area for each obj
        for row_cell in range(n_ays):
            for col_cell in range(n_axs):
                anchor = anchors.at(row_cell, col_cell)
                cell_xmin, cell_xmax = anchor.x - cell_w_half, anchor.x + cell_w_half
                cell_ymin, cell_ymax = anchor.y - cell_h_half, anchor.y + cell_h_half
                
                obj_inter_cell = U.intersection_of_two_rects(obj['xmin'],obj['xmax'],obj['ymin'],obj['ymax'],cell_xmin, cell_xmax,cell_ymin,cell_ymax)

                area[0, row_cell, col_cell] = obj_inter_cell/cell_area

    return target, area













# def decoding(target:np.array, img:np.array, anchors:U.Anchors, w_normalize = 416, h_normalize = 416, obj_th=0.8, grid=True):
#     x_normalize, y_normalize = (anchors.at(0,0).x - anchors.at(0,1).x)/2, (anchors.at(1,0).y - anchors.at(0,0).y)/2
#     objness = np.where(target[0, :, :]>=obj_th)

#     num_objs = objness[0].shape[0]

#     if grid:
#         img = D.make_grid(img, P.grid_x, P.grid_y)

#     for obj_i in range(num_objs):
#         row, col = objness[0][obj_i], objness[1][obj_i]
#         # cx, cy, w, h
#         cx_en, cy_en = target[1, row, col], target[2, row, col]
#         w_en, h_en = target[3, row, col], target[4, row, col]

#         cx_de = cx_en*x_normalize + anchors.at(row, col).x
#         cy_de = cy_en*y_normalize + anchors.at(row, col).y
#         w_de = w_en*w_normalize
#         h_de = h_en*h_normalize
        

#         img = D.draw_rectangle(img, cx_de, cy_de, w_de, h_de)
#         cv2.circle(img, (int(cx_de), int(cy_de)), 3, (0,255,0),-1,8,0)


#     return img

# def decoding_return(target, img, anchors:U.Anchors, w_normalize = P.resize[0], h_normalize=P.resize[1], obj_th=0.8, grid=True):
#     x_normalize, y_normalize = (anchors.at(0,0).x - anchors.at(0,1).x)/2, (anchors.at(1,0).y - anchors.at(0,0).y)/2
#     objness = np.where(target[0, :, :]>=obj_th)
#     return_ratio_w = 1280/768
#     return_ratio_h = 720/768
#     num_objs = objness[0].shape[0]

#     if grid:
#         img = D.make_grid(img, P.grid_x, P.grid_y)

#     for obj_i in range(num_objs):
#         row, col = objness[0][obj_i], objness[1][obj_i]
#         # cx, cy, w, h
#         cx_en, cy_en = target[1, row, col], target[2, row, col]
#         w_en, h_en = target[3, row, col], target[4, row, col]

#         cx_de = (cx_en*x_normalize + anchors.at(row, col).x)*return_ratio_w
#         cy_de = (cy_en*y_normalize + anchors.at(row, col).y)*return_ratio_h
#         w_de = w_en*w_normalize*return_ratio_w
#         h_de = h_en*h_normalize*return_ratio_h
#         # print(w_de>10, w_de)
        

#         img = D.draw_rectangle(img, cx_de, cy_de, w_de, h_de,(0,255,0))
        
#         w_de_str, h_de_str = str(round(w_de, 0)), str(round(h_de, 0))
#         cv2.putText(img, w_de_str + '_' + h_de_str, (int(cx_de)-50,int(cy_de)-50), cv2.FONT_HERSHEY_PLAIN, 3, (0,0,255),3, cv2.LINE_AA)
    

#         cv2.circle(img, (int(cx_de), int(cy_de)), 3, (0,255,0),-1,8,0)


#     return img




