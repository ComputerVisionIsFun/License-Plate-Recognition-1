from termcolor import colored
from itertools import product
import numpy as np
import cv2
import debug as D
import XML
import utils as U
from termcolor import colored

class Point():
    '''
    ## member: 
        * x:int
        * y:int
    ## member funciton: 
        * show(self, img:np.array): draw (x, y) on the img. 
    '''
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def show(self, img:np.array):
        cv2.circle(img, (int(self.x), int(self.y)), 3, (0,255,0), -1, 8, 0)
        return img
        
class Rectangle():
    '''
    ## members:
        * cx, cy, w, h, xmin, xmax, ymin, ymax.
    ## member function:
        * show(self, img:np.array): draw the rectangle on the img.
    '''
    def __init__(self, cx, cy, w, h):
        self.cx, self.cy = cx, cy
        self.w, self.h = w, h
        self.xmin, self.xmax = cx - w/2, cx + w/2
        self.ymin, self.ymax = cy - h/2, cy + h/2

    def show(self, img:np.array):
        cv2.rectangle(img, (int(self.xmin), int(self.ymin)), (int(self.xmax), int(self.ymax)), (0,255,0), 3, 8, 0)
        return img

class Anchors():
    '''
    ## member:
        * rows, cols, ays, axs.
    ## member funciton: 
        * at(self, row, col): return anchor in the position (row, col) which of type Point.
        * show(self, img:np.array): draw anchors on the img.
    '''
    def __init__(self, ays:list, axs:list):
        self.rows = len(ays)
        self.cols = len(axs)
        self.ays = ays
        self.axs = axs
        self.anchors = []
        for row in range(len(ays)):
            for col in range(len(axs)):
                self.anchors.append((ays[row], axs[col]))



    def at(self, row:int, col:int):
        return Point(self.axs[col], self.ays[row])

    def show(self, img:np.array):
        for row in range(self.rows):
            for col in range(self.cols):
                anchor = self.at(row, col)
                cv2.circle(img, (int(anchor.x), int(anchor.y)), 2, (255,0,0),-1,8,0)

        return img

def find_the_nearest_anchor(anchors:U.Anchors, obj:dict):
    '''
    ## Given anchors and an obj of type dict with 'xmin', 'xmax', 'ymin', 'ymax', 
    ## find the nearest anchor close to the center of the object.  
    
    '''
    rows, cols = anchors.rows, anchors.cols
    distances = np.zeros(shape = (rows, cols), dtype=float)
    cx = (obj['xmin'] + obj['xmax'])/2. 
    cy = (obj['ymin'] + obj['ymax'])/2.

    for row in range(rows):
        for col in range(cols):
            anchor = anchors.at(row, col)
            distances[row, col] = (anchor.x - cx)**2 + (anchor.y - cy)**2
            
    return np.where(distances==np.min(distances))


def intersection_of_two_rects(xmin_1:float, xmax_1:float, ymin_1:float, ymax_1:float, xmin_2:float, xmax_2:float, ymin_2:float, ymax_2:float):
    dx = min(xmax_1, xmax_2) - max(xmin_1, xmin_2)
    dy = min(ymax_1, ymax_2) - max(ymin_1, ymin_2)
    
    if dx>0 and dy>0:
        # print('dx, dy = ',dx, dy)
        return dx*dy
    else:
        return 0

def compute_iou(xmin_1, xmax_1, ymin_1, ymax_1, xmin_2, xmax_2, ymin_2, ymax_2):
    intersection_area = intersection_of_two_rects(xmin_1, xmax_1, ymin_1, ymax_1, xmin_2, xmax_2, ymin_2, ymax_2)
    union_area = (xmax_1 - xmin_1)*(ymax_1 - ymin_1) + (xmax_2 - xmin_2)*(ymax_2 - ymin_2)
    # print(colored('inter and uni', 'red'), intersection_area, union_area)
    # print(xmin_1, xmax_1, ymin_1, ymax_1, xmin_2, xmax_2, ymin_2, ymax_2)
    return intersection_area/(union_area)
    
def argmax_iou(anchors:Anchors, cell_width:float, cell_height:float, 
               obj_xmin:float, obj_xmax:float, obj_ymin:float, obj_ymax:float, iou_th=.5):
    rows, cols = anchors.rows, anchors.cols
    half_cell_width, half_cell_height = cell_width/2., cell_height/2.
    iou_array = np.zeros((rows, cols), dtype='float')
    
    for row in range(rows):
        for col in range(cols):
            # if row==3 and col==4:
            #     print('row, col-----------------------------------', row, col)
            pt = anchors.at(row=row, col=col)
            cell_cx, cell_cy = pt.x, pt.y
            cell_xmin, cell_xmax = cell_cx - half_cell_width, cell_cx + half_cell_width
            cell_ymin, cell_ymax = cell_cy - half_cell_height, cell_cy + half_cell_height

            iou = compute_iou(cell_xmin, cell_xmax, cell_ymin, cell_ymax, obj_xmin, obj_xmax, obj_ymin, obj_ymax)
            # print(colored('iou score', 'red'), iou)
            iou_array[row, col] = iou
    # print(iou_array)

    return np.where(iou_array>=iou_th)





def show_train_info(title:str, parameters:dict):
    '''
    ## Print information and parameters of the training processing.
    '''
    print(colored('*'*13 + title + '*'*13, 'yellow'))
    for para in parameters:
        print(colored(para + ': ', 'cyan') + str(parameters[para]))
    

    print(colored('*'*13 +'*'*13, 'yellow'))