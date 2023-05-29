from termcolor import colored
from itertools import product
import numpy as np
import cv2
import debug as D
import XML


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

    def at(self, row:int, col:int):
        return Point(self.axs[col], self.ays[row])

    def show(self, img:np.array):
        for row in range(self.rows):
            for col in range(self.cols):
                anchor = self.at(row, col)
                cv2.circle(img, (int(anchor.x), int(anchor.y)), 2, (255,0,0),-1,8,0)

        return img

# class LprData():
#     def __init__(self, XmlImgFolder = 'D:/Dataset/LPR/cars_on_the_road/',anchors:Anchors, resize = P.resize, osize = P.osize):

#         self.XmlImgFolder =  XmlImgFolder
#         self.annos, self.objs = XML.xmlDecodingInTheFolder(XmlImgFolder)
#         self.anchors = anchors
#         self.resize = resize
        
#         self.rannos = []
#         ratioWidth, ratioHeight = resize[0]/osize[0], resize[1]/osize[1]

#         for path_i in range(len(self.annos)):
#             ranno = {'filename':self.annos[path_i]['filename'], 'objs':self.annos[path_i]['objs']}
#             for obj_i, obj in enumerate(ranno['objs']):
#                 ranno['objs'][obj_i]['xmin'], ranno['objs'][obj_i]['xmax'] = obj['xmin']*ratioWidth, obj['xmax']*ratioWidth
#                 ranno['objs'][obj_i]['ymin'], ranno['objs'][obj_i]['ymax'] = obj['ymin']*ratioHeight, obj['ymax']*ratioHeight
                
#             self.rannos.append(ranno)

#     def DrawAnnos(self, ind:int, show=True)->np.array:
#         '''
#         resize = (rWidth, rHeight)
#         '''
#         if ind>len(self.rannos) - 1:
#             print('ind is out of the range!')
#         else:
#             img = cv2.resize(cv2.imread(self.XmlImgFolder + self.rannos[ind]['filename']), self.resize, 1)
#             # img = D.make_grid(img, P.grid_x, P.grid_y)
#             objs = self.rannos[ind]['objs']
#             for obj in objs:
#                 xmin, xmax, ymin, ymax = int(obj['xmin']), int(obj['xmax']),int(obj['ymin']),int(obj['ymax'])
#                 cv2.rectangle(img, (xmin,ymin), (xmax,ymax),(0, 200, 0), 3, 8, 0)
#                 cx, cy = int((obj['xmin'] + obj['xmax'])/2), int((obj['ymin'] + obj['ymax'])/2)
#                 cv2.circle(img, (cx,cy),3,(0,255,0),-1,8,0)
#             if show:
#                 cv2.imshow("", img)
#                 cv2.waitKey(0)

#             return img

#     def DrawAnchors(self, ind:int, show:bool=True):
#         if ind>len(self.rannos) - 1:
#             print('ind is out of the range!')
#         else:
#             img = cv2.resize(cv2.imread(self.XmlImgFolder + self.rannos[ind]['filename']), self.resize, 1)
#             # img = D.make_grid(img, P.grid_x, P.grid_y)
#             img = self.anchors.show(img)
            
#             if show:
#                 cv2.imshow("", img)
#                 cv2.waitKey(0)

#             return img

#     def DrawBoth(self, ind:int, show:bool=True)->None:
#         if ind>len(self.annos) - 1:
#             print('ind is out of the range!')
#         else:
#             img = self.DrawAnnos(ind, False)
#             # anchors
#             img = self.anchors.show(img)

#             if show:
#                 cv2.imshow("", img)
#                 cv2.waitKey(0)

#             return img











def show_train_info(title:str, parameters:dict):
    '''
    ## Print information and parameters of the training processing.
    '''
    print(colored('*'*13 + title + '*'*13, 'yellow'))
    for para in parameters:
        print(colored(para + ': ', 'cyan') + str(parameters[para]))
    

    print(colored('*'*13 +'*'*13, 'yellow'))