from termcolor import colored
from itertools import product
import numpy as np



def find_the_nearest_anchor(axs:list, ays:list, px:int, py:int)->tuple:
    '''
    For each anchor in anchors, anchor = (ay, ax)
    
    '''
    distance_min = px**2 + py**2
    argmin_anchor = (0,0)

    for ay in ays:
        for ax in axs:
            distance = (ay - py)**2 + (ax - px)**2
            
            if distance<=distance_min:
                argmin_anchor = (ay, ax)
                distance_min = distance

    return argmin_anchor


def compute_iou():






def show_train_info(title:str, parameters:dict):
    print(colored('*'*13 + title + '*'*13, 'yellow'))
    for para in parameters:
        print(colored(para + ': ', 'cyan') + str(parameters[para]))
    

    print(colored('*'*13 +'*'*13, 'yellow'))