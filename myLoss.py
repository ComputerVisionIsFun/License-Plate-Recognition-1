import torch.nn as nn
import torch


def myLoss(A_gt, A_pred, O_gt, O_pred, Cx_gt, Cx_pred, Cy_gt, Cy_pred):
    '''
    arguments are of size N x 13 x 13
    
    '''
    mse = nn.MSELoss()
    loss_area = mse(A_gt, A_pred)
    loss_objness = mse(O_gt, O_pred)

    ns, _, rows, cols = torch.where(A_gt>0)
    num_objs = len(ns)

    loss_coord = torch.tensor(0.)

    for obj_i in range(num_objs):
        n, row, col = ns[obj_i], rows[obj_i], cols[obj_i]
        loss_coord = loss_coord + (Cx_gt[n, 0, row, col] - Cx_pred[n, 0, row, col])**2 + (Cy_gt[n, 0, row, col] - Cy_pred[n, 0, row, col])**2 
    
    loss_coord = loss_coord/num_objs

    loss_total = loss_area + loss_objness + loss_coord

    loss = {'area':loss_area, 'objness':loss_objness, 'coord':loss_coord, 'total':loss_total}

    return loss