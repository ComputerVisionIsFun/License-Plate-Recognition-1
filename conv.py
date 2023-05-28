import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F



def conv3x3(in_planes:int,out_planes:int,stride:int=1,groups:int=1,dilation:int=1)->nn.Conv2d:
    '''3x3 conv with padding'''
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride = stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )


def conv1x1(in_planes:int,out_planes:int,stride:int=1)->nn.Conv2d:
    '''1x1 conv'''
    return nn.Conv2d(in_planes,out_planes,kernel_size=1,stride=stride,bias=False)


class BasicBlock(nn.Module):
    def __init__(
        self,
        inplanes:int,
        outplanes:int,
        stride:int = 1,
        groups:int = 1,
        dilation:int = 1
        )->None:

        super().__init__()
        self.conv1 = conv3x3(inplanes, outplanes, stride)
        self.bn1 = nn.BatchNorm2d(outplanes) 
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(outplanes, outplanes)
        self.bn2 = nn.BatchNorm2d(outplanes)     
        self.stride = stride

    def forward(self, x:Tensor)->Tensor:
        # identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        # out+=identity
        out = self.relu(out)

        return out
    

