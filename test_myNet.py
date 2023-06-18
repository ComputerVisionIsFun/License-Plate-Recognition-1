import myNet as mn
import torchvision 
import torch.nn as nn
import torch
import torchvision

a = torch.zeros((5, 1, 9, 6))
print(torch.squeeze(a, dim = 1).size())


# create backbone from resnet18
resnet18 = torchvision.models.resnet18()
backbone_list = []

for child_i, child in enumerate(resnet18.children()):
    if child_i>=8:
        break

    backbone_list.append(child)

backbone = nn.Sequential(*backbone_list)

# 
model = mn.myNet(backbone=backbone, bridge_channels=512)
input = torch.zeros((1, 3, 416, 416))
area, objness, cxs, cys = model(input)
print(area.size(), objness.size(), cxs.size(), cys.size())





