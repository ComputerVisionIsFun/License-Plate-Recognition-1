import torch.nn as nn
import torch
import torch.nn as nn



class myNet(nn.Module):
    '''
    forward:
        return area, objness, cxs, cys of shapes N x 13 x 13.
    
    '''
    def __init__(self, backbone, bridge_channels = 512):
        super(myNet, self).__init__()
        self.backbone = backbone
        self.bridge = nn.Conv2d(in_channels=bridge_channels, out_channels=1,kernel_size=1)
        self.lstm_area = nn.LSTM(input_size=13,hidden_size=13,num_layers=1,batch_first=True)
        self.lstm_objness = nn.LSTM(input_size=13,hidden_size=13,num_layers=1,batch_first=True)
        self.lstm_cx = nn.LSTM(input_size=13,hidden_size=13,num_layers=1,batch_first=True)
        self.lstm_cy = nn.LSTM(input_size=13,hidden_size=13,num_layers=1,batch_first=True)


    def forward(self, x):
        fmap = self.bridge(self.backbone(x))# shape: N x 1 x 13 x 13
        fmap = torch.squeeze(fmap, dim = 1)# shape: N x 13 x 13 
        area, _ = self.lstm_area(fmap)# shape: N x 13 x 13
        # print(area.size(), '---')
        objness, _ = self.lstm_objness(area)# shape: N x 13 x 13
        cxs, _ = self.lstm_cx(area)# shape: N x 13 x 13
        cys, _ = self.lstm_cy(area)# shape: N x 13 x 13

        return area, objness, cxs, cys