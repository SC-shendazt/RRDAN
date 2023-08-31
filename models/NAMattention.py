import torch.nn as nn
import torch
from torch.nn import functional as F



class Channel_Att(nn.Module):
    def __init__(self, channels, t=16):
        super(Channel_Att, self).__init__()
        self.channels = channels
        self.bn2 = nn.BatchNorm2d(self.channels, affine=True)

    def forward(self, x):
        residual = x
        x = self.bn2(x)

        weight_bn = self.bn2.weight.data.abs() / torch.sum(self.bn2.weight.data.abs())
        x = x.permute(0, 2, 3, 1).contiguous()
        x = torch.mul(weight_bn, x)
        x = x.permute(0, 3, 1, 2).contiguous()
        x = torch.sigmoid(x) * residual  #
        return x



if __name__ =='__main__':
    input = torch.randn(50, 512).unsqueeze(2).unsqueeze(3)
    print(input)
    se = Channel_Att(input.shape[1])
    output = se(input)
    output=output.squeeze(2).squeeze(2)

    print(output)
