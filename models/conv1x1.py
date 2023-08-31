import torch.nn as nn
import torch
import torch.nn.functional as F


class Conv1x1(nn.Module):
    """
    Conv -> BN -> ReLU
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=1,
                 stride=1,
                 transpose=False,
                 padding_mode='zeros',
                 bn=False,
                 activation=None):
        super(Conv1x1, self).__init__()

        bias = False if bn else True
        conv_fn = nn.ConvTranspose2d if transpose else nn.Conv2d
        self.conv = conv_fn(in_channels,out_channels,kernel_size,stride=stride, bias=bias, padding_mode=padding_mode)

        self.batch_norm = nn.BatchNorm2d(out_channels, eps=1e-6, momentum=0.99) if bn else None
        if activation is not None:
            self.activation = nn.LeakyReLU(negative_slope=0.1)
        else:
            self.activation = None

    def forward(self, x):
        """
        :param x: [B, C, N, K]
        :return: [B, C, N, K]
        """
        x = x.transpose(0, -1)
        x = self.conv(x)
        if self.batch_norm is not None:
            x = self.batch_norm(x)
        if self.activation is not None:
            x = self.activation(x)
        x = x.transpose(0, -1)

        return x


# class ECC_CRFModule(nn.Module):
#     """
#     Adapted "Conditional Random Fields as Recurrent Neural Networks" (https://arxiv.org/abs/1502.03240)
#     `propagation` should be ECC with Filter generating network producing 2D matrix.
#     """
#     def __init__(self, propagation, nrepeats=1):
#         super(ECC_CRFModule, self).__init__()
#         self._propagation = propagation
#         self._nrepeats = nrepeats
#
#     def forward(self, input):
#         Q = F.softmax(input)
#         for i in range(self._nrepeats):
#             Q = self._propagation(Q) #
#             Q = input - Q
#             if i < self._nrepeats-1:
#                 Q = F.softmax(Q) # last softmax will be part of cross-entropy loss
#         return Q


if __name__=='__main__':
    a = torch.rand((32, 9))
    b = torch.rand((32, 3))
    idx = torch.rand((32, 16)).type(torch.long)
    c=torch.rand((32, 9))
    # # c = KPConv(15, 9, 64, 0.01, 0.5)(a, b, a[:, :3], idx)
    # print(c)
    c=c.unsqueeze(2).unsqueeze(3)
    c=Conv1x1(c.shape[1],c.shape[1],bn=True,activation=True)(c)
    c=c.squeeze(2).squeeze(2)
    print(c)