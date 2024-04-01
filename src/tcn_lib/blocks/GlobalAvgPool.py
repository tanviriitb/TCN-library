import torch.nn as nn

class GlobalAvgPool1d(nn.Sequential):

    def __init__(self, kernel_size: int = 201):

        layer = nn.AvgPool1d(kernel_size=kernel_size)

        super(GlobalAvgPool1d, self).__init__(layer)

   
class GlobalAvgPool(nn.Module):

    def __init__(self, kernel_size: int = 201):
        super(GlobalAvgPool, self).__init__()

        self.layer = GlobalAvgPool1d(kernel_size)

    def forward(self, x):
        return self.layer(x)