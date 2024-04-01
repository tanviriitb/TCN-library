import torch.nn as nn

class GlobalAvgPool(nn.Module):

    def __init__(self, kernel_size: int = 201):
        super(GlobalAvgPool, self).__init__()

        self.layer = nn.AvgPool1d(kernel_size=kernel_size)

    def forward(self, x):
        return self.layer(x)