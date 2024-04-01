from torch import nn


class GlobalAvgPool1d(nn.Sequential):

    def __init__(self, kernel_size: int = 200):
        conv = nn.AvgPool1d(kernel_size=kernel_size)
        super(GlobalAvgPool1d, self).__init__(conv)
