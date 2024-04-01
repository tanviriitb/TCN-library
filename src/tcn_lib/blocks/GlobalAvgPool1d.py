from torch import nn


class GlobalAvgPool1d(nn.Module):
    def __init__(self):
        super(GlobalAvgPool1d, self).__init__()

    def forward(self, x):
        # Calculate kernel size dynamically based on the input tensor shape
        kernel_size = x.size(-1)
        # Define and apply the AvgPool1d layer with dynamically calculated kernel size
        global_avg_pool = nn.AvgPool1d(kernel_size=kernel_size)
        return global_avg_pool(x)
