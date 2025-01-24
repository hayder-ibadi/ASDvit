import torch.nn as nn

class SEBlock(nn.Module):
    def __init__(self, channel_num, reduction_ratio=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(
            nn.Linear(channel_num, channel_num // reduction_ratio, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel_num // reduction_ratio, channel_num, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        s = self.avg_pool(x).view(b, c)
        z = self.fc(s).view(b, c, 1, 1)
        return x * z