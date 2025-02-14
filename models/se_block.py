import torch

class SEBlock(torch.nn.Module):
    def __init__(self, channel_num, reduction_ratio=16):
        super(SEBlock, self).__init__()
        self.avg_pool = torch.nn.AdaptiveAvgPool2d((1, 1))
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(channel_num, channel_num // reduction_ratio, bias=False),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(channel_num // reduction_ratio, channel_num, bias=False),
            torch.nn.Sigmoid()
        )

    def forward(self, x):
        s = self.avg_pool(x)
        z = self.fc(s.squeeze(-1).unsqueeze(-1))
        return x * z
