import torch
from transformers import ViTForImageClassification

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

class ViTForImageClassificationWithSEBlock(ViTForImageClassification):
    def __init__(self, config):
        super().__init__(config)
        self.config = config

    def forward(self, pixel_values, labels=None, attention_mask=None, head_mask=None):
        encoder_layers = self.vit.encoder.layer
        hidden_size = self.config.hidden_size

        for i in range(len(encoder_layers)):
            encoder_layers[i].attention.add_module('se_block', SEBlock(hidden_size))

        if attention_mask is not None:
            outputs = super().forward(
                pixel_values, labels=labels, attention_mask=attention_mask, head_mask=head_mask
            )
        else:
            outputs = super().forward(pixel_values, labels=labels, head_mask=head_mask)

        return outputs
