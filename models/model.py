from transformers import ViTForImageClassification
from .se_block import SEBlock

class ViTWithSE(ViTForImageClassification):
    def __init__(self, config):
        super().__init__(config)
        hidden_size = config.hidden_size
        for layer in self.vit.encoder.layer:
            layer.attention.se_block = SEBlock(hidden_size)