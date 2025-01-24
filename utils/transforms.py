from torchvision.transforms import (
    Compose,
    Resize,
    RandomRotation,
    ToTensor,
    Normalize
)

def get_transforms(processor):
    normalize = Normalize(mean=processor.image_mean, std=processor.image_std)
    return {
        "train": Compose([
            Resize((processor.size["height"], processor.size["width"])),
            RandomRotation(30),
            ToTensor(),
            normalize
        ]),
        "val": Compose([
            Resize((processor.size["height"], processor.size["width"])),
            ToTensor(),
            normalize
        ])
    }