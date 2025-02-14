from torchvision.transforms import (
    Compose,
    Normalize,
    RandomRotation,
    RandomResizedCrop,
    RandomHorizontalFlip,
    RandomAdjustSharpness,
    Resize,
    ToTensor
)

def create_transforms(processor):
    image_mean, image_std = processor.image_mean, processor.image_std
    size = processor.size["height"]

    normalize = Normalize(mean=image_mean, std=image_std)

    _train_transforms = Compose(
        [
            Resize((size, size)),
            RandomRotation(30),
            RandomAdjustSharpness(2),
            ToTensor(),
            normalize
        ]
    )

    _val_transforms = Compose(
        [
            Resize((size, size)),
            ToTensor(),
            normalize
        ]
    )

    def train_transforms(examples):
        examples['pixel_values'] = [_train_transforms(image.convert("RGB")) for image in examples['image']]
        return examples

    def val_transforms(examples):
        examples['pixel_values'] = [_val_transforms(image.convert("RGB")) for image in examples['image']]
        return examples

    return train_transforms, val_transforms
