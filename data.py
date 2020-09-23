import os

import torch
from torchvision import datasets, transforms

DATA_ROOT = "./data"
default_data_transform = transforms.Compose(
    [
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.3337, 0.3064, 0.3171), (0.2672, 0.2564, 0.2629)),
    ]
)


def data_loaders(batch_size=64, data_transforms=None):
    if data_transforms is None:
        data_transforms = {
            "train": default_data_transform,
            "val": default_data_transform,
        }
    else:
        data_transforms = {"train": data_transforms, "val": default_data_transform}
    assert isinstance(data_transforms, dict)

    train_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(
            DATA_ROOT + "/train_images", transform=data_transforms["train"]
        ),
        batch_size=batch_size,
        shuffle=True,
    )
    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(
            DATA_ROOT + "/val_images", transform=data_transforms["val"]
        ),
        batch_size=batch_size,
        shuffle=False,
    )

    return train_loader, val_loader


def initialize_data(folder):
    train_folder = folder + "/train_images"
    # make validation_data by using images 00000*, 00001* and 00002* in each class
    val_folder = folder + "/val_images"
    if not os.path.isdir(val_folder):
        print(val_folder + " not found, making a validation set")
        os.mkdir(val_folder)
        for dirs in os.listdir(train_folder):
            if dirs.startswith("000"):
                os.mkdir(val_folder + "/" + dirs)
                for f in os.listdir(train_folder + "/" + dirs):
                    if (
                        f.startswith("00000")
                        or f.startswith("00001")
                        or f.startswith("00002")
                    ):
                        # move file to validation folder
                        os.rename(
                            train_folder + "/" + dirs + "/" + f,
                            val_folder + "/" + dirs + "/" + f,
                        )


class AddGaussianNoise(object):
    def __init__(self, mean=0.0, std=0.1):
        self.std = std
        self.mean = mean

    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean

    def __repr__(self):
        return self.__class__.__name__ + f"(mean={self.mean}, std={self.std})"


if __name__ == "__main__":
    initialize_data(DATA_ROOT)
