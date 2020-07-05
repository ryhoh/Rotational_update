# Official packages
from typing import Tuple

# PyPI packages
from PIL import Image
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torchvision import transforms


def _load_cifar10(transform: transforms) -> Tuple[CIFAR10, CIFAR10]:
    return CIFAR10(root='./data', train=True,  download=True, transform=transform), \
           CIFAR10(root='./data', train=False, download=True, transform=transform)


def cifar_10_for_vgg_loaders() -> Tuple[DataLoader, DataLoader]:
    # テンソル化, RGB毎に平均と標準偏差を用いて正規化
    transform = transforms.Compose([
        # VGGは元々ImageNetを想定しているので、cifarをリサイズする
        transforms.Resize((224, 224), interpolation=Image.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # datasets
    train_set, test_set = _load_cifar10(transform=transform)

    # data_loader
    train_loader = DataLoader(train_set, batch_size=32, shuffle=True,  num_workers=4)
    test_loader  = DataLoader(test_set,  batch_size=32, shuffle=False, num_workers=4)

    return train_loader, test_loader


def cifar10_loaders() -> Tuple[DataLoader, DataLoader]:
    # テンソル化, RGB毎に平均と標準偏差を用いて正規化
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # datasets
    train_set, test_set = _load_cifar10(transform=transform)

    # data_loader
    train_loader = DataLoader(train_set, batch_size=32, shuffle=True,  num_workers=4)
    test_loader  = DataLoader(test_set,  batch_size=32, shuffle=False, num_workers=4)

    return train_loader, test_loader
