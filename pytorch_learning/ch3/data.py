import torch
from torchvision.datasets import CIFAR10
from pathlib import Path
from torchvision import transforms

current_dir = Path(__file__).resolve().parent
data_dir = current_dir.parent / "data"

train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.4914, 0.4822, 0.4465),
                         std=(0.2023, 0.1994, 0.2010))])
test_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        mean=(0.4914, 0.4822, 0.4465),
        std=(0.2023, 0.1994, 0.2010))])

train_data = CIFAR10(root=data_dir, train=True, download=True, transform=train_transform)
test_data = CIFAR10(root=data_dir, train=False, download=True, transform=test_transforms)

print("train data:", train_data)
print("len:", len(train_data))
print("shape: ", train_data.data.shape)
# print("targets: ", train_data.targets)
print("classes: ", train_data.classes)
print("index: ", train_data.class_to_idx)

train_loader = torch.utils.data.DataLoader(train_data, batch_size=16, shuffle=True, num_workers=2)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=16, shuffle=False, num_workers=2)


