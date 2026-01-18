import torch
from torch import nn
from torchvision import transforms, datasets
from torchvision.transforms import ToTensor
from torch.utils.data import random_split
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
NUM_WORKERS = 0
BATCH_SIZE = 32
class ImageFolderCustom(Dataset):
  def __init__(self, subset, transform):
    """
    Overrides the default PyTorch ImageFolder to create training and testing splits
    """
    self.subset = subset
    self.transforms = transform

  def __len__(self):
    """returns the length of a split"""
    return len(self.subset)

  def __getitem__(self, idx: int):
    """gets the image and the label at a specific index, and will apply transforms if neccesary and return the image and label"""
    img, label = self.subset[idx]
    if self.transforms:
      return self.transforms(img), label
    else:
      return img, label

train_transforms = transforms.Compose([
    transforms.Lambda(lambda img: img.convert("RGB")),
    transforms.Resize(size = (224,224)),
    transforms.TrivialAugmentWide(num_magnitude_bins = int(31/3)),
    transforms.ToTensor()
])

test_transforms = transforms.Compose([
    transforms.Lambda(lambda img: img.convert("RGB")),
    transforms.Resize(size = (224, 224)),
    transforms.ToTensor()
])

data = datasets.Caltech101(
    root = "data",
    download = True,
    transform = None
)

train_split, test_split = random_split(data, [.8,.2])

training_data = ImageFolderCustom(train_split, train_transforms)
testing_data = ImageFolderCustom(test_split, test_transforms)

train_dataloader = DataLoader(dataset = training_data,
                              batch_size = BATCH_SIZE,
                              num_workers = NUM_WORKERS,
                              shuffle = True)

test_dataloader = DataLoader(dataset = testing_data,
                             batch_size = BATCH_SIZE,
                             num_workers = NUM_WORKERS,
                             shuffle = False)
