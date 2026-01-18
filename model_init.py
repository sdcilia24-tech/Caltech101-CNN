import torch
from torch import nn
KERNEL_SIZE = 3
POOLING_KERNEL = 2


class PracticeModel(nn.Module):
  def __init__(self, input_shape: int, hidden_units: int, output_shape: int):
    """
    Defines a Convulational neural Network able to classify and predict on images in the CalTech101 Dataset.

    :params:
      input_shape: the number of colour channels coming into the neural network (RGB)
      hidden_units: the base amount hidden neurons within the network
      output_shape: the models logits

    :returns: None
    """
    super().__init__()
    self.conv_block_one = nn.Sequential(
        nn.Conv2d(
            in_channels = input_shape,
            out_channels = hidden_units,
            kernel_size = KERNEL_SIZE,
            padding = 1,
            stride = 1),
        nn.BatchNorm2d(hidden_units),
        nn.ReLU())

    self.conv_block_two = nn.Sequential(
        nn.Conv2d(
            in_channels = hidden_units,
            out_channels = 2 * hidden_units,
            kernel_size = KERNEL_SIZE,
            padding = 1,
            stride = 1),
        nn.BatchNorm2d(2 * hidden_units),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size = POOLING_KERNEL,
                     stride = 2),
        nn.Dropout(p = .1))

    self.conv_block_three = nn.Sequential(
        nn.Conv2d(
            in_channels = 2 * hidden_units,
            out_channels = 4 *  hidden_units,
            kernel_size = KERNEL_SIZE,
            padding = 1,
            stride = 1),
        nn.BatchNorm2d(4 * hidden_units),
        nn.ReLU())

    self.conv_block_four = nn.Sequential(
        nn.Conv2d(
            in_channels = 4 * hidden_units,
            out_channels = 8 * hidden_units,
            kernel_size = KERNEL_SIZE,
            padding = 1,
            stride = 1),
        nn.BatchNorm2d(8 * hidden_units),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size = POOLING_KERNEL,
                     stride = 2),
        nn.Dropout(p = .1))

    self.conv_block_five = nn.Sequential(
        nn.Conv2d(
            in_channels = 8 * hidden_units,
            out_channels = 16 * hidden_units,
            kernel_size = KERNEL_SIZE,
            padding = 1,
            stride = 1),
        nn.BatchNorm2d(16 * hidden_units),
        nn.ReLU())

    self.conv_block_six = nn.Sequential(
        nn.Conv2d(
            in_channels = 16 * hidden_units,
            out_channels = 32 * hidden_units,
            kernel_size = KERNEL_SIZE,
            padding = 1,
            stride = 1),
        nn.BatchNorm2d(32 * hidden_units),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size= POOLING_KERNEL,
                     stride = 2),
        nn.Dropout(p = .1))

    self.conv_block_seven = nn.Sequential(
        nn.Conv2d(
            in_channels = 32 * hidden_units,
            out_channels = 48 * hidden_units,
            kernel_size = KERNEL_SIZE,
            padding = 1,
            stride = 1),
        nn.BatchNorm2d(48 * hidden_units),
        nn.ReLU())

    self.classifier = nn.Sequential(
        nn.AdaptiveAvgPool2d(output_size = (1,1)),
        nn.Flatten(),
        nn.Dropout(p = .5),
        nn.Linear(in_features = 48 * hidden_units,
                  out_features = output_shape)
    )

  def forward(self, x):
    """
    The forward pass of the model, which takes a Tensor and predicts on it.
    :params: x -> Tensor
    :returns: torch.Tensor: models logits
    """
    return self.classifier(self.conv_block_seven(self.conv_block_six(self.conv_block_five(self.conv_block_four(self.conv_block_three(self.conv_block_two(self.conv_block_one(x))))))))
