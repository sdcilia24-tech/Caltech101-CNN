import torch
from torch import nn

def training_step(model: nn.Module, optimizer: torch.optim.Optimizer, dataloader: torch.utils.data.DataLoader,
                  loss_fn: torch.nn.Module, 
                  scheduler: torch.optim.lr_scheduler,
                  device = 'cuda' if torch.cuda.is_available() else 'cpu'):
  """
  Defines a training loop for the given model, which will perform backprogation and gradient descent
  :params:
    model: a CNN multilabel image classification model
    optimizer: One of the many provided torch optimizers, ie: Adam, SGD, etc.
    dataloder: an iterable version of data that is inhereited from torch.utils.data
    loss_fn: one of the provided loss functions, ie CrossEntropyLoss()
    device: device to run the computation of the model cpu by default
  :returns:
    tuple -> total training loss and training accuracy as floating point numbers
  """
  model.train()
  train_loss, train_acc, total = 0, 0, 0
  for batch, (input, label) in enumerate(dataloader):
    input, label = input.to(device), label.to(device)

    logits = model(input)
    loss = loss_fn(logits, label)

    train_loss += loss.item()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    scheduler.step()

    predicted_class = torch.argmax(logits, dim = 1)
    train_acc += (predicted_class == label).sum().item()
    total += label.size(0)

  train_loss = train_loss / len(dataloader)
  train_acc = train_acc / total

  return train_loss, train_acc

def testing_step(model: nn.Module, dataloader: torch.utils.data.DataLoader,
                 loss_fn: torch.nn.Module, device = 'cuda' if torch.cuda.is_available() else 'cpu'):
  """
  defines a testing loop for a given model
  :params:
    model: a CNN multilabel image classification model
    dataloder: an iterable version of data that is inhereited from torch.utils.data
    loss_fn: one of the provided loss functions, ie CrossEntropyLoss()
    device: device to run the computation of the model cpu by default 
  :returns:
    tuple -> total testing loss and testing accuracy as floating point numbers
  """
  model.eval()
  test_loss, test_acc, total = 0, 0, 0
  with torch.inference_mode():
    for batch, (image, label) in enumerate(dataloader):
      image, label = image.to(device), label.to(device)

      logits = model(image)
      loss = loss_fn(logits, label)

      test_loss += loss.item()

      test_pred_class = torch.argmax(logits, dim = 1)
      test_acc += (test_pred_class == label).sum().item()
      total += label.size(0)

    test_loss /= len(dataloader)
    test_acc = test_acc / total
    return test_loss, test_acc
