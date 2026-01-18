import torch
from torch import nn
import mlxtend
try:
  from torchmetrics.classification import ConfusionMatrix
except:
  !pip install torchmetrics 
  from torchmetrics.classification import ConfusionMatrix
from mlxtend.plotting import plot_confusion_matrix
import image_folder.py

def plot_heat_map(model: nn.Module, dataloader: torch.utils.data.dataset, data, classes,
                  device = 'cuda' if torch.cuda.is_available() else 'cpu'):
  """
  will plot a heatmap of what the model was confused on throughout testing, 
  by first retesting and then plotting a heatmap

  :params: 
          model: the model to plot the heat map for
          dataloader: an iterable version of data that is inhereited from torch.utils.data
          data: A downloadable dataset that was most likely used for model training,
                must be in standard Image folder formatting
          classes: A list of labels of the images in the dataset
          device: the device to run the computation cpu by default 
  :returns:
          None
  """
  y_preds = []
  y_true = []
  model.eval()
  with torch.inference_mode():
    for batch, (X, y) in enumerate(dataloader):
      X, y = X.to(device), y.to(device)
      y_logit = model(X)

      y_pred = torch.softmax(y_logit.squeeze(), dim = 0)
      y_pred = y_pred.argmax(dim = 1)
      y_preds.append(y_pred.cpu())
      y_true.append(y.cpu())
  y_pred_tensor = torch.cat(y_preds)
  y_true_tensor = torch.cat(y_true)

  confmat = ConfusionMatrix(task = "multiclass",
                            num_classes = len(classes))
  confmat_tensor = confmat(preds = y_pred_tensor,
                           target = y_true_tensor)
  fig, ax = plot_confusion_matrix(
    conf_mat = confmat_tensor.numpy(),
    class_names =  data.categories,
    figsize = (24, 24),
)





import matplotlib.pyplot as plt

def plot_curves(results):
  """
  plots both the train loss, testing loss curves
  and the train accuracy and testing accuracy curves side by side
  :params:
    A dictionary that has string keys, and Lists of floats as a value ex: {"string": [float, float, float]}
  returns -> none
  """
  train_loss = results["train_loss_list"]
  train_acc = results["train_acc_list"]

  test_loss = results["test_loss_list"]
  test_acc = results["test_acc_list"]

  epochs = range(len(results["train_loss_list"]))

  plt.figure(figsize=(15, 7))

  plt.subplot(1, 2, 1)
  plt.plot(epochs, train_loss, label='train_loss')
  plt.plot(epochs, test_loss, label='test_loss')
  plt.title('Loss')
  plt.xlabel('Epochs')
  plt.ylabel('Loss')
  plt.legend()


  plt.subplot(1, 2, 2)
  plt.plot(epochs, train_acc, label='train_accuracy')
  plt.plot(epochs, test_acc, label='test_accuracy')
  plt.title('Accuracy')
  plt.xlabel('Epochs')
  plt.ylabel('Accuracy % out of 1')
  plt.legend();
