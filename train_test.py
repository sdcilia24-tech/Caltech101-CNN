import torch
from torch import nn
from torch.optim.lr_scheduler import OneCycleLR
from tqdm.auto import tqdm
from train_test_step.py import training_step, testing_step

def train_test_loop(model: nn.Module, training_dataloader: torch.utils.data.DataLoader,
                    testing_dataloader: torch.utils.data.DataLoader,
                    device = 'cuda' if torch.cuda.is_available() else 'cpu'):
  """
  Defines a training and testing loop for this practice model, using the AdamW optimizer 
  and the OneCycleLR learning rate scheduler, as well  and will ouput the training loss training accuracy testing loss
  and testing accuracy over every epoch in order to get an easy visual on what is happening during training

  :params:
          model: the model to be trained and tested.
          training_dataloader: the iterable data to be trained on
          testing_dataloader: the iterable data to be tested on
          device: device to train and test the model on cpu by default
  :returns:
  Dict -> {epochs: int, training_loss_list: (List[float]), test_loss_list: List[float],
  train_acc_list: List[float], test_acc_list: List[float]}
  """
  EPOCHS = 50
  LEARNING_RATE = .001

  results = {
  "train_loss_list": [],
  "test_loss_list": [],
  "train_acc_list": [],
  "test_acc_list": []}

  loss_fn = nn.CrossEntropyLoss()
  optimizer = torch.optim.AdamW(model.parameters(), lr = LEARNING_RATE)
  #scheduler = CyclicLR(optimizer = optimizer, base_lr = LEARNING_RATE, max_lr = .010, step_size_up = 100, mode = "triangular2")
  #scheduler = CosineAnnealingLR(optimizer = optimizer, T_max = EPOCHS)
  scheduler = OneCycleLR(optimizer = optimizer, max_lr = .010, epochs = EPOCHS,
                         steps_per_epoch = len(training_dataloader), anneal_strategy = 'cos')



  for epoch in tqdm(range(EPOCHS)):
    train_loss, train_acc = training_step(model, optimizer, training_dataloader, loss_fn, device, scheduler)

    results["train_loss_list"].append(train_loss)
    results["train_acc_list"].append(train_acc)

    test_loss, test_acc = testing_step(model, testing_dataloader, loss_fn, device)

    results["test_loss_list"].append(test_loss)
    results["test_acc_list"].append(test_acc)

    print(f"Epoch: {epoch} | train loss: {train_loss:.4f} | train acc: {train_acc:.4f} | test loss: {test_loss:.4f}, test acc: {test_acc:.4f}")
    print("==================================================================================================================\n")

  return results
