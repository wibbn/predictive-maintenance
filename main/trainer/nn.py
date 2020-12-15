import numpy as np
import pandas as pd
import copy
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

import data
from models import LSTM

def train_loop(model, train_dataset, val_dataset, n_epochs, log_interval=5):
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

  optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
  criterion = nn.L1Loss(reduction='sum').to(device)
  history = dict(train_mean_losses=[], train_losses=[], val_mean_losses=[], val_loases=[], trains=[], vals=[])

  best_model_wts = copy.deepcopy(model.state_dict())
  best_loss = 10000.0
  
  for epoch in range(1, n_epochs + 1):
    model = model.train()

    train_losses = []
    for idx, (seq, label) in enumerate(train_dataset):
      optimizer.zero_grad()

      seq = seq.to(device)
      label = label.to(device)

      model.hidden_cell = (torch.zeros(1, 1, model.hidden_layer_size).to(device),
                        torch.zeros(1, 1, model.hidden_layer_size).to(device))

      pred = model(seq)
      loss = criterion(pred, label)

      loss.backward()
      optimizer.step()

      train_losses.append(loss.item())

      history['trains'].append((seq, pred, label))

      if idx % log_interval == 0:
        print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            epoch, idx, len(train_dataset),
            100. * idx / len(train_dataset), loss.item()))
        print(pred.shape)

    val_losses = []
    model = model.eval()
    with torch.no_grad():
      for (seq, label) in val_dataset:

        seq = seq.to(device)
        label = label.to(device)
        pred = model(seq)

        loss = criterion(pred, label)
        val_losses.append(loss.item())

        history['vals'].append((seq, pred, label))

    train_loss = np.mean(train_losses)
    val_loss = np.mean(val_losses)

    history['train_mean_losses'].append(train_loss)
    history['train_losses'].append(train_losses)
    history['val_mean_losses'].append(val_loss)
    history['val_loases'].append(val_losses)


    if val_loss < best_loss:
      best_loss = val_loss
      best_model_wts = copy.deepcopy(model.state_dict())

    print(f'Epoch {epoch}: train loss {train_loss} val loss {val_loss}')

  model.load_state_dict(best_model_wts)
  return model.eval(), history

def train_nn(n_epochs=2, log_interval=10):
  df = pd.read_csv('./data/df_super.csv')
  train_df, val_df, test_df = data.data_split(data.create_nn_dataset(df))

  model = LSTM(output_size=64)

  model, history = train_loop(model, train_df, val_df, n_epochs=n_epochs, log_interval=log_interval)

  torch.save({'model_state_dict': model.state_dict()}, 'nn.hdf5')