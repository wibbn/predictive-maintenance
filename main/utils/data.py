import pandas as pd
import numpy as np
import torch
import datetime

def create_gbm_dataset(df, test=False):
    drop_cols = ['machineID', 'datetime', 'past_vib', 'future_vib', 'failure', 'model']

    target = df['failure']

    return df.drop(drop_cols, axis=1), target

def create_nn_dataset(df):
    dataset = []
    for ID in range(1, 101):
      df_len = df[df['machineID'] == ID].shape[0]

      for i in range(64, df_len - 64, 64):
        seq = torch.tensor(df[df['machineID'] == ID]['past_vib'].iloc[i][1:-1].split(', ')).float()
        target = torch.tensor(df[df['machineID'] == ID]['future_vib'].iloc[i][1:-1].split(', ')).float()
    
        dataset.append((seq, target))
    return dataset

def data_split(df):
    train_size = int(len(df) * 0.75)
    val_size = (len(df) - train_size) // 2

    train_df = df[: train_size]
    val_df = df[train_size : train_size + val_size]
    test_df = df[train_size + val_size :]

    return train_df, val_df, test_df

