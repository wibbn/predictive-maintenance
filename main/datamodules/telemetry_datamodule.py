import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
import pytorch_lightning as pl

from ..datasets import TelemetryDataset


class TelemetryDataModule(pl.LightningDataModule):
    def __init__(self,
                 path: str,
                 seq_len: int = 1,
                 batch_size: int = 128,
                 num_workers: int = 0
                 ):
        super().__init__()
        self.path = path
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.X_train = None
        self.y_train = None
        self.X_val = None
        self.y_val = None
        self.X_test = None
        self.X_test = None
        self.columns = None
        self.preprocessing = None

    def prepare_date(self):
        pass

    def setup(self, stage=None):
        if stage == 'fit' and self.X_train is not None:
            return
        if stage == 'test' and self.X_test is not None:
            return
        if stage is None and self.X_train is not None and self.X_test is not None:
            return

        df = pd.read_csv(self.path,
                         usecols=['datetime', 'vibration', 'machineID'],
                         parse_dates=True,
                         infer_datetime_format=True,
                         low_memory=False,
                         error_bad_lines=False,
                         index_col='datetime',
                        #  dtype={'vibration': np.float}
        )

        df_one = df[df['machineID'] == 1].drop(columns=['machineID'])
        df_resample = df_one.astype(float).resample('h').mean()

        X = df_resample.dropna().copy()
        y = X.shift(-1).ffill()

        self.columns = X.columns

        X_cv, X_test, y_cv, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
        X_train, X_val, y_train, y_val = train_test_split(X_cv, y_cv, test_size=0.25, shuffle=False)

        preprocessing = StandardScaler()
        preprocessing.fit(X_train)

        if stage == 'fit' or stage is None:
            self.X_train = preprocessing.transform(X_train)
            self.y_train = y_train.values.reshape((-1, 1))
            self.X_val = preprocessing.transform(X_val)
            self.y_val = y_val.values.reshape((-1, 1))

        if stage == 'test' or stage is None:
            self.X_test = preprocessing.transform(X_test)
            self.y_test = y_test.values.reshape((-1, 1))

    def train_dataloader(self):
        train_dataset = TelemetryDataset(self.X_train, 
                                          self.y_train, 
                                          seq_len=self.seq_len)
        train_loader = DataLoader(train_dataset, 
                                  batch_size = self.batch_size, 
                                  shuffle = False, 
                                  num_workers = self.num_workers)
        
        return train_loader

    def val_dataloader(self):
        val_dataset = TelemetryDataset(self.X_val, 
                                        self.y_val, 
                                        seq_len=self.seq_len)
        val_loader = DataLoader(val_dataset, 
                                batch_size = self.batch_size, 
                                shuffle = False, 
                                num_workers = self.num_workers)

        return val_loader

    def test_dataloader(self):
        test_dataset = TelemetryDataset(self.X_test, 
                                         self.y_test, 
                                         seq_len=self.seq_len)
        test_loader = DataLoader(test_dataset, 
                                 batch_size = self.batch_size, 
                                 shuffle = False, 
                                 num_workers = self.num_workers)

        return test_loader