from torch.optim import Adam
from torch import Tensor
import torch.nn as nn
import pytorch_lightning as pl


class LSTM(pl.LightningModule):
    def __init__(self,
                 n_features,
                 hidden_size,
                 seq_len,
                 batch_size,
                 num_layers,
                 dropout,
                 learning_rate,
                 criterion):

        super().__init__()
        self.n_features = n_features
        self.hidden_size = hidden_size
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.criterion = criterion
        self.learning_rate = learning_rate

        self.lstm = nn.LSTM(input_size=n_features, hidden_size=hidden_size,
                            num_layers=num_layers, dropout=dropout, batch_first=True)

        self.linear = nn.Linear(hidden_size, 1)

        # self.hidden_cell = (torch.zeros(1, 1, self.hidden_layer_size),
        #                     torch.zeros(1, 1, self.hidden_layer_size))

    def forward(self,
                x: Tensor
                ) -> Tensor:
        lstm_out, _ = self.lstm(x)
        y_pred = self.linear(lstm_out[:, -1])
        return y_pred

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=self.learning_rate)

    def training_step(self,
                      batch: Tensor,
                      batch_idx: int
                      ) -> Tensor:
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        result = pl.TrainResult(loss)
        result.log('train_loss', loss)
        return result

    def validation_step(self,
                        batch: Tensor,
                        batch_idx: int
                        ) -> Tensor:
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        result = pl.EvalResult(checkpoint_on=loss)
        result.log('val_loss', loss)
        return result

    def test_step(self,
                  batch: Tensor,
                  batch_idx: int
                  ) -> Tensor:
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        result = pl.EvalResult()
        result.log('test_loss', loss)
        return result
