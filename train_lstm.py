import os
from argparse import ArgumentParser
import pandas as pd
import numpy as np
import torch

import pytorch_lightning as pl
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import NeptuneLogger

from main.models import LSTM
from main.datamodules import TelemetryDataModule

from config import (
    CommonArguments,
    DataArguments,
    TrainArguments
)


def main(args):
    seed_everything(args.seed)

    logger = NeptuneLogger(api_key=os.environ.get("NEPTUNE_API_TOKEN"),
                           project_name="wibbn/predictive-maintenance",
                           params=vars(args),
                           experiment_name="lstm_logs",
                           )
    trainer = Trainer(max_epochs=args.max_epochs,
                      logger=logger,
                      gpus=0,
                      progress_bar_refresh_rate=2,
                      )
    model = LSTM(n_features=args.n_features,
                 hidden_size=args.hidden_size,
                 seq_len=args.seq_len,
                 out_seq_len=args.out_seq_len,
                 batch_size=args.batch_size,
                 criterion=args.criterion,
                 num_layers=args.num_layers,
                 dropout=args.dropout,
                 learning_rate=args.learning_rate,
                 )
    dm = TelemetryDataModule(path=args.telemetry_path,
                             seq_len=args.seq_len,
                             out_seq_len=args.out_seq_len,
                             batch_size=args.batch_size,
                             num_workers=args.num_workers,
                             )

    trainer.fit(model, dm)
    trainer.test(model, datamodule=dm)
    model.save_hyperparameters()
    trainer.save_checkpoint(args.checkpoint_path)

if __name__ == "__main__":
    parser = ArgumentParser()
    default_args_dict = {
        **vars(CommonArguments()),
        **vars(DataArguments()),
        **vars(TrainArguments()),
    }

    for arg, value in default_args_dict.items():
        parser.add_argument(
            f'--{arg}',
            type=type(value),
            default=value,
            help=f'<{arg}>, default: {value}'
        )

    args = parser.parse_args()
    main(args)
