from argparse import ArgumentParser
from catboost import CatBoostClassifier
import torch
from pytorch_lightning import LightningModule

from main.datamodules import TelemetryDataModule
from main.models import LSTM
from main.utils import get_gbm_database, get_lstm_feature

from config import (
    CommonArguments,
    DataArguments,
    TrainArguments
)


def main(args):
    # get data
    X, y = get_gbm_database(args.telemetry_path,
                            args.maint_path,
                            args.machines_path,
                            args.errors_path,
                            args.failures_path,
                            seq_len=args.out_seq_len,
                            machine_id=args.machine_id,
                            )
    X_gbm = X.iloc[args.seq_len:-args.out_seq_len]
    y_target = y.iloc[args.seq_len:-args.out_seq_len]

    dm = TelemetryDataModule(path=args.telemetry_path,
                             seq_len=args.seq_len,
                             out_seq_len=args.out_seq_len,
                             batch_size=X_gbm.shape[0],
                             num_workers=args.num_workers,)
    dm.setup(stage="prodaction")
    X_lstm = dm.prodaction_dataset()
    
    # load models
    lstm = LSTM.load_from_checkpoint(checkpoint_path=args.checkpoint_path + '/lstm.ckpt',
                                     n_features=args.n_features,
                                     hidden_size=args.hidden_size,
                                     seq_len=args.seq_len,
                                     out_seq_len=args.out_seq_len,
                                     batch_size=X_gbm.shape[0],
                                     criterion=args.criterion,
                                     num_layers=args.num_layers,
                                     dropout=args.dropout,
                                     learning_rate=args.learning_rate,
                                     )
    lstm.freeze()
    
    gbm = CatBoostClassifier()
    gbm.load_model(args.checkpoint_path + '/gbm.cbm')

    # prediction
    y_hat_lstm = None
    for (x, _) in X_lstm:
        y_hat_lstm = lstm(x)

    X_gbm = get_lstm_feature(X_gbm, y_hat_lstm)

    score = gbm.score(X_gbm, y_target)

    print('Model accuracy: {0:.2f}%'.format(score*100))


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
