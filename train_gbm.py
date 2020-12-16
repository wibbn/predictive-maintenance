from argparse import ArgumentParser
from sklearn.model_selection import train_test_split
from catboost import CatBoostClassifier

from main.utils import get_gbm_database

from config import (
    CommonArguments,
    DataArguments,
    TrainArguments
)


def main(args):
    X, y = get_gbm_database(args.telemetry_path,
                            args.maint_path,
                            args.machines_path,
                            args.errors_path,
                            args.failures_path,
                            seq_len=args.out_seq_len,
                            machine_id=args.machine_id,
                            )
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.9)

    model = CatBoostClassifier(
        iterations=args.n_iterations,
        learning_rate=args.gbm_learning_rate
    )

    model.fit(X_train, y_train, eval_set=(X_test, y_test))
    model.save_model(args.checkpoint_path)


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
