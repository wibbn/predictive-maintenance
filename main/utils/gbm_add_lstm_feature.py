import pandas as pd
import torch

def get_lstm_feature(
    X_target: pd.DataFrame,
    lstm_out: torch.Tensor,
):

    result = X_target.copy()

    result['lstm_mean'] = lstm_out.mean(axis=1)
    result['lstm_std'] = lstm_out.std(axis=1)

    return result