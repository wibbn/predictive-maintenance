import pandas as pd
import numpy as np


def get_gbm_database(
    telemetry_path: str,
    maint_path: str,
    machines_path: str,
    errors_path: str,
    failures_path: str,
    seq_len: int,
    machine_id: int = 1,
    all_ids: bool = False,
):
    # constants
    machine_ids = list(range(1, 101)) if all_ids else [machine_id]
    cols = ['volt', 'rotate', 'pressure', 'vibration']

    # import data
    telemetry = pd.read_csv(telemetry_path,
                            parse_dates=['datetime'],
                            infer_datetime_format=True,
                            low_memory=False,
                            error_bad_lines=False,
                            )
    maint = pd.read_csv(maint_path,
                        parse_dates=['datetime'],
                        infer_datetime_format=True,
                        error_bad_lines=False,
                        )
    machines = pd.read_csv(machines_path,
                           error_bad_lines=False,
                           )
    errors = pd.read_csv(errors_path,
                         parse_dates=['datetime'],
                         infer_datetime_format=True,
                         error_bad_lines=False,
                         )
    failures = pd.read_csv(failures_path,
                           parse_dates=['datetime'],
                           infer_datetime_format=True,
                           error_bad_lines=False,
                           )
    telemetry = telemetry.loc[telemetry['machineID'] == machine_id]
    failures = failures.loc[failures['machineID'] == machine_id]
    errors = errors.loc[errors['machineID'] == machine_id]
    maint = maint.loc[maint['machineID'] == machine_id]

    # drop duplicates
    telemetry.drop_duplicates(['datetime'], inplace=True, keep='last')
    failures.drop_duplicates(['datetime'], inplace=True, keep='last')
    maint.drop_duplicates(['datetime'], inplace=True, keep='last')
    errors.drop_duplicates(['datetime'], inplace=True, keep='last')

    # data-processing
    failures['failure'] = 1
    errors['errorID'] = 1
    maint['date'] = maint['datetime']
    maint.drop('comp', axis=1, inplace=True)

    # create statistics
    stats = telemetry.copy()

    # time since last service
    stats = stats.merge(maint, on=['machineID', 'datetime'], how='left')

    for mId in machine_ids:
        stats.loc[stats['machineID'] == mId, 'date'] = stats.loc[stats['machineID'] == mId, 'date'].ffill().fillna(stats['datetime'])

    stats['maint_delta'] = (stats['datetime'].astype(int) - stats['date'].astype(int))

    # feature agrigations
    for col in cols:
        stats['{}_mean_8hrs'.format(col)] = 0
        stats['{}_mean_32hrs'.format(col)] = 0
        stats['{}_mean_128hrs'.format(col)] = 0

        stats['{}_std_8hrs'.format(col)] = 0
        stats['{}_std_32hrs'.format(col)] = 0
        stats['{}_std_128hrs'.format(col)] = 0

        for mId in machine_ids:
            stats.loc[stats['machineID'] == mId, '{}_mean_8hrs'.format(col)] = stats.loc[stats['machineID'] == mId, col].rolling(
                8, min_periods=1).mean()
            stats.loc[stats['machineID'] == mId, '{}_mean_32hrs'.format(col)] = stats.loc[stats['machineID'] == mId, col].rolling(
                32, min_periods=1).mean()
            stats.loc[stats['machineID'] == mId, '{}_mean_128hrs'.format(col)] = stats.loc[stats['machineID'] == mId, col].rolling(
                128, min_periods=1).mean()

            stats.loc[stats['machineID'] == mId, '{}_std_8hrs'.format(col)] = stats.loc[stats['machineID'] == mId, col].rolling(
                8, min_periods=1).std()
            stats.loc[stats['machineID'] == mId, '{}_std_32hrs'.format(col)] = stats.loc[stats['machineID'] == mId, col].rolling(
                32, min_periods=1).std()
            stats.loc[stats['machineID'] == mId, '{}_std_128hrs'.format(col)] = stats.loc[stats['machineID'] == mId, col].rolling(
                128, min_periods=1).std()

    stats = stats.ffill()

    # recent breakdowns
    stats = stats.merge(errors, on=['machineID', 'datetime'], how='left').fillna(0)

    stats['error_sum_8hrs'] = 0
    stats['error_sum_32hrs'] = 0

    for mId in machine_ids:
        stats.loc[stats['machineID'] == mId, 'error_sum_8hrs'] = stats.loc[stats['machineID'] == mId, 'errorID'].rolling(8, min_periods=1).sum()
        stats.loc[stats['machineID'] == mId, 'error_sum_32hrs'] = stats.loc[stats['machineID'] == mId, 'errorID'].rolling(32, min_periods=1).sum()

    stats = stats.fillna(0)

    # machines info
    stats = stats.merge(machines, on=['machineID'], how='left')
    one_hot_models = pd.get_dummies(stats['model'])
    stats = pd.concat([stats, one_hot_models], axis=1)

    # LSTM features
    stats['lstm_mean'] = 0
    stats['lstm_std'] = 0

    for mId in machine_ids:
        stats.loc[stats['machineID'] == mId, 'lstm_mean'] = stats.loc[stats['machineID'] == mId, 'vibration'].rolling(seq_len, min_periods=1).mean()
        stats.loc[stats['machineID'] == mId, 'lstm_std'] = stats.loc[stats['machineID'] == mId, 'vibration'].rolling(seq_len, min_periods=1).std()

        stats.loc[stats['machineID'] == mId, 'lstm_mean'] = stats.loc[stats['machineID'] == mId, 'lstm_mean'].shift(-seq_len, fill_value=0)
        stats.loc[stats['machineID'] == mId, 'lstm_std'] = stats.loc[stats['machineID'] == mId, 'lstm_std'].shift(-seq_len, fill_value=0)

    # get target
    stats = stats.merge(failures, on=['machineID', 'datetime'], how='left').fillna(0)

    for mId in machine_ids:
        stats.loc[stats['machineID'] == mId, 'failure'] = stats.loc[stats['machineID'] == mId, 'failure'].rolling(64, min_periods=1).max()

    stats['failure'] = stats['failure'].shift(-64, fill_value=0)

    X = stats.drop(columns=['machineID', 'vibration', 'rotate', 'volt', 'pressure', 'errorID', 'date', 'model', 'failure']).set_index('datetime')
    y = stats[['failure', 'datetime']].set_index('datetime')

    return (X, y)