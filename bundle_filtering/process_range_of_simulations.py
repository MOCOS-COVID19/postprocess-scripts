import pickle
import click
import pandas as pd
import numpy as np
import os


def detected_cases(df_r1):
    """ This method should be moved to utils """
    cond1 = ~df_r1.tdetection.isna()
    cond2a = ~df_r1.trecovery.isna()
    cond2b = df_r1.tdetection > df_r1.trecovery
    cond2 = ~np.logical_and(cond2a, cond2b)
    if len(df_r1[~df_r1.tdeath.isna()]) > 0:
        cond3a = ~df_r1.tdeath.isna()
        cond3b = df_r1.tdetection > df_r1.tdeath
        cond3 = ~np.logical_and(cond3a, cond3b)
        cond23 = np.logical_and(cond2, cond3)
    else:
        cond23 = cond2
    cond = np.logical_and(cond1, cond23)
    df = df_r1[cond]
    return df.sort_values(by='tdetection').tdetection


@click.command()
@click.option('--path', type=click.Path(exists=True))
@click.option('--zero', type=int, default=133)
@click.option('--minus', type=int, default=81)
@click.option('--minus-days', type=int, default=7)
@click.option('--minus-tolerance', type=float, default=0.1)
@click.option('--bundle-days', type=int, default=60)
@click.option('--bundle-prefix', default='')
def runner(path, zero, minus, minus_days, minus_tolerance, bundle_days, bundle_prefix):
    """
    Calculates how many sample paths are fitting 2-points criteria and saves bundles to files.

    :param path: path to set of simulations e.g. "<outputdir>/<experiment_root>/grid_X_Y/outputs/<outputs_id>/"
    :param zero: number of detected cases at zero time (synchronization point)
    :param minus: number of detected cases at past time (2-point checkpoint)
    :param minus_days: how many days are between minus day and zero day (e.g. 7)
    :param minus_tolerance: fraction of "minus" that is tolerated (e.g. 0.1*minus)
    :param bundle_days: time horizon used for saving forecast bundle files (e.g. 60)
    :param bundle_prefix: file prefix for storing bundle coordinations (may be blank)
    :return:
    """

    d = path
    list_subfolders_with_paths = [f.path for f in os.scandir(d) if f.is_dir()]
    zero_time = zero
    minus_time = minus
    successes = 0
    x_ = []
    y_ = []
    tries = 0
    fails = []
    for sub_ in list_subfolders_with_paths:
        if os.path.basename(sub_).startswith('agg'):
            continue

        tries += 1
        progression_path = os.path.join(sub_, 'output_df_progression_times.csv')
        df = pd.read_csv(progression_path)
        detected = detected_cases(df).values
        if len(detected) <= zero_time:
            continue

        detected = detected - detected[zero_time]
        t0 = detected[zero_time]
        filt_detected = detected[detected <= t0 - minus_days]
        if len(filt_detected) == 0:
            continue

        arg_tminus = np.argmax(filt_detected)

        if np.abs(arg_tminus - minus_time) > minus_tolerance * minus_time:
            fails.append(arg_tminus)
            continue

        successes += 1
        detected = detected[detected >= 0]
        x = detected[detected <= bundle_days]
        y = np.arange(zero_time, zero_time + len(x))
        x_.extend(list(x))
        y_.extend(list(y))

    print(fails)
    print(successes)
    print(tries)
    bundle_x = os.path.join(d, f'{bundle_prefix}bundle_x.pkl')
    bundle_y = os.path.join(d, f'{bundle_prefix}bundle_y.pkl')
    with open(bundle_x, 'wb') as f:
        pickle.dump(x_, f)
    with open(bundle_y, 'wb') as f:
        pickle.dump(y_, f)


if __name__ == '__main__':
    runner()

