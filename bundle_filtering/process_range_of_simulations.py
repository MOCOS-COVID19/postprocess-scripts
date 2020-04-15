import pickle
import click
import pandas as pd
import numpy as np
import os
from matplotlib import pyplot as plt


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

def hospitalized_cases(df_r1):
    """ This method should be moved to utils """
    cond1 = ~df_r1.t2.isna()
    cond2a = ~df_r1.trecovery.isna()
    cond2b = df_r1.t2 > df_r1.trecovery
    cond2 = ~np.logical_and(cond2a, cond2b)
    if len(df_r1[~df_r1.tdeath.isna()]) > 0:
        cond3a = ~df_r1.tdeath.isna()
        cond3b = df_r1.t2 > df_r1.tdeath
        cond3 = ~np.logical_and(cond3a, cond3b)
        cond23 = np.logical_and(cond2, cond3)
    else:
        cond23 = cond2
    cond = np.logical_and(cond1, cond23)
    df = df_r1[cond]
    return df.sort_values(by='t2').t2

def infected_cases(df_r2):
    """ This method should be moved to utils """
    return df_r2.sort_values(by='contraction_time').contraction_time


def n_days_back(x, ref, n):
    filtered = x[x <= (ref - n)]
    if len(filtered) == 0:
        return 0
    return 1 + np.argmax(filtered)


def n_avg(x, ref, n):
    avg = 0.0
    for i in np.arange(n):
        avg += n_days_back(x, ref, i) / n
    return avg


def avg_array(x, n, max_len):
    avg = [n_avg(x, elem, n) for elem in x[:int(max_len)]]
    return np.array(avg)

def every_nth(seq, seq2, step=0.1):
    assert len(seq) == len(seq2)
    ret = []
    ret2 = []
    while len(seq) > 0:
        ret.append(seq[0])
        ret2.append(seq2[0])
        id = np.argmax(seq >= seq[0] + step)
        if id == 0:
            break
        seq = seq[id:]
        seq2 = seq2[id:]
    return np.array(ret), np.array(ret2)

#@click.option('--data', type=click.Path(exists=True))
#@click.option('--zero-date', default='20200414')
@click.command()
@click.option('--path', type=click.Path(exists=True))
@click.option('--zero', type=float, default=0.0)
@click.option('--minus', type=float, default=0.0)
@click.option('--minus2', type=float, default=-1.0)
@click.option('--minus-days', type=float, default=7.0)
@click.option('--minus2-days', type=float, default=14.0)
@click.option('--minus-tolerance', type=float, default=0.1)
@click.option('--minus2-tolerance', type=float, default=0.21)
@click.option('--days', type=int, default=60)
@click.option('--prefix', default='')
@click.option('--sliding-window-length', type=int, default=1)
def runner(path, zero, minus, minus2, minus_days, minus2_days, minus_tolerance, minus2_tolerance, days, prefix, sliding_window_length):
    """
    Calculates how many sample paths are fitting 2-points criteria and saves bundles to files.

    :param path: path to set of simulations e.g. "<outputdir>/<experiment_root>/grid_X_Y/outputs/<outputs_id>/"
    :param zero: number of detected cases at zero time (synchronization point)
    :param minus: number of detected cases at past time (2- or 3-point checkpoint)
    :param minus2: number of detected cases at second past time (3-point checkpoint); if minus2 < 0, we ignore this and use 2-point checkpoint
    :param minus_days: how many days are between minus day and zero day (e.g. 7)
    :param minus2_days: how many days are between minus2 day and zero day (e.g. 14)
    :param minus_tolerance: fraction of "minus" that is tolerated (e.g. 0.1*minus)
    :param minus2_tolerance: fraction of "minus2" that is tolerated (e.g. 0.21*minus)
    :param days: time horizon used for saving forecast bundle files (e.g. 60)
    :param prefix: file prefix for storing bundle coordinations (may be blank)
    :param sliding_window_length: if 1, use values for zero and minus, if >1, then use avg over last sliding_window_length days
    :return:
    """
    assert sliding_window_length >= 1
    d = path
    list_subfolders_with_paths = [f.path for f in os.scandir(d) if f.is_dir()]
    zero_time = zero
    minus_time = minus
    minus2_time = minus2
    successes = 0
    tries = 0
    fails = []
    fails2 = []
    detected_check_ = []
    hospitalized_ = []
    infected_ = []
    detected_ = []
    arrs = [detected_, hospitalized_, infected_]
    for i, sub_ in enumerate(list_subfolders_with_paths):
        print(sub_)
        if os.path.basename(sub_).startswith('agg'):
            continue
        progression_path = os.path.join(sub_, 'output_df_progression_times.csv')
        if not os.path.exists(progression_path):
            continue
        contractions_path = os.path.join(sub_, 'output_df_potential_contractions.csv')
        if not os.path.exists(contractions_path):
            continue

        tries += 1

        df = pd.read_csv(progression_path, na_values='None')
        detected = detected_cases(df).values

        if n_avg(detected, detected[-1], sliding_window_length) <= zero_time:
            continue
        avg_detected = avg_array(detected, sliding_window_length, zero_time*1.2)
        zero_time_av = np.argmax(avg_detected[avg_detected <= zero_time])
        t0 = detected[zero_time_av]
        detected = detected - t0

        filt_detected = detected[detected <= - minus_days]
        if len(filt_detected) == 0:
            continue

        arg_tminus = np.argmax(filt_detected)

        if np.abs(avg_detected[arg_tminus] - minus_time) > minus_tolerance * minus_time:
            fails.append(avg_detected[arg_tminus])
            continue

        if minus2 > 0.0:

            filt_detected = detected[detected <= - minus2_days]
            if len(filt_detected) == 0:
                continue

            arg_tminus = np.argmax(filt_detected)

            if np.abs(avg_detected[arg_tminus] - minus2_time) > minus2_tolerance * minus2_time:
                fails2.append(avg_detected[arg_tminus])
                continue

        successes += 1

        hospitalized = hospitalized_cases(df).values
        df = pd.read_csv(contractions_path, na_values='None')
        infected = infected_cases(df).values
        df = None
        hospitalized = hospitalized - t0
        infected = infected - t0

        for i, arr in enumerate([detected, hospitalized, infected]):
            start_y = np.argmax(arr >= 0) + 1
            x = arr[arr >= 0]
            x = x[x <= days]
            # TODO: if we want to display bundles for averages instead:
            # y = avg_detected[zero_time_av:zero_time_av + len(x)]
            y = np.arange(start_y, start_y + len(x))
            print(f'{len(x)} {len(y)}')
            x, y = every_nth(np.array(x), y, step=0.05)
            print(f'{len(x)} {len(y)}')
            arrs[i].append(zip(x, y))
        #coeff = np.polyfit(x, np.log(y), 5)
        #coeffs.append(coeff)
        #print(f'{sub_},{coeff}')
        x = detected[detected <= 0]
        y = np.arange(1, 1 + len(x))
        print(f'{len(x)} {len(y)}')
        x, y = every_nth(np.array(x), y, step=0.05)
        print(f'{len(x)} {len(y)}')
        detected_check_.append(zip(x, y))
        # TODO: if we want to display bundles for averages instead, we need to store y_ as well
        detected = None
        infected = None
        hospitalized = None
        x = None
        y = None

    for i, arr_str in enumerate(['detected', 'hospitalized', 'infected']):
        save_path = os.path.join(d, f'{prefix}{arr_str}_{sliding_window_length}.pkl')
        with open(save_path, 'wb') as f:
            pickle.dump(arrs[i], f)

    x_path = os.path.join(d, f'{prefix}x_{sliding_window_length}.pkl')
    with open(x_path, 'wb') as f:
        pickle.dump(detected_check_, f)

    too_small = (1 - minus_tolerance) * minus_time
    too_large = (1 + minus_tolerance) * minus_time
    print(f'bundle condition failing values: {fails}, '
          f'smaller than {too_small:.1f}: {len([fail for fail in fails if fail < too_small])}, '
          f'larger than {too_large:.1f}: {len([fail for fail in fails if fail > too_large])}')
    if minus2 > 0.0:
        too_small = (1 - minus2_tolerance) * minus2_time
        too_large = (1 + minus2_tolerance) * minus2_time
        print(f'bundle condition failing values: {fails2}, '
              f'smaller than {too_small:.1f}: {len([fail for fail in fails2 if fail < too_small])}, '
              f'larger than {too_large:.1f}: {len([fail for fail in fails2 if fail > too_large])}')

    print(f'bundle success ratio: {successes}/{tries}')


if __name__ == '__main__':
    runner()

