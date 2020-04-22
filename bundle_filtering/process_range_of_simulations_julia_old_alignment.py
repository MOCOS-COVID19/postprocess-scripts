import pickle
import click
import pandas as pd
import numpy as np
import os
import h5py
from matplotlib import pyplot as plt


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

def delayed_array(x, n, max_len):
    count = np.arange(1, 1 + max_len)
    delayed = [count[i] - n_days_back(x, elem, n) for i, elem in enumerate(x[:int(max_len)])]
    return np.array(delayed)

def every_nth(seq, seq2, step=0.1):
    assert len(seq) == len(seq2), f'{len(seq)} vs {len(seq2)}'
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


def compress(x, y, step, verbose=True):
    x = np.array(x)
    y = np.array(y)
    if verbose:
        print(f'Length before compression: {len(x)} {len(y)}')
    if step > 0:
        x, y = every_nth(x, y, step=step)
        if verbose:
            print(f'After compressing: {len(x)} {len(y)}')
    elif verbose:
        print(f'step: {step} <= 0, so we are not compressing trajectories')
    return x, y


#@click.option('--data', type=click.Path(exists=True))
#@click.option('--zero-date', default='20200414')
@click.command()
@click.option('--path', type=click.Path(exists=True))
@click.option('--offset-days', type=int, default=7)
@click.option('--offset-tolerance', type=float, default=0.10)
@click.option('--days', type=int, default=60)
@click.option('--prefix', default='')
@click.option('--sliding-window-length', type=int, default=1)
@click.option('--groundtruth-path')
@click.option('--file-id', default='output.jld2')
@click.option('--step', type=float, default=0.05)
def runner(path, offset_days, offset_tolerance, days, prefix, sliding_window_length, groundtruth_path, file_id, step):
    """
    Calculates how many sample paths are fitting 2-points criteria and saves bundles to files.

    :param path: path to set of simulations e.g. "<outputdir>/<experiment_root>/grid_X_Y/outputs/<outputs_id>/"
    :param offset_days: how many days are between minus day and zero day (e.g. 7)
    :param offset_tolerance: fraction of "minus" that is tolerated (e.g. 0.1*minus)
    :param days: time horizon used for saving forecast bundle files (e.g. 60)
    :param prefix: file prefix for storing bundle coordinations (may be blank)
    :param sliding_window_length: if 1, use values for zero and minus, if >1, then use avg over last sliding_window_length days
    :param groundtruth_path
    :param file_id
    :param step: what is the distance between two points in bundle (we are compressing the trajectory). Put 0 or less to not compress
    :return:
    """
    assert sliding_window_length >= 1
    d = path
    #list_subfolders_with_paths = [f.path for f in os.scandir(d) if f.is_file()]
    successes = 0
    tries = 0
    fails = []
    fails2 = []
    detected_check_ = []
    detected_check_slide_ = []
    infected_ = []
    detected_ = []
    arrs = [detected_, infected_]
    julia_path = os.path.join(d, file_id)
    if not os.path.exists(julia_path):
        return
    f = h5py.File(julia_path, "r")
    dw_dets = pd.read_csv(groundtruth_path)
    minus2_cases = dw_dets.average4.values[-1 - offset_days - offset_days]
    zero_time = dw_dets.average4.values[-1]
    minus_cases = dw_dets.average4.values[-1 - offset_days]
    dw_dets = None
    for key in f.keys():

        tries += 1

        detected = f[key]['detection_times'][()]
        infected = f[key]['infection_times'][()]

        if n_avg(detected, detected[-1], sliding_window_length) <= zero_time:
            continue
        avg_detected = avg_array(detected, sliding_window_length, zero_time*1.4)
        zero_time_av = np.argmax(avg_detected[avg_detected <= zero_time])
        t0 = detected[zero_time_av]
        detected = detected - t0

        filt_detected = detected[detected <= - offset_days]
        if len(filt_detected) == 0:
            continue

        arg_tminus = np.argmax(filt_detected)

        if len(avg_detected) < arg_tminus:
            continue

        if np.abs(avg_detected[arg_tminus] - minus_cases) > offset_tolerance * minus_cases:
            fails.append(avg_detected[arg_tminus])
            continue

        filt_detected = detected[detected <= - 2 * offset_days]
        if len(filt_detected) == 0:
            continue

        arg_tminus2 = np.argmax(filt_detected)

        if len(avg_detected) < arg_tminus2:
            continue

        if np.abs(avg_detected[arg_tminus2] - minus2_cases) > 2 * offset_tolerance * minus2_cases:
            fails2.append(avg_detected[arg_tminus2])
            continue

        successes += 1

        infected = infected - t0
        for ij, arr in enumerate([detected, infected]):
            start_y = np.argmax(arr >= 0) + 1 #
            x = arr[arr >= 0]
            x = x[x <= days]
            # TODO: if we want to display bundles for averages instead:
            # y = avg_detected[zero_time_av:zero_time_av + len(x)]
            y = np.arange(start_y, start_y + len(x))
            x, y = compress(x, y, step)
            arrs[ij].append(zip(x, y))

        x = detected[detected <= 0]
        y = np.arange(1, 1 + len(x))
        x, y = compress(x, y, step)
        detected_check_.append(zip(x, y))
        x = detected[detected <= 0]
        y = avg_detected[:len(x)]
        x, y = compress(x, y, step)
        detected_check_slide_.append(zip(x, y))
        # TODO: if we want to display bundles for averages instead, we need to store y_ as well
        detected = None
        infected = None
        x = None
        y = None

    for i, arr_str in enumerate(['detected', 'infected']):
        save_path = os.path.join(d, f'{prefix}{arr_str}_{sliding_window_length}.pkl')
        with open(save_path, 'wb') as f:
            pickle.dump(arrs[i], f)

    x_path = os.path.join(d, f'{prefix}x_{sliding_window_length}.pkl')
    with open(x_path, 'wb') as f:
        pickle.dump(detected_check_, f)

    x_path = os.path.join(d, f'{prefix}x_slide_{sliding_window_length}.pkl')
    with open(x_path, 'wb') as f:
        pickle.dump(detected_check_slide_, f)

    too_small = (1 - offset_tolerance) * minus_cases
    too_large = (1 + offset_tolerance) * minus_cases
    print(f'bundle condition failing values: {fails}, '
          f'smaller than {too_small:.1f}: {len([fail for fail in fails if fail < too_small])}, '
          f'larger than {too_large:.1f}: {len([fail for fail in fails if fail > too_large])}')

    too_small = (1 - 2 * offset_tolerance) * minus2_cases
    too_large = (1 + 2 * offset_tolerance) * minus2_cases
    print(f'bundle condition failing values: {fails2}, '
          f'smaller than {too_small:.1f}: {len([fail for fail in fails2 if fail < too_small])}, '
          f'larger than {too_large:.1f}: {len([fail for fail in fails2 if fail > too_large])}')

    print(f'bundle success ratio: {successes}/{tries}')


if __name__ == '__main__':
    runner()

