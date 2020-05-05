import pickle
import click
import pandas as pd
import numpy as np
import os
import h5py
import re
import csv


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


class Processor:

    def _get_iterations(self, path):
        raise NotImplementedError('Children implement this')

    def get_detected(self, path, key):
        raise NotImplementedError('Children implement this')

    def get_infection_time(self, path, key):
        raise NotImplementedError('Children implement this')

    def run(self, f, d, offset_days, offset_tolerance, days, prefix, sliding_window_length, groundtruth_path, step):
        # list_subfolders_with_paths = [f.path for f in os.scandir(d) if f.is_file()]
        successes = 0
        tries = 0
        fails = []
        fails2 = []
        detected_check_ = []
        detected_check_slide_ = []
        infected_ = []
        detected_ = []
        arrs = [detected_, infected_]

        dw_dets = pd.read_csv(groundtruth_path)
        minus_cases = dw_dets.average4.values[-1 - offset_days - offset_days]
        zero_time = dw_dets.average4.values[-1 - offset_days]
        plus_cases = dw_dets.average4.values[-1]
        dw_dets = None
        for key in self._get_iterations(f):

            tries += 1

            detected = self.get_detected(f, key)
            infected = self.get_infection_time(f, key)

            if n_avg(detected, detected[-1], sliding_window_length) <= zero_time:
                continue
            avg_detected = avg_array(detected, sliding_window_length, plus_cases * 1.4)
            zero_time_av = np.argmax(avg_detected[avg_detected <= zero_time])
            t0 = detected[zero_time_av]
            detected = detected - (t0 + offset_days)

            filt_detected = detected[detected <= - 2 * offset_days]
            if len(filt_detected) == 0:
                continue

            arg_tminus = np.argmax(filt_detected)

            if len(avg_detected) < arg_tminus:
                continue

            if np.abs(avg_detected[arg_tminus] - minus_cases) > offset_tolerance * minus_cases:
                fails.append(avg_detected[arg_tminus])
                continue

            filt_detected = detected[detected <= 0]
            if len(filt_detected) == 0:
                continue

            arg_tplus = np.argmax(filt_detected)

            if len(avg_detected) < arg_tplus:
                continue

            if np.abs(avg_detected[arg_tplus] - plus_cases) > offset_tolerance * plus_cases:
                fails2.append(avg_detected[arg_tplus])
                continue

            successes += 1

            infected = infected - (t0 + offset_days)
            for ij, arr in enumerate([detected, infected]):
                start_y = np.argmax(arr >= -offset_days) + 1  #
                x = arr[arr >= -offset_days]
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

        too_small = (1 - offset_tolerance) * plus_cases
        too_large = (1 + offset_tolerance) * plus_cases
        print(f'bundle condition failing values: {fails2}, '
              f'smaller than {too_small:.1f}: {len([fail for fail in fails2 if fail < too_small])}, '
              f'larger than {too_large:.1f}: {len([fail for fail in fails2 if fail > too_large])}')

        print(f'bundle success ratio: {successes}/{tries}')


class JLD2Processor(Processor):
    def _get_iterations(self, path):
        return path.keys()

    def get_detected(self, path, key):
        return path[key]['detection_times'][()]

    def get_infection_time(self, path, key):
        return path[key]['infection_times'][()]


class CSVProcessor(Processor):
    def _get_iterations(self, path):
        pattern = re.compile(r'_(\d+)\.csv')
        iterations = []
        for file in os.listdir(path):
            iterations.append(int(pattern.search(file).group(1)))
        return sorted(list(set(iterations)))

    def _read_from_csv(self, file_path, column):
        output = []
        with open(file_path, 'r') as csvfile:
            output_reader = csv.reader(csvfile)
            for idx, line in enumerate(output_reader):
                if idx == 0:
                    continue
                output.append(float(line[column]))
        return np.array(sorted(output))

    def get_detected(self, path, key):
        file_path = os.path.join(path, f'detections_{key}.csv')
        return self._read_from_csv(file_path, 1)

    def get_infection_time(self, path, key):
        file_path = os.path.join(path, f'progressions_{key}.csv')
        return self._read_from_csv(file_path, 0)


# @click.option('--data', type=click.Path(exists=True))
# @click.option('--zero-date', default='20200414')
@click.command()
@click.option('--path', type=click.Path(exists=True))
@click.option('--offset-days', type=int, default=7)
@click.option('--offset-tolerance', type=float, default=0.05)
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
    # list_subfolders_with_paths = [f.path for f in os.scandir(d) if f.is_file()]

    julia_path = os.path.join(path, file_id)

    if os.path.isfile(julia_path):
        f = h5py.File(julia_path, "r")
        return JLD2Processor().run(f, path, offset_days, offset_tolerance, days, prefix, sliding_window_length,
                                   groundtruth_path, step)
    if os.path.isdir(julia_path):
        return CSVProcessor().run(julia_path, path, offset_days, offset_tolerance, days, prefix, sliding_window_length,
                                  groundtruth_path, step)
    return


if __name__ == '__main__':
    runner()
