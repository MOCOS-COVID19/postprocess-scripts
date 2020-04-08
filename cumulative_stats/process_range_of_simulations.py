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


def infected_cases(df_r2):
    """ This method should be moved to utils """
    return df_r2.sort_values(by='contraction_time').contraction_time


""" # TODO
def icu_cases():
    cond = [k for k, v in self._expected_case_severity.items() if v == ExpectedCaseSeverity.Critical]
    critical = df_r1.loc[df_r1.index.isin(cond)]
    plus = critical.t2.values
    deceased = critical[~critical.tdeath.isna()]
    survived = critical[critical.tdeath.isna()]
    minus1 = survived.trecovery.values
    minus2 = deceased.tdeath.values
    max_time = df_r2.contraction_time.max(axis=0)
    df_plus = pd.DataFrame({'t': plus, 'd': np.ones_like(plus)})
    df_minus1 = pd.DataFrame({'t': minus1, 'd': -np.ones_like(minus1)})
    df_minus2 = pd.DataFrame({'t': minus2, 'd': -np.ones_like(minus2)})
    df = df_plus.append(df_minus1).append(df_minus2).sort_values(by='t')
    df = df[df.t <= max_time]
"""


@click.command()
@click.option('--path', type=click.Path(exists=True))
@click.option('--zero', type=int, default=133)
@click.option('--minus', type=int, default=81)
@click.option('--minus-days', type=int, default=7)
@click.option('--minus-tolerance', type=float, default=0.1)
@click.option('--days', type=int, default=60)
@click.option('--prefix', default='')
def runner(path, zero, minus, minus_days, minus_tolerance, days, prefix):
    """
    Calculates how many sample paths are fitting 2-points criteria and saves bundles to files.

    :param path: path to set of simulations e.g. "<outputdir>/<experiment_root>/grid_X_Y/outputs/<outputs_id>/"
    :param zero: number of detected cases at zero time (synchronization point)
    :param minus: number of detected cases at past time (2-point checkpoint)
    :param minus_days: how many days are between minus day and zero day (e.g. 7)
    :param minus_tolerance: fraction of "minus" that is tolerated (e.g. 0.1*minus), set to 1 or larger number to not filter at all
    :param days: time horizon after which total number of detected is calculated (e.g. 60)
    :param prefix: file prefix for storing results (may be blank)
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
    print('path,detected,infected')
    for sub_ in list_subfolders_with_paths:
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
        df = pd.read_csv(contractions_path, na_values='None')
        infected = infected_cases(df).values
        df = None

        if len(detected) <= zero_time:
            continue

        t0 = detected[zero_time]
        detected = detected - detected[zero_time]
        filt_detected = detected[detected <= - minus_days]
        if len(filt_detected) == 0:
            continue

        arg_tminus = np.argmax(filt_detected)

        if np.abs(arg_tminus - minus_time) > minus_tolerance * minus_time:
            fails.append(arg_tminus)
            continue

        successes += 1
        detected = detected[detected <= days]
        detected = len(detected)

        infected = infected[infected <= days + t0]
        infected = len(infected)

        print(f'{sub_},{detected},{infected}')
        x_.append(detected)
        y_.append(infected)

    if len(x_) > 0:
        print(f'statistics,detected,infected,icu')
        print(f'mean,{np.array(x_).mean()},{np.array(y_).mean()}')
        print(f'std,{np.array(x_).std()},{np.array(y_).std()}')

    path_detected_pkl = os.path.join(d, f'{prefix}stats_detected.pkl')
    path_infected_pkl = os.path.join(d, f'{prefix}stats_infected.pkl')
    with open(path_detected_pkl, 'wb') as f:
        pickle.dump(x_, f)
    with open(path_infected_pkl, 'wb') as f:
        pickle.dump(y_, f)
    too_small = (1 - minus_tolerance) * minus_time
    too_large = (1 + minus_tolerance) * minus_time
    print(f'bundle condition failing values: {fails}, '
          f'smaller than {too_small:.1f}: {len([fail for fail in fails if fail < too_small])}, '
          f'larger than {too_large:.1f}: {len([fail for fail in fails if fail > too_large])}')
    print(f'bundle success ratio: {successes}/{tries}')


if __name__ == '__main__':
    runner()

