import pickle
import click
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.dates as mdates
from dateutil import parser
import datetime as dt
import os
import pandas as pd

def value_to_id(value, max_value, resolution):
    if value < 0:
        raise ValueError(f'{value} {max_value} {resolution}')
    if value > max_value:
        raise ValueError(f'{value} {max_value} {resolution}')
    return int(value / max_value * resolution)


def x_to_xid(x, max_x, plot_resolution_x):
    return value_to_id(x, max_x, plot_resolution_x)


def y_to_yid(y, max_y, plot_resolution_y):
    return value_to_id(y, max_y, plot_resolution_y)


def draw_back_in_time(reverse_array, df_y_column, title, ylabel, filename_fig):
    # now draw back in time
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_title(title, fontsize=18)
    for reverse_time in reverse_array:
        x, y = zip(*reverse_time)
        x = x.astype(np.float64)
        now = parser.parse(begin_date)
        x = [now + dt.timedelta(days=el) for el in x if el <= 0]
        ax.plot(x, y, 'r-')
    if groundtruth_path is not None and os.path.exists(groundtruth_path):
        dat = pd.read_csv(groundtruth_path, converters={'date': (lambda x: parser.parse(x, dayfirst=True))})
        ax.plot('date', df_y_column, 'k.', data=dat)
    else:
        print(f'groundtruth path {groundtruth_path} not found or not specified, ignoring')
    ax.format_xdata = mdates.DateFormatter('%d/%m/%y')
    ax.get_xaxis().set_major_locator(mdates.DayLocator(interval=7))
    ax.get_xaxis().set_major_formatter(mdates.DateFormatter('%d/%m/%y'))
    xlabel_pl = 'Data'
    xlabel_en = 'Days from today'
    xlabel = xlabel_pl

    ax.set_ylabel(ylabel, fontsize=18)
    ax.set_xlabel(xlabel, fontsize=18, labelpad=16)
    plt.tight_layout()
    plt.savefig(os.path.join(d, filename_fig), dpi=300)
    plt.close(fig)


def draw(item, maxy, ylabel, filename_fig):
    array = np.zeros((plot_resolution_x, plot_resolution_y))
    # x1 = np.arange(max_x * 1000)/1000
    for detections in item:
        x1, y1 = zip(*detections)
        # p = np.poly1d(coeffs)
        # y1 = np.exp(p(x1))
        zer = np.zeros_like(array)
        prev_point = None
        for x_elem, y_elem in zip(x1, y1):
            if y_elem >= maxy:
                continue
            if x_elem >= max_x:
                continue
            x_p = x_to_xid(x_elem, max_x, plot_resolution_x)
            y_p = y_to_yid(y_elem, maxy, plot_resolution_y)
            if prev_point is None:
                zer[x_p, y_p] = 1.0
            else:
                zer[prev_point[0]:(x_p + 1), prev_point[1]:(y_p + 1)] = 1.0
            prev_point = (x_p, y_p)
        array += zer
    fig, ax = plt.subplots(figsize=(10, 6))
    # s = ndimage.generate_binary_structure(2, 1)
    # array = ndimage.grey_dilation(array, footprint=s)
    pa = ax.imshow(np.rot90(array), cmap='BuPu', vmin=0, vmax=np.maximum(5, np.percentile(array, 99)),
                   aspect='auto')
    ax.set_title(f'q={q_id}, ({bundle_prefix.strip("_")})', fontsize=18)
    cbb = plt.colorbar(pa, shrink=0.35)
    cbarlabel = 'Zagęszczenie trajektorii'
    cbb.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom", fontsize=18)

    ax.set_xticks(np.arange(0, plot_resolution_x + 1, 7 * plot_resolution_x / max_x))
    now = parser.parse(begin_date)
    then = now + dt.timedelta(days=max_x + 1)
    days = mdates.drange(now, then, dt.timedelta(days=7))
    t = [dt.datetime.fromordinal(int(day)).strftime('%d/%m/%y') for day in days]
    ax.set_xticklabels([t[i] for i, v in enumerate(range(0, max_x + 1, 7))], rotation=30)
    ax.set_yticks([v for v in np.arange(plot_resolution_y, -1, -plot_resolution_y / 10.0)])
    ax.set_yticklabels(
        [int(v) for v in np.arange(0, maxy + 1, maxy / 10.0)])  # , list(np.arange(20)))

    ax.set_ylabel(ylabel, fontsize=18)
    xlabel_pl = 'Data'
    xlabel_en = 'Days from today'
    xlabel = xlabel_pl
    ax.set_xlabel(xlabel, fontsize=18, labelpad=16)
    plt.tight_layout()
    plt.savefig(os.path.join(d, filename_fig), dpi=300)
    plt.close(fig)


@click.command()
@click.option('--path', type=click.Path(exists=True))
@click.option('--bundle-prefix', default='')
@click.option('--max-x', type=int, default=60)
@click.option('--max-y', type=int, default=80000)
@click.option('--max-y-hospitalized', type=int, default=80000)
@click.option('--max-y-infections', type=int, default=80000)
@click.option('--plot-resolution-x', type=int, default=800)
@click.option('--plot-resolution-y', type=int, default=40000)
@click.option('--begin-date', default='20200414')
@click.option('--sliding-window-length', type=int, default=1)
@click.option('--groundtruth-path')
def runner(path, bundle_prefix, max_x, max_y, max_y_hospitalized, max_y_infections,
           plot_resolution_x, plot_resolution_y, begin_date, sliding_window_length, groundtruth_path=None):
    """
    Plots aggregated bundle for all q bundles.

    :param path: path to set of simulations e.g. "<outputdir>/<experiment_root>/"
    :param bundle_prefix: string prefix for reading bundle coordinations (may be blank)
                          - method searches for files <bundle_prefix>bundle_x.pkl and <bundle_prefix>bundle_y.pkl
    :param max_x: time horizon in days for visualization (e.g. 60 [days]) - longer trajectories will be trimmed
    :param max_y: max value for visualization (e.g. 20000) - trajectories with larger than max_y cases will be trimmed
    :param plot_resolution_x: number of points on x axis (e.g. 200)
    :param plot_resolution_y: number of points on y axis (e.g. 200)
    :param begin_date: begin date for xaxis
    :return:
    """
    q_id = 'all'
    d = path
    detected_path = os.path.join(d, f'{bundle_prefix}detected_{q_id}_{sliding_window_length}.pkl')
    infected_path = os.path.join(d, f'{bundle_prefix}infected_{q_id}_{sliding_window_length}.pkl')
    hospitalized_path = os.path.join(d, f'{bundle_prefix}hospitalized_{q_id}_{sliding_window_length}.pkl')
    hospitalized_current_path = os.path.join(d, f'{bundle_prefix}hospitalized_current_{q_id}_{sliding_window_length}.pkl')
    reverse_time_path = os.path.join(d, f'{bundle_prefix}x_{q_id}_{sliding_window_length}.pkl')
    reverse_time_slide_path = os.path.join(d, f'{bundle_prefix}x_slide_{q_id}_{sliding_window_length}.pkl')
    detections_ = []
    infections_ = []
    hospitalizations_ = []
    hospitalizations_current_ = []
    detections_reverse_time_ = []
    detections_reverse_time_slide_ = []

    coeffs_ = []
    successes = 0
    if os.path.exists(detected_path):
        with open(detected_path, 'rb') as f:
            detections_ = pickle.load(f)
        if os.path.exists(infected_path):
            with open(infected_path, 'rb') as f:
                infections_ = pickle.load(f)
        if os.path.exists(hospitalized_path):
            with open(hospitalized_path, 'rb') as f:
                hospitalizations_ = pickle.load(f)
        if os.path.exists(hospitalized_current_path):
            with open(hospitalized_current_path, 'rb') as f:
                hospitalizations_current_ = pickle.load(f)
        if os.path.exists(reverse_time_path):
            with open(reverse_time_path, 'rb') as f:
                detections_reverse_time_ = pickle.load(f)
        if os.path.exists(reverse_time_slide_path):
            with open(reverse_time_slide_path, 'rb') as f:
                detections_reverse_time_slide_ = pickle.load(f)
        successes = 1
    else:
        list_files_with_paths = [f.path for f in os.scandir(d) if f.is_file()]
        list_files_with_paths.sort()
        sim_filter_detected = f'{bundle_prefix}detected_'
        sim_filter_infected = f'{bundle_prefix}infected_'
        sim_filter_hospitalized = f'{bundle_prefix}hospitalized_'
        sim_filter_hospitalized_current = f'{bundle_prefix}hospitalized_current_'
        sim_filter_x = f'{bundle_prefix}x_'
        sim_filter_x_slide = f'{bundle_prefix}x_slide_'
        sims = [sim_filter_detected, sim_filter_infected, sim_filter_hospitalized,
                sim_filter_hospitalized_current, sim_filter_x, sim_filter_x_slide]
        arrays = [detections_, infections_, hospitalizations_,
                  hospitalizations_current_, detections_reverse_time_, detections_reverse_time_slide_]
        for sub_ in list_files_with_paths:
            for sim, arr in zip(sims, arrays):
                if os.path.basename(sub_).startswith(sim):
                    if sim==sim_filter_x:
                        if os.path.basename(sub_).startswith(sim_filter_x_slide):
                            continue
                    with open(sub_, 'rb') as f:
                        elem = pickle.load(f)
                        arr.extend(elem)
                    successes += 1
                    continue

        #########################

        with open(detected_path, 'wb') as f:
            pickle.dump(detections_, f)
        with open(infected_path, 'wb') as f:
            pickle.dump(infections_, f)
        with open(hospitalized_path, 'wb') as f:
            pickle.dump(hospitalizations_, f)
        with open(hospitalized_current_path, 'wb') as f:
            pickle.dump(hospitalizations_current_, f)
        with open(reverse_time_path, 'wb') as f:
            pickle.dump(detections_reverse_time_, f)
        with open(reverse_time_slide_path, 'wb') as f:
            pickle.dump(detections_reverse_time_slide_, f)

    if successes > 0:
        q_id = 'all'

        draw(detections_, max_y, 'Liczba zdiagnozowanych przypadków', f'bundle_{q_id}_{bundle_prefix}_detections.png')
        draw(infections_, max_y_infections, 'Liczba zakażonych', f'bundle_{q_id}_{bundle_prefix}_infections.png')
        draw(hospitalizations_, max_y_hospitalized, 'Liczba hospitalizacji (sumaryczna)', f'bundle_{q_id}_{bundle_prefix}_hospitalizations.png')
        draw(hospitalizations_current_, max_y_hospitalized, 'Liczba hospitalizacji (bieżąca)',
             f'bundle_{q_id}_{bundle_prefix}_hospitalizations_current.png')

        draw_back_in_time(detections_reverse_time_, 'detected', 'Weryfikacja dla poprzednich dni',
                          'Liczba zdiagnozowanych przypadków', f'check_bundle_{q_id}_{bundle_prefix}_test.png')
        draw_back_in_time(detections_reverse_time_slide_, 'average4', 'Weryfikacja dla poprzednich dni - średnia',
                          'Średnia z 4 dni \n liczby zdiagnozowanych przypadków',
                          f'check_slide_bundle_{q_id}_{bundle_prefix}_test.png')


if __name__ == '__main__':
    runner()
