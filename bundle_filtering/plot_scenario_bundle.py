import pickle
import click
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.dates as mdates
from dateutil import parser
import datetime as dt
import os


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


@click.command()
@click.option('--path', type=click.Path(exists=True))
@click.option('--bundle-prefix', default='')
@click.option('--max-x', type=int, default=60)
@click.option('--max-y', type=int, default=80000)
@click.option('--plot-resolution-x', type=int, default=800)
@click.option('--plot-resolution-y', type=int, default=40000)
@click.option('--begin-date', default='20200410')
def runner(path, bundle_prefix, max_x, max_y, plot_resolution_x, plot_resolution_y, begin_date):
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

    d = path
    coeffs_path = os.path.join(d, f'{bundle_prefix}coeffs.pkl')

    coeffs_ = []
    successes = 0
    if os.path.exists(coeffs_path):
        with open(coeffs_path, 'rb') as f:
            coeffs_ = pickle.load(f)
        successes = 1
    else:
        list_files_with_paths = [f.path for f in os.scandir(d) if f.is_file()]
        list_files_with_paths.sort()
        sim_filter = f'{bundle_prefix}coeffs'

        for sub_ in list_files_with_paths:
            if not os.path.basename(sub_).startswith(sim_filter):
                continue

            with open(sub_, 'rb') as f:
                coeffs = pickle.load(f)
                coeffs_.extend(coeffs)

            successes += 1

        print(successes)
        with open(coeffs_path, 'wb') as f:
            pickle.dump(coeffs_, f)

    if successes > 0:
        array = np.zeros((plot_resolution_x, plot_resolution_y))
        x1 = np.arange(max_x * 1000) / 1000
        for coeffs in coeffs_:
            p = np.poly1d(coeffs)
            y1 = np.exp(p(x1))
            zer = np.zeros_like(array)
            prev_point = None
            for x_elem, y_elem in zip(x1, y1):
                if y_elem > max_y:
                    continue
                if x_elem > max_x:
                    continue
                x_p = x_to_xid(x_elem, max_x, plot_resolution_x)
                y_p = y_to_yid(y_elem, max_y, plot_resolution_y)
                if prev_point is None:
                    zer[x_p, y_p] = 1.0
                else:
                    zer[prev_point[0]:(x_p + 1), prev_point[1]:(y_p + 1)] = 1.0
                prev_point = (x_p, y_p)
            array += zer
            print(prev_point[1])

        fig, ax = plt.subplots(figsize=(10, 6))
        pa = ax.imshow(np.rot90(array), cmap='BuPu', vmin=0, vmax=np.maximum(5, np.percentile(array, 99)),
                       aspect='auto')
        #ax.set_title("Prognozowane scenariusze rozwoju choroby", fontsize=18)
        cbb = fig.colorbar(pa, shrink=0.35, location='left')
        cbarlabel = 'Zagęszczenie trajektorii'
        cbb.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom", fontsize=18)
        ax.yaxis.tick_right()
        ax.set_xticks(np.arange(0, plot_resolution_x + 1, 7 * plot_resolution_x / max_x))
        now = parser.parse(begin_date)
        then = now + dt.timedelta(days=max_x + 1)
        days = mdates.drange(now, then, dt.timedelta(days=7))
        t = [dt.datetime.fromordinal(int(day)).strftime('%d/%m/%y') for day in days]
        ax.set_xticklabels([t[i] for i, v in enumerate(range(0, max_x + 1, 7))], rotation=30)
        ax.set_yticks([v for v in np.arange(plot_resolution_y, -1, -plot_resolution_y / 10.0)])
        def format_num(n):
            a = str(n)
            l = len(a)
            x = []
            while l > 3:
                x.append(a[-3:])
                a = a[:-3]
                l -= 3
            x.append(a)
            return ' '.join(x[::-1])
        ax.set_yticklabels([format_num(int(v)) for v in np.arange(0, max_y + 1, max_y / 10.0)])  # , list(np.arange(20)))
        ylabel_pl = 'Liczba zdiagnozowanych przypadków'
        ylabel_en = 'detected cases'
        ylabel = ylabel_pl
        xlabel_pl = 'Data'
        xlabel_en = 'Days from today'
        xlabel = xlabel_pl
        ax.set_ylabel(ylabel, fontsize=18)
        ax.set_xlabel(xlabel, fontsize=18, labelpad=16)
        plt.tight_layout()
        plt.savefig(os.path.join(d, f'bundle_all_{bundle_prefix}_test.png'), dpi=300)
        plt.close(fig)


if __name__ == '__main__':
    runner()
