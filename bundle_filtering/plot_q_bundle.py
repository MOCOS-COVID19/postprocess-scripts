import pickle
import click
import numpy as np
from matplotlib import pyplot as plt
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
@click.option('--simulation-prefix', default='grid')
@click.option('--q-id', type=int, default=0)
@click.option('--outputs-id', default='')
@click.option('--bundle-prefix', default='')
@click.option('--max-x', type=int, default=60)
@click.option('--max-y', type=int, default=80000)
@click.option('--plot-resolution-x', type=int, default=400)
@click.option('--plot-resolution-y', type=int, default=20000)
def runner(path, simulation_prefix, q_id, outputs_id, bundle_prefix, max_x, max_y, plot_resolution_x, plot_resolution_y):
    """
    Calculates how many sample paths are fitting 2-points criteria and saves bundles to files.

    :param path: path to set of simulations e.g. "<outputdir>/<experiment_root>/"
    :param simulation_prefix: prefix used for set of simulations, e.g. "grid"
    :param q_id: q id value used for filtering the proper simulations, e.g. "2" for filtering on "<simulation_prefix>_2_*"
    :param outputs_id: identifier for simulation outputs,
                       e.g. "wroclaw" for "<path>/<simulation_prefix>_<q_id>_*/outputs/<outputs_id>"
    :param bundle_prefix: string prefix for reading bundle coordinations (may be blank)
                          - method searches for files <bundle_prefix>bundle_x.pkl and <bundle_prefix>bundle_y.pkl
    :param max_x: time horizon in days for visualization (e.g. 60 [days]) - longer trajectories will be trimmed
    :param max_y: max value for visualization (e.g. 20000) - trajectories with larger than max_y cases will be trimmed
    :param plot_resolution_x: number of points on x axis (e.g. 200)
    :param plot_resolution_y: number of points on y axis (e.g. 200)
    :return:
    """
    d = path
    coeffs_path = os.path.join(d, f'{bundle_prefix}coeffs_{q_id}.pkl')
    coeffs_ = []
    successes = 0
    if os.path.exists(coeffs_path):
        with open(coeffs_path, 'rb') as f:
            coeffs_ = pickle.load(f)
        successes = 1
    else:
        list_subfolders_with_paths = [f.path for f in os.scandir(d) if f.is_dir()]
        sim_filter = f'{simulation_prefix}_{q_id}_'
        for sub_ in list_subfolders_with_paths:
            print(f'entering {sub_}')
            print(f'basename: {os.path.basename(sub_)} vs filter: {sim_filter}')
            if not os.path.basename(sub_).startswith(sim_filter):
                continue
            bundle_dir = os.path.join(sub_, 'outputs', outputs_id)
            coeff_path = os.path.join(bundle_dir, f'{bundle_prefix}coeffs.pkl')
            if os.path.exists(coeff_path):
                with open(coeff_path, 'rb') as f:
                    coeffs = pickle.load(f)
                    coeffs_.extend(coeffs)
            else:
                print(f'cannot read - {coeff_path} do not exist!')
                continue
        print(successes)
        with open(coeffs_path, 'wb') as f:
            pickle.dump(coeffs_, f)
    if successes > 0:
        array = np.zeros((plot_resolution_x, plot_resolution_y))
        x1 = np.arange(60000)/1000
        for coeffs in coeffs_:
            p = np.poly1d(coeffs)
            y1 = np.exp(p(x1))
            for x_elem, y_elem in zip(x1, y1):
                if y_elem > max_y:
                    continue
                if x_elem > max_x:
                    continue
                array[x_to_xid(x_elem, max_x, plot_resolution_x)][y_to_yid(y_elem, max_y, plot_resolution_y)] += 1.0

        fig, ax = plt.subplots(figsize=(10, 6))
        pa = ax.imshow(np.rot90(array), cmap='BuPu', vmin=0, vmax=np.percentile(array, 99), aspect='auto')
        ax.set_title("Wiązka prawdopodobnych krzywych", fontsize=18)
        cbb = plt.colorbar(pa, shrink=0.35)

        ax.set_xticks(np.arange(0, plot_resolution_x, plot_resolution_x / 10))
        ax.set_xticklabels([f'dzień {v}' for v in range(0, 60, 6)], rotation=30)
        ax.set_yticks([v for v in np.arange(plot_resolution_y, 0, -plot_resolution_y / 10.0)])
        ax.set_yticklabels(
            [int(v) for v in np.arange(0, plot_resolution_y, plot_resolution_y / 10.0)])  # , list(np.arange(20)))
        ylabel_pl = 'Liczba zdiagnozowanych przypadków'
        ylabel_en = 'detected cases'
        ylabel = ylabel_pl
        xlabel_pl = 'Liczba dni od dziś'
        xlabel_en = 'Days from today'
        xlabel = xlabel_pl
        ax.set_ylabel(ylabel, fontsize=18)
        ax.set_xlabel(xlabel, fontsize=18, labelpad=16)
        plt.tight_layout()
        plt.savefig(os.path.join(d, f'bundle_{q_id}_{bundle_prefix}_test.png'), dpi=300)
        plt.close(fig)


if __name__ == '__main__':
    runner()
