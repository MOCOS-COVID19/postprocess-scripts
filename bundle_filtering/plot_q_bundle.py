import pickle
import click
import numpy as np
from matplotlib import pyplot as plt
import os
import scipy.stats

@click.command()
@click.option('--path', type=click.Path(exists=True))
@click.option('--simulation-prefix', default='grid')
@click.option('--q-id', type=int, default=0)
@click.option('--outputs-id', default='')
@click.option('--bundle-prefix', default='')
def runner(path, simulation_prefix, q_id, outputs_id, bundle_prefix):
    """
    Calculates how many sample paths are fitting 2-points criteria and saves bundles to files.

    :param path: path to set of simulations e.g. "<outputdir>/<experiment_root>/"
    :param simulation_prefix: prefix used for set of simulations, e.g. "grid"
    :param q_id: q id value used for filtering the proper simulations, e.g. "2" for filtering on "<simulation_prefix>_2_*"
    :param outputs_id: identifier for simulation outputs,
                       e.g. "wroclaw" for "<path>/<simulation_prefix>_<q_id>_*/outputs/<outputs_id>"
    :param bundle_prefix: string prefix for reading bundle coordinations (may be blank)
                          - method searches for files <bundle_prefix>bundle_x.pkl and <bundle_prefix>bundle_y.pkl
    :return:
    """

    d = path
    list_subfolders_with_paths = [f.path for f in os.scandir(d) if f.is_dir()]
    sim_filter = f'{simulation_prefix}_{q_id}_'
    x_ = []
    y_ = []
    successes = 0
    for sub_ in list_subfolders_with_paths:
        print(f'entering {sub_}')
        print(f'basename: {os.path.basename(sub_)} vs filter: {sim_filter}')
        if not os.path.basename(sub_).startswith(sim_filter):
            continue
        bundle_dir = os.path.join(sub_, 'outputs', outputs_id)
        bundle_x = os.path.join(bundle_dir, f'{bundle_prefix}bundle_x.pkl')
        bundle_y = os.path.join(bundle_dir, f'{bundle_prefix}bundle_y.pkl')
        print(f'checking if exists: {bundle_x} and {bundle_y}')
        if os.path.exists(bundle_x) and os.path.exists(bundle_y):
            with open(bundle_x, 'rb') as f:
                x = pickle.load(f)
                x_.extend(x)
            with open(bundle_y, 'rb') as f:
                y = pickle.load(f)
                y_.extend(y)
            successes += 1
            print(f'added x and y: {bundle_x} and {bundle_y}')
        else:
            print(f'cannot read - either {bundle_x} or {bundle_y} do not exist!')
            continue
    print(successes)
    bundle_x = os.path.join(d, f'{bundle_prefix}bundle_x_{q_id}.pkl')
    bundle_y = os.path.join(d, f'{bundle_prefix}bundle_y_{q_id}.pkl')
    if os.path.exists(bundle_x):
        print(f'cannot save bundle_x to {bundle_x} - file already exists!')
    else:
        with open(bundle_x, 'wb') as f:
            pickle.dump(x_, f)
    if os.path.exists(bundle_y):
        print(f'cannot save bundle_y to {bundle_y} - file already exists!')
    else:
        with open(bundle_y, 'wb') as f:
            pickle.dump(y_, f)
    if successes > 0:
        xedges = np.arange(0, 60, 0.1)
        yedges = np.arange(0, 20000, 30)
        H, xedges, yedges = np.histogram2d(x_, y_, bins=(xedges, yedges))
        H = H.T  # Let each row list bins with common y range.

        fig, ax = plt.subplots()

        X, Y = np.meshgrid(xedges, yedges)
        flat = H.flatten()
        flat.sort()

        ax.pcolormesh(X, Y, H, cmap='Blues', vmin=0, vmax=flat[-int(len(flat) / 1000)])
        fig.tight_layout()
        plt.savefig(os.path.join(d, f'bundle_{q_id}_{bundle_prefix}_test.png'), dpi=300)
        plt.close(fig)


if __name__ == '__main__':
    runner()
