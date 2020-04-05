import pickle
import click
import numpy as np
from matplotlib import pyplot as plt
import os
import scipy.stats

@click.command()
@click.option('--path', type=click.Path(exists=True))
@click.option('--bundle-prefix', default='')
@click.option('--max-x', type=int, default=60)
@click.option('--max-y', type=int, default=20000)
@click.option('--plot-resolution', type=int, default=200)
def runner(path, bundle_prefix, max_x, max_y, plot_resolution):
    """
    Calculates how many sample paths are fitting 2-points criteria and saves bundles to files.

    :param path: path to set of simulations e.g. "<outputdir>/<experiment_root>/"
    :param bundle_prefix: string prefix for reading bundle coordinations (may be blank)
                          - method searches for files <bundle_prefix>bundle_x.pkl and <bundle_prefix>bundle_y.pkl
    :param max_x: time horizon in days for visualization (e.g. 60 [days]) - longer trajectories will be trimmed
    :param max_y: max value for visualization (e.g. 20000) - trajectories with larger than max_y cases will be trimmed
    :param plot_resolution: number of points on one axis (e.g. 200)
    :return:
    """

    d = path
    bundle_x = os.path.join(d, f'{bundle_prefix}bundle_x.pkl')
    bundle_y = os.path.join(d, f'{bundle_prefix}bundle_y.pkl')

    x_ = []
    y_ = []
    successes = 0
    if os.path.exists(bundle_x) and os.path.exists(bundle_y):
        with open(bundle_x, 'rb') as f:
                x_ = pickle.load(f)
        with open(bundle_y, 'rb') as f:
                y_ = pickle.load(f)
        successes = 1
    else:
        list_files_with_paths = [f.path for f in os.scandir(d) if f.is_file()]
        list_files_with_paths.sort()
        sim_filter = f'{bundle_prefix}bundle'

        for sub_ in list_files_with_paths:
            if not os.path.basename(sub_).startswith(sim_filter):
                continue

            if os.path.basename(sub_).startswith(f'{sim_filter}_x'):
                with open(sub_, 'rb') as f:
                    x = pickle.load(f)
                    x_.extend(x)
            elif os.path.basename(sub_).startswith(f'{sim_filter}_y'):
                with open(sub_, 'rb') as f:
                    y = pickle.load(f)
                    y_.extend(y)
            successes += 1

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
        xedges = np.arange(0, max_x, max_x/plot_resolution)
        yedges = np.arange(0, max_y, max_y/plot_resolution)
        H, xedges, yedges = np.histogram2d(x_, y_, bins=(xedges, yedges))
        H = H.T  # Let each row list bins with common y range.

        fig, ax = plt.subplots()

        X, Y = np.meshgrid(xedges, yedges)
        flat = H.flatten()
        flat.sort()
        max_value = flat[-int(len(flat) / 1000)]
        print(f'max_value for coloring (0.1 percentile) : {max_value}')
        ax.pcolormesh(X, Y, H, cmap='Blues', vmin=0, vmax=flat[-int(len(flat) / max_value)])
        fig.tight_layout()
        plt.savefig(os.path.join(d, f'bundle_all_{bundle_prefix}_test.png'), dpi=300)
        plt.close(fig)


if __name__ == '__main__':
    runner()
