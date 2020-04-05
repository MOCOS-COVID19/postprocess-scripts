import pickle
import click
import numpy as np
from matplotlib import pyplot as plt
import os
import scipy.stats

@click.command()
@click.option('--path', type=click.Path(exists=True))
@click.option('--bundle-prefix', default='')
def runner(path, bundle_prefix):
    """
    Calculates how many sample paths are fitting 2-points criteria and saves bundles to files.

    :param path: path to set of simulations e.g. "<outputdir>/<experiment_root>/"
    :param bundle_prefix: string prefix for reading bundle coordinations (may be blank)
                          - method searches for files <bundle_prefix>bundle_x.pkl and <bundle_prefix>bundle_y.pkl
    :return:
    """

    d = path
    list_files_with_paths = [f.path for f in os.scandir(d) if f.is_file()]
    list_files_with_paths.sort()
    sim_filter = f'{bundle_prefix}bundle'
    x_ = []
    y_ = []
    successes = 0
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

    bundle_x = os.path.join(d, f'{bundle_prefix}bundle_x.pkl')
    bundle_y = os.path.join(d, f'{bundle_prefix}bundle_y.pkl')
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
        plt.savefig(os.path.join(d, f'bundle_all_{bundle_prefix}_test.png'), dpi=300)
        plt.close(fig)


if __name__ == '__main__':
    runner()
