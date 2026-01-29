import os
import logging

import numpy as np
from matplotlib import pyplot as plt


logger = logging.getLogger('Plotting')


def lighten_color(color, amount=0.5):
    """
    Lightens the given color by multiplying (1-luminosity) by the given amount.
    Input can be matplotlib color string, hex string, or RGB tuple.

    Examples:
    >> lighten_color('g', 0.3)
    >> lighten_color('#F034A3', 0.6)
    >> lighten_color((.3,.55,.1), 0.5)
    """
    import matplotlib.colors as mc
    import colorsys
    try:
        c = mc.cnames[color]
    except KeyError:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], 1 - amount * (1 - c[1]), c[2])


def is_sequence(item):
    return isinstance(item, (tuple, list))


def is_path(item):
    return isinstance(item, (str, os.PathLike))


def mkdir(dirname, **kwargs):
    """Try to create ``dirname`` and catch :class:`OSError`."""
    try:
        os.makedirs(dirname, **kwargs)  # MPI...
    except OSError:
        return

def compute_tension(mean1, sigma1, mean2=73.04, sigma2=1.04):
    """Compute the level of Gaussian tension between two sets of measurements."""
    diff= np.abs(mean2 - mean1)
    std = np.sqrt(sigma1**2 + sigma2**2)
    return diff / std


def plot_fill_between(ax, x, samples, label=None, color='gray', lw=2., alpha=0.5, q=(2.3, 16, 50, 84, 97.7)):
    """Plot median and +/- 2 sigma regions for a given (flatten) array of samples"""
    qs = np.percentile(samples, q=q, axis=0)
    idx = len(qs) // 2
    median = qs[idx]
    for i in range(1, idx + 1):
        ax.fill_between(x.flatten(), qs[idx - i].flatten(), qs[idx + i].flatten(), color=color, lw=lw, alpha=alpha / i)
    ax.plot(x, median, label=label, c=color, ls='-', lw=lw)


class FakeFigure(object):

    def __init__(self, axes):
        if not hasattr(axes, '__iter__'):
            axes = [axes]
        self.axes = list(axes)


def savefig(filename, fig=None, bbox_inches='tight', pad_inches=0.1, dpi=200, **kwargs):
    """
    Save figure to ``filename``.

    Warning
    -------
    Take care to close figure at the end, ``plt.close(fig)``.

    Parameters
    ----------
    filename : string
        Path where to save figure.

    fig : matplotlib.figure.Figure, default=None
        Figure to save. Defaults to current figure.

    kwargs : dict
        Optional arguments for :meth:`matplotlib.figure.Figure.savefig`.

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    mkdir(os.path.dirname(filename))
    logger.info('Saving figure to {}.'.format(filename))
    if fig is None:
        fig = plt.gcf()
    fig.savefig(filename, bbox_inches=bbox_inches, pad_inches=pad_inches, dpi=dpi, **kwargs)
    return fig


def suplabel(axis, label, shift=0, labelpad=5, ha='center', va='center', **kwargs):
    """
    Add global x-coordinate or y-coordinate label to the figure. Similar to matplotlib.suptitle.
    Taken from https://stackoverflow.com/questions/6963035/pyplot-axes-labels-for-subplots.

    Parameters
    ----------
    axis : str
        'x' or 'y'.

    label : string
        Label string.

    shift : float, optional
        Shift along ``axis``.

    labelpad : float, optional
        Padding perpendicular to ``axis``.

    ha : str, optional
        Label horizontal alignment.

    va : str, optional
        Label vertical alignment.

    kwargs : dict
        Arguments for :func:`matplotlib.pyplot.text`.
    """
    fig = plt.gcf()
    xmin, ymin = [], []
    for ax in fig.axes:
        xmin.append(ax.get_position().xmin)
        ymin.append(ax.get_position().ymin)
    xmin, ymin = min(xmin), min(ymin)
    dpi = fig.dpi
    if axis.lower() == 'y':
        rotation = 90.
        x = xmin - float(labelpad) / dpi
        y = 0.5 + shift
    elif axis.lower() == 'x':
        rotation = 0.
        x = 0.5 + shift
        y = ymin - float(labelpad) / dpi
    else:
        raise ValueError('Unexpected axis {}; chose between x and y'.format(axis))
    plt.text(x, y, label, rotation=rotation, transform=fig.transFigure, ha=ha, va=va, **kwargs)


def plotter(*args, **kwargs):

    from functools import wraps

    use_interactive = False

    def get_wrapper(func):
        """
        Return wrapper for plotting functions, that adds the following (optional) arguments to ``func``:

        Parameters
        ----------
        fn : str, Path, default=None
            Optionally, path where to save figure.
            If not provided, figure is not saved.

        kw_save : dict, default=None
            Optionally, arguments for :meth:`matplotlib.figure.Figure.savefig`.

        show : bool, default=False
            If ``True``, show figure.
        """
        @wraps(func)
        def wrapper(*args, fn=None, kw_save=None, show=False, fig=None, **kwargs):

            if fig is not None:

                if not isinstance(fig, plt.Figure):  # create fake figure that has axes
                    fig = FakeFigure(fig)
                
                elif not fig.axes:
                    fig.add_subplot(111)

                kwargs['fig'] = fig

            fig = func(*args, **kwargs)
            if fn is not None:
                savefig(fn, **(kw_save or {}))
            if show: plt.show()
            return fig

        return wrapper

    if kwargs or not args:
        if args:
            raise ValueError('unexpected args: {}, {}'.format(args, kwargs))
        return get_wrapper

    if len(args) != 1:
        raise ValueError('unexpected args: {}'.format(args))

    return get_wrapper(args[0])
