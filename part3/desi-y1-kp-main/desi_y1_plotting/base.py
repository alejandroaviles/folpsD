import matplotlib as mpl


class BaseStyle(object):
    """
    Context for base plotting style.
    To be used as a context:

    .. code-block:: python

        with BaseStyle() as style:

            ...

    """
    def __init__(self, **kwargs):
        """
        Initialize :class:`BaseStyle`.

        Parameters
        ----------
        kwargs : dict
            Either arguments for ``mpl.rcParams``, or attributes for this class.
            Passed to :meth:`update`.
        """
        self._rcparams = mpl.rcParams.copy()
        self.update(**kwargs)

    def update(self, *args, **kwargs):
        """
        Update attributes.

        Parameters
        ----------
        kwargs : dict
            Either arguments for ``mpl.rcParams``, or attributes for this class.
            Passed to :meth:`update`.
        """
        tmp = {}
        tmp.update(*args, **kwargs)
        for name, value in tmp.items():
            if name in mpl.rcParams:
                mpl.rcParams[name] = value
            else:
                setattr(self, name, value)

    def __enter__(self):
        """Enter context."""
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        """Exit context."""
        mpl.rcParams = self._rcparams