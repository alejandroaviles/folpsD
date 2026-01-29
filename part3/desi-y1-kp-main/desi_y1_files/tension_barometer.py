# Driver class and methods to estimate tension and significance metrics for DESI-Y1 KP7.
# All errors are mine.
# Nhat-Minh Nguyen (UofM) -- nguyenmn@umich.edu

import logging
import warnings
import numpy as np
from scipy.stats import uniform, norm

class TensionBarometer:
    r"""
    A class to compute:
    - significance of a point in parameter space (from theory or observation) given the data constraint.
    - tension between two data constraints (within the same theoretical model).
    """
    logger = logging.getLogger('TensionBarometer')
   
    def __init__(self, *args):
        r"""
        Class constructor.

        Parameters
        ----------
        samples : list[getdist.MCSamples]
            A list of ``getdist.MCSamples`` instance(s).
        """
        self.samples = args
        self.significances = None
        self.n_datapoints = None
        if self.samples is None:
            raise Exception(f"Receive empty input. Provide >=1 instance(s) of ``getdist.MCSamples``.")
        elif self.samples == 1:
            warnings.warn(f"Receive a single ``getdist.MCSamples`` as input.", category=Warning)
        else:
            self.logger.info(f"Receive {len(self.samples)} instances of ``getdist.MCSamples`` as input.")
            
    def compute_point_pdf_2D(self, par_xy, point_xy, normalized=False, **kwargs):
        r"""
        Helper function.
        Return the posterior densities at points corresponding to input parameters ``par_xy``.
        
        Parameters
        ----------
        par_xy : [str, str]
            The indexes or names of the two parameters, ``[par_x, par_y]``.
        point_xy : [float, float]
            The position of the prediction/observation in ``[par_x, par_y]`` coordinates.
        normalized : bool, default=False
            Whether to normalize the :class:`getdist.densities.Density2D` estimate.
        **kwargs:
            Keyword arguments for :func:`getdist.mcsamples.get2DDensityGridData`.
        """
        point_pdf_2D = []
        for sample in self.samples:
            density2D = sample.get2DDensity(par_xy[0], par_xy[1], normalized=normalized, **kwargs)
            point_pdf_2D.append(density2D.Prob(point_xy[0], point_xy[1]))
        return point_pdf_2D
    
    def get_density_contours_2D(self, par_xy, contour_levels, normalized=False, **kwargs):
        r"""
        Helper function.
        Return the posterior densities corresponding to input `contour_levels`.
        
        Parameters
        ----------
        par_xy : [str, str]
            The indexes or names of the two parameters ``[par_x, par_y]``.
        contour_levels : list[float]
            The contour levels, i.e. confidence limits per each sample.
        normalized : bool, default=False
            Whether to normalize the :class:`getdist.densities.Density2D` estimate.
        **kwargs:
            Keyword arguments for :func:`getdist.mcsamples.get2DDensity`.
        """
        density_contours_2D = []
        for sample in self.samples:
            density2D = sample.get2DDensity(par_xy[0], par_xy[1], normalized=normalized, **kwargs)
            density_contours_2D.append(density2D.getContourLevels(contours=contour_levels))
        return np.asarray(density_contours_2D).reshape(len(self.samples), len(contour_levels))
    
    def compute_deltachi2(self):
        r"""
        Compute delta chi2 where chi2 = -2 * log(posterior).
        Assume the first `getdist.mcsamples` instance in :attr:`samples` as baseline.
        Try to load -log(Like) from a `.minimum` or `.bestfit` file.
        If neither can be found, fall back to -log(Like).max() in `getdist.mcsamples`.
        """
        MAP_chi2s = []
        for sample in self.samples:
            try:
                # NOTE: getdist.mcsamples.getBestFit().logLike returns bestfit -log(posterior)
                MAP_minusloglike = sample.getBestFit().logLike
            except FileNotFoundError as err:
                warnings.warn("Cannot load `BestFit` from file. No `.minimum` or `.bestfit` file found. Use a *noisy* estimate of MAP from chain.", UserWarning)
                # NOTE: getdist.mcsamples.getLikeStats().logLike_sample returns MCMC sample bestfit -log(Like)
                MAP_minusloglike = sample.getLikeStats().logLike_sample
            MAP_chi2s.append(2 * MAP_minusloglike)
        MAP_chi2s = np.asarray(MAP_chi2)
        return MAP_chi2s - MAP_chi2s[0]

    def compute_deltak(self):
        r"""
        Return the difference between number of degrees of freedom between samples.
        Assume the first `getdist.mcsamples` instance in :attr:`samples` as baseline.
        """
        ks = []
        for sample in self.samples:
            ks.append(len(sample.getParamSampleDict(0, want_derived=False, want_fixed=False)))
        ks = np.asarray(ks)
        # We only care about the difference, all universal parameters should cancel out
        return ks - ks[0]

    def compute_deltachi2_over_deltak(self):
        r"""
        Compute delta chi2 / delta k.
        Assume the first `getdist.mcsamples` instance in :attr:`samples` as baseline.
        """
        deltachi2 = self.compute_deltachi2()
        deltak = self.get_deltak()
        return deltachi2 / deltak

    def compute_deltaAIC(self):
        r"""
        Compute delta Akaike Information Criterion (AIC).
        AIC = 2*(-logLike) + 2*k.
        Reference: [DES-Y3 extension paper](https://arxiv.org/abs/2207.05766), Eq.(F5)
        In practice, delta AIC = delta chi2 + (2 * delta k)
        """
        return self.compute_deltachi2() + 2.0 * self.get_deltak()

    def compute_deltaDIC(self):
        r"""
        Compute delta Deviance Information Criterion (DIC).
        DIC = 2*(-logLike) + 2*pDIC, where pDIC = 2*logLike - 2*<logL>
        Reference: [DES-Y3 extension paper](https://arxiv.org/abs/2207.05766), Eq.(F6-F7)
        In practice, delta DIC = -delta chi2 - (2 * delta <logL>)
        """
        DICs = []
        for sample in self.samples:
            try:
                # NOTE: getdist.mcsamples.getBestFit().logLike returns bestfit -log(Like)
                pDIC = 2.0 * (sample.getBestFit().logLike - sample.getLikeStats().meanLogLike)
                DIC = 2.0 * (sample.getBestFit().logLike + pDIC)
            except:
                warnings.warn("Cannot load `BestFit` from file. No `.minimum` or `.bestfit` file found. Use a *noisy* estimate of MAP log(Like) and pDIC from chain.", UserWarning)
                pDIC = 2.0 * sample.getLikeStats().varLogLike
                # NOTE: getdist.mcsamples.getLikeStats().logLike_sample returns MCMC sample bestfit -log(Like)
                DIC = 2.0 * (sample.getLikeStats().logLike_sample + pDIC)
            DICs.append(DIC)
        DICs = np.asarray(DICs)
        return DICs - DICs[0]

    def compute_deltaBIC(self, n_datapoints=None):
        r"""
        Compute delta Bayesian Information Criterion (BIC).
        BIC = k*log(n) - 2*log(Like), where n is the number of data points in the data vector.
        Reference: [`astropy.stats.bayesian_info_criterion` doc](https://docs.astropy.org/en/stable/api/astropy.stats.bayesian_info_criterion.html), Eq.(F6-F7)

        Parameters
        ----------
        n_datapoints : list(int)
            List of numbers of data points in the data vector of each sample in :attr:`samples`.
        """
        if self.n_datapoints is None and n_datapoints is None:
            raise ValueError("Must pass a list of numbers of data points (in the data vectors), per each sample in `self.samples`.")
        elif self.n_datapoints is None and n_datapoints is not None:
            self.n_datapoints = n_datapoints
        else:
            pass
        BICs = []
        for sample in self.samples:
            # NOTE: minus the `weights` and `logpost` parameters
            ks.append(len(sample.getParamSampleDict(0, want_derived=False, want_fixed=False)) - 2)
        klogns = np.asarray(ks) * np.asarray(self.n_datapoints)
        delta_klogns = klogns - klogns[0]
        delta_chi2 = self.compute_deltachi2()
        return delta_klogns + delta_chi2

    def compute_posterior_density_ratio_2D(self, par_xy, point_xy, normalized=False, **kwargs):
        r"""
        Main function.
        Compute the ratio of the posterior density at `point_xy` over that at the MAP point.
        Require a `BestFit` object loaded from a `.minimum` or `.bestfit` file.
        Those are outputs from a posterior/likelihood minimization run on top of the MCMC chain.
        
        Parameters
        ----------
        par_xy : [str, str]
            The indexes or names of the two parameters, [par_x, par_y].
        point_xy : [float, float]
            The position of the prediction/observation in [par_x, par_y] coordinates.
        normalized : bool, default=False
            Whether to normalize the :class:`getdist.densities.Density2D` estimate.
        **kwargs :
            Keyword arguments for :meth:`compute_point_pdf_2D` and :func:`getdist.mcsamples.get2DDensity`
        """
        point_pdf_2Ds = self.compute_point_pdf_2D(par_xy, point_xy, normalized=normalized, **kwargs)
        MAP_pdf_2Ds = []
        for sample in self.samples:
            try:
                MAP_pdf_2D = self.compute_point_pdf_2D(par_xy,\
                            [sample.getBestFit().getParamDict()[par_xy[0]], sample.getBestFit().getParamDict()[par_xy[1]]],\
                            normalized=normalized, **kwargs)
            except FileNotFoundError as err:
                warnings.warn("Cannot load `BestFit` from file. No `.minimum` or `.bestfit` file found. Use a *noisy* estimate of MAP from chain.", UserWarning)
                density2D = sample.get2DDensity(par_xy[0], par_xy[1], normalized=normalized, **kwargs)
                MAP_pdf_2D = density2D.P.max()
            MAP_pdf_2Ds.append(MAP_pdf_2D)
        return [(point_pdf_2D / MAP_pdf_2D) for point_pdf_2D, MAP_pdf_2D in zip(point_pdf_2Ds, MAP_pdf_2Ds)]
    
    def compute_Bayes_factor_SD_ratio_2D(self, par_xy, point_xy, prior_loc_xy, prior_scale_xy, prior_dist_xy='uniform', normalized=False, **kwargs):
        r"""
        Main function.
        Compute the Bayes factor from MCMC chains through the Savage-Dickey density ratio.
        Note that S-D estimator only works for nested models.
        See, e.g. [here](https://statproofbook.github.io/P/bf-sddr.html)
        
        Parameters
        ----------
        par_xy : [str, str]
            The indexes or names of the two parameters, [par_x, par_y].
        point_xy: [float, float]
            The position of the prediction/observation in [par_x, par_y] coordinates.
        prior_loc_xy, prior_scale_xy : array_like
            The location and scale parameters of the prior distribution(s).
        prior_dist_xy : str, default='uniform'
            Prior distribution of [par_x, par_y].
            Currently support either `uniform` or `normal`.
        normalized : bool, default=False
            Whether to normalize the :class:`getdist.densities.Density2D` estimate.
        **kwargs :
            Keyword arguments for :attr:`compute_point_pdf_2D`.
        """
        if prior_dist_xy == 'uniform':
            prior_pdf = np.prod(uniform.pdf(point_xy, loc=prior_loc_xy, scale=prior_scale_xy))
        elif prior_dist_xy == 'normal':
            prior_pdf = np.prod(norm.pdf(point_xy, loc=prior_loc_xy, scale=prior_scale_xy))
        else:
            prior_pdf = 1.0
            warnings.warn("Invalid choice of prior distribution. ``prior_dist_xy`` should be either `uniform` or `normal`. Defaults to 1.0.",category=Warning)
        return np.log10(np.array(self.compute_point_pdf_2D(par_xy, point_xy, normalized=normalized, **kwargs)) / prior_pdf)

    @staticmethod
    def _bisect_search(arr, point):
        r"""
        Helper function.
        Search the array for the closest *higher* element, by bisection.
        Scale as O(log n), more efficient for large array.
        
        Parameters
        ----------
        arr : array_like
            The input array. Assumed to be sorted in ascending order.
        point : float
            The input point value.
        """
        import bisect
        return len(arr) - bisect.bisect_left(arr, point) - 1

    def compute_posterior_pvalue_2D(self, par_xy, point_xy, normalized=False, inner_CL=1e-6, outer_CL=1.0, n_contours=100, convergence_threshold=1e-4, **kwargs):
        r"""
        Main function.
        Compute the significance of the prediction or observation given the 2D posterior density.
        Return the integral of the posterior density over the area with P(`par_xy`|data) >= P(`point_xy`|data).
        
        Parameters
        ----------
        par_xy : [str, str]
            The indexes or names of the two parameters, [par_x, par_y].
        point_xy : [float, float]
            The position of the prediction/observation in [par_x, par_y] coordinates.
        contour_levels : list[float]
            The contour levels, i.e. confidence limits per each sample.
        normalized : bool, default=False
            Whether to normalize the :class:`getdist.densities.Density2D` estimate.
        inner_CL : float, default=1e-6
        outer_CL : float, default=1.0
            Initial inner and outer confidence levels to begin searching.
            If you have a good idea of P([point_x, point_y]), adjust these two accordingly to improve efficiency.
        n_contours : int, default=100
            Number of contour levels to define.
            Default to `100`.
        convergence_threshold : float, default=1e-4
            Threshold for convergence of P search
        **kwargs :
            Keyword arguments for :meth:`compute_point_pdf_2D` and :meth:`get_density_contours_2D`.
        """
        point_pdf_2Ds = self.compute_point_pdf_2D(par_xy, point_xy, normalized=normalized, **kwargs)
        contour_levels_init = np.linspace(inner_CL, outer_CL, n_contours)
        density_contours_2Ds_init = self.get_density_contours_2D(par_xy, contour_levels=contour_levels_init)
        sig2Ds = []
        for p, pdf2D_point in enumerate(point_pdf_2Ds):
            # Initialize distance from CL to the point
            distance = density_contours_2Ds_init[p][0] - pdf2D_point
            if distance < 0:
                raise ValueError(f"Input point is within the innermost contour. Try define initial contours starting closer from the peak. Current distance P={distance}.")
            else:
                contour_levels = contour_levels_init
                density_contours_2Ds = density_contours_2Ds_init
                try:
                    while distance > convergence_threshold:
                        # Find closest CL with P([par_x, par_y])>=P([point_x, point_y])
                        CL_index = self._bisect_search(np.flip(density_contours_2Ds[p, :]), pdf2D_point)
                        # Check whether CL_index corresponds to the first or last element of contour_levels
                        if CL_index == 0:
                            raise IndexError(f"Input point is closest to the innermost contour. Try define initial contours starting closer from the peak and/or finer contours. Current distance P={distance}.")
                        elif CL_index == n_contours - 1:
                            raise IndexError(f"Input point is closest to the outermost contour. Try define intial contours reaching further out to the tail. Current distance P={distance}.")
                        else:
                            # Update inner_CL and outer_CL
                            inner_CL = contour_levels[CL_index]
                            outer_CL = contour_levels[CL_index + 1]
                            # Update CLs - Zoom in the vicinity of point_xy
                            contour_levels = np.linspace(inner_CL, outer_CL, n_contours)
                            # Compute new densities at updated CLs
                            density_contours_2Ds = self.get_density_contours_2D(par_xy, contour_levels=contour_levels, **kwargs)
                            # Update distance
                            distance = density_contours_2Ds[p, 0] - pdf2D_point
                    self.logger.info("Convergence condition reached.")
                    sig2Ds.append(contour_levels[CL_index])
                except ValueError as e:
                    self.logger.info(f"Convergence condition never met.")
        return np.asarray(sig2Ds)
    
    def get_latex_table(self, sample_names=None, sample_significances=None):
        """
        Return a LaTeX table of sample names and their corresponding significance values.
        Compute and fill significances before calling this function.
        
        Parameters
        ----------
        sample_names : list[str], default=None
            Names of samples/data.
            Either pass here as input or grab from ``getdist.mcsamples``.
        
        sample_significances : list[float], default=None
            Significances of null hypothesis rejections per each sample/data.
            Either pass here as input or pre-compute and store in ``self.significances``.
        """
        if len(self.samples) != len(self.significances):
            raise ValueError("Number of samples and significances must be the same.")
        if sample_names is None:
            warnings.warn("No input sample names. Try to grab names from ``getdist.mcsamples``.", category=Warning)
            sample_names = []
            for sample in self.samples:
                name = sample.getName().replace("_", r"\textunderscore ")
                sample_names.append(name)
        if sample_significances is None:
            warnings.warn("No input sample significances. Try to grab names from ``self.significances``.", category=Warning)
            sample_significances = self.significances
        # Table header - 2 columns [sample name | significance]
        toret = []
        toret.append(r"\begin{table}")
        toret.append(r"\begin{center}")
        toret.append(r"\begin{tabular}{|l l|}")
        toret.append(r"\hline")
        toret.append(r"Data & Significance \\")
        toret.append(r"\hline")

        # Table values - individual rows correspond to individual samples
        for sample_name, sample_significance in zip(sample_names, sample_significances):
            toret.append(f"{sample_name} & {sample_significance * 100:.1f}% \\")

        # Table footer
        toret.append(r"\hline")
        toret.append(r"\end{tabular}")
        toret.append(r"\label{tab:data_significance}")
        toret.append(r"\end{center}")
        toret.append(r"\end{table}")
        return '\n'.join(toret)