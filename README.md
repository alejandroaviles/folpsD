<!-- 
<p align="center">
</p>
-->

# FOLPS 

## This is Folps version 2. FOLPS-v1 is located at [here](https://github.com/alejandroaviles/folpsD/tree/folpsD_v1) or [here](https://github.com/alejandroaviles/folpsD/tree/folps-nu).

FOLPS computes the galaxy power spectrum and bispectrum (Sugiyama and Scocimarro basis) for cosmologies in the presence of massive neutrinos. This version can be ran with numpy or JAX (see also [Folpsax](https://github.com/cosmodesi/folpsax) for jax with FOLPS-v1)

The official repositorie for folpsD is (https://github.com/cosmodesi/FolpsD)

[![arXiv](https://img.shields.io/badge/arXiv-2208.02791-red)](https://arxiv.org/abs/2208.02791)
[![arXiv](https://img.shields.io/badge/arXiv-2604.08895-red)](https://arxiv.org/abs/2604.08895)



## Tests and Timing Benchmarks

The [folps](folps/) directory includes several `test_*.py` scripts that demonstrate end-to-end execution for both the power spectrum and bispectrum using NumPy and JAX.

Main scripts:

- [folps/test_folps_numpy.py](folps/test_folps_numpy.py)
- [folps/test_folps_jax.py](folps/test_folps_jax.py)
- [folps/test_compare_folps_numpy_vs_jax.py](folps/test_compare_folps_numpy_vs_jax.py)

## Notebooks

The [notebooks](notebooks/) directory contains worked examples showing how to run:

- Power spectrum calculations
- Bispectrum calculations in the Scoccimarro and Sugiyama bases
- Windowed bispectrum calculations including survey geometry effects

Main notebooks:

- [notebooks/example_folps_numpy.ipynb](notebooks/example_folps_numpy.ipynb)
- [notebooks/example_folps_jax.ipynb](notebooks/example_folps_jax.ipynb)
- [notebooks/B000_B202_windowing.ipynb](notebooks/B000_B202_windowing.ipynb)

## Developers

- [Hernan E. Noriega](mailto:henoriega@icf.unam.mx)
- [Alejandro Aviles](mailto:avilescervantes@gmail.com)

Arnaud de Mattia: support with JAX-related development

Prakhar Bansal: integration with desilike.

## References

FOLPS theory: [https://arxiv.org/abs/2007.06508](https://arxiv.org/abs/2007.06508), [https://arxiv.org/abs/2106.13771](https://arxiv.org/abs/2106.13771)  

Folps v1 original release: [https://arxiv.org/abs/2208.02791](https://arxiv.org/abs/2208.02791)   

Including bispectrum and JAX capabilities: [https://arxiv.org/abs/2604.08895](https://arxiv.org/abs/2604.08895)   


## Acknowledgements

We acknowledge financial support from grants DGAPA-PAPIIT IA101825 and SECIHITI CBF2023-2024-162
