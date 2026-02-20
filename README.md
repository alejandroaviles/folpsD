<!-- 
<p align="center">
</p>
-->

# FOLPS 

## (This is version 2. FOLPS-v1 is located at [here](https://github.com/alejandroaviles/folpsD/tree/folpsD_v1) or [here](https://github.com/alejandroaviles/folpsD/tree/folps-nu).

FOLPS computes the galaxy power spectrum and bispectrum (Sugiyama and Scocimarro basis) for cosmologies in the presence of massive neutrinos. This version can be ran with numpy or JAX (see also [Folpsax](https://github.com/cosmodesi/folpsax) for jax with FOLPS-v1)

[![arXiv](https://img.shields.io/badge/arXiv-2208.02791-red)](https://arxiv.org/abs/2208.02791)


## Developers: 
- [Hernán E. Noriega](mailto:henoriega@estudiantes.fisica.unam.mx)
- [Alejandro Aviles](mailto:avilescervantes@gmail.com)


*Special thanks to Arnaud de Mattia for helping with the [Jax](https://github.com/cosmodesi/folpsax) version of this code.* 


## Run

**Dependences**

The code employs the standard libraries:
- NumPy 
- SciPy

We recommend to use NumPy versions ≥ 1.20.0. For older versions, one needs to rescale by a factor 1/N the [FFT computation](https://github.com/henoriega/FOLPS-nu/blob/main/FOLPSnu.py#L626). 

To run the code, first use git clone:

```
git clone https://github.com/alejandroaviles/folpsD.git
```

<!-- 
or install via pip by:

```
pip install git+https://github.com/henoriega/FOLPS-nu
```
-->


Once everything is ready, please check this [Jupyter Notebook](https://github.com/alejandroaviles/folpsD/blob/main/notebooks/example_detailed.ipynb), which contains some helpful examples. 

Attribution
-----------

Please cite <https://arxiv.org/abs/2208.02791> if you find this code useful in your research. We encourage you to cite also <https://arxiv.org/abs/2106.13771>.

<!--
@article{Noriega:2022nhf,
    author = "Noriega, Hern{\'a}n E. and Aviles, Alejandro and Fromenteau, Sebastien and Vargas-Maga{\~n}a, Mariana",
    title = "{Fast computation of non-linear power spectrum in cosmologies with massive neutrinos}",
    eprint = "2208.02791",
    archivePrefix = "arXiv",
    primaryClass = "astro-ph.CO",
    doi = "10.1088/1475-7516/2022/11/038",
    journal = "JCAP",
    volume = "11",
    pages = "038",
    year = "2022"
}

    @article{Aviles:2021que,
    author = "Aviles, Alejandro and Banerjee, Arka and Niz, Gustavo and Slepian, Zachary",
    title = "{Clustering in massive neutrino cosmologies via Eulerian Perturbation Theory}",
    eprint = "2106.13771",
    archivePrefix = "arXiv",
    primaryClass = "astro-ph.CO",
    reportNumber = "FERMILAB-PUB-21-280-T",
    doi = "10.1088/1475-7516/2021/11/028",
    journal = "JCAP",
    volume = "11",
    pages = "028",
    year = "2021"
}
-->


## Acknowledgements: 

-This code was financially supported by grants UNAM PAPIIT IA101825 and SECIHTI CBF2023-2024-162.


