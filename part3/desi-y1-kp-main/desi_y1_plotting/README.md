# Plotting

Definition of plotting styles for Y1KPs.

```
from desipipe import setup_logging
from desipipe.file_manager import BaseFile
from desi_y1_plotting import KP3StylePaper
setup_logging()

correlation_fn = BaseFile(path='/global/cfs/cdirs/desi//survey/catalogs/Y1/LSS/iron/LSScats/v0.6/blinded/xi/smu/allcounts_LRG_GCcomb_0.4_0.6_default_FKP_lin_njack0_nran4_split20.npy', filetype='correlation', options=dict(tracer='LRG', zrange=(0.4, 0.6)))
power_fn = BaseFile(path='/global/cfs/cdirs/desi//survey/catalogs/Y1/LSS/iron/LSScats/v0.6/blinded/pk/pkpoles_LRG_GCcomb_0.4_0.6_default_FKP_lin.npy', filetype='power', options=dict(tracer='LRG', zrange=(0.4, 0.6)))

with KP3StylePaper() as style:
    style.plot_correlation_multipoles(data=correlation_fn, fn='_tests/correlation.png')
    style.plot_power_multipoles(data=power_fn, fn='_tests/power.png')

```

See plots here https://data.desi.lbl.gov/desi/survey/catalogs/Y1/LSS/iron/LSScats/test/blinded/plots_kp3/ from the ```scripts/plot_y1kp3_key_paper.py``` script.
See plots here https://data.desi.lbl.gov/desi/science/cpe/y1kp7/test/mock_blinded_y1/plots_kp7/ from the ```scripts/plot_y1kp7_key_paper.py``` script.