# MontePython

Copy-paste the relevant automatically-generated likelihood directory in the MontePython /path/to/montepython_public/montepython/likelihoods folder.
Copy-paste the content of the likelihood directory's *.param file (which can be empty) in the nuisance parameter section of the MontePython *.param file.


For BAO + SN, run:
```
python /path/to/montepython_public/montepython/MontePython.py run --conf /path/to/montepython_public/default.conf -p config_bao_all_sn.param -o _chains_bao_all_sn/
```

For FS + BAO, run:
```
python /path/to/montepython_public/montepython/MontePython.py run --conf /path/to/montepython_public/default.conf -p config_fs_bao_all.param -o _chains_fs_bao_all/
```
