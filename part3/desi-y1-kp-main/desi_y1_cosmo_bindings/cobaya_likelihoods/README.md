# Cobaya

## Path

Export the Python path to the desi-y1-kp directory:
```
cd ../..
export PYTHONPATH=$(pwd):$PYTHONPATH
```

## BAO

To test the Cobaya implementation of BAO likelihoods:
```
cobaya-run config_bao.yaml
```
Note that you can add all tracer likelihoods one-by-one , e.g. 'desi_bao_bgs', as in the above example, single-out one redshift bin, e.g. 'desi_bao_lrg_z0', or use the all-tracer likelihoods directly, 'desi_bao_all', 

```'path'``` should indicate the path where bao_data is.

## SN

Cobaya and desilike implementations of SN likelihoods can be compared with:
```
cobaya-run compare_cobaya_desilike_sn.yaml
```

# DESY3 3x2pt

To test the Cobaya implementation of DES-Y3 3x2pt likelihoods:
```
cobaya-run config_desy3.yaml
```