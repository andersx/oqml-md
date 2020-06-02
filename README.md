# oqml-md
Example for running MD simulation for OQML

# How to run
## 1) Training the model

To train a model and save it as a pickle with an ASE calculator:

```
    # syntax:
    # ./train_oqml.py <npz_filename> <train_idx_filename> <kernel_width> <regularizer>
    ./train_oqml.py ../data/h2co_ccsdt_avtz_4001.npz ../data/idx_train_100_0.dat 2.0 1e-10
```

# 2) Run MD simulation

To run un an MD simulation with the saved calculator


```
    ./run_md_oqml.py
```

# 3) Requirements:

``` 
    # Install ASE
    pip install ase --user -U

    # Install QML development version
    pip install git+https://github.com/qmlcode/qml@develop --user -U --verbose
    
    # Install QML development version with Intel compilers (MUCH faster)
    pip install git+https://github.com/qmlcode/qml@develop --user -U --global-option="build" --global-option="--compiler=intelem" --global-option="--fcompiler=intelem" --verbose
```
