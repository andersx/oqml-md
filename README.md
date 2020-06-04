# oqml-md
Example for running MD simulation for OQML

# How to run MD
## 1) Requirements:

``` 
    # Install ASE
    pip install ase --user -U

    # Install QML development version
    pip install git+https://github.com/qmlcode/qml@develop --user -U --verbose
    
    # Install QML development version with Intel compilers (MUCH faster)
    pip install git+https://github.com/qmlcode/qml@develop --user -U --global-option="build" --global-option="--compiler=intelem" --global-option="--fcompiler=intelem" --verbose
```
## 2) Training the model

To train a model and save it as a pickle with an ASE calculator:

```
    # syntax:
    # ./train_oqml.py <npz_filename> <train_idx_filename> <kernel_width> <regularizer>
    ./train_oqml.py ../data/h2co_ccsdt_avtz_4001.npz ../data/idx_train_100_0.dat 2.0 1e-10
```

## 3) Run MD simulation

To run un an MD simulation with the saved calculator


```
    ./run_md_oqml.py
```


# How to predict dipoles:

## 1) Requirements:

First install Python3 from Anaconda (only way to get the correct Openbabel)


``` 
    # Install Openbabel
    conda install -c openbabel openbabel
   
    # Install QML development version with Intel compilers (MUCH faster)
    ~/opt/anaconda3/bin/pip install --install-option="--compiler=intelem" --install-option="--fcompiler-intelem" git+https://github.com/qmlcode/qml@develop --verbose --user -U

    # Install tqdm
    ~/opt/anaconda3/bin/pip install tqdm --user -U

```
Important to note: Use the pip executable from the anaconda library! Mine is located in `~/opt/anaconda3/bin/pip` but yours my differ/




## 2) To train a model for dipoles: (default parameters are 0.64 and 1e-09)
```
    # syntax:
    # ./train_dipole.py <npz_filename> <train_idx_filename> <kernel_width> <regularizer>
    ./train_dipole.py ../data/h2co_ccsdt_avtz_4001.npz ../data/idx_train_100_0.dat 0.64 1e-09
```

## 3) To predict dipoles: (use the same kernel width as for the training step above!)

```
    # syntax:
    # ./predict_fchl.py <trajectory.xyz> <kernel_width>
    ./predict_fchl.py pos_oqml_aligned_new.xyz 0.64
```
