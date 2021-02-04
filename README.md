# ReadMe for Patch RNN demo program

**Description**: A demo program of security patch identification using RNN model.

**Date**: 2021-02-02

**Version**: V4-demo

**OS**: Linux or Windows.

**Prerequisites**: 

Run on CPU only: RAM >= 2GB. 

Run on GPU: RAM >= 2GB, GPU memory >= 2GB.

**Environment**: Python 3.6 or higher. ([Anaconda](https://www.anaconda.com/products/individual) is highly recommended to provide Python 3 environment.)

**Dependencies**:

Use the following commands to install all dependencies.

```shell script
pip install clang == 6.0.0.2
pip install nltk  == 3.3
pip install natsort
pip install torch
```

IMPORTANT Note: You might receive the following error message when running this demo program on ***Linux*** system!

```
clang.cindex.LibclangError: libclang.so: cannot open shared object file: No such file or directory. 
To provide a path to libclang use Config.set_library_path() or Config.set_library_file().
```

**Usage**:
```shell script
python SecurityPatchIdentificationRNN.py
```
