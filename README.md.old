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
> pip install clang == 6.0.0.2
> pip install nltk  == 3.3
> pip install natsort
> pip install torch
```

IMPORTANT Note: You might receive the following error message when running this demo program on ***Linux*** system!

```
clang.cindex.LibclangError: libclang.so: cannot open shared object file: No such file or directory. 
To provide a path to libclang use Config.set_library_path() or Config.set_library_file().
```

To solve this problem, you can make a simbolic link for `libclang-6.0.so.1` in `/usr/lib/x86_64-linux-gnu/`.

```shell script
> cd /usr/lib/x86_64-linux-gnu/
> ll | grep clang
  -rw-r--r--   1 root root 27976816 Apr  5  2018 libclang-6.0.so.1
> sudo ln -s libclang-6.0.so.1 libclang.so
> ll | grep clang
  -rw-r--r--   1 root root 27976816 Apr  5  2018 libclang-6.0.so.1
  lrwxrwxrwx   1 root root       17 Feb  3 23:03 libclang.so -> libclang-6.0.so.1
```

**File Structure**:

```
PatchRNN
    |-- logs                                      
        |-- PatchRNNdemo.log                        # log file recording the running information.
    |-- model                                       # folder used to store pre-trained parameters and model.
        |-- model_TwinRNN.pth                       # a pretrained model.
        |-- tokenDict.pickle                        # dictionary file for diff code.
        |-- tokenDictMsg.pickle                     # dictionary file for commit message.
    |-- results                                     # folder used to store the prediction results.
        |-- results.txt                             # result file.
    |-- testdata                                    # folder used to store input test data.
        |-- nginx
            |-- release-1.19.0_release-1.19.1
                |-- *.patch                         # test patch file.
    |-- tmp                                         # folder used to store temporary results.
        |-- *.npy
    |-- demo.py                                     # main python file.
    |-- README.md                                   # this file.
```

**How to Run**:

***Step 1:*** put the test patch data into the folder `./testdata/`.

***Step 2:*** run the demo using the following command:

```shell script
> python3 ./demo.py
```

***Step 3:*** check the prediction result in the file `./results/results.txt`.

```shell script
> cat ./results/results.txt
```

The output would like the following format `[path_to_file],[0 or 1]`.

```
.//testdata/nginx/release-1.19.0_release-1.19.1/0a683fdd.patch,1
.//testdata/nginx/release-1.19.0_release-1.19.1/1bbc37d3.patch,1
.//testdata/nginx/release-1.19.0_release-1.19.1/2afc050b.patch,0
.//testdata/nginx/release-1.19.0_release-1.19.1/2d4f04bb.patch,0
.//testdata/nginx/release-1.19.0_release-1.19.1/6bb43361.patch,1
...
```

If the prediction is 1, the sample is identified as a security patch; otherwise it is identified as non-security one.
