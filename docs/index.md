---
layout: default
---

PatchRNN is a deep learning based model to identify security patches in software using recurrent neural networks. 
Due to the different structures of patch components, PatchRNN tends to process the commit message and the code revision individually hence conducting a more comprehensive analysis.
The code revision in patches is processed using a twin RNN model with a static analytic tool, while the commit message is processed with a TextRNN model with NLP techniques. 
Afterward, the information from the commit message and the code revision is fused to get the final prediction using a multi-layer perceptron.

More details about the PatchRNN can be found in the paper "[PatchRNN: A Deep Learning-Based System for Security Patch Identification](https://shuwang127.github.io/papers/milcom21_PatchRNN.pdf)", to appear in the IEEE/AFCEA MILCOM 2021, San Diego, USA, November 29–December 2, 2021.

If you are using PatchRNN for work that will result in a publication (thesis, dissertation, paper, article), please use the following citation:

```bibtex
@article{PatchRNN2021Wang,
  author    = {Xinda Wang and
               Shu Wang and
               Pengbin Feng and
               Kun Sun and
               Sushil Jajodia and
               Sanae Benchaaboun and
               Frank Geck},
  title     = {PatchRNN: A Deep Learning-Based System for Security Patch Identification},
  journal   = {CoRR},
  volume    = {abs/2108.03358},
  year      = {2021},
  url       = {https://arxiv.org/abs/2108.03358},
  eprinttype = {arXiv},
  eprint    = {2108.03358},
  timestamp = {Wed, 11 Aug 2021 15:24:08 +0200},
  biburl    = {https://dblp.org/rec/journals/corr/abs-2108-03358.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
```

To download the PatchRNN demo program, please use this [link](https://github.com/shuwang127/PatchRNN-demo).
This demo program is based on the [PatchRNN Developer Edition (S2020.08.08-V4)](https://github.com/shuwang127/PatchRNN), developed by [Shu Wang](https://shuwang127.github.io/). 
You can follow the provided [instructions](#instructions) to run this demo.

## PatchRNN design

### 1. Overview

Figure 1 presents the architecture of our PatchRNN toolkit.
Since a commit (i.e., a patch file) is composed of code revision and commit message, we utilize both parts to capture more comprehensive features. 
For code revision, we reconstruct the unpatched code and patched code and process them separately with the same tokenization and abstraction strategies.
Then a twin RNN model is adopted to generate the code vector representation. 
For the commit message, we utilize the NLP toolkit to process the text sequences and employ a TextRNN model to obtain vector representation for commit message.
The final results are derived from the feature vectors of both parts. 
We use a large-scale patch dataset [PatchDB](https://sunlab-gmu.github.io/PatchDB/) to train the model. The size of the dataset is 38K.

![architecture](https://shuwang127.github.io/PatchRNN-demo/img/architecture.png)
<p align="center">Figure 1: The architecture of PatchRNN model.</p>

From the perspective of process flow, our system can be divided into 4 phases: text processing, featurization, RNN-based feature vector extraction, and finial identification. From the perspective of process objects, our system can be divided into 3 modules: code revision processing, commit message processing, and fused predictive modeling.

### 2. Code revision processing


### 3. Commit message processing


### 4. Fused predictive modeling


## How to run PatchRNN-demo <span id="instructions"></span>

### 1. Install OS

We use Ubuntu 20.04.3 LTS (Focal Fossa) desktop version. \
Download Link: [https://releases.ubuntu.com/20.04/ubuntu-20.04.3-desktop-amd64.iso](https://releases.ubuntu.com/20.04/ubuntu-20.04.3-desktop-amd64.iso).

The virtualization software in our experiments is VirtualBox 5.2.24. \
Download Link: [https://www.virtualbox.org/wiki/Download_Old_Builds_5_2](https://www.virtualbox.org/wiki/Download_Old_Builds_5_2). \
You can use other software like VMware Workstation.

**System configurations:**\
RAM: 2GB\
Disk: 25GB\
CPU: 1 core in i7-7700HQ @ 2.8GHz

### 2. Download the source code from github

We use `home` directory to store the project folder.

```shell
cd ~
```

Install `git` tool.

```shell 
sudo apt install git
```

Download `PatchRNN-demo` project to local disk. (You may need to enter your github account/password.)

```shell 
git clone https://github.com/shuwang127/PatchRNN-demo
```

### 3. Install the dependencies.

Install `pip` tool for `python3`.

```shell 
sudo apt install python3-pip
```

Install common dependencies.

```shell 
pip3 install nltk==3.3
pip3 install natsort
pip3 install pandas
pip3 install sklearn
```

Install CPU-version PyTorch. Official website: https://pytorch.org/.

```shell 
pip3 install torch==1.7.1+cpu torchvision==0.8.2+cpu torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html
```

Install `clang` tool.

```shell 
pip3 install clang==6.0.0.2
```

Configurate the clang environment.

```shell 
sudo apt install clang
cd /usr/lib/x86_64-linux-gnu/
sudo ln -s libclang-*.so.1 libclang.so
```

### 4. Run the demo program.

```shell 
cd ~/PatchRNN-demo/
python3 demo.py
```

There are 56 input test samples stored in `~/PatchRNN-demo/testdata/`, the output results are saved in `~/PatchRNN-demo/results/results.txt`.

```shell 
cat results/results.txt
```

You can find the results.

```shell 
.//testdata/nginx/release-1.19.0_release-1.19.1/0a683fdd.patch,1
.//testdata/nginx/release-1.19.0_release-1.19.1/1bbc37d3.patch,1
.//testdata/nginx/release-1.19.0_release-1.19.1/2afc050b.patch,0
.//testdata/nginx/release-1.19.0_release-1.19.1/2d4f04bb.patch,0
.//testdata/nginx/release-1.19.0_release-1.19.1/6bb43361.patch,1
...
```
