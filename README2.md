## ReadMe for running PatchRNN


### 1. Install OS

We use Ubuntu 20.04.2.0 LTS (Focal Fossa) desktop version. \
Download Link: https://releases.ubuntu.com/20.04/ubuntu-20.04.2.0-desktop-amd64.iso

The virtualization software in our experiments is VirtualBox 5.2.24. \
Download Link: https://www.virtualbox.org/wiki/Download_Old_Builds_5_2. \
You can use other software like VMware Workstation.

System configurations:\
RAM: 2GB\
Disk: 10GB\
CPU: 1 core in i7-7700

### 2. Download the source code from github

We use `home` directory to store the project folder.

```shell scripts
cd ~
```

Install `git` tool.

```shell scripts
sudo apt install git
```

Download `PatchRNN-demo` project to local disk.

```shell scripts
git clone https://github.com/SunLab-GMU/PatchRNN-demo
```

### 3. Install the dependencies.

Install `pip` tool for `python3`.

```shell scripts
sudo apt install python3-pip
```

Install common dependencies.

```shell scripts
pip3 install nltk==3.3
pip3 install natsort
```

Install CPU-version PyTorch. Official website: https://pytorch.org/.

```shell scripts
pip3 install torch==1.7.1+cpu torchvision==0.8.2+cpu torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html
```

