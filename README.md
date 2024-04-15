
<h1  align="center" > ℕ𝔸𝕊-𝔹𝕖𝕟𝕔𝕙-𝕊𝕦𝕚𝕥𝕖-ℤ𝕖r𝕠 for SiGeo and ZiCoPro </h1>



The code is modified on [`NAS-Bench-Suite-Zero`](https://github.com/automl/NASLib/tree/zerocost).

<h3> Quick Links: </h3>

[**Setup**](#setup)
| [**Data**](#data)
| [**Experiments**](#experiments)

# Setup

While installing the repository, creating a new conda environment is recomended. [Install PyTorch GPU/CPU](https://pytorch.org/get-started/locally/) for your setup.

```bash
git clone -b zerocost https://github.com/automl/NASLib/
cd NASLib
conda create -n zerocost  python=3.7
conda activate zerocost
```

Run setup.py file with the following command, which will install all the packages listed in [`requirements.txt`](requirements.txt).
```bash
pip install --upgrade pip setuptools wheel
pip install -e .
```
# Data

Download all the ZC proxies evaluations, which contains the scores for each proxy and validation accuracy for each architecutre. The ```gdown (pip install gdown)``` package is required to download from google drive. The following command will download the data.

```bash
source scripts/bash_scripts/download_nbs_zero.sh <search_space>
source scripts/bash_scripts/download_nbs_zero.sh nb201
source scripts/bash_scripts/download_nbs_zero.sh all
```

Alternative to gdown, here are the google drive links to the ZC proxy evaluations:
- [NB101](https://drive.google.com/file/d/1Rkse44EWgYdBS34iyhjSs9Y2l0fxPCpU/view?usp=share_link)
- [NB201](https://drive.google.com/file/d/1R7n7GpFHAjUZpPISzbhxH0QjubnvZM5H/view?usp=share_link)
- [NB301](https://drive.google.com/file/d/1RddgmwqjWJ1czGT8gEPB8qqhUHazp92G/view?usp=share_link)
- [TNB101-Macro](https://drive.google.com/file/d/1teH8JcQsamZngUD_DMQyNkCoUYYSTM0M/view?usp=share_link)
- [TNB101-Micro](https://drive.google.com/file/d/1SBOVAyhLCBTAJiU_fo7hLRknNrGNqFk7/view?usp=share_link)



Download all the NAS benchmarks and their associated datasets ( for mac users, please make sure you have wget installed).
```bash
source scripts/bash_scripts/download_data.sh all 
```
Alternatively, you can download the benchmark for a specific search space and dataset/task as follows:
```bash
source scripts/bash_scripts/download_data.sh <search_space> <dataset> 
source scripts/bash_scripts/download_data.sh nb201 cifar10
source scripts/bash_scripts/download_data.sh nb201 all 
```

<!---
Download the TransNAS-Bench-101 benchmark from [here](https://www.noahlab.com.hk/opensource/vega/page/doc.html?path=datasets/transnasbench101) unzip the folder and place the benchmark `transnas-bench_v10141024.pth` from this folder in `NASLib/naslib/data/..`

If you face issues downloading the datasets please follow the steps [here](dataset_preparation/).
-->

# Experiments 
See [`naslib/runners`](naslib/runners) for specific experiment scripts. 
Here we provide instructions for running experiments en masse. Note that the correlation experiments requires SLURM on your machine. 
Please contact us if you have any questions.

## Reproduce ZC proxy correlation experiments in Section 5.3 (SiGeo)
```bash
cd NASLib
NASbenchmark=nasbench201-9000
dataset=cifar10
config=config_10pct_warmup
python naslib/runners/benchmarks/runner.py --config-file configs/correlation/SiGeo/${NASbenchmark}/${dataset}/${config}.yaml
```

| Benchmarks |               | NB101-CF10 | NB101-CF10 | NB201-CF10 | NB201-CF10 | NB201-CF100 | NB201-CF100 | NB201-IMGNT | NB201-IMGNT | NB301-CF10 | NB301-CF10 |
|------------|---------------|------------|------------|------------|------------|-------------|-------------|-------------|-------------|------------|------------|
| Method     | Warm-up Level | Spearman   | Kendall    | Spearman   | Kendall    | Spearman    | Kendall     | Spearman    | Kendall     | Spearman   | Kendall    |
| ZiCo       | 0%            | 0.63       | 0.46       | 0.74       | 0.54       | 0.78        | 0.58        | 0.79        | 0.60        | 0.5        | 0.35       |
| ZiCo       | 10%           | 0.63       | 0.46       | 0.78       | 0.58       | 0.81        | 0.61        | 0.80        | 0.60        | 0.51       | 0.36       |
| ZiCo       | 20%           | 0.64       | 0.46       | 0.77       | 0.57       | 0.81        | 0.62        | 0.79        | 0.59        | 0.51       | 0.36       |
| ZiCo       | 40%           | 0.64       | 0.46       | 0.78       | 0.58       | 0.80        | 0.61        | 0.79        | 0.59        | 0.52       | 0.37       |
| ZiCo       | 60%           | 0.64       | 0.47       | 0.78       | 0.58       | 0.81        | 0.62        | 0.79        | 0.59        | 0.53       | 0.38       |
| ZiCo       | 100%          | 0.63       | 0.46       | 0.77       | 0.57       | 0.80        | 0.61        | 0.78        | 0.59        | 0.53       | 0.37       |
| SiGeo      | 0%            | 0.63       | 0.46       | 0.78       | 0.58       | 0.82        | 0.62        | 0.80        | 0.61        | 0.5        | 0.35       |
| SiGeo      | 10%           | 0.68       | 0.48       | 0.83       | 0.64       | 0.85        | 0.66        | 0.85        | 0.67        | 0.53       | 0.37       |
| SiGeo      | 20%           | 0.69       | 0.51       | 0.84       | 0.65       | 0.87        | 0.69        | 0.86        | 0.68        | 0.55       | 0.40       |
| SiGeo      | 40%           | 0.70       | 0.52       | 0.83       | 0.64       | 0.88        | 0.70        | 0.87        | 0.69        | 0.56       | 0.41       |
| SiGeo      | 60%           | 0.69       | 0.51       | 0.83       | 0.65       | 0.87        | 0.70        | 0.87        | 0.69        | 0.57       | 0.41       |
| SiGeo      | 100%          | 0.69       | 0.51       | 0.83       | 0.64       | 0.88        | 0.71        | 0.86        | 0.68        | 0.57       | 0.42       |
