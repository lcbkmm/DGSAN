# DGSAN: Dual-Graph Spatiotemporal Attention Network for Pulmonary Nodule Malignancy Prediction

The overall framework of our proposed DGSAN:

![model](img/model.png)

Comparison of different SOTA methods on the NLST-cmst dataset and CLST dataset:

![renderings](img/table.png)

## Pre-requisties

- Linux

- Python>=3.7

- NVIDIA GPU + CUDA12.1 cuDNN8.9

## Installation

Clone this repository and navigate to it in your terminal. Then run:

```
pip install -r requirements.
```

## Data Preparation
The NLST-cmst dataset, derived from the National Lung Screening Trial (NLST, you can find it in [Link](https://cdas.cancer.gov/datasets/nlst/)) initiated by the National Cancer Institute, includes 433 subjects with Regions of Interest (ROI) of pulmonary nodules annotated by physicians. Each participant has at least two longitudinal CT scans and clinical data (age, sex, smoking status, screening results). Nodule malignancy is confirmed by pathological standards, and 3D ROIs are sized 16×64×64. The dataset is divided into 347 training and 86 testing cases in a 4:1 ratio.

The CLST dataset includes 109 patients with 317 CT sequences and 2,295 annotated nodules ([Link](https://www.nature.com/articles/s41597-024-03851-7)). Nodules were categorized as: malignant (invasive adenocarcinoma, microinvasive adenocarcinoma, in situ adenocarcinoma, and other malignant subtypes) and benign (inflammation and other benign subtypes). We selected 36 cases with two time points (72 data points total) and used the entire dataset for testing. 3D ROIs (16×64×64) were extracted based on nodule locations, and nodule diameters were recorded as clinical features.

## Training

- training for Encoder:

```
python train_Encoder.py
```

- training for DGSAN Model:

```
python train_model.py
```
