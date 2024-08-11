# Understanding Cross-Platform Referral Traffic for Illicit Drug Promotion
> Our paper has been accepted by [CCS'24](https://www.sigsac.org/ccs/CCS2024/)

<a href="https://opensource.org/license/mit"><img src="https://img.shields.io/badge/license-MIT-yello" alt="The MIT Lucense"/></a>
<a href="https://opensource.org/license/mit"><img src="https://img.shields.io/badge/CCS'24-paper-yello" alt="The MIT Lucense"/></a>
<a href="https://opensource.org/license/mit"><img src="https://img.shields.io/badge/dataset-released-blue" alt="The MIT Lucense"/></a>

> **Notice for AEC**
```diff
- Some files in this GitHub repository are incomplete. Due to restrictions on large files, for example, the size of the best performance model (`model/best`) exceeds 1GB.;
- Please download the complete files from Zenodo
```

## Introduction

This artifact presents the implementation and experimental results from our paper proposed at CCS'24. Our approach focuses on identifying drug-referral comments using a text classifier based on BERT. Initially, we preprocessed the comment text and gathered a sample of 18,000 comments across 6,638 unique videos to train our classifier with BERT. This dataset, known as the ground truth dataset, comprised 9,000 drug-referral comments (Badset) and 9,000 benign comments (Goodset).

To assess the classifier's efficacy, we created a separate test dataset by randomly selecting 9,000 video comments not included in the ground truth dataset. Our text classifier achieved outstanding results on this test dataset, with an accuracy of 99.92%, a precision of 94.59%, and a recall of 99.06%.

## Code Structure

The following is a brief introduction to the directory structure of this artifact:

```
- code/
    - train_test.py                   ; the code for training our text classifier with BERT on the ground truth dataset, as well as the code for testing the classifier on the test dataset
- models/
    - results                         ; the folder for saving model used in train_test.py
    - best                            ; the best performance model used in train_test.py
    - model.png                       ; the figure of our model's architecture
- dataset/
    - Ground_Truth_Dataset.csv        ; the ground truth dataset used for training
    - Test_Dataset.csv                ; the test dataset used for testing
- experiment/
    - best-model.png                  ; the result of our classifier on the test dataset
- papers/
    - artifact.pdf                    ; the artifact appendix
    - CCS24_camera_ready.pdf          ; our camera ready version paper
- setup/
    - requirements.txt                ; enviroment setup file
    - requirements_windows_gpu.txt    ; enviroment setup file for windows gpu
- README.md                           ; instructions of this artifact
```
## Model Architecture 

The architecture of our model is illustrated in the following picture:

![image](https://github.com/demo123attack/Artifact-CCS24/blob/main/models/model.png)

## Environment Setup
> This implementation has been successfully tested on Windows 11 with Python 3.7.16 and TensorFlow 2.9.3, utilizing an NVIDIA GeForce RTX 3070 Ti Laptop GPU 8GB, a 12th Gen Intel(R) Core(TM) i7-12700H processor, and 16GB of RAM. For optimal performance, we recommend using GPU acceleration (e.g., NVIDIA GeForce RTX 3070 Ti Laptop GPU 8GB). If you choose to use GPU acceleration on Windows, please install the necessary components by following the `setup/requirements_windows_gpu.txt` file.

To ensure the proper functioning of this artifact, please follow the commands belows:
1. Please ensure that `conda` is installed on your system. If `conda` is not already installed, we recommend installing it as part of the Anaconda distribution or Miniconda.
2. Open a terminal or command prompt.
3. Create a new conda environment with a name of your choice (e.g., `bert`) and specify the desired Python version for configuration.
   ```
   conda create -n bert python=3.7.16
   ```
4. Once the environment is created, activate it by running:
   ```
   conda activate bert
   ```
   This command switches your command line environment to use the newly created conda environment, including all necessary packages.
5. Run the following command to install all the required packages:
   ```
   pip install -r setup/requirements.txt
   ```
   This command instructs pip to install all packages listed in the setup/requirements.txt file.
> If you choose to use GPU acceleration on Windows, please install the necessary components by running `pip install -r setup/requirements_windows_gpu.txt`.

## Experiment Results (Avaiable on Zenodo)

We used [code/train_test.py](https://github.com/demo123attack/Artifact-CCS24/blob/main/code/train_test.py) with the [models/best](https://github.com/demo123attack/Artifact-CCS24/blob/main/models/best/README.md) to evaluate the effectiveness of our classifier on the *test* dataset. **Some files in this GitHub repository are incomplete. Due to constraints on file size, such as the best performing model (models/best) exceeding 1GB, please download the complete files from Zenodo.**

Our text classifier achieved an accuracy of 99.92%, a precision of 94.59%, and a recall of 99.06% as demonstrated in:

![image](https://github.com/demo123attack/Artifact-CCS24/blob/main/experiment/best-model.png)


## Important Notes for AEC Review

**To facilitate AEC review, we have summarized the materials and instructions related to the three badges as follows:**

### 1. Artifacts Available

This artifact will be uploaded to Zenodo and made public on GitHub.

### 2. Artifacts Evaluated

To train our classifier, please run `code/train_test.py`, which which utilizes 5-fold cross-validation on the *Ground truth* dataset and saves our models in the `models/results/` folder. 
```
python train_test.py --train
```

To evaluate the effectiveness of our classifier, please run `code/train_test.py`, the results are displayed in `experiment/best-model.png`.
```
python train_test.py --test
```

### 3. Results Reproduced

Please see the above section in this README ("**Experiment Results**").
