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
    - train.py                    ; code of training our text classifier with BERT on the ground truth dataset
    - test.py                     ; code of testing our text classifier on the test dataset
- models/
    - results                     ; the folder for saving model used in train.py
    - best                        ; the best performance model used in test.py
- dataset/
    - Ground_Truth_Dataset.csv    ; the ground truth dataset used for training
    - Test_Dataset.csv            ; the test dataset used for testing
- experiment/
    - best-model.png              ; the result of our classifier on the test dataset
    
- setup/                          ; enviroment setup files
- README.md                       ; instructions of this artifact
```

## Environment Setup
> This implementation has been successfully tested on Windows 11 using Python 3.7.16 and TensorFlow 2.9.3. For optimal performance, we recommend running this artifact with GPU acceleration.

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
   This command instructs pip to install all packages listed in the requirements.txt file.

## Experiment Results (Avaiable on Zenodo)

We used [code/test.py](https://github.com/demo123attack/DatasetCode/blob/main/train.py) with the [models/best](https://github.com/demo123attack/DatasetCode/blob/main/dataset.zip) to evaluate the effectiveness of our classifier on the *test* dataset. 
Our text classifier achieved an accuracy of 99.92%, a precision of 94.59%, and a recall of 99.06% as demonstrated in:
```
experiment/best-model.png
```

## Important Notes for AEC Review

**To facilitate AEC review, we have summarized the materials and instructions related to the three badges as follows:**

### 1. Artifacts Available

This artifact will be uploaded to Zenodo and made public on GitHub.

### 2. Artifacts Evaluated

To train our classifier, please run `code/train.py`, which which utilizes 5-fold cross-validation on the *Ground truth* dataset and saves our models in the `models/results/` folder. 
```
python train.py
```

To evaluate the effectiveness of our classifier, please run `code/test.py`, the results are displayed in `experiment/best-model.png`.
```
python test.py
```

### 3. Results Reproduced

Please see the above section in this README ("**Experiment Results**").
