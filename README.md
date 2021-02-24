# Feature_relevance_optimization
This repo implement a optimization method that use the similarity distance and feature ranking to remove irrelevant generic image features from the pre-trained models with large dissimilarities to the medical imaging dataset.

## Requirements
- [Scikit-learn](http://scikit-learn.org/stable/)
- [Pytorch](https://pytorch.org/) (Recommended version 9.2)
- [Python 3](https://www.python.org/)
- [tqdm](https://github.com/tqdm/tqdm)

## Dataset
We download dataset from following link: 
[imageclef](https://www.imageclef.org/2016)
[isic](https://www.isic-archive.com/#!/topWithHeader/wideContentTop/main)
[TNSCUI](https://tn-scui2020.grand-challenge.org/Dataset/)

## Quick start
Please easily run the RunAll.py file or use example in recipes to have a quick start.

## Result

![image](https://github.com/FredericChai/Feature_relevance_optimization/blob/main/src/1.png)
Comparative analysis of feature maps from the first layer after ReLU activation; Left image represents sample feature and filters which are removed in our method. Right images refer to features and same filters after recover; Middle images was the filter in the middle of the similarity rank as a standard for comparison

![image](https://github.com/FredericChai/Feature_relevance_optimization/blob/main/src/2.png)
The results of classification and segmentation accuracy using different sparse ratio during SLFS based on (i) ImageClef2016 – left image (ii) TNSCUI2020 – middle image (iii) ISIC2016 – right image

![image](https://github.com/FredericChai/Feature_relevance_optimization/blob/main/src/4.png)
Compare impact from feature representations during training; Red curve - directly finetuning; Green curve – remove redundant filters and finetune; Yellow curve – remove redundant filters and re-activate the compressed filters then finetune;

# Result on Imageclef2016 measured by accuracy based on three models
|     Trainining method  | Vgg-16 | ResNet-50 | DenseNet-121 |
|:--------------:|:----------------:|:----------------:|:----------------:|
77.3
85.2
86.1
86.3
87.4

| Train from scratch|  79.3    | 79.5   | 77.3    |
| Finetuning |  84.5    | 85.2 | 85.2    |
| Ensemble |  85.1    | 85.7   | 86.1    |
| Dense-sparse-Dense |  85.9    | 85.9    | 86.3  |
| Ours |  86.5    | 87.5  |87.4  |
