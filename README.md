# COVID-19 Prediction using VGG-16, VGG-19, and AlexNet

This repository contains code for predicting COVID-19 cases using the COVID-19 dataset and three different deep learning models: VGG-16, VGG-19, and AlexNet. The models are trained on a dataset of 224 samples for training and 60 samples for testing.

## Problem Description

The task is to predict COVID-19 cases using deep learning models. The COVID-19 dataset is used for training and evaluation. The goal is to train models that can accurately classify COVID-19 cases based on input images. Three different models, namely VGG-16, VGG-19, and AlexNet, are implemented and compared in terms of their performance.

## Dataset

The COVID-19 dataset consists of medical images related to COVID-19 cases. The dataset includes a total of 284 images, with 224 images for training and 60 images for testing. Each image in the dataset is labeled as COVID-19 positive or negative. The dataset is divided into training and testing sets to train the models and evaluate their performance.

## Model Architectures

Three different deep learning models are implemented for COVID-19 prediction:
- VGG-16: A convolutional neural network (CNN) architecture with 16 layers, including convolutional, pooling, and fully connected layers.
- VGG-19: Similar to VGG-16, but with 19 layers, providing a deeper network for improved feature extraction.
- AlexNet: Another CNN architecture with 5 convolutional layers, local response normalization, and 3 fully connected layers.

## Implementation

The code is implemented using Python and deep learning libraries such as TensorFlow and Keras. The models are built using the Keras API and trained on the COVID-19 dataset. The models are trained from scratch, without using pretrained weights. The training process involves feeding the images to the models and optimizing their parameters using appropriate loss functions and optimization techniques.


## Evaluation and Results

Evaluate the performance of the models using metrics such as accuracy, precision, recall, and F1-score. Provide a comparison of the models' performance and discuss their strengths and weaknesses. Visualize the results, such as confusion matrices or ROC curves, to better understand the models' predictions. Discuss any challenges faced during training, potential limitations, and suggestions for improvement.

