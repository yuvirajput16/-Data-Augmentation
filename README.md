# Data-Augmentation to tackle Overfitting

## Flower Classification Using CNN with Data Augmentation

This Jupyter Notebook demonstrates the process of training a Convolutional Neural Network (CNN) to classify images from the Flower dataset. The notebook highlights the issue of overfitting in machine learning models and shows how data augmentation can be used to mitigate this problem.

## Table of Contents
1. [Introduction](#introduction)
2. [Dataset](#dataset)
3. [Dependencies](#dependencies)
4. [Training the Model Without Data Augmentation](#training-the-model-without-data-augmentation)
5. [Applying Data Augmentation](#applying-data-augmentation)
6. [Training the Model With Data Augmentation](#training-the-model-with-data-augmentation)
7. [Results](#results)
8. [Conclusion](#conclusion)

## Introduction
In this project, we use the Flower dataset to train a CNN for image classification. Initially, the model overfits the data, which we address by incorporating data augmentation techniques. This notebook demonstrates the improvement in model performance and generalization due to data augmentation.

## Dataset
The Flower dataset consists of images of various flower species. Each image is labeled with the corresponding flower type, making it a suitable dataset for multi-class classification tasks.

## Dependencies
To run this notebook, you need to install the following dependencies:
- Python 3.x
- TensorFlow
- Keras
- NumPy
- Matplotlib
- scikit-learn
- opelnCV
- Pillow


## Training the Model Without Data Augmentation
In this section, we train the CNN model on the Flower dataset without using any data augmentation techniques. We observe that the model quickly overfits the training data, achieving high accuracy on the training set but poor performance on the test set.

## Applying Data Augmentation
Data augmentation is a technique to artificially increase the size of the training dataset by applying random transformations such as rotations, flips, and zooms to the images. This helps the model generalize better to unseen data.

## Training the Model With Data Augmentation
We re-train the CNN model using the augmented data. This section demonstrates the improvement in model performance and reduction in overfitting.

## Results
Displayed in the jupyter notebook.

## Conclusion
Data augmentation is an effective technique to tackle overfitting in image classification tasks. By applying data augmentation, we significantly improved the generalization of our CNN model on the Flower dataset.
