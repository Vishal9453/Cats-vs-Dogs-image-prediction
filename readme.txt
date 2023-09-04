
Dataset used for this project: "https://www.kaggle.com/datasets/kunalgupta2616/dog-vs-cat-images-data" 

# Cats vs Dogs Image Classification Project Documentation

**Table of Contents:**

1. **Introduction**
2. **Project Overview**
3. **Requirements**
4. **Code Files and Functionalities**
    - `main.py`: Data Preprocessing
    - `training.py`: CNN Model Building and Training
    - `tuning.py`: Hyperparameter Tuning
    - `predicting.py`: Model Testing
    - `Evaluation.py`: Model Evaluation
    - `app.py`: App Deployment using flask
    - `index.html`: Frontend
5. **Usage Instructions**

## 1. Introduction

This document provides a detailed overview of the project, its purpose, code functionalities, and usage instructions. The project aims to build and train a Convolutional Neural Network (CNN) model to classify images as either cats or dogs. The process involves data preprocessing, model building, hyperparameter tuning, and testing.

## 2. Project Overview

The project involves several steps, starting from preprocessing the dataset to testing the trained model on new images. The main focus is on training a CNN model to achieve accurate image classification.

## 3. Requirements

To run the project, you need:

- Python 3
- Libraries: numpy, OpenCV (cv2), Keras, pickle, os

## 4. Code Files and Functionalities

### `main.py`: Data Preprocessing

This file is responsible for preprocessing the dataset. It reads the images, resizes them, converts to grayscale, and stores the processed data in serialized files.

### `training.py`: CNN Model Building and Training

This file builds, compiles, and trains the CNN model using Keras. It uses the processed data from `main.py` for training and saves the trained model to a file.

### `tuning.py`: Hyperparameter Tuning

This file explores hyperparameter configurations for the CNN model. It iterates through different numbers of dense layers, convolutional layers, and neurons, trains models for each configuration, and saves the best model.

### `predicting.py`: Model Testing

This file loads the trained model and tests it on a set of test images. It predicts the classes of the images and prints the results.

### `Evaluation.py`: Model Evaluation

This file loads the testing data and make evaluations scores based on the moedel's perfromance.

### `app.py`: App Deployment using Flask

This file deploys the app on local system using flask and can be deployed on web after altering the adresses.

## 5. Usage Instructions

1. Place the dataset images in the appropriate folders as per the structure defined in the code files.

2. Run `main.py` to preprocess the dataset and create serialized files 'X.pkl' and 'y.pkl'.

3. Execute `training.py` to build, compile, and train the CNN model. The trained model will be saved as 'Cats_vs_Dogs.model'.

4. Optionally(recommended), run `tuning.py` to explore hyperparameter configurations and save the best model.

5. For model testing, run `testing.py`. This will load the trained model and predict classes for randomly selected test images.

6. Adjust the code files or experiment with hyperparameters to achieve the desired performance.

## Model Evalution scores:

Accuracy: 0.9123043478260869
Precision: 0.9322636521104932
Recall: 0.8892173913043478
F1 Score: 0.9102318750278161
Confusion Matrix:
[[10757   743]
 [ 1274 10226]]
## Classification Report:
              precision    recall  f1-score   support

           0       0.89      0.94      0.91     11500
           1       0.93      0.89      0.91     11500

    accuracy                           0.91     23000
   macro avg       0.91      0.91      0.91     23000
weighted avg       0.91      0.91      0.91     23000


## Conclusion

The Cats vs Dogs Image Classification project demonstrates the process of building and training a CNN model for image classification. It covers data preprocessing, model building, hyperparameter tuning, and testing, providing a comprehensive learning experience in the field of machine learning and computer vision.

For any further questions or assistance, feel free to reach out.

---

