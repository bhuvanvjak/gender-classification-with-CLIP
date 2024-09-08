# gender-classification-with-CLIP

This project implements a gender classification system using the CLIP model for feature extraction and a custom neural network for classification. The dataset consists of male and female images, and the model predicts the gender based on the provided image.

## Table of Contents
- [Overview](#overview)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Model Architecture](#model-architecture)
- [Results](#results)
- [Contributing](#contributing)


## Overview
This project leverages the OpenAI CLIP model to extract features from images and a custom TensorFlow neural network to classify gender as either male or female. The system is trained on a dataset of images categorized by gender and achieves a high level of accuracy.

## Dataset
The dataset used for this project is the **CCTV Gender Classifier Dataset**. It contains two folders, `FEMALE` and `MALE`, each holding `.bmp` images representing female and male individuals respectively.

**Note**: You need to download and place the dataset in your local directory. Update the `data_folder` variable in the code with the correct path to your dataset.

## Installation
To run this project, you will need to have the following libraries installed:

- `tensorflow`
- `torch`
- `transformers`
- `numpy`
- `Pillow`
- `scikit-learn`

You can install these dependencies using pip:

```bash
pip install tensorflow torch transformers numpy Pillow scikit-learn
```


## Usage
Download the dataset and place it in a folder (e.g., CCTV Gender Classifier Dataset).
Update the data_folder variable in the code with the path to your dataset.
Run the script to train the model
```bash
python gender_classification.py
```
After training, you can test the model on a new image by calling the predict_gender(image_path) function.

## Model Architecture
The model architecture consists of two main parts:

1. Feature Extraction with CLIP:

-The CLIP model (openai/clip-vit-base-patch32) is used to extract features from the images. 

-The extracted features are passed into a custom neural network for classification.

2. Custom Neural Network:

-Input Layer: Receives the extracted features from the CLIP model.

-Dense Layer 1: Fully connected layer with 128 units and ReLU activation.

-Dropout Layer 1: Dropout layer with a rate of 0.3 to prevent overfitting.

-Dense Layer 2: Fully connected layer with 64 units and ReLU activation.

-Dropout Layer 2: Dropout layer with a rate of 0.3 to further reduce overfitting.

-Output Layer: A dense layer with a single unit and a sigmoid activation function for binary classification (male or female).

-The model is compiled with the Adam optimizer and binary cross-entropy as the loss function, which is suitable for binary classification tasks. It also tracks accuracy as a performance metric.

-The training process includes early stopping to prevent overfitting, monitoring the validation loss and restoring the best weights after the training is completed.

## Results
The model achieves around 80% accuracy on the test set. Below are the sample results:

Test Loss: 1.384
Test Accuracy: 80.4%

## Contributing
Contributions are welcome! Please feel free to submit a pull request or open an issue to discuss your ideas for improving the project.

