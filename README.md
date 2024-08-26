
# Fashion MNIST Classification with CNN and ANN Models

This project focuses on classifying images from the Fashion MNIST dataset using both Artificial Neural Networks (ANN) and Convolutional Neural Networks (CNN). The goal is to explore the effectiveness of these models on image data, with a particular focus on preprocessing, model architecture, and training optimization.

## Table of Contents

- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Models Implemented](#models-implemented)
- [Data Preprocessing](#data-preprocessing)
- [Model Architectures](#model-architectures)
- [Training and Evaluation](#training-and-evaluation)
- [Results](#results)
- [Data Augmentation](#data-augmentation)
- [Conclusion](#conclusion)
- [References](#references)

## Project Overview

This project demonstrates the application of ANN and CNN models to the Fashion MNIST dataset. Fashion MNIST is a dataset consisting of 70,000 grayscale images in 10 categories, with 60,000 images in the training set and 10,000 images in the test set. The images are 28x28 pixels, and each image is associated with a label from one of the 10 categories.

## Dataset

The Fashion MNIST dataset is a well-known benchmark for machine learning algorithms. It contains images of various fashion items like T-shirts, trousers, and sneakers. Below is an example image from the dataset:

<picture>
 <source media="(prefers-color-scheme: dark)" srcset="https://github.com/KimiyaVahidMotlagh/Handwritten_Kmeans/blob/main/Pictures/DataDarkmode.jpg"></picture> <br/>

*Figure: Sample images from the Fashion MNIST dataset showing different categories.*

This dataset is included in the `tensorflow.keras.datasets` module, making it easy to load and use.

## Models Implemented

1. **ANN Model**:
   - A simple feedforward neural network with one hidden layer.
   - Activation functions: ReLU for the hidden layer and Softmax for the output layer.

2. **CNN Model**:
   - A more complex model designed to capture spatial features in the image data.
   - Consists of convolutional layers, max-pooling layers, dropout for regularization, and dense layers at the end for classification.

## Data Preprocessing

- **Normalization**: All pixel values in the images are scaled to a range of 0 to 1 to facilitate better convergence during training.
- **Reshaping**: The images are reshaped to include a single channel, making them compatible with the CNN model input.

## Model Architectures

### ANN Model

- **Input Layer**: Flattens the 28x28 image into a 784-element vector.
- **Hidden Layer**: Dense layer with 128 units and ReLU activation.
- **Output Layer**: Dense layer with 10 units and Softmax activation for classification.

### CNN Model

- **First Convolutional Layer**: 64 filters, 5x5 kernel size, ReLU activation, followed by max-pooling.
- **Second Convolutional Layer**: 32 filters, 3x3 kernel size, ReLU activation, followed by max-pooling.
- **Dropout Layer**: Dropout rate of 20% to prevent overfitting.
- **Dense Layers**: Two dense layers, the first with 256 units and ReLU activation, and the second with 10 units and Softmax activation for classification.

## Training and Evaluation

Both models were trained using the Adam optimizer with categorical cross-entropy loss. Training involved multiple epochs, with performance metrics such as accuracy and loss tracked for both training and validation datasets.

### ANN Model Training

- **Epochs**: 20
- **Batch Size**: 256
- **Validation Accuracy**: 88.67%

### CNN Model Training

- **Epochs**: 50
- **Batch Size**: 256
- **Validation Accuracy**: 91.75% after 10 epochs

## Results

The CNN model outperformed the ANN model, achieving higher accuracy on the test data. This result aligns with expectations, as CNNs are better suited for image data due to their ability to capture spatial hierarchies.

## Data Augmentation

To further improve the model’s performance, data augmentation was applied using `ImageDataGenerator`. This helped in generating more varied training samples by applying transformations like zooming and flipping. However, some models experienced overfitting, indicating a need for careful tuning of augmentation parameters.

## Conclusion

This project demonstrates the effectiveness of CNNs in image classification tasks compared to ANNs. While ANNs are simpler and faster to train, CNNs provide superior performance by leveraging spatial features. Data augmentation, when applied correctly, can further enhance model performance but requires careful tuning to avoid overfitting.

## References

- [Fashion MNIST Dataset](https://github.com/zalandoresearch/fashion-mnist)
- [TensorFlow Documentation](https://www.tensorflow.org/api_docs/python/tf/keras)
