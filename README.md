# Handwritten Digit Recognition using CNN

This project implements a Convolutional Neural Network (CNN) to classify handwritten digits using the MNIST dataset. The system achieves high accuracy and can predict handwritten digits from both the dataset and user-uploaded images.

## Project Overview

- **Dataset**: MNIST (contains 60,000 training images and 10,000 test images of handwritten digits from 0 to 9)
- **Model**: Convolutional Neural Network (CNN)
- **Input Shape**: 28x28 grayscale images
- **Accuracy**: Achieved high accuracy on the test set after 10 epochs of training

### Features:
- Preprocessing of dataset images (normalization and reshaping)
- CNN model architecture with convolutional, pooling, and fully connected layers
- Training and evaluation of the model on the MNIST dataset
- User image upload functionality for handwritten digit prediction
- Visualization of training performance with accuracy and loss plots

## Installation

### Requirements
Ensure you have the following packages installed:
- `tensorflow`
- `numpy`
- `matplotlib`
- `pillow`

You can install these using:
```bash
pip install tensorflow numpy matplotlib pillow
```

### Steps to Run

1. **Clone the repository**:
   ```bash
   git clone https://github.com/ahmdmohamedd/MNIST-handwritten-digits-recognition
   cd MNIST-handwritten-digits-recognition
   ```

2. **Run the Jupyter Notebook**:  
   Open `handwritten_digit_recognition.ipynb` in Jupyter and execute all the cells to train the model and make predictions.

3. **Test with your own images**:
   You can upload a custom handwritten digit image, preprocess it, and the model will predict the digit.

## Usage

- **Training**: The model is trained on the MNIST dataset with 60,000 images for 10 epochs.
- **Prediction**: After training, you can upload a 28x28 grayscale image of a handwritten digit for prediction.

## Model Architecture

The CNN architecture includes:
- Convolutional layers with ReLU activation
- Max Pooling layers
- Fully connected Dense layers with Softmax output for classification

### Model Summary:
```
Layer (type)                 Output Shape              Param #
=================================================================
conv2d (Conv2D)              (None, 26, 26, 32)        320
max_pooling2d (MaxPooling2D) (None, 13, 13, 32)        0
conv2d_1 (Conv2D)            (None, 11, 11, 64)        18496
max_pooling2d_1 (MaxPooling2D)(None, 5, 5, 64)         0
conv2d_2 (Conv2D)            (None, 3, 3, 64)          36928
flatten (Flatten)            (None, 576)               0
dense (Dense)                (None, 64)                36928
dense_1 (Dense)              (None, 10)                650
=================================================================
Total params: 93,322
Trainable params: 93,322
Non-trainable params: 0
```

## Results

- The model achieves high accuracy on the test data.
- The user-uploaded image prediction also works well.

## Example Prediction

You can upload an image (28x28 pixels, grayscale) and predict the digit. Here’s an example:

```python
predicted_digit = predict_digit('path_to_image.png')
print(f'The predicted digit is: {predicted_digit}')
```

## Contributing

Feel free to fork this repository, open issues, and submit pull requests to improve the model or add more features.
