# MNIST Digit Classifier – NumPy from Scratch 🧠

This project implements a Convolutional Neural Network (CNN) **entirely from scratch using NumPy**, designed to classify handwritten digits from the MNIST dataset. It includes a complete training pipeline, testing script, and an interactive GUI where users can draw digits and get predictions from the trained model.

## Features
- CNN built without any deep learning libraries (no TensorFlow or PyTorch)
- Manual forward and backward propagation
- Training on the MNIST dataset using only NumPy
- Evaluation on test data
- Drawing interface to test predictions on user-generated digits
- Model saving/loading using `.npy` files

## Architecture
- Conv2D: 8 filters of size 3x3
- Activation: ReLU
- MaxPooling: 2x2
- Flatten
- Dense Layer: Fully connected layer with 10 output neurons (one per digit)
- Activation: Softmax for classification


## How to Use

Install dependencies using: `pip install numpy pandas pillow matplotlib`. Then, train the model using `python train.py`, which uses 1000 MNIST samples over 20 epochs and saves the trained weights to the `model/` directory. Once trained, you can evaluate model performance on the first 1000 MNIST test samples using `python test_set.py`, which prints the accuracy. To test your own digits, launch the drawing interface using `python draw_digit_gui.py`. This opens a 280x280 black canvas where you draw with white strokes; after clicking the button to save, the drawn digit is saved as `sample_input.npy` (auto-centered). Then, predict the digit using `python predict_from_drawing.py`, which loads the drawing, runs it through the model, prints the predicted digit, and displays the drawn image alongside the softmax scores.
