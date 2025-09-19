# MNIST Digit Classification with PyTorch

## Project Overview
This project demonstrates building, training, and evaluating a neural network to classify handwritten digits from the MNIST dataset. The goal is to create a model that can accurately recognize digits (0–9) in images, and explore the impact of different architectures and hyperparameters on performance.

The MNIST dataset is a benchmark dataset in computer vision, containing 60,000 training images and 10,000 test images of handwritten digits, each of size 28x28 pixels.

---

## Project Goals
- Load and preprocess the MNIST dataset.
- Build a neural network (MLP and optionally CNN) using PyTorch.
- Train the network and monitor training and validation loss.
- Evaluate model performance on the test set.
- Experiment with different hyperparameters to improve accuracy.
- Save the trained model for future use.

---

## Dataset
- Dataset: [MNIST](http://yann.lecun.com/exdb/mnist/)
- Number of classes: 10 (digits 0–9)
- Training samples: 60,000
- Test samples: 10,000
- Image size: 28x28 pixels, grayscale

---

## Methodology

### 1. Data Preprocessing
- The images are converted to PyTorch tensors using `transforms.ToTensor()`.
- No normalization is applied for visualization purposes, but normalization can be added for better training performance.
- Data is loaded using `torch.utils.data.DataLoader` with batching and shuffling.

### 2. Model Architecture
- **MLP (Multi-Layer Perceptron)**:
  - Flatten input 28x28 images into a vector of 784 elements.
  - Two hidden layers with ReLU activations and optional Dropout for regularization.
  - Output layer with 10 neurons (one for each digit) using logits.
- **Optional CNN (for higher accuracy)**:
  - Convolutional layers to extract spatial features.
  - Max-pooling layers for downsampling.
  - Fully connected layers for classification.

### 3. Loss Function and Optimizer
- Loss Function: `CrossEntropyLoss` (suitable for multi-class classification)
- Optimizer: `Adam` (adaptive learning rate for efficient training)

### 4. Training and Validation
- Model trained for multiple epochs (default: 10–15)
- Training loss recorded per epoch.
- Validation loss and accuracy computed using the test set.
- GPU acceleration used if available via `torch.device` and `.to(device)`.

### 5. Hyperparameter Tuning
- Batch size, learning rate, number of hidden units, and dropout rate were tuned to improve accuracy.
- Multiple runs tested to achieve optimal performance.

### 6. Model Saving
- Trained model is saved using `torch.save`:
  - Entire model: `"mnist_model_full.pth"`
  - State dictionary: `"mnist_model_state.pth"`

---

## Results
- Training and validation loss recorded for each epoch.
- Test accuracy calculated on unseen data.
- Achieved accuracy depends on chosen architecture and hyperparameters:
  - Simple MLP: ~95–97% accuracy
  - CNN (optional): >98% accuracy

---
