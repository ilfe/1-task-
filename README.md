# 1-task-
# Convolutional Neural Network (CNN) Training for Image Classification

## Overview
This project demonstrates how to build and train a Convolutional Neural Network (CNN) to classify images into one of 10 classes. The model is implemented using TensorFlow/Keras and includes data preprocessing, data augmentation, and evaluation metrics visualization.

## Features
- **Data Preprocessing:** Normalizes image pixel values to the range [0, 1].
- **Data Augmentation:** Randomly rotates, shifts, and flips training images to improve model generalization.
- **Model Architecture:**
  - Three convolutional layers with ReLU activation.
  - MaxPooling after each convolutional layer.
  - Dropout regularization to reduce overfitting.
  - Dense layers for final classification.
- **Callbacks:** Early stopping and model checkpointing for efficient training.
- **Metrics and Visualizations:**
  - Accuracy and loss curves during training.
  - Classification report and confusion matrix.

## Requirements
- Python 3.x
- TensorFlow
- NumPy
- Matplotlib
- scikit-learn

Install the dependencies using:
```bash
pip install tensorflow numpy matplotlib scikit-learn
```

## Data
The training and testing datasets are stored in `.npz` files containing image data and labels:
- **train_small.npz**: Training dataset.
- **test_small.npz**: Testing dataset.

The data arrays are preprocessed to normalize pixel values to the range [0, 1]. Labels are assumed to be integers corresponding to class indices.

## Model Parameters
The model configuration is defined in the `params` dictionary:
```python
params = {
    "input_shape": (224, 224, 3),
    "num_classes": 10,
    "batch_size": 32,
    "epochs": 50,
    "dropout_conv": 0.25,
    "dropout_dense": 0.5,
    "learning_rate": 0.001
}
```

## Training
Run the following script to train the model:
```python
python cnn_training.py
```

Key training steps:
1. Data augmentation is applied using `ImageDataGenerator`.
2. The CNN model is defined and compiled with the Adam optimizer and categorical cross-entropy loss.
3. Training is performed using the augmented data.

### Callbacks
- **EarlyStopping**: Stops training when validation accuracy stops improving for 5 epochs.
- **ModelCheckpoint**: Saves the best model during training to a file (`best_model.keras`).

## Evaluation
After training, the model is evaluated on the test dataset:
- Accuracy is printed to the console.
- A classification report and confusion matrix are generated to assess performance per class.

## Visualization
Training and validation metrics are visualized:
- **Accuracy curves**: Show training and validation accuracy over epochs.
- **Loss curves**: Show training and validation loss over epochs.
- **Confusion matrix**: Visualizes prediction errors across classes.

### Example Results
```
Точность улучшенной модели CNN: 0.89
Отчет классификации:
              precision    recall  f1-score   support

           0       0.90      0.85      0.88       100
           1       0.88      0.91      0.89       100
           ...

   accuracy                           0.89      1000
  macro avg       0.89      0.89      0.89      1000
weighted avg       0.89      0.89      0.89      1000
```

## Functions
### Data Preprocessing
```python
def preprocess_data(data, labels):
    data = data.astype(np.float32) / 255.0
    return data, labels
```

### Model Definition
```python
def create_model(input_shape, num_classes, dropout_conv, dropout_dense):
    model = Sequential([
        Input(shape=input_shape),
        Conv2D(32, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Dropout(dropout_conv),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Dropout(dropout_conv),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Dropout(dropout_conv),
        Flatten(),
        Dense(256, activation='relu'),
        Dropout(dropout_dense),
        Dense(num_classes, activation='softmax')
    ])
    return model
```

### Plotting Training History
```python
def plot_history(history):
    plt.figure(figsize=(12, 6))
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.legend()
    plt.title("Training and Validation Metrics")
    plt.show()
```

## License
This project is open source and available under the MIT License.
