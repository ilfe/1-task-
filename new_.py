import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

params = {
    "input_shape": (224, 224, 3),
    "num_classes": 10,
    "batch_size": 32,
    "epochs": 50, # 30-40 эпох вполне достаточно 
    "dropout_conv": 0.25,
    "dropout_dense": 0.5,
    "learning_rate": 0.001
}

train_data = np.load('/content/drive/MyDrive/train_small.npz')
test_data = np.load('/content/drive/MyDrive/test_small.npz')

def preprocess_data(data, labels):
    data = data.astype(np.float32) / 255.0
    return data, labels

X_train, y_train = preprocess_data(train_data['data'], train_data['labels'])
X_test, y_test = preprocess_data(test_data['data'], test_data['labels'])

datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)
datagen.fit(X_train)

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

model = create_model(params["input_shape"], params["num_classes"], params["dropout_conv"], params["dropout_dense"])
optimizer = tf.keras.optimizers.Adam(learning_rate=params["learning_rate"])
model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

callbacks = [
    tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
    tf.keras.callbacks.ModelCheckpoint("best_model.keras", save_best_only=True)
]

history = model.fit(
    datagen.flow(X_train, y_train, batch_size=params["batch_size"]),
    epochs=params["epochs"],
    validation_data=(X_test, y_test),
    callbacks=callbacks
)

loss, accuracy = model.evaluate(X_test, y_test)
print(f"Точность улучшенной модели CNN: {accuracy:.2f}")

y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
print("Отчет классификации:")
print(classification_report(y_test, y_pred_classes, zero_division=0))
ConfusionMatrixDisplay.from_predictions(y_test, y_pred_classes, cmap="Blues")
plt.title("Confusion Matrix")
plt.show()

def plot_history(history):
    plt.figure(figsize=(12, 6))
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.legend()
    plt.title("Training and Validation Metrics")
    plt.show()

plot_history(history)
