import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.layers import Input
import numpy as np


# Paths to train and test folders (dataset)
train_data = r"C:\Users\vicky\.cache\kagglehub\datasets\brsdincer\cell-images-parasitized-or-not\versions\1\cell_images\test"
test_data = r"C:\Users\vicky\.cache\kagglehub\datasets\brsdincer\cell-images-parasitized-or-not\versions\1\cell_images\train"

# Hyperparameters
# activation_functions = ['sigmoid', 'relu', 'tanh']
batch_num = 32
epochs_num = 18
learning_rate = 0.001

# Normalize data to the range [0, 1]
datagen = ImageDataGenerator(rescale=1.0/255)

# Converts all images to 130 x 130, batch size of 32, and into a binary classification (0 and 1s)
train_data = datagen.flow_from_directory(
    train_data, target_size=(130, 130), batch_size=batch_num, class_mode='binary'
)
test_data = datagen.flow_from_directory(
    test_data, target_size=(130, 130), batch_size=batch_num, class_mode='binary'
)

# Create a simple Deep Neural Network
model = Sequential([
    Input(shape=(130, 130, 3)),  # Define the input shape here
    Flatten(),
    Dense(256, activation='relu'),
    Dense (128, activation='relu'),
    Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer=SGD(learning_rate=learning_rate),
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(train_data, epochs=epochs_num, validation_data=test_data)

# Evaluate the model
test_loss, test_accuracy = model.evaluate(test_data)
print(f"Test accuracy: {test_accuracy:.4f}")

# Convert the data to arrays for the Perceptron model
def get_data_from_generator(generator):
    X, y = [], []
    for images, labels in generator:
        X.append(images.reshape(images.shape[0], -1))
        y.append(labels)
        if len(X) >= 1:
            break
    return np.vstack(X), np.hstack(y)

# Get a batch of data from the generator
X_train, y_train = get_data_from_generator(train_data)
X_test, y_test = get_data_from_generator(test_data)

# Perceptron Model for Binary Classification
class Perceptron:
    def __init__(self, input_size, lr=0.1):
        self.weights = np.zeros(input_size + 1)  # +1 for bias
        self.lr = lr
    
    def predict(self, x):
        summation = np.dot(x, self.weights[1:]) + self.weights[0]
        return 1 if summation >= 0 else 0
    
    def train(self, X, y, epochs=10):
        for _ in range(epochs):
            for i, x in enumerate(X):
                prediction = self.predict(x)
                # Update weights
                self.weights[1:] += self.lr * (y[i] - prediction) * x
                self.weights[0] += self.lr * (y[i] - prediction)

# Create and train the Perceptron
perceptron = Perceptron(input_size=X_train.shape[1])
perceptron.train(X_train, y_train, epochs=10)

# Test the Perceptron model
perceptron_predictions = [perceptron.predict(x) for x in X_test]
perceptron_accuracy = np.mean(perceptron_predictions == y_test)
print(f"Perceptron Test accuracy: {perceptron_accuracy:.4f}")