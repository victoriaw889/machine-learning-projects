# Task 1: Build a GAN for Image Generation
import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt

# Function to build the Generator model
def build_generator():
    model = tf.keras.Sequential()
    model.add(layers.Dense(128, activation='relu', input_dim=100))
    model.add(layers.Dense(28 * 28, activation='sigmoid'))
    model.add(layers.Reshape((28, 28)))
    return model

# Function to build the Discriminator model
def build_discriminator():
    model = tf.keras.Sequential()
    model.add(layers.Flatten(input_shape=(28, 28)))
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))
    return model

# Instantiate models
generator = build_generator()
discriminator = build_discriminator()

# Compile the discriminator
discriminator.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Combine generator and discriminator to create GAN
discriminator.trainable = False
gan = tf.keras.Sequential([generator, discriminator])
gan.compile(optimizer='adam', loss='binary_crossentropy')

# Training parameters
epochs = 1000
batch_size = 32
noise_dim = 100

# Function to generate random noise
def generate_noise(batch_size, noise_dim):
    return np.random.normal(0, 1, (batch_size, noise_dim))

# Placeholder for real data (replace with actual training data)
real_data = np.random.rand(batch_size, 28, 28)

# Function to visualize generated images
def plot_generated_images(generator, epoch, examples=10, dim=(1, 10), figsize=(10, 1)):
    noise = generate_noise(examples, noise_dim)
    generated_images = generator.predict(noise)
    generated_images = generated_images.reshape(examples, 28, 28)
    plt.figure(figsize=figsize)
    for i in range(examples):
        plt.subplot(dim[0], dim[1], i + 1)
        plt.imshow(generated_images[i], interpolation='nearest', cmap='gray')
        plt.axis('off')
    plt.tight_layout()
    plt.savefig(f'gan_generated_epoch_{epoch}.png')
    plt.show()

# Training loop
for epoch in range(epochs):
    # Generate fake images
    noise = generate_noise(batch_size, noise_dim)
    fake_images = generator.predict(noise)

    # Train discriminator on real and fake data
    d_loss_real = discriminator.train_on_batch(real_data, np.ones((batch_size, 1)))
    d_loss_fake = discriminator.train_on_batch(fake_images, np.zeros((batch_size, 1)))

    # Train generator via GAN to fool discriminator
    noise = generate_noise(batch_size, noise_dim)
    g_loss = gan.train_on_batch(noise, np.ones((batch_size, 1)))

    # Print progress and visualize every 100 epochs
    if epoch % 100 == 0:
        print(f'Epoch {epoch}, D Loss Real: {d_loss_real[0]:.4f}, D Loss Fake: {d_loss_fake[0]:.4f}, G Loss: {g_loss:.4f}')
        plot_generated_images(generator, epoch)


# Task 2: CNN for Image Classification
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.datasets import mnist

# Load and preprocess data
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(-1, 28, 28, 1) / 255.0
x_test = x_test.reshape(-1, 28, 28, 1) / 255.0

# Define the CNN model
model = Sequential([
    Conv2D(32, kernel_size=(3, 3), activation="relu", input_shape=(28, 28, 1)),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(128, activation="relu"),
    Dense(10, activation="softmax")
])

# Compile the model
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

# Train the model
model.fit(x_train, y_train, epochs=5, batch_size=32)


# Task 3 & 4: Ensemble Learning for Model Improvement
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_breast_cancer, load_wine
from sklearn.model_selection import train_test_split

# Breast cancer dataset example
data_bc = load_breast_cancer()
x_train_bc, x_test_bc, y_train_bc, y_test_bc = train_test_split(data_bc.data, data_bc.target, test_size=0.2)

# Build and train Random Forest classifier
model_bc = RandomForestClassifier(n_estimators=100)
model_bc.fit(x_train_bc, y_train_bc)

# Evaluate model
accuracy_bc = model_bc.score(x_test_bc, y_test_bc)
print(f"Breast Cancer Model Accuracy: {accuracy_bc:.4f}")

# Wine dataset example (Task 4)
data_wine = load_wine()
X_train, X_test, Y_train, Y_test = train_test_split(data_wine.data, data_wine.target, test_size=0.2, random_state=42)

model_wine = RandomForestClassifier(n_estimators=200)
model_wine.fit(X_train, Y_train)

accuracy_wine = model_wine.score(X_test, Y_test)
print(f"Wine Dataset Model Accuracy: {accuracy_wine:.2f}")


# Task 5: Transfer Learning for Faster Training
import os
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping

# Load pre-trained VGG16 model without top layers
base_model = VGG16(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False

# Build a model for binary classification
model_transfer = Sequential([
    base_model,
    Flatten(),
    Dense(128, activation="relu"),
    Dropout(0.5),
    Dense(1, activation="sigmoid")
])

# Compile the model
model_transfer.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

# Prepare image data with augmentation
train_datagen = ImageDataGenerator(
    rescale=1.0/255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode="nearest"
)

# Set directory for training data
train_dir = r"C:\Users\0651747\AppData\Local\Programs\Python\Python311\snakes and lizards"
assert os.path.exists(train_dir), "The training directory path does not exist."

# Create data generator
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode="binary"
)

print("Class indices:", train_generator.class_indices)

# Train the model with early stopping
early_stopping = EarlyStopping(monitor="loss", patience=3, restore_best_weights=True)
model_transfer.fit(train_generator, epochs=10, callbacks=[early_stopping])

# Save the trained model
model_transfer.save("transfer_learning_vgg16.h5")
