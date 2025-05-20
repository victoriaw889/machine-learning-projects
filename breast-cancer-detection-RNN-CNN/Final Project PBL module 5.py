# Task 1: Build a GAN for Image Generation---------------------------------------------------------------------------------->
import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt

# Function to build the Generator model. Generator and Discriminator Models: Functions build_generator and build_discriminator define the architecture of the generator and discriminator, respectively, using fully connected (dense) layers.
def build_generator():
    model = tf.keras.Sequential()  # Create a sequential model
    model.add(layers.Dense(128, activation='relu', input_dim=100))  # Dense layer with 128 neurons and ReLU activation
    model.add(layers.Dense(28 * 28, activation='sigmoid'))  # Output layer with 28x28 neurons, output values between 0 and 1
    model.add(layers.Reshape((28, 28)))  # Reshape output to a 28x28 image
    return model  # Return the generator model

# Function to build the Discriminator model
def build_discriminator():
    model = tf.keras.Sequential()  # Create a sequential model
    model.add(layers.Flatten(input_shape=(28, 28)))  # Flatten the 28x28 image into a 1D array
    model.add(layers.Dense(128, activation='relu'))  # Dense layer with 128 neurons and ReLU activation
    model.add(layers.Dense(1, activation='sigmoid'))  # Output layer with 1 neuron, output is between 0 and 1 (real or fake)
    return model  # Return the discriminator model

# Define the models. Create and Compile the model: The discriminator is compiled first, specifying the optimizer, loss function, and metrics. 
generator = build_generator()  

# Instantiate the generator
discriminator = build_discriminator()  # Instantiate the discriminator

# Compile the discriminator
discriminator.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])  # Set optimizer, loss function, and metrics

# Combine the models to create the GAN
discriminator.trainable = False  # Freeze the discriminator during generator training
gan = tf.keras.Sequential([generator, discriminator])  # Create the GAN model by stacking generator and discriminator

# Compile the GAN model
gan.compile(optimizer='adam', loss='binary_crossentropy')  # Set optimizer and loss function for the GAN

# Training parameters.  Parameters such as the number of epochs, batch size, and noise dimension must defined to control the training process.
epochs = 1000  # Number of training epochs
batch_size = 32  # Number of samples per gradient update
noise_dim = 100  # Dimension of the noise vector input to the generator

# Function to generate random noise. Noise Generation: A function to create random noise vectors is provided. This noise serves as input to the generator to create fake images.
def generate_noise(batch_size, noise_dim):
    return np.random.normal(0, 1, (batch_size, noise_dim))  # Generate random noise from a normal distribution

# Placeholder for real data (replace this with actual training data)
real_data = np.random.rand(batch_size, 28, 28)  # Generate dummy real data for testing

# Function to visualize generated images. we visualize with The plot_generated_images function to generate images from the generator to create images from noise, reshaping them, and displaying them in a grid.
def plot_generated_images(generator, epoch, examples=10, dim=(1, 10), figsize=(10, 1)):
    noise = generate_noise(examples, noise_dim)  # Generate noise for the specified number of examples
    generated_images = generator.predict(noise)  # Use the generator to produce images from noise
    generated_images = generated_images.reshape(examples, 28, 28)  # Reshape the generated images to 28x28

    plt.figure(figsize=figsize)  # Set the figure size for the plot
    for i in range(examples):  # Loop through each generated image
        plt.subplot(dim[0], dim[1], i + 1)  # Create subplots
        plt.imshow(generated_images[i], interpolation='nearest', cmap='gray')  # Display the image in grayscale
        plt.axis('off')  # Hide axis ticks and labels
    plt.tight_layout()  # Adjust layout to prevent overlap
    plt.savefig(f'gan_generated_epoch_{epoch}.png')  # Save the generated image as a PNG file
    plt.show()  # Show the generated images

# Training loop. The main loop performs training over a specified number of epochs. The discriminator is trained on both real and fake images, while the generator is trained via the GAN model to improve its ability to produce convincing fake images. The loss values for both models are printed periodically, and generated images are visualized every 100 epochs.
for epoch in range(epochs):  # Loop through each epoch
    # Generate fake images
    noise = generate_noise(batch_size, noise_dim)  # Generate noise for a batch
    fake_images = generator.predict(noise)  # Generate fake images using the generator

    # Train the discriminator on real and fake data
    d_loss_real = discriminator.train_on_batch(real_data, np.ones((batch_size, 1)))  # Train on real images with label 1
    d_loss_fake = discriminator.train_on_batch(fake_images, np.zeros((batch_size, 1)))  # Train on fake images with label 0

    # Train the generator (via the GAN model, which tries to fool the discriminator)
    noise = generate_noise(batch_size, noise_dim)  # Generate new noise for the generator
    g_loss = gan.train_on_batch(noise, np.ones((batch_size, 1)))  # Train the generator to make the discriminator predict 1

    # Print progress and visualize images every 100 epochs
    if epoch % 100 == 0:  # Every 100 epochs
        print(f'Epoch {epoch}, D Loss Real: {d_loss_real[0]:.4f}, D Loss Fake: {d_loss_fake[0]:.4f}, G Loss: {g_loss[0]:.4f}')
        plot_generated_images(generator, epoch)  # Call the visualization function

# Task 2: CNN for Image Classification------------------------------------------------------------------>
# imports the necessary libraries
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.datasets import mnist

# defines training and testing data
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# reshapes the training and testing data to converge faster
x_train = x_train.reshape(-1, 28, 28, 1) / 255.0
x_test = x_test.reshape(-1, 28, 28, 1) / 255.0

# defines the model with 10 possible outcomes
model = Sequential([Conv2D(32, kernel_size = (3, 3), activation = "relu", input_shape = (28, 28, 1)),
                     MaxPooling2D(pool_size = (2, 2)), 
                     Flatten(), 
                     Dense(128, activation = "relu"), 
                     Dense(10, activation = "softmax")])

# compiles the model
model.compile(optimizer = "adam", loss = "sparse_categorical_crossentropy", metrics = ["accuracy"])

# trains the model
model.fit(x_train, y_train, epochs = 5, batch_size = 32)

# Ensemble
# imports libraries
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

# loads dataset
data = load_breast_cancer()

# splits data into training and testing
x_train, x_test, y_train, y_test = train_test_split(data.data, data.target, test_size = 0.2)

# builds the model
model = RandomForestClassifier(n_estimators = 100)

# trains the model
model.fit(x_train, y_train)

# evaluates the model
accuracy = model.score(x_test, y_test)

# displays the accuracy of the model
print(f"Model Accuracy: {accuracy}")

# Task 3: RNN for Time-Series Forecasting----------------------------------------------------------------->
# Ensemble
# imports libraries
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

# loads dataset
data = load_breast_cancer()

# splits data into training and testing
x_train, x_test, y_train, y_test = train_test_split(data.data, data.target, test_size = 0.2)

# builds the model
model = RandomForestClassifier(n_estimators = 100)

# trains the model
model.fit(x_train, y_train)

# evaluates the model
accuracy = model.score(x_test, y_test)

# displays the accuracy of the model
print(f"Model Accuracy: {accuracy}")

# Task 4: Ensemble Learning for Model Improvement ------------------------------------------------------->
from sklearn.ensemble import RandomForestClassifier  # Importing the Random Forest Classifier
from sklearn.datasets import load_wine  # Importing the Wine dataset
from sklearn.model_selection import train_test_split  # Function to split the dataset into training and testing sets

# Load the Wine dataset
data = load_wine()

# Split the dataset into training and testing sets (80% training, 20% testing)
X_train, X_test, Y_train, Y_test = train_test_split(data.data, data.target, test_size=0.2, random_state=42)

# Build and train the Random Forest classifier
model = RandomForestClassifier(n_estimators=200)  # Creating a Random Forest classifier with 200 decision trees
model.fit(X_train, Y_train)  # Training the Random Forest classifier on the training data

# Evaluate the model
accuracy = model.score(X_test, Y_test)  # Evaluating the model's accuracy on the testing data
print(f"Model Accuracy: {accuracy:.2f}")  # Printing the model's accuracy

# Task 5: Transfer Learning for Faster Training------------------------------------------------------------------------>
# Transfer
# import libraries
import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping
import os

# load the pre-trained model w/o top layers
base_model = VGG16(weights = "imagenet", include_top = False, input_shape = (224, 224, 3))
base_model.trainable = False

# builds a model for binary classification (two classes)
model = Sequential([
    base_model,
    Flatten(),
    Dense(128, activation = "relu"),
    Dropout(0.5),
    Dense(1, activation = "sigmoid")])

# compiles the model
model.compile(optimizer = "adam", loss = "binary_crossentropy", metrics = ["accuracy"])

# prepares image data
train_datagen = ImageDataGenerator(
    rescale = 1.0 / 255,
    rotation_range = 20,
    width_shift_range = 0.2,
    height_shift_range = 0.2,
    shear_range = 0.2,
    zoom_range = 0.2,
    horizontal_flip = True,
    fill_mode = "nearest"
)

# finds the data's location and loads it
train_dir = r"C:\Users\0651747\AppData\Local\Programs\Python\Python311\snakes and lizards"

# confirm the path exists
assert os.path.exists(train_dir), "The training directory path."

# creates the generator to train data
train_generator = train_datagen.flow_from_directory(train_dir, target_size = (224, 224), batch_size = 32, class_mode = "binary")

# checks class indices
print("Class indices:", train_generator.class_indices)

# trains the model with early stopping
early_stopping = EarlyStopping(monitor = "loss", patience = 3, restore_best_weights = True)
model.fit(train_generator, epochs = 10, callbacks = [early_stopping])

# saves the model after training it
model.save("transfer_learning_vgg16.h5")


