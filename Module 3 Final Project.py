import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt

# Load in the dataset

dataset, metadata = tfds.load('stanford_dogs', as_supervised=True, with_info=True)
train_dataset, test_dataset = dataset['train'], dataset['test']

# Get the name function for labels

get_name = metadata.features['label'].int2str

# Display the first image and label from the training dataset

for image, label in train_dataset.take(1):
    plt.figure()
    plt.title(get_name(label))  # Display the label as a title
    plt.imshow(image)  # Display the image
    plt.show()


# Preprocessing: normalize images by dividing by 255 (pixel values between 0 and 1)

def normalize_img(image, label):
    image = tf.cast(image, tf.float32) / 255.0
    return image, label

train_dataset = train_dataset.map(normalize_img)
test_dataset = test_dataset.map(normalize_img)


# Batch and shuffle the data

train_dataset = train_dataset.shuffle(1024).batch(32)
test_dataset = test_dataset.batch(32)

# Building the neural network model (based on 224x224 dog images, not 28x28)

model = tf.keras.models.Sequential([

    # Resize images to a uniform size (Stanford Dogs dataset uses various sizes)
    tf.keras.layers.Resizing(224, 224),
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(metadata.features['label'].num_classes, activation='softmax')

])

# Compile the model

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

print("Model built successfully.")

# Train the model

history = model.fit(train_dataset, epochs=5, validation_data=test_dataset)
 
# Evaluate the model on test data

test_loss, test_accuracy = model.evaluate(test_dataset)
print(f"Test accuracy: {test_accuracy:.4f}")