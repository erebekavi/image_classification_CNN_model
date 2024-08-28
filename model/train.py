import tensorflow as tf
import keras
from keras import layers, models
import os

#Preprocess Your Data
# Resize images
IMG_SIZE = (150, 150)   
BATCH_SIZE = 1000

# Load training data
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    'model/dataset/train',
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

# Load validation data
val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    'model/dataset/validation',
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

# Load test data
test_ds = tf.keras.preprocessing.image_dataset_from_directory(
    'model/dataset/test',
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)



# Normalize images
normalization_layer = keras.layers.Rescaling(1./255)

def preprocess_image(image, label):
    image = normalization_layer(image)
    return image, label

train_ds = train_ds.map(preprocess_image)
val_ds = val_ds.map(preprocess_image)
test_ds = test_ds.map(preprocess_image)

# Data augmentation


data_augmentation = keras.Sequential([
    keras.layers.RandomFlip("horizontal"),
    keras.layers.RandomRotation(0.2)
])

def augment_image(image, label):
    image = data_augmentation(image)
    return image, label

train_ds = train_ds.map(augment_image)

#model

def get_class_names_from_directory(directory):
    return sorted([d.name for d in os.scandir(directory) if d.is_dir()])


class_names = get_class_names_from_directory('model/dataset/train')
num_classes = len(class_names)

model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(512, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(num_classes, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# train
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=10
)

#Evaluate
test_loss, test_acc = model.evaluate(test_ds)
print(f"Test accuracy: {test_acc}")



#save and load model

# Save the model
model.save('model/CNN_model.h5')

