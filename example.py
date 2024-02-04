import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

def load_and_preprocess_data(path, img_size, batch_size):
    """
    Load and preprocess the dataset.
    
    Parameters:
    - path: Path to the dataset directory.
    - img_size: Tuple of (height, width) for image resizing.
    - batch_size: Number of samples per batch.
    
    Returns:
    - train_generator: Training data generator.
    - validation_generator: Validation data generator.
    """
    datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=0.2,  # using 20% of the data for validation
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    train_generator = datagen.flow_from_directory(
        path,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='training'  # Set as training data
    )

    validation_generator = datagen.flow_from_directory(
        path,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation'  # Set as validation data
    )

    return train_generator, validation_generator


def create_model(input_shape, num_classes):
    """
    Create and compile a CNN model.
    
    Parameters:
    - input_shape: Shape of the input images (height, width, channels).
    - num_classes: Number of classes in the dataset.
    
    Returns:
    - model: Compiled CNN model.
    """
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D(2, 2),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Flatten(),
        Dense(512, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    
    return model


def train_and_evaluate(model, train_generator, validation_generator, epochs):
    """
    Train and evaluate the CNN model.
    
    Parameters:
    - model: Compiled CNN model.
    - train_generator: Training data generator.
    - validation_generator: Validation data generator.
    - epochs: Number of epochs to train the model.
    
    Returns:
    - history: Training history object.
    """
    early_stopping = EarlyStopping(monitor='val_loss', patience=10)

    history = model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // train_generator.batch_size,
        epochs=epochs,
        validation_data=validation_generator,
        validation_steps=validation_generator.samples // validation_generator.batch_size,
        callbacks=[early_stopping]
    )

    return history

def main():
    # Define dataset parameters
    dataset_path = 'path/to/your/dataset'
    img_size = (150, 150)  # Example image size
    batch_size = 32
    num_classes = 10  # Example number of classes
    epochs = 50  # Number of epochs to train

    # Load and preprocess the dataset
    print("Loading and preprocessing data...")
    train_generator, validation_generator = load_and_preprocess_data(dataset_path, img_size, batch_size)

    # Create and compile the CNN model
    print("Creating and compiling the model...")
    model = create_model(input_shape=img_size + (3,), num_classes=num_classes)

    # Train and evaluate the model
    print("Training and evaluating the model...")
    history = train_and_evaluate(model, train_generator, validation_generator, epochs)

    # Optionally, save the trained model and print out the evaluation results
    model.save('artwork_classification_model.h5')
    print("Model training complete and saved.")

if __name__ == '__main__':
    main()
