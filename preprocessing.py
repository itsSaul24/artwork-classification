import os
from PIL import Image
import torch
from torchvision import transforms
import matplotlib.pyplot as plt
import pickle
import random
from tqdm import tqdm




def preprocess_images(dataset_directory):
    transformations = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.RandomRotation(10),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    preprocessed_data = []
    class_labels = {}  # Maps class names to numeric labels
    class_names = []   # List of class names for reference
    current_label = 0
    
    # Iterate over each class directory in the dataset directory
    for class_folder in tqdm(sorted(os.listdir(dataset_directory)), desc="Processing classes"):
        class_path = os.path.join(dataset_directory, class_folder)
        if os.path.isdir(class_path):
            if class_folder not in class_labels:
                class_labels[class_folder] = current_label
                class_names.append(class_folder)
                current_label += 1
                
            # Process each image file within the class directory
            for image_filename in tqdm([f for f in os.listdir(class_path) if f.lower().endswith('.jpg')], desc=f"Processing {class_folder}", leave=False):
                # Skip files that are not JPEG images
                if not image_filename.lower().endswith('.jpg'):
                    continue
                
                image_path = os.path.join(class_path, image_filename)
                image = Image.open(image_path)
                image = transformations(image)
                image = (image - image.min()) / (image.max() - image.min())  # Normalize to 0-1 range
                
                # Append the processed image and its class label
                preprocessed_data.append((image, class_labels[class_folder]))
                
    return preprocessed_data, class_labels, class_names

# Example of using the function:
# dataset_directory = "data"
# preprocessed_data, class_labels, class_names = preprocess_images(dataset_directory)



def preprocess_image_class(class_directory):
    """
    Processes images from a single class directory, assigning labels based on the directory name.
    
    Parameters:
    - class_directory: Path to the directory containing images of a single class.
    
    Returns:
    - A list of tuples, each containing a processed image tensor and its numeric label.
    - A dictionary mapping the class name to a numeric label.
    - A list containing a single class name.
    """
    transformations = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.RandomRotation(10),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    preprocessed_data = []
    class_name = os.path.basename(class_directory)  # Derive class name from folder name
    class_labels = {class_name: 0}  # Assign 0 as the label for simplicity in this context
    class_names = [class_name]
    
    for image_filename in tqdm([f for f in os.listdir(class_directory) if f.lower().endswith('.jpg')], desc=f"Processing {os.path.basename(class_directory)}"):
        # Only process JPEG images
        if not image_filename.lower().endswith('.jpg'):
            continue  # Skip non-JPEG files
        
        image_path = os.path.join(class_directory, image_filename)
        image = Image.open(image_path)
        image = transformations(image)
        image = (image - image.min()) / (image.max() - image.min())  # Normalize to 0-1 range
        
        preprocessed_data.append((image, 0))  # Append image with label 0
        
    return preprocessed_data, class_labels, class_names
# Example of using the function:
# class_directory = "/path/to/Abstract_Expressionism"
#processed_images, class_labels, class_names = preprocess_image_class(class_directory)


#IGNORE THIS ONE I DUNNO WHY IT NO WORK
# def show_image_from_dataset(preprocessed_data, class_labels, index):
#     img_tensor, label = preprocessed_data[index]
#     img = img_tensor.permute(1, 2, 0).numpy()
    
#     # Debug: Print available labels if the label is not found
#     label_textual = class_labels.get(label, f'Label {label} not found. Available labels: {class_labels}')
    
#     plt.imshow(img)
#     plt.title(f"Label: {label_textual}")
#     plt.show()


def show_image_from_dataset(preprocessed_data, class_labels, index):
    img_tensor, label = preprocessed_data[index]
    
    # Assuming img_tensor is a PyTorch tensor; convert to numpy for matplotlib
    img = img_tensor.permute(1, 2, 0).numpy()
    
    # If img_tensor is already a numpy array or similar, you might not need the conversion
    # img = img_tensor  # Use this line if no conversion is needed

    # Attempt to get the textual label from the class_labels mapping
    label_textual = class_labels.get(label, f'Label {label} not found. Available labels: {class_labels}')

    # Print the label information to the terminal
    print(f'Label (numeric): {label}')
    print(f'Label (textual): {label_textual}')

    # Show the image with matplotlib
    plt.imshow(img)
    plt.title(f"Label: {label_textual}")
    plt.show()

# Example of using the function:
# show_image_from_dataset(preprocessed_data, class_labels, 100)
 

def split_dataset(processed_images, train_ratio, test_ratio, val_ratio):
    """
    Splits the dataset into training, testing, and validation sets based on the given ratios.
    
    Parameters:
    - processed_images: A list of tuples, each containing an image and its label.
    - train_ratio: The proportion of the dataset to include in the train set.
    - test_ratio: The proportion of the dataset to include in the test set.
    - val_ratio: The proportion of the dataset to include in the validation set.
    
    Returns:
    - train_dataset: List of tuples for the training set.
    - test_dataset: List of tuples for the testing set.
    - val_dataset: List of tuples for the validation set.
    
    Note: train_ratio + test_ratio + val_ratio should ideally equal 1.
    """
    # Ensure the ratios sum up to 1
    assert 0.999 <= train_ratio + test_ratio + val_ratio <= 1.001, "The sum of ratios must be 1."
    
    # Shuffle the dataset to ensure random distribution
    random.shuffle(processed_images)
    
    # Calculate the number of samples for each set
    train_count = int(len(processed_images) * train_ratio)
    test_count = int(len(processed_images) * test_ratio)
    
    # Split the dataset
    train_dataset = processed_images[:train_count]
    test_dataset = processed_images[train_count:train_count + test_count]
    val_dataset = processed_images[train_count + test_count:]
    
    return train_dataset, test_dataset, val_dataset
# Example of using the function:
# processed_images = # Your dataset of processed images and labels
# train_dataset, test_dataset, val_dataset = split_dataset(processed_images, 0.7, 0.2, 0.1)


def save_data(data, file_name):
    with open(file_name, 'wb') as file:
        pickle.dump(data, file)

def save_labels(labels, file_name):
    with open(file_name, 'wb') as file:
        pickle.dump(labels, file)

def save_class_names(class_names, file_name):
    with open(file_name, 'wb') as file:
        pickle.dump(class_names, file)


def load_data(file_name):
    with open(file_name, 'rb') as file:
        data = pickle.load(file)
    return data



def main():
    """
    main() is split into 2 parts
    part before the lot of hashmarks is the stuff to process all the images in the folder
    takes around 10-20 minutes i forgot and didn't track time mb, but there is tqdn as a progress bar.
    also train test val splits it

    second part loads in the processed data that's saved as a pickle file and now u can do the things
    """


    # Define the relative or absolute path to your dataset
    dataset_directory = "/home/emeans/git/artwork-genre-classifier/data"
    
    # print("Preprocessing images...")
    # preprocessed_data, class_labels, class_names = preprocess_images(dataset_directory)

    # # If you want to process a single class directory, uncomment the following line and provide the path
    # class_directory = "/home/emeans/git/artwork-genre-classifier/data/Abstract_Expressionism"
    # processed_images, class_labels, class_names = preprocess_image_class(class_directory)

    # print("Splitting dataset into training, testing, and validation sets...")
    # train_dataset, test_dataset, val_dataset = split_dataset(preprocessed_data, 0.7, 0.2, 0.1)
    
    # # Optionally show an image from the train dataset
    # # Uncomment the line below to display the first image from the training dataset
    # show_image_from_dataset(train_dataset, {v: k for k, v in class_labels.items()}, 0)

    # # Save the preprocessed images and datasets
    # print("Saving processed data...")
    # save_data(preprocessed_data, 'processed_data.pkl')
    # save_data(train_dataset, 'train_dataset.pkl')
    # save_data(test_dataset, 'test_dataset.pkl')
    # save_data(val_dataset, 'val_dataset.pkl')

    # # Save the label mappings and class names
    # save_labels(class_labels, 'class_labels.pkl')
    # save_class_names(class_names, 'class_names.pkl')


    # ###################################################
    # #Load in data
    # preprocessed_data = load_data('processed_data.pkl')
    # class_labels = load_data('class_labels.pkl')
    # class_names = load_data('class_names.pkl')
    # train_dataset = load_data('train_dataset.pkl')
    # test_dataset = load_data('test_dataset.pkl')
    # val_dataset = load_data('val_dataset.pkl')

    # label_to_class_name = {v: k for k, v in class_labels.items()}
    # show_image_from_dataset(preprocessed_data, label_to_class_name, 0)




##RUN THING BELOW TO GET PREPROCESSING THING##
if __name__ == '__main__':
    tqdm.write("Starting preprocessing...")
    main()