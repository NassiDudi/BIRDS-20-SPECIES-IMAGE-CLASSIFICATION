# BIRDS-20-SPECIES-IMAGE-CLASSIFICATION

This project involves the need for an automated and scalable method to classify bird species from images. Manual identification is time-consuming and requires expert knowledge. An automated solution can streamline this process, making it more accessible and efficient for various stakeholders such as researchers, conservationists, and hobbyists.


## Table of Contents
1. [Project Overview](#project-overview)  
2. [Technologies Used](#technologies-used)  
3. [Dataset](#dataset)  
4. [Data Augmentation](#data-augmentation)  
5. [How to Run](#how-to-run)  
6. [Code Breakdown](#code-breakdown)  
7. [Visualization](#visualization)  
8. [Results](#results)

## Project Overview
This repository contains a project for classifying images of 20 bird species using deep learning. The project leverages a custom CNN architecture and fine-tunes pre-trained models such as ResNet50, DenseNet121, and VGG16 for accurate bird species classification.

## Technologies Used
- Python 3.x
- PyTorch for deep learning
- Torchvision for pre-trained models and dataset utilities
- Matplotlib for data visualization
- NumPy for numerical computations
- PIL for image processing

## Dataset
The dataset is structured into folders for each bird species:
```
/content/train/
    ├── species_1/
    ├── species_2/
    └── ...
/content/val/
/content/test/
```
- **Train Set:** Contains 3208 labeled images for training.
- **Validation Set:** Contains 100 labeled images of each specie. Used to tune model hyperparameters and prevent overfitting.
- **Test Set:** Contains 100 labeled images of each specie. Used for final evaluation.
  
## Data Augmentation
Data augmentation is crucial in ensuring the model performs well in real-world scenarios, where bird images may vary in orientation, lighting, and appearance. For instance, birds might be seen from different angles, in varying lighting conditions, or even in different environments. Augmentation helps the model recognize birds regardless of their orientation or the lighting they are captured in. It also prepares the model to handle real-life variations, such as changes in the bird's color due to lighting or camera settings. By simulating these real-world conditions, the model becomes more robust and accurate, even with a limited number of training images(3208).
i used to following techniques:

| Augmentation Technique     | Key Role in Training                                     |
|----------------------------|----------------------------------------------------------|
| **RandomGrayscale**         | Encourages focus on texture and shape, not color.        |
| **Resize**                  | Ensures uniformity and compatibility with pre-trained models. |
| **RandomHorizontalFlip**    | Handles variations in bird orientations.                |
| **RandomRotation**          | Increases robustness to different angles.                |
| **ColorJitter**             | Trains the model for varied lighting and color conditions. |
| **RandomVerticalFlip**      | Adds uncommon variations for improved generalization.    |


## How to Run
There are two ways you can run this project:

1. Using Google Colab (Recommended for easy setup)
   Download the notebook:

   Visit the GitHub repository and download the BIRDS 20 SPECIES- IMAGE CLASSIFICATION.ipynb file.
   Upload the notebook to Colab:

   Open Google Colab.
   Click on "File" > "Upload notebook" and select the downloaded .ipynb file.

2. Clone this repository:
   ```bash
   git clone https://github.com/NassiDudi/BIRDS-20-SPECIES-IMAGE-CLASSIFICATION.git
   ```
   - Install the required packages
   - Use any IDE that supports Jupyter Notebooks


## Code Breakdown
### 1. Data Augmentation
Before training the model, the dataset is augmented and divided to dataloades. The following augmentation techniques are applied to the training images:
- Random Grayscale: Converts images to grayscale with a 50% probability.
- Resize: Resizes all images to 224x224 pixels to ensure consistency.
- Random Horizontal Flip: Randomly flips images horizontally.
- Random Rotation: Rotates images randomly within a ±30 degree range.
- Color Jitter: Randomly adjusts the brightness, contrast, and saturation of images.
- Random Vertical Flip: Randomly flips images vertically.

### 2. **Custom CNN Architecture**
Defines a CNN with three convolutional layers followed by fully connected layers for classification.

### 3. **Pre-Trained Models**
Fine-tunes the following pre-trained models:
- **ResNet50**
- **DenseNet121**
- **VGG16**

The final layers are modified to output probabilities for 20 classes.

### 4. **Training and Validation**
The models are trained with:
- **CrossEntropyLoss**: For multi-class classification.
- **Adam Optimizer**: For efficient learning.

### 5. **Evaluation**
Each model is evaluated on the test set using accuracy and a classification report.


## Visualization
### Visual Outputs:
1. **Augmented vs. Real Images**: A comparison between augmented images and the original images, demonstrating the effect of data augmentation techniques.
2. **Class Distribution**: A bar plot showing the number of samples per bird species in the training set.
3. **Training Metrics**: Loss and accuracy curves over epochs on the validation set.
4. **Classification report**: A visualization of model performance, including confusion matrix metrics on the test set.

## Results
The project achieved the following test accuracies:
| Model         | Test Accuracy |
|---------------|---------------|
| Custom CNN    |    53%        |
| ResNet50      |   100%        |
| DenseNet121   |    95%        |
| VGG16         |    77%        |

The project successfully implemented a multi-class image classification task, aiming to recognize 20 bird species using deep learning models.
These results highlight the power of fine-tuning pre-trained models, such as DenseNet121 and ResNet50, to achieve high accuracy in real-world image classification problems. The data augmentation techniques further enhanced the model's robustness, making it resilient to variations in lighting, orientation, and other real-world factors.

