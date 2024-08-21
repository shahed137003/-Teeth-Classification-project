# Teeth-Classification-project
## Project Overview
Our goal is to develop a comprehensive teeth classification solution that includes  preprocessing and visualizing dental images, and a robust computer vision model  capable of accurately classifying teeth into 7 distinct categories.

This project aims to classify different types of teeth using machine learning techniques. The model takes input images of teeth and categorizes them into predefined classes. The purpose of this classification is to assist in dental diagnostics and automate the process of identifying tooth types from images.

### Dataset
The dataset consists of labeled images of teeth, classified into various categories (e.g., molars, canines, incisors, premolars, etc.). Each image has an associated label indicating its class.

### Model
The model architecture for teeth classification is built using convolutional neural networks (CNN) with the following layers:

- Convolutional Layers: For feature extraction from teeth images.
- MaxPooling Layers: For reducing the spatial dimensions of the feature maps.
- Fully Connected Layers: For classification into respective categories.
Sample Architecture
- Input: 256x256 RGB image of a tooth
- Layers:
  
  Conv2D (32 filters, 3x3 kernel, ReLU activation)
  
  MaxPooling2D (2x2)
  
  Conv2D (64 filters, 3x3 kernel, ReLU activation)

  MaxPooling2D (2x2)
- Flatten
  
   Dense (64 units, ReLU activation)

   Dense (number of classes, softmax activation)

### Model Training
The model is trained using the Adam or SGD optimizer and sparse_categorical_crossentropy loss function for multi-class classification. The dataset is split into training and testing sets.
